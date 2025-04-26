from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state


class RNNModel(nn.Module):
    hidden_size: int
    output_dim: int

    @nn.compact
    def __call__(self, carry, x):
        rnn = nn.SimpleCell(features=self.hidden_size)
        carry, hidden = rnn(carry, x)
        y = nn.Dense(features=self.output_dim)(hidden)
        return carry, y


def bptt_loss_fn(params, model, x, y):
    # x shape: [seq_len, batch_size, input_dim]
    batch_size = x.shape[1]

    # 最初のキャリー状態を初期化
    hidden_size = params["params"]["SimpleCell_0"]["h"]["kernel"].shape[0]
    initial_carry = jnp.zeros((batch_size, hidden_size))

    # scanでシーケンス全体に対して処理を適用
    step_fn = partial(model, params)
    _, y_pred = jax.lax.scan(step_fn, initial_carry, x)

    # MSE損失を計算
    loss = jnp.mean((y_pred - y) ** 2)
    return loss


@jax.jit
def bptt_grads(state, batch_x, batch_y):
    loss, grads = jax.value_and_grad(bptt_loss_fn)(
        state.params, state.apply_fn, batch_x, batch_y
    )
    return loss, grads


@jax.jit
def dtanh(x):
    return 1 - jnp.tanh(x) ** 2


def rtrl_grads(state, batch_x, batch_y):
    params = state.params
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]
    batch_size = batch_x.shape[1]
    hidden_size = params["params"]["SimpleCell_0"]["h"]["kernel"].shape[0]

    R = params["params"]["SimpleCell_0"]["h"]["kernel"]
    W = params["params"]["SimpleCell_0"]["i"]["kernel"]
    B = params["params"]["SimpleCell_0"]["i"]["bias"]

    Wy = params["params"]["Dense_0"]["kernel"]
    Wy_b = params["params"]["Dense_0"]["bias"]

    input_size = W.shape[0]

    # 最初のキャリー状態を初期化
    curr_s = jnp.zeros((batch_size, hidden_size))
    curr_h = jnp.tanh(curr_s)

    # RNNの更新式は
    #   h(t+1) = \tanh(W x + B + R h(t))

    # 感度行列 : 隠れ状態に対するパラメータの偏微分を保持 (batch, hidden_size, n_params)
    S_R = jnp.zeros((batch_size, hidden_size, *R.shape))
    S_W = jnp.zeros((batch_size, hidden_size, *W.shape))
    S_B = jnp.zeros((batch_size, hidden_size, *B.shape))
    print(f"{S_R.shape=}")
    print(f"{S_W.shape=}")
    print(f"{S_B.shape=}")

    grad_structured = unravel_fn(jnp.zeros(n_params))
    seq_len = batch_x.shape[0]

    loss = 0.0

    for t in range(seq_len):
        print(f"{t=}")
        curr_x = batch_x[t]
        curr_y_ref = batch_y[t]

        prev_s = curr_s
        prev_h = curr_h

        curr_s = jnp.dot(curr_x, W) + B + jnp.dot(curr_h, R)
        curr_h = jnp.tanh(curr_s)
        curr_y_prd = jnp.dot(curr_h, Wy) + Wy_b
        curr_loss = jnp.mean((curr_y_prd - curr_y_ref) ** 2)
        loss += curr_loss

        dl_dy = 2 * (curr_y_prd - curr_y_ref) / (batch_size * seq_len * input_size)
        grad_structured["params"]["Dense_0"]["kernel"] += jnp.einsum(
            "bd,bh->hd", dl_dy, curr_h
        )
        dl_dh = jnp.dot(dl_dy, Wy.T)
        dl_ds = dl_dh * (1 - curr_h**2)

        # 感度行列の更新
        d_s = dtanh(prev_s)
        eye = jnp.eye(hidden_size)
        S_W = jnp.einsum("bi,jk->bkij", curr_x, eye) + jnp.einsum(
            "nk,bn,bnij->bkij", R, d_s, S_W
        )

        S_B = eye[None, :, :] + jnp.einsum("nk,bn,bnj->bkj", R, d_s, S_B)

        S_R = jnp.einsum("bi,jk->bkij", prev_h, eye) + jnp.einsum(
            "nk,bn,bnij->bkij", R, d_s, S_R
        )

        curr_grad_W = jnp.einsum("bh,bhij->bij", dl_ds, S_W)
        curr_grad_B = jnp.einsum("bh,bhj->bj", dl_ds, S_B)
        curr_grad_R = jnp.einsum("bh,bhij->bij", dl_ds, S_R)

        grad_structured["params"]["SimpleCell_0"]["i"]["kernel"] += curr_grad_W
        grad_structured["params"]["SimpleCell_0"]["i"]["bias"] += curr_grad_B
        grad_structured["params"]["SimpleCell_0"]["h"]["kernel"] += curr_grad_R

    # 平均損失を返す
    loss = loss / seq_len

    return loss, grad_structured


if __name__ == "__main__":
    # グローバルなシードを設定
    SEED = 0
    np.random.seed(SEED)
    rng_key = jax.random.PRNGKey(SEED)

    seq_len = 30
    batch_size = 1
    input_dim = 5
    hidden_size = 20
    num_steps = 100

    # ダミーなランダムデータとして入力データとターゲットデータを作成
    data_x = np.random.randn(seq_len, batch_size, input_dim).astype(np.float32)
    data_y = np.random.randn(seq_len, batch_size, input_dim).astype(np.float32)

    # JAXのPRNGキーを分割して使用
    rng_key, init_key = jax.random.split(rng_key)

    # モデルの初期化
    model = RNNModel(hidden_size=hidden_size, output_dim=input_dim)
    params = model.init(
        init_key,
        jnp.ones((batch_size, hidden_size)),
        jnp.ones((batch_size, input_dim)),
    )

    # 訓練状態の作成
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
    )

    loss_bptt, grads_bptt = bptt_grads(state, data_x, data_y)
    loss_rtrl, grads_rtrl = rtrl_grads(state, data_x, data_y)

    print("BPTT Loss:", loss_bptt)
    print("RTRL Loss:", loss_rtrl)

    loss_diff = jnp.abs(loss_bptt - loss_rtrl)
    print(f"{loss_diff=}")

    grads_bptt_Wy = grads_bptt["params"]["Dense_0"]["kernel"]
    grads_rtrl_Wy = grads_rtrl["params"]["Dense_0"]["kernel"]
    grads_diff_Wy = jnp.abs(grads_bptt_Wy - grads_rtrl_Wy)
    grads_diff_Wy = jnp.mean(grads_diff_Wy)
    print(f"{grads_diff_Wy=}")

    grads_bptt_W = grads_bptt["params"]["SimpleCell_0"]["i"]["kernel"]
    grads_rtrl_W = grads_rtrl["params"]["SimpleCell_0"]["i"]["kernel"]
    grads_diff_W = jnp.abs(grads_bptt_W - grads_rtrl_W)
    grads_diff_W = jnp.mean(grads_diff_W)
    print(f"{grads_diff_W=}")

    grads_bptt_B = grads_bptt["params"]["SimpleCell_0"]["i"]["bias"]
    grads_rtrl_B = grads_rtrl["params"]["SimpleCell_0"]["i"]["bias"]
    grads_diff_B = jnp.abs(grads_bptt_B - grads_rtrl_B)
    grads_diff_B = jnp.mean(grads_diff_B)
    print(f"{grads_diff_B=}")

    grads_bptt_R = grads_bptt["params"]["SimpleCell_0"]["h"]["kernel"]
    grads_rtrl_R = grads_rtrl["params"]["SimpleCell_0"]["h"]["kernel"]
    grads_diff_R = jnp.abs(grads_bptt_R - grads_rtrl_R)
    grads_diff_R = jnp.mean(grads_diff_R)
    print(f"{grads_diff_R=}")
