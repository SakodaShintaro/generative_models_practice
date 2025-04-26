from functools import partial

import flax
import flax.linen
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen import initializers
from flax.linen.linear import Dense, default_kernel_init
from flax.training import train_state
from flax.typing import Dtype, Initializer

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, ANN205, ERA001

# documents for custom_vjp
# jax) https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
# flax) https://flax.readthedocs.io/en/v0.5.3/_autosummary/flax.linen.vjp.html


def print_dict_tree(d, indent=0):
    for key, value in d.items():
        print("  " * indent + str(key))
        if isinstance(value, dict):
            print_dict_tree(value, indent + 1)


def print_tuple_tree(t, indent=0):
    for i, value in enumerate(t):
        print("  " * indent + str(i))
        if isinstance(value, tuple):
            print_tuple_tree(value, indent + 1)
        else:
            print("  " * (indent + 1) + str(value.shape))


@jax.jit
def dtanh(x):
    return 1 - jnp.tanh(x) ** 2


class RtrlCell(nn.Module):
    hidden_size: int
    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()
    residual: bool = False

    def setup(self):
        self.dense_h = Dense(
            features=self.hidden_size,
            use_bias=False,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            name="h",
        )
        self.dense_i = Dense(
            features=self.hidden_size,
            use_bias=True,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="i",
        )

    def forward(self, carry, x):
        prev_h, prev_sensitivity_matrices = carry
        prev_s = jnp.arctanh(prev_h)

        # update hidden state
        curr_s = self.dense_i(x) + self.dense_h(prev_h)
        curr_h = flax.linen.activation.tanh(curr_s)

        # update sensitivity matrices
        R = self.variables["params"]["h"]["kernel"]
        (S_R, S_W, S_B) = prev_sensitivity_matrices
        d_s = dtanh(prev_s)
        eye = jnp.eye(hidden_size)
        S_W = jnp.einsum("bi,jk->bkij", x, eye) + jnp.einsum(
            "nk,bn,bnij->bkij", R, d_s, S_W
        )
        S_B = eye[None, :, :] + jnp.einsum("nk,bn,bnj->bkj", R, d_s, S_B)
        S_R = jnp.einsum("bi,jk->bkij", prev_h, eye) + jnp.einsum(
            "nk,bn,bnij->bkij", R, d_s, S_R
        )
        S_W = jax.lax.stop_gradient(S_W)
        S_B = jax.lax.stop_gradient(S_B)
        S_R = jax.lax.stop_gradient(S_R)
        curr_sensitivity_matrices = (S_R, S_W, S_B)

        return (curr_h, curr_sensitivity_matrices), curr_h

    def __call__(self, carry, x_t):
        def fwd(mdl, carry, x_t):
            f_out, vjp_func = nn.vjp(RtrlCell.forward, mdl, carry, x_t)
            # f_out = (curr_h, curr_sensitivity_matrices), curr_out
            # we need curr_h and curr_sensitivity_matrices for backward pass
            return f_out, (vjp_func, f_out[0])

        def bwd(residuals, y_t):
            vjp_func, (curr_h, sensitivity_matrices) = residuals
            params_t, *inputs_t = vjp_func(y_t)
            params_t1 = flax.core.unfreeze(params_t)
            (S_R, S_W, S_B) = sensitivity_matrices
            dl_dy = y_t[1]
            dl_ds = dl_dy * (1 - curr_h**2)
            curr_grad_W = jnp.einsum("bh,bhij->bij", dl_ds, S_W)
            curr_grad_B = jnp.einsum("bh,bhj->bj", dl_ds, S_B)
            curr_grad_R = jnp.einsum("bh,bhij->bij", dl_ds, S_R)
            params_t1["params"]["i"]["kernel"] = jnp.sum(curr_grad_W, 0)
            params_t1["params"]["i"]["bias"] = jnp.sum(curr_grad_B, 0)
            params_t1["params"]["h"]["kernel"] = jnp.sum(curr_grad_R, 0)
            return (params_t1, *inputs_t)

        vjp_fn = nn.custom_vjp(RtrlCell.forward, forward_fn=fwd, backward_fn=bwd)
        carry, hidden = vjp_fn(self, carry, x_t)
        return carry, hidden

    @staticmethod
    def initialize_state(batch_size, d_rec, d_input):
        hidden_init = jnp.zeros((batch_size, d_rec))
        S_R = jnp.zeros((batch_size, d_rec, d_rec, d_rec))
        S_W = jnp.zeros((batch_size, d_rec, d_rec, d_input))
        S_B = jnp.zeros((batch_size, d_rec, d_input))
        memory_grad_init = (S_R, S_W, S_B)
        return (hidden_init, memory_grad_init)


def bptt_loss_fn(params, model, x, y):
    # x shape: [seq_len, batch_size, input_dim]
    batch_size = x.shape[1]

    # 最初のキャリー状態を初期化
    hidden_size = params["params"]["h"]["kernel"].shape[0]
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


def rtrl_grads1(state, batch_x, batch_y):
    params = state.params
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]
    batch_size = batch_x.shape[1]
    hidden_size = params["params"]["h"]["kernel"].shape[0]

    R = params["params"]["h"]["kernel"]
    W = params["params"]["i"]["kernel"]
    B = params["params"]["i"]["bias"]

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

    grad_structured = unravel_fn(jnp.zeros(n_params))
    seq_len = batch_x.shape[0]

    loss = 0.0

    for t in range(seq_len):
        curr_x = batch_x[t]
        curr_y_ref = batch_y[t]

        prev_s = curr_s
        prev_h = curr_h

        curr_s = jnp.dot(curr_x, W) + B + jnp.dot(curr_h, R)
        curr_h = jnp.tanh(curr_s)
        curr_y_prd = curr_h
        curr_loss = jnp.mean((curr_y_prd - curr_y_ref) ** 2) / seq_len
        loss += curr_loss

        dl_dy = 2 * (curr_y_prd - curr_y_ref) / (batch_size * seq_len * input_size)
        dl_dh = dl_dy
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

        curr_grad_W = jnp.sum(curr_grad_W, 0)
        curr_grad_B = jnp.sum(curr_grad_B, 0)
        curr_grad_R = jnp.sum(curr_grad_R, 0)

        grad_structured["params"]["i"]["kernel"] += curr_grad_W
        grad_structured["params"]["i"]["bias"] += curr_grad_B
        grad_structured["params"]["h"]["kernel"] += curr_grad_R

    return loss, grad_structured


def rtrl_grads2(state, batch_x, batch_y):
    params = state.params
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]
    grads_flat = jnp.zeros(n_params)
    seq_len = batch_x.shape[0]

    # 最初のキャリー状態を初期化
    hidden_size = params["params"]["h"]["kernel"].shape[0]
    carry = RtrlCell.initialize_state(batch_size, hidden_size, hidden_size)

    loss = 0.0

    def step_loss_fn(params, carry, x_t, y_t_ref):
        carry, out = state.apply_fn(params, carry, x_t)
        curr_loss = jnp.mean((out - y_t_ref) ** 2) / seq_len
        return curr_loss, carry

    for t in range(seq_len):
        x_t = batch_x[t]
        y_t_ref = batch_y[t]
        (curr_loss, carry), grads = jax.value_and_grad(step_loss_fn, has_aux=True)(
            params, carry, x_t, y_t_ref
        )
        curr_flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
        grads_flat += curr_flat_grads
        loss += curr_loss

    return loss, unravel_fn(grads_flat)


if __name__ == "__main__":
    # グローバルなシードを設定
    SEED = 0
    np.random.seed(SEED)
    rng_key = jax.random.PRNGKey(SEED)

    seq_len = 30
    batch_size = 2
    hidden_size = 20
    num_steps = 100

    # ダミーなランダムデータとして入力データとターゲットデータを作成
    data_x = np.random.randn(seq_len, batch_size, hidden_size).astype(np.float32)
    data_y = np.random.randn(seq_len, batch_size, hidden_size).astype(np.float32)

    # JAXのPRNGキーを分割して使用
    rng_key, init_key = jax.random.split(rng_key)

    # モデルの初期化
    model_cell_bptt = nn.SimpleCell(features=hidden_size)
    model_cell_rtrl = RtrlCell(hidden_size=hidden_size)

    params_bptt = model_cell_bptt.init(
        init_key,
        jnp.ones((batch_size, hidden_size)),
        jnp.ones((batch_size, hidden_size)),
    )
    params_rtrl = model_cell_rtrl.init(
        init_key,
        model_cell_rtrl.initialize_state(batch_size, hidden_size, hidden_size),
        jnp.ones((batch_size, hidden_size)),
    )

    # 訓練状態の作成
    state_bptt = train_state.TrainState.create(
        apply_fn=model_cell_bptt.apply, params=params_bptt, tx=optax.adam(1e-3)
    )
    state_rtrl = train_state.TrainState.create(
        apply_fn=model_cell_rtrl.apply, params=params_rtrl, tx=optax.adam(1e-3)
    )

    # copy params from bptt to rtrl
    params_rtrl["params"] = params_bptt["params"]

    loss_bptt, grads_bptt = bptt_grads(state_bptt, data_x, data_y)
    loss_rtrl1, grads_rtrl1 = rtrl_grads1(state_rtrl, data_x, data_y)
    loss_rtrl2, grads_rtrl2 = rtrl_grads2(state_rtrl, data_x, data_y)

    print("BPTT Loss:", loss_bptt)
    print("RTRL Loss1:", loss_rtrl1)
    print("RTRL Loss2:", loss_rtrl2)

    loss_diff1 = jnp.abs(loss_bptt - loss_rtrl1)
    print(f"{loss_diff1=}")
    loss_diff2 = jnp.abs(loss_bptt - loss_rtrl2)
    print(f"{loss_diff2=}")

    def compare_grads(grads_rtrl):
        grads_bptt_W = grads_bptt["params"]["i"]["kernel"]
        grads_rtrl_W = grads_rtrl["params"]["i"]["kernel"]
        grads_diff_W = jnp.abs(grads_bptt_W - grads_rtrl_W)
        grads_diff_W = jnp.mean(grads_diff_W)
        print(f"{grads_diff_W=}")

        grads_bptt_B = grads_bptt["params"]["i"]["bias"]
        grads_rtrl_B = grads_rtrl["params"]["i"]["bias"]
        grads_diff_B = jnp.abs(grads_bptt_B - grads_rtrl_B)
        grads_diff_B = jnp.mean(grads_diff_B)
        print(f"{grads_diff_B=}")

        grads_bptt_R = grads_bptt["params"]["h"]["kernel"]
        grads_rtrl_R = grads_rtrl["params"]["h"]["kernel"]
        grads_diff_R = jnp.abs(grads_bptt_R - grads_rtrl_R)
        grads_diff_R = jnp.mean(grads_diff_R)
        print(f"{grads_diff_R=}")

    print("Grads1:")
    compare_grads(grads_rtrl1)
    print("Grads2:")
    compare_grads(grads_rtrl2)
