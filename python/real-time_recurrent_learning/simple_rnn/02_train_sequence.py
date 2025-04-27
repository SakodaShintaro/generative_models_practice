from copy import deepcopy
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

SEQ_LEN = 10
INPUT_DIM = 10


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


class SimpleModel(nn.Module):
    features_hidden: int
    features_out: int

    def setup(self):
        self.simple_cell = nn.SimpleCell(features=self.features_hidden)
        self.dense_out = nn.Dense(features=self.features_out)

    def __call__(self, carry, x_t):
        carry, y_t = self.simple_cell(carry, x_t)
        y_t = self.dense_out(y_t)
        return carry, y_t

    def initialize_carry(self, batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))


class RtrlCell(nn.Module):
    hidden_size: int
    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()
    residual: bool = False

    @staticmethod
    def dtanh(x):
        return 1 - jnp.tanh(x) ** 2

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
        d_s = self.dtanh(prev_s)
        eye = jnp.eye(hidden_size)
        S_W = jnp.einsum("bi,jk->bkij", x, eye) + jnp.einsum("nk,bn,bnij->bkij", R, d_s, S_W)
        S_B = eye[None, :, :] + jnp.einsum("nk,bn,bnj->bkj", R, d_s, S_B)
        S_R = jnp.einsum("bi,jk->bkij", prev_h, eye) + jnp.einsum("nk,bn,bnij->bkij", R, d_s, S_R)
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
    def initialize_carry(batch_size, d_rec, d_input):
        hidden_init = jnp.zeros((batch_size, d_rec))
        S_R = jnp.zeros((batch_size, d_rec, d_rec, d_rec))
        S_W = jnp.zeros((batch_size, d_rec, d_rec, d_input))
        S_B = jnp.zeros((batch_size, d_rec, d_input))
        memory_grad_init = (S_R, S_W, S_B)
        return (hidden_init, memory_grad_init)


class RtrlModel(nn.Module):
    features_hidden: int
    features_out: int

    def setup(self):
        self.simple_cell = RtrlCell(hidden_size=self.features_hidden)
        self.dense_out = nn.Dense(features=self.features_out)

    def __call__(self, carry, x_t):
        carry, y_t = self.simple_cell(carry, x_t)
        y_t = self.dense_out(y_t)
        return carry, y_t

    @staticmethod
    def initialize_carry(batch_size, d_rec, d_inp):
        hidden_init = jnp.zeros((batch_size, d_rec))
        S_R = jnp.zeros((batch_size, d_rec, d_rec, d_rec))
        S_W = jnp.zeros((batch_size, d_rec, d_inp, d_rec))
        S_B = jnp.zeros((batch_size, d_rec, d_rec))
        memory_grad_init = (S_R, S_W, S_B)
        return (hidden_init, memory_grad_init)


def bptt_loss_fn(params, model, batch_carry, batch_x, batch_y, batch_mask):
    # scanでシーケンス全体に対して処理を適用
    step_fn = partial(model, params)
    _, y_pred = jax.lax.scan(step_fn, batch_carry, batch_x)

    # CrossEntropyを計算
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(y_pred, batch_y)
    loss = loss * batch_mask
    loss = jnp.mean(loss)
    return loss


@jax.jit
def bptt_grads(state, batch_carry, batch_x, batch_y, batch_mask):
    loss, grads = jax.value_and_grad(bptt_loss_fn)(
        state.params, state.apply_fn, batch_carry, batch_x, batch_y, batch_mask
    )
    return loss, grads


def make_data(batch_size):
    # seq_len // 2の長さのシーケンスをコピーするタスク
    # INPUT_DIM - 1はコピー中のxの入力に使うので、INPUT_DIM - 2までで生成
    data = np.random.randint(low=0, high=INPUT_DIM - 2, size=(SEQ_LEN // 2, batch_size))
    data = np.concatenate([data, data], axis=0)

    mask = np.ones((SEQ_LEN, batch_size))
    mask[: SEQ_LEN // 2] = 0
    mask = jnp.array(mask)

    batch_x = deepcopy(data)
    batch_y = deepcopy(data)

    batch_x[SEQ_LEN // 2 :] = INPUT_DIM - 1  # コピー中のxの入力
    batch_x = jax.nn.one_hot(batch_x, INPUT_DIM)
    batch_x = jnp.array(batch_x)
    batch_y = jnp.array(batch_y)
    return batch_x, batch_y, mask


if __name__ == "__main__":
    # グローバルなシードを設定
    SEED = 0
    np.random.seed(SEED)
    rng_key = jax.random.PRNGKey(SEED)

    batch_size = 16
    hidden_size = INPUT_DIM - 1

    # JAXのPRNGキーを分割して使用
    rng_key, init_key = jax.random.split(rng_key)

    # モデルの初期化
    model_bptt = SimpleModel(features_hidden=hidden_size, features_out=INPUT_DIM)
    model_rtrl = RtrlModel(features_hidden=hidden_size, features_out=INPUT_DIM)

    params_bptt = model_bptt.init(
        init_key,
        model_bptt.initialize_carry(batch_size, hidden_size),
        jnp.ones((batch_size, INPUT_DIM)),
    )
    params_rtrl = model_rtrl.init(
        init_key,
        model_rtrl.initialize_carry(batch_size, hidden_size, INPUT_DIM),
        jnp.ones((batch_size, INPUT_DIM)),
    )

    # 訓練状態の作成
    state_bptt = train_state.TrainState.create(
        apply_fn=model_bptt.apply, params=params_bptt, tx=optax.adam(1e-2)
    )
    state_rtrl = train_state.TrainState.create(
        apply_fn=model_rtrl.apply, params=params_rtrl, tx=optax.adam(1e-2)
    )

    STEP_NUM = 1000

    # BPTT
    print("BPTT")
    loss_list = []
    for i in range(1, STEP_NUM + 1):
        batch_x, batch_y, mask = make_data(batch_size)
        carry = model_bptt.initialize_carry(batch_size, hidden_size)
        loss, grads = bptt_grads(state_bptt, carry, batch_x, batch_y, mask)
        state_bptt = state_bptt.apply_gradients(grads=grads)
        loss_list.append(loss.item())
        if i % (STEP_NUM / 10) == 0:
            loss_avg = sum(loss_list) / len(loss_list)
            loss_std = np.std(loss_list)
            loss_list = []
            print(f"{i:08d} {loss_avg:.5f} {loss_std:.5f}")

    # 確認
    initial_carry = model_bptt.initialize_carry(batch_size, hidden_size)
    step_fn = partial(state_bptt.apply_fn, state_bptt.params)
    _, y_pred = jax.lax.scan(step_fn, initial_carry, batch_x)
    y_pred_int = jnp.argmax(y_pred, axis=-1)
    acc = jnp.mean(jnp.equal(y_pred_int[SEQ_LEN // 2 :], batch_y[SEQ_LEN // 2 :])).item() * 100
    print(f"{acc=:5.1f}%")

    # RTRL
    print("RTRL")

    @jax.jit
    def step_loss_fn(params, carry, x_t, y_t_ref):
        carry, out = state_rtrl.apply_fn(params, carry, x_t)
        curr_loss = optax.losses.softmax_cross_entropy_with_integer_labels(out, y_t_ref)
        curr_loss = jnp.mean(curr_loss) / SEQ_LEN
        return curr_loss, carry

    loss_list = []
    for i in range(1, STEP_NUM + 1):
        batch_x, batch_y, mask = make_data(batch_size)
        loss = 0.0
        carry = model_rtrl.initialize_carry(batch_size, hidden_size, INPUT_DIM)
        for j in range(SEQ_LEN):
            x_t = batch_x[j]
            y_t_ref = batch_y[j]
            if j < SEQ_LEN // 2:
                carry, y_t = model_rtrl.apply(state_rtrl.params, carry, x_t)
            else:
                (curr_loss, carry), grads = jax.value_and_grad(step_loss_fn, has_aux=True)(
                    state_rtrl.params, carry, x_t, y_t_ref
                )
                state_rtrl = state_rtrl.apply_gradients(grads=grads)
                loss += curr_loss.item()

        loss_list.append(loss)
        if i % (STEP_NUM / 10) == 0:
            loss_avg = sum(loss_list) / len(loss_list)
            loss_std = np.std(loss_list)
            loss_list = []
            print(f"{i:08d} {loss_avg:.5f} {loss_std:.5f}")

    # 確認
    batch_x, batch_y, mask = make_data(batch_size)
    carry = model_rtrl.initialize_carry(batch_size, hidden_size, INPUT_DIM)
    equal_num = 0
    for j in range(SEQ_LEN):
        x_t = batch_x[j]
        y_t_ref = batch_y[j]
        carry, y_t = model_rtrl.apply(state_rtrl.params, carry, x_t)
        if j >= SEQ_LEN // 2:
            y_t_int = jnp.argmax(y_t, axis=-1)
            equal_num += jnp.sum(jnp.equal(y_t_int, y_t_ref)).item()
    acc = equal_num / (SEQ_LEN // 2 * batch_size) * 100
    print(f"{acc=:5.1f}%")
