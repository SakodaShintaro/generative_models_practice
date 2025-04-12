import jax
import jax.numpy as jnp

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, ANN205, ERA001, N816


def print_tuple_tree(t, indent=0):
    for i, value in enumerate(t):
        print("  " * indent + str(i))
        if isinstance(value, tuple):
            print_tuple_tree(value, indent + 1)
        else:
            print("  " * (indent + 1) + str(value.shape))


def f1(S, w, z, b, v, k):
    return S * w.mT + S @ z * b.mT + v * k.mT


def f2(S, w, z, b, v, k):
    w = w.squeeze(-1)
    z = z.squeeze(-1)
    b = b.squeeze(-1)
    v = v.squeeze(-1)
    k = k.squeeze(-1)
    return (
        jnp.einsum("hij,hj->hij", S, w)
        + jnp.einsum("hik,hk,hj->hij", S, z, b)
        + jnp.einsum("hi,hj->hij", v, k)
    )


def f_impl(S, sensitivity_mats, w, z, b, v, k):
    prev_S = S
    S = f1(S, w, z, b, v, k)
    sw, sz, sb, sv, sk = sensitivity_mats

    ones = jnp.ones(sw.shape[1])
    identity = jnp.eye(sw.shape[1])

    w = w.squeeze(-1)
    z = z.squeeze(-1)
    b = b.squeeze(-1)
    v = v.squeeze(-1)
    k = k.squeeze(-1)

    def recursive(x):
        return jnp.einsum("hpij,hj->hpij", x, w) + jnp.einsum("hpik,hk,hj->hpij", x, z, b)

    # Update sensitivity matrices
    # x: (HEAD_NUM, HEAD_SIZE, 1)
    # sx: (HEAD_NUM, HEAD_SIZE(w param), HEAD_SIZE(si), HEAD_SIZE(sj))
    sw = recursive(sw) + jnp.einsum("hij,pj->hpij", prev_S, identity)
    sz = recursive(sz) + jnp.einsum("hij,p,hj->hpij", prev_S, ones, b)
    sb = recursive(sb) + jnp.einsum("hik,hk,pj->hpij", prev_S, z, identity)
    sv = recursive(sv) + jnp.einsum("pi,hj->hpij", identity, k)
    sk = recursive(sk) + jnp.einsum("hi,pj->hpij", v, identity)

    sensitivity_mats = (sw, sz, sb, sv, sk)

    return (S, sensitivity_mats)


@jax.custom_vjp
def custum_f(S, sensitivity_mats, z, w, b, v, k):
    return f_impl(S, sensitivity_mats, w, z, b, v, k)


def custum_fwd(S, sensitivity_mats, w, z, b, v, k):
    (S, sensitivity_mats), vjp_func = jax.vjp(f_impl, S, sensitivity_mats, w, z, b, v, k)
    return (S, sensitivity_mats), (sensitivity_mats, w, z, b, v, k)


def custum_bwd(res, g):
    dL_dS, d_sensitivity_mats = g[0], g[1]
    sensitivity_mats, w, z, b, v, k = res

    sw, sz, sb, sv, sk = sensitivity_mats

    # vjp
    vw = jnp.einsum("hij,hpij->hp", dL_dS, sw)[..., jnp.newaxis]
    vz = jnp.einsum("hij,hpij->hp", dL_dS, sz)[..., jnp.newaxis]
    vb = jnp.einsum("hij,hpij->hp", dL_dS, sb)[..., jnp.newaxis]
    vv = jnp.einsum("hij,hpij->hp", dL_dS, sv)[..., jnp.newaxis]
    vk = jnp.einsum("hij,hpij->hp", dL_dS, sk)[..., jnp.newaxis]

    result = (dL_dS, d_sensitivity_mats, vw, vz, vb, vv, vk)

    return result


def compute_loss_bptt(params, curr_S, y):
    sum_loss = 0

    w, z, b, v, k, q = params

    for t in range(w.shape[0]):
        curr_S = f1(curr_S, w[t], z[t], b[t], v[t], k[t])
        curr_pred = curr_S @ q[t]
        curr_loss = jnp.mean((curr_pred - y[t]) ** 2)
        sum_loss += curr_loss

    sum_loss /= w.shape[0]
    return sum_loss


def compute_loss_rtrl(params, curr_S, sensitivity_mats, y):
    w, z, b, v, k, q = params
    curr_S, sensitivity_mats = custum_f(curr_S, sensitivity_mats, w, z, b, v, k)
    curr_pred = curr_S @ q
    curr_loss = jnp.mean((curr_pred - y) ** 2)
    return curr_loss, (curr_S, sensitivity_mats)


if __name__ == "__main__":
    HEAD_NUM = 2
    HEAD_SIZE = 8
    TIMESTEP = 1

    curr_S = jnp.ones((HEAD_NUM, HEAD_SIZE, HEAD_SIZE))
    y = jnp.ones((TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))

    rng = jax.random.PRNGKey(0)
    rng, rng_w, rng_z, rng_b, rng_v, rng_k, rng_q = jax.random.split(rng, 7)
    w = jax.random.normal(rng_w, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    z = jax.random.normal(rng_z, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    b = jax.random.normal(rng_b, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    v = jax.random.normal(rng_v, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    k = jax.random.normal(rng_k, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    q = jax.random.normal(rng_q, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))

    S1 = S2 = jnp.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE))
    for t in range(TIMESTEP):
        S1 = f1(S1, w[t], z[t], b[t], v[t], k[t])
        S2 = f2(S2, w[t], z[t], b[t], v[t], k[t])
        assert jnp.allclose(S1, S2), "S1 and S2 are not equal"

    # BPTT (Backpropagation Through Time)
    params = (w, z, b, v, k, q)
    loss_bptt, grads_bptt = jax.value_and_grad(compute_loss_bptt, argnums=0)(params, curr_S, y)

    # RTRL (Real-Time Recurrent Learning)
    custum_f.defvjp(custum_fwd, custum_bwd)
    sensitivity_w = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_z = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_b = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_v = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_k = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_mats = (sensitivity_w, sensitivity_z, sensitivity_b, sensitivity_v, sensitivity_k)
    loss_rtrl = 0
    grads_rtrl = (
        jnp.zeros_like(w),
        jnp.zeros_like(z),
        jnp.zeros_like(b),
        jnp.zeros_like(v),
        jnp.zeros_like(k),
        jnp.zeros_like(q),
    )
    for t in range(TIMESTEP):
        params = (w[t], z[t], b[t], v[t], k[t], q[t])
        (curr_loss, (curr_S, sensitivity_mats)), curr_grads_rtrl = jax.value_and_grad(
            compute_loss_rtrl, argnums=0, has_aux=True
        )(params, curr_S, sensitivity_mats, y[t])
        loss_rtrl += curr_loss / TIMESTEP
        grads_rtrl = tuple(
            grads_rtrl[i] + curr_grads_rtrl[i] / TIMESTEP for i in range(len(grads_rtrl))
        )

    print(f"{loss_bptt.item()}")
    print(f"{loss_rtrl.item()}")
    diff_loss = jnp.abs(loss_bptt - loss_rtrl)
    jnp.allclose(loss_bptt, loss_rtrl), "Losses are not equal"

    # Check if the gradients are equal
    for i, (grad_bptt, grad_rtrl) in enumerate(zip(grads_bptt, grads_rtrl)):
        close = jnp.allclose(grad_bptt, grad_rtrl)
        print(f"{i=}, {close=}")
        if not close:
            print("Gradients are not equal")
            print(f"{grad_bptt.shape=}, {grad_rtrl.shape=}")
            print(f"{grad_bptt=}")
            print(f"{grad_rtrl=}")
