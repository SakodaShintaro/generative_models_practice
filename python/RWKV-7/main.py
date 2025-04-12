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
    return (
        jnp.einsum("hij,hj->hij", S, w.squeeze(-1))
        + jnp.einsum("hik,hk,hj->hij", S, z.squeeze(-1), b.squeeze(-1))
        + jnp.einsum("hi,hj->hij", v.squeeze(-1), k.squeeze(-1))
    )


def f_impl(S, sensitivity_mats, w, z, b, v, k):
    S = S * w.mT + S @ z * b.mT + v * k.mT
    return (S, sensitivity_mats)


@jax.custom_vjp
def custum_f(S, sensitivity_mats, z, w, b, v, k):
    return f_impl(S, sensitivity_mats, w, z, b, v, k)


def custum_fwd(S, sensitivity_mats, w, z, b, v, k):
    prev_S = S
    (S, sensitivity_mats), vjp_func = jax.vjp(f_impl, S, sensitivity_mats, w, z, b, v, k)
    return (S, sensitivity_mats), (vjp_func, prev_S, sensitivity_mats, w, z, b, v, k)


def custum_bwd(res, g):
    dS, d_sensitivity_mats = g[0], g[1]
    vjp_func, prev_S, sensitivity_mats, w, z, b, v, k = res

    sw, sz, sb, sv, sk = sensitivity_mats

    # vjp
    vw = jnp.einsum("hij,hkij->hk", dS, sw)[..., jnp.newaxis]
    vz = jnp.einsum("hij,hkij->hk", dS, sz)[..., jnp.newaxis]
    vb = jnp.einsum("hij,hkij->hk", dS, sb)[..., jnp.newaxis]
    vv = jnp.einsum("hij,hkij->hk", dS, sv)[..., jnp.newaxis]
    vk = jnp.einsum("hij,hkij->hk", dS, sk)[..., jnp.newaxis]

    result = (dS, d_sensitivity_mats, vw, vz, vb, vv, vk)

    return result


def compute_loss(params, init_S, y):
    w, z, b, v, k, q = params
    curr_S = init_S

    sum_loss = 0

    for i in range(w.shape[0]):
        curr_S = f1(curr_S, w[i], z[i], b[i], v[i], k[i])
        curr_pred = curr_S @ q[i]
        curr_loss = jnp.mean((curr_pred - y[i]) ** 2)
        sum_loss += curr_loss

    sum_loss /= w.shape[0]
    return sum_loss


def compute_loss2(curr_S, sensitivity_mats, w, z, b, v, k):
    curr_S, sensitivity_mats = custum_f(curr_S, sensitivity_mats, w, z, b, v, k)
    return jnp.mean(curr_S)


if __name__ == "__main__":
    HEAD_NUM = 2
    HEAD_SIZE = 8
    TIMESTEP = 10

    curr_S = jnp.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE))
    y = jnp.ones((TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))

    rng = jax.random.PRNGKey(0)
    rng, rng_w, rng_z, rng_b, rng_v, rng_k, rng_q = jax.random.split(rng, 7)
    w = jax.random.normal(rng_w, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    z = jax.random.normal(rng_z, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    b = jax.random.normal(rng_b, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    v = jax.random.normal(rng_v, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    k = jax.random.normal(rng_k, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    q = jax.random.normal(rng_q, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))

    S1 = f1(curr_S, w[0], z[0], b[0], v[0], k[0])
    S2 = f2(curr_S, w[0], z[0], b[0], v[0], k[0])
    assert jnp.allclose(S1, S2), "S1 and S2 are not equal"

    # BPTT (Backpropagation Through Time)
    params = (w, z, b, v, k, q)
    loss_val, grads = jax.value_and_grad(compute_loss, argnums=0)(params, curr_S, y)
    grad_w, grad_z, grad_b, grad_v, grad_k, grad_q = grads
    print(f"Loss: {loss_val}")
    print(f"w: {grad_w.shape=}")
    print(f"z: {grad_z.shape=}")
    print(f"b: {grad_b.shape=}")
    print(f"v: {grad_v.shape=}")
    print(f"k: {grad_k.shape=}")
    print(f"q: {grad_q.shape=}")

    # RTRL (Real-Time Recurrent Learning)
    custum_f.defvjp(custum_fwd, custum_bwd)
    sensitivity_w = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_z = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_b = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_v = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_k = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_mats = (sensitivity_w, sensitivity_z, sensitivity_b, sensitivity_v, sensitivity_k)
    for i in range(w.shape[0]):
        custum_f(curr_S, sensitivity_mats, w[i], z[i], b[i], v[i], k[i])
        grad = jax.grad(compute_loss2)(curr_S, sensitivity_mats, w[i], z[i], b[i], v[i], k[i])
