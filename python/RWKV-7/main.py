import jax
import jax.numpy as jnp

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, ANN205, ERA001, N816


def f(S, w, z, b, v, k):
    S = S * w.mT - S @ z * b.mT + v * k.mT
    return S


def f_impl(S, sensitivity_mats, w, z, b, v, k):
    S = S * w.mT - S @ z * b.mT + v * k.mT
    return (S, sensitivity_mats)


@jax.custom_vjp
def custum_f(S, sensitivity_mats, z, w, b, v, k):
    return f_impl(S, sensitivity_mats, w, z, b, v, k)


def custum_fwd(S, sensitivity_mats, w, z, b, v, k):
    f_out, vjp_func = jax.vjp(f_impl, S, sensitivity_mats, w, z, b, v, k)
    return f_out, (vjp_func, f_out)


def custum_bwd(res, g):
    vjp_func, f_out = res
    return vjp_func(g)


def compute_loss(params, init_S):
    w, z, b, v, k = params
    curr_S = init_S

    for i in range(w.shape[0]):
        curr_S = f(curr_S, w[i], z[i], b[i], v[i], k[i])

    return jnp.mean(curr_S)


def compute_loss2(curr_S, sensitivity_mats, w, z, b, v, k):
    curr_S, sensitivity_mats = custum_f(curr_S, sensitivity_mats, w, z, b, v, k)
    return jnp.mean(curr_S)


if __name__ == "__main__":
    HEAD_NUM = 2
    HEAD_SIZE = 8
    TIMESTEP = 10

    rng = jax.random.PRNGKey(0)
    curr_S = jax.random.normal(rng, (HEAD_NUM, HEAD_SIZE, HEAD_SIZE))
    w = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    z = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    b = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    v = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    k = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))

    params = (w, z, b, v, k)

    loss_val, grads = jax.value_and_grad(compute_loss, argnums=0)(params, curr_S)

    grad_w, grad_z, grad_b, grad_v, grad_k = grads
    print(f"Loss: {loss_val}")
    print(f"w: {grad_w.shape=}")
    print(f"z: {grad_z.shape=}")
    print(f"b: {grad_b.shape=}")
    print(f"v: {grad_v.shape=}")
    print(f"k: {grad_k.shape=}")

    custum_f.defvjp(custum_fwd, custum_bwd)

    sensitivity_w = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_z = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_b = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_v = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_k = jax.numpy.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE))
    sensitivity_mats = (sensitivity_w, sensitivity_z, sensitivity_b, sensitivity_v, sensitivity_k)
    carry = (curr_S, sensitivity_mats)

    for i in range(w.shape[0]):
        print(f"{i=}")
        custum_f(curr_S, sensitivity_mats, w[i], z[i], b[i], v[i], k[i])
        grad = jax.grad(compute_loss2)(curr_S, sensitivity_mats, w[i], z[i], b[i], v[i], k[i])
        exit(0)
