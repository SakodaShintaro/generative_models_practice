import torch
import torch.nn.functional as F

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, ANN205, ERA001, N816


def print_tuple_tree(t, indent=0):
    for i, value in enumerate(t):
        print("  " * indent + str(i))
        if isinstance(value, tuple):
            print_tuple_tree(value, indent + 1)
        else:
            print("  " * (indent + 1) + str(value.shape))


def f1(S, w, z, b, v, k):
    return S * w.transpose(-1, -2) + S @ z * b.transpose(-1, -2) + v @ k.transpose(-1, -2)


def f2(S, w, z, b, v, k):
    w = w.squeeze(-1)
    z = z.squeeze(-1)
    b = b.squeeze(-1)
    v = v.squeeze(-1)
    k = k.squeeze(-1)
    return (
        torch.einsum("hij,hj->hij", S, w)
        + torch.einsum("hik,hk,hj->hij", S, z, b)
        + torch.einsum("hi,hj->hij", v, k)
    )


def f_impl(S, sensitivity_mats, w, z, b, v, k):
    prev_S = S
    S = f1(S, w, z, b, v, k)
    sw, sz, sb, sv, sk = sensitivity_mats
    identity = torch.eye(sw.shape[1], device=S.device)

    w = w.squeeze(-1)
    z = z.squeeze(-1)
    b = b.squeeze(-1)
    v = v.squeeze(-1)
    k = k.squeeze(-1)

    def recursive(x):
        return torch.einsum("hpij,hj->hpij", x, w) + torch.einsum("hpik,hk,hj->hpij", x, z, b)

    sw = recursive(sw) + torch.einsum("hij,pj->hpij", prev_S, identity)
    sz = recursive(sz) + torch.einsum("hik,pk,hj->hpij", prev_S, identity, b)
    sb = recursive(sb) + torch.einsum("hik,hk,pj->hpij", prev_S, z, identity)
    sv = recursive(sv) + torch.einsum("pi,hj->hpij", identity, k)
    sk = recursive(sk) + torch.einsum("hi,pj->hpij", v, identity)

    sensitivity_mats = (sw, sz, sb, sv, sk)
    return S, sensitivity_mats


class CustomF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, sensitivity_mats, w, z, b, v, k):
        S_out, sensitivity_out = f_impl(S, sensitivity_mats, w, z, b, v, k)
        ctx.save_for_backward(S_out, *sensitivity_out)
        return S_out, sensitivity_out

    @staticmethod
    def backward(ctx, grad_S, grad_sens):
        S_out, *sensitivity_mats = ctx.saved_tensors
        sw, sz, sb, sv, sk = sensitivity_mats

        vw = torch.einsum("hij,hpij->hp", grad_S, sw).unsqueeze(-1)
        vz = torch.einsum("hij,hpij->hp", grad_S, sz).unsqueeze(-1)
        vb = torch.einsum("hij,hpij->hp", grad_S, sb).unsqueeze(-1)
        vv = torch.einsum("hij,hpij->hp", grad_S, sv).unsqueeze(-1)
        vk = torch.einsum("hij,hpij->hp", grad_S, sk).unsqueeze(-1)

        return grad_S, grad_sens, vw, vz, vb, vv, vk


def custom_f(S, sensitivity_mats, w, z, b, v, k):
    return CustomF.apply(S, sensitivity_mats, w, z, b, v, k)


def compute_loss_bptt(params, curr_S, y):
    w, z, b, v, k, q = params
    sum_loss = 0
    for t in range(w.shape[0]):
        curr_S = f1(curr_S, w[t], z[t], b[t], v[t], k[t])
        curr_pred = curr_S @ q[t]
        curr_loss = F.mse_loss(curr_pred, y[t])
        sum_loss += curr_loss
    return sum_loss / w.shape[0]


def compute_loss_rtrl(params, curr_S, sensitivity_mats, y):
    w, z, b, v, k, q = params
    curr_S, sensitivity_mats = custom_f(curr_S, sensitivity_mats, w, z, b, v, k)
    curr_pred = curr_S @ q
    curr_loss = F.mse_loss(curr_pred, y)
    return curr_loss, (curr_S, sensitivity_mats)


if __name__ == "__main__":
    HEAD_NUM = 2
    HEAD_SIZE = 8
    TIMESTEP = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    curr_S = torch.ones((HEAD_NUM, HEAD_SIZE, HEAD_SIZE), device=device)
    y = torch.ones((TIMESTEP, HEAD_NUM, HEAD_SIZE, 1), device=device)

    w = torch.randn((TIMESTEP, HEAD_NUM, HEAD_SIZE, 1), device=device, requires_grad=True)
    z = torch.randn((TIMESTEP, HEAD_NUM, HEAD_SIZE, 1), device=device, requires_grad=True)
    b = torch.randn((TIMESTEP, HEAD_NUM, HEAD_SIZE, 1), device=device, requires_grad=True)
    v = torch.randn((TIMESTEP, HEAD_NUM, HEAD_SIZE, 1), device=device, requires_grad=True)
    k = torch.randn((TIMESTEP, HEAD_NUM, HEAD_SIZE, 1), device=device, requires_grad=True)
    q = torch.randn((TIMESTEP, HEAD_NUM, HEAD_SIZE, 1), device=device, requires_grad=True)

    S1 = S2 = torch.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE), device=device)
    for t in range(TIMESTEP):
        S1 = f1(S1, w[t], z[t], b[t], v[t], k[t])
        S2 = f2(S2, w[t], z[t], b[t], v[t], k[t])
        assert torch.allclose(S1, S2), "S1 and S2 are not equal"

    # BPTT
    params = (w, z, b, v, k, q)
    loss_bptt = compute_loss_bptt(params, curr_S.clone(), y)
    loss_bptt.backward()
    grads_bptt = tuple(p.grad.clone() for p in params)

    # RTRL
    sensitivity_zero = torch.zeros((HEAD_NUM, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE), device=device)
    sensitivity_mats = (sensitivity_zero.clone(),) * 5
    loss_rtrl = 0
    grads_rtrl = [torch.zeros_like(x[0]) for x in params]

    for t in range(TIMESTEP):
        single_params = tuple(p[t].detach().clone().requires_grad_() for p in params)
        loss, (curr_S, sensitivity_mats) = compute_loss_rtrl(
            single_params, curr_S, sensitivity_mats, y[t]
        )
        loss_rtrl += loss / TIMESTEP
        loss.backward(retain_graph=True)
        for i, p in enumerate(single_params):
            grads_rtrl[i] += p.grad.detach().clone() / TIMESTEP
            p.grad.zero_()

    print(loss_bptt.item())
    print(loss_rtrl.item())
    print(torch.allclose(loss_bptt, loss_rtrl))

    for i, (gb, gr) in enumerate(zip(grads_bptt, grads_rtrl)):
        gb_sum = gb.mean(dim=0)
        gr_sum = gr / TIMESTEP
        close = torch.allclose(gb_sum, gr_sum)
        print(f"i={i}, close={close}")
        if not close:
            print(f"gb_sum: {gb_sum}\ngr_sum: {gr_sum}")
