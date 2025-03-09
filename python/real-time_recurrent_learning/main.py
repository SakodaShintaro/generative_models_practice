"""BPTTとRTRLで勾配が一致することを検証するスクリプト."""

import torch
from models.minimal_manba2 import InferenceCache, Mamba2LMHeadModel

if __name__ == "__main__":
    device = torch.device("cuda")
    model = Mamba2LMHeadModel()
    model.to(device)

    torch.autograd.set_detect_anomaly(True)

    d_model = model.args.d_model
    print(d_model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ダミーの系列を準備
    # (batch_size, seq_len)の形状の整数テンソルを生成
    src = torch.randint(
        0, model.args.vocab_size, (1, model.args.chunk_size), dtype=torch.long, device=device
    )
    print(src.shape)

    ###########################################
    # (1) BPTT(Back Propagation Through Time) #
    ###########################################

    # モデルに入力
    output_bptt, h_bptt = model(src)
    print(f"{output_bptt.shape=}")

    # 損失は最後の出力の自己エントロピーとする
    last = output_bptt[:, -1]  # (b, vocab_size)
    prob = torch.nn.functional.softmax(last, dim=-1)
    loss = -prob * torch.log(prob)
    loss = loss.sum(dim=-1).mean()
    print(loss)

    loss.backward()
    grad_bptt = [param.grad for param in model.parameters() if param.grad is not None]
    optim.zero_grad()

    ###########################################
    # (2) RTRL(Real-Time Recurrent Learning)  #
    ###########################################
    # RTRL
    h_rtrl = [InferenceCache.alloc(1, model.args, device)]
    ssm_state_shape = h_rtrl[0].ssm_state.shape
    conv_state_shape = h_rtrl[0].conv_state.shape
    print(f"{h_rtrl[0].ssm_state.shape=}, {h_rtrl[0].conv_state.shape=}")

    ssm_state_num = h_rtrl[0].ssm_state.numel()
    conv_state_num = h_rtrl[0].conv_state.numel()
    total_state_num = ssm_state_num + conv_state_num
    print(f"{ssm_state_num=:,}, {conv_state_num=:,}, {total_state_num=:,}")

    # ssm_state_shape=torch.Size([1, 8, 128, 256]).
    # torch.zeros(batch_size, args.nheads, args.headdim, args.d_state, device=device).

    # conv_state_shape=torch.Size([1, 1536, 4]).
    # torch.zeros(batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device).

    params = [torch.zeros_like(param) for param in model.parameters()]
    num_params = 0
    for param in model.parameters():
        print(param.shape, param.names)
        num_params += param.numel()
    print(f"{num_params=:,}")
    total_array_num = total_state_num * num_params
    print(f"{total_array_num=:,}")

    dssm_state_dw = []
    for batch_size in range(ssm_state_shape[0]):
        dssm_state_dw.append([])
        for nhead in range(ssm_state_shape[1]):
            dssm_state_dw[batch_size].append([])
            for headdim in range(ssm_state_shape[2]):
                dssm_state_dw[batch_size][nhead].append([])
                for _ in range(ssm_state_shape[3]):
                    dssm_state_dw[batch_size][nhead][headdim].append(
                        [torch.zeros_like(param) for param in model.parameters()]
                    )

    dconv_state_dw = []
    for batch_size in range(conv_state_shape[0]):
        dconv_state_dw.append([])
        for d_inner in range(conv_state_shape[1]):
            dconv_state_dw[batch_size].append([])
            for _ in range(conv_state_shape[2]):
                dconv_state_dw[batch_size][d_inner].append(
                    [torch.zeros_like(param) for param in model.parameters()]
                )

    for t in range(src.shape[1]):
        print(f"{t=}")
        h_rtrl = [
            InferenceCache(
                h_rtrl[0].conv_state.clone().detach(), h_rtrl[0].ssm_state.clone().detach()
            )
        ]
        output_rtrl, h_rtrl = model(src[:, t : t + 1], h_rtrl)

        for batch_size in range(ssm_state_shape[0]):
            dssm_state_dw.append([])
            for nhead in range(ssm_state_shape[1]):
                dssm_state_dw[batch_size].append([])
                for headdim in range(ssm_state_shape[2]):
                    print(f"{batch_size=}, {nhead=}, {headdim=}")
                    dssm_state_dw[batch_size][nhead].append([])
                    for d_state in range(ssm_state_shape[3]):
                        optim.zero_grad()
                        loss = h_rtrl[0].ssm_state[batch_size, nhead, headdim, d_state]
                        loss.backward(retain_graph=True)
                        for param in model.parameters():
                            if param.grad is None:
                                continue
                            dssm_state_dw[batch_size][nhead][headdim][d_state] += param.grad
                        optim.zero_grad()

    print(f"{output_rtrl.shape=}")
    diff_output = output_rtrl - output_bptt[:, -1]

    max_diff = diff_output.abs().max()
    print(f"{max_diff=}")
