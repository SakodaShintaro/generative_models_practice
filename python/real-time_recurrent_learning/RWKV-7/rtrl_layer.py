import torch
import torch.nn.functional as F
from kernel_torch import custom_f


class GeneralizedDeltaRtrlLayer(torch.nn.Module):
    def __init__(self, input_size, head_num, head_size):
        super().__init__()
        self.head_num = head_num
        self.head_size = head_size
        output_size = head_num * head_size
        self.linear_w = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear_z = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear_b = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear_v = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear_k = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear_q = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear_o = torch.nn.Linear(output_size, input_size, bias=True)

    def _apply_linear(self, x, l):
        x = l(x)
        return x.view(self.head_num, self.head_size, 1)

    def forward(self, x, S, sensitivity_mats):
        """
        Args:
            x: (batch_size=1, input_size)
            S: (head_num, head_size, head_size)
            sensitivity_mats: (head_num, head_size, head_size, head_size)
        Return:
            y: (batch_size=1, input_size)
            S: (head_num, head_size, head_size)
            sensitivity_mats: (head_num, head_size, head_size, head_size)
        """
        x = x.squeeze(0)  # (input_size)
        w = self._apply_linear(x, self.linear_w)  # (head_num, head_size, 1)
        z = self._apply_linear(x, self.linear_z)  # (head_num, head_size, 1)
        b = self._apply_linear(x, self.linear_b)  # (head_num, head_size, 1)
        v = self._apply_linear(x, self.linear_v)  # (head_num, head_size, 1)
        k = self._apply_linear(x, self.linear_k)  # (head_num, head_size, 1)
        q = self._apply_linear(x, self.linear_q)  # (head_num, head_size, 1)
        S, sensitivity_mats = custom_f(S, sensitivity_mats, w, z, b, v, k)
        # S : (head_num, head_size, head_size)
        q = q.view(self.head_num, self.head_size)

        y = torch.einsum("hij,hj->hi", S, q)
        y = y.view(self.head_num * self.head_size)
        y = self.linear_o(y)
        y = torch.unsqueeze(y, 0)  # (1, input_size)

        sensitivity_mats = [s.clone().detach() for s in sensitivity_mats]

        return y, S, sensitivity_mats

    def initialize_state(self):
        S = torch.zeros(self.head_num, self.head_size, self.head_size, requires_grad=False)
        t = torch.zeros(
            self.head_num, self.head_size, self.head_size, self.head_size, requires_grad=False
        )
        sensitivity_mats = (
            t.clone(),  # w
            t.clone(),  # z
            t.clone(),  # b
            t.clone(),  # v
            t.clone(),  # k
        )
        return S, sensitivity_mats


def make_data(batch_size, seq_len, input_dim):
    half_seq = seq_len // 2
    data = torch.randint(low=0, high=input_dim - 2, size=(half_seq, batch_size))
    data = torch.cat([data, data], dim=0)

    mask = torch.ones(seq_len, batch_size)
    mask[:half_seq] = 0

    batch_x = data.clone()
    batch_y = data.clone()

    batch_x[half_seq:] = input_dim - 1
    batch_x = F.one_hot(batch_x, num_classes=input_dim).float()
    return batch_x, batch_y, mask


if __name__ == "__main__":
    HEAD_NUM = 2
    HEAD_SIZE = 8
    TIMESTEP = 10
    INPUT_DIM = 9
    SEQ_LEN = 10
    STEP_NUM = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GeneralizedDeltaRtrlLayer(INPUT_DIM, HEAD_NUM, HEAD_SIZE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 1

    loss_list = []
    acc = 0.0
    num = 0
    for i in range(1, STEP_NUM + 1):
        batch_x, batch_y, _ = make_data(batch_size, SEQ_LEN, INPUT_DIM)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        loss = 0.0
        S, sensitivity_mats = model.initialize_state()
        S = S.to(device)
        sensitivity_mats = [s.to(device) for s in sensitivity_mats]
        for j in range(SEQ_LEN):
            x_t = batch_x[j]
            y_t_ref = batch_y[j]
            y, S, sensitivity_mats = model(x_t, S, sensitivity_mats)
            if j >= SEQ_LEN // 2:
                curr_loss = F.cross_entropy(y, y_t_ref, reduction="sum")
                acc += (torch.argmax(y, dim=1) == y_t_ref).float().sum().item()
                num += 1
                loss += curr_loss.item()
                curr_loss /= SEQ_LEN
                optim.zero_grad()
                curr_loss.backward()
                optim.step()
                S = S.detach().clone()

        loss_list.append(loss)
        if i % (STEP_NUM / 10) == 0:
            loss_list_t = torch.tensor(loss_list)
            loss_avg = torch.mean(loss_list_t)
            loss_std = torch.std(loss_list_t)
            acc *= 100 / num
            print(f"{i:08d} {acc:5.1f} {loss_avg:.5f} {loss_std:.5f}")
