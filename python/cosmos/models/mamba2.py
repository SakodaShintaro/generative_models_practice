import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .config import ModelArgs
from .llama_transformer import apply_rotary_pos_emb

# ruff: noqa: N803, N806


# ref. https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
def segsum(x: torch.Tensor) -> torch.Tensor:
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
    which is equivalent to a scalar SSM.
        x = [a, b, c, d]
        x_cumsum = [a, a+b, a+b+c, a+b+c+d]
        x_segsum(before masking) = [
            [  a-a,    a-(a+b),     a-(a+b+c),     a-(a+b+c+d)     ]
            [  a+b-a,  a+b-(a+b),   a+b-(a+b+c),   a+b-(a+b+c+d)   ]
            [  a+b+c-a, a+b+c-(a+b), a+b+c-(a+b+c), a+b+c-(a+b+c+d) ]
            [  a+b+c+d-a, ...                                       ]
        ]
        x_segsum(after masking) = [
            [  0,    -inf,   -inf,   -inf  ]
            [  a+b-a,  0,   -inf,   -inf   ]
            [  a+b+c-a, a+b+c-(a+b), 0, -inf ]
            [  a+b+c+d-a, ...                ]
        ]
    """
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_cumsum_v = x_cumsum[..., :, None]  # (..., T, 1)
    x_cumsum_h = x_cumsum[..., None, :]  # (..., 1, T)
    x_segsum = x_cumsum_v - x_cumsum_h  # (..., T, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(
    X: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    block_len: int = 64,
    initial_states: torch.Tensor = None,
) -> torch.Tensor:
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


def simple_matrix(
    X: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    A = rearrange(A, "b l h -> b h l")
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(segsum(A))
    Y = torch.einsum("blhn,bshn,bhls,bshp->blhp", C, B, L, X)
    decay_states = torch.exp(A_cumsum[:, :, -1:] - A_cumsum)
    states = torch.einsum("blhn,bhl,blhp->bhpn", B, decay_states, X)
    return Y, states


def simple_recurrsive(
    X: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    b, l, h, d_head = X.shape
    b, l, h, d_state = B.shape
    state = torch.zeros(b, h, d_head, d_state, device=X.device, dtype=X.dtype)
    y_list = []
    for i in range(l):
        curr_A = A[:, i, :, None, None]  # (b, h, 1, 1)
        curr_B = B[:, i, :, :]  # (b, h, d)
        curr_C = C[:, i, :, :]  # (b, h, d)
        curr_X = X[:, i, :, :]  # (b, h, d_head)
        state = state * torch.exp(curr_A) + torch.einsum("bhf,bhg->bhgf", curr_B, curr_X)
        curr_Y = torch.einsum("bhfg,bhg->bhf", state, curr_C)
        y_list.append(curr_Y)
    Y = torch.stack(y_list, dim=1)
    return Y, state


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        dim = args.dim
        assert dim % args.n_heads == 0

        self.n_heads = args.n_heads
        self.head_dim = dim // args.n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # Additional for Mamba
        self.A_log = nn.Parameter(torch.zeros(args.n_heads))
        self.w_dt = nn.Linear(dim, args.n_heads, bias=True)

    def prepare_values(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        A = -torch.exp(self.A_log)  # (nheads,)
        A = A.view(1, 1, -1, 1)
        dt = self.w_dt(x)  # (batch, seqlen, nheads)
        dt = F.softplus(dt)  # (batch, seqlen, nheads)
        dt.unsqueeze_(-1)

        # SSM notation
        X = xv
        B = xk
        C = xq

        # conv1d (skip).
        # SwiGLU (skip).
        X = X * dt
        A = A * dt
        A = A.squeeze(-1)
        return X, A, B, C

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        X, A, B, C = self.prepare_values(x, cos, sin)
        y, state = ssd(X, A, B, C, block_len=64)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(y), state

    def simple_matrix(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        X, A, B, C = self.prepare_values(x, cos, sin)
        y, state = simple_matrix(X, A, B, C)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(y), state

    def simple_recurrsive(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        X, A, B, C = self.prepare_values(x, cos, sin)
        y, state = simple_recurrsive(X, A, B, C)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(y), state
