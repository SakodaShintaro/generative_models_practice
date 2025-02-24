import numpy as np
import torch


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()


class FSQ:
    """Quantizer."""

    def __init__(self, levels: list[int], eps: float = 1e-3) -> None:
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(np.uint32)

        self._implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size))

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return int(np.prod(self._levels))

    @property
    def codebook(self) -> torch.Tensor:
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        half_l = torch.tensor(half_l, dtype=z.dtype, device=z.device)
        offset = torch.where(torch.tensor(self._levels_np % 2 == 1, device=z.device), 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1]
        half_width = torch.tensor(self._levels_np // 2, dtype=z.dtype, device=z.device)
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        """Scale and shift to range [0, ..., L-1]."""
        half_width = torch.tensor(
            self._levels_np // 2,
            dtype=zhat_normalized.dtype,
            device=zhat_normalized.device,
        )
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        """Inverse scaling and shifting."""
        half_width = torch.tensor(self._levels_np // 2, dtype=zhat.dtype, device=zhat.device)
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        basis = torch.tensor(self._basis, dtype=zhat.dtype, device=zhat.device)
        return (zhat * basis).sum(dim=-1).to(torch.uint32)

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indexes`."""
        indices = indices.unsqueeze(-1)
        basis = torch.tensor(self._basis, dtype=indices.dtype, device=indices.device)
        codes_non_centered = torch.floor_divide(indices, basis) % torch.tensor(
            self._levels_np,
            dtype=indices.dtype,
            device=indices.device,
        )
        return self._scale_and_shift_inverse(codes_non_centered)
