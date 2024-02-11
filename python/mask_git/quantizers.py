"""
https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb
の写経
"""

"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import jax
import jax.numpy as jnp
import numpy as np
Codeword = jax.Array
Indices = jax.Array


def round_ste(z):
    """Round with straight through gradients."""
    zhat = jnp.round(z)
    return z + jax.lax.stop_gradient(zhat - z)


class FSQ:
    """Quantizer."""

    def __init__(self, levels: list[int], eps: float = 1e-3):
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(
            ([1], np.cumprod(self._levels_np[:-1]))).astype(np.uint32)

        self._implicit_codebook = self.indexes_to_codes(
            np.arange(self.codebook_size))

    def __call__(self, x) -> Codeword:
        return self.quantize(x), dict()

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: jax.Array) -> jax.Array:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = jnp.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = jnp.tan(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    def quantize(self, z: jax.Array) -> Codeword:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: Codeword) -> Indices:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).astype(jnp.uint32)

    def indexes_to_codes(self, indices: Indices) -> Codeword:
        """Inverse of `indexes_to_codes`."""
        indices = indices[..., jnp.newaxis]
        codes_non_centered = np.mod(
            np.floor_divide(indices, self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(codes_non_centered)


class FSQ_Level2:
    def __init__(self, dim: int, eps: float = 1e-3):
        self._eps = eps
        self._dim = dim
        self._basis = np.array([2 ** i for i in range(dim)]).astype(np.uint32)
        self._implicit_codebook = self.indexes_to_codes(
            np.arange(self.codebook_size))

    def __call__(self, x) -> Codeword:
        return self.quantize(x), dict()

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return self._dim

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return 2 ** self._dim

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (2^self._dim, num_dimensions)."""
        return self._implicit_codebook

    def quantize(self, z: jax.Array) -> Codeword:
        """Quantizes z, returns quantized zhat, same shape as z."""
        z = jax.nn.sigmoid(z)  # Map to [0, 1]
        z = round_ste(z)       # Quantize
        z = z * 2 - 1          # Map back to [-1, 1]
        return z

    def codes_to_indexes(self, zhat: Codeword) -> Indices:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = zhat * 0.5 + 0.5  # Map to [0, 1]
        return (zhat * self._basis).sum(axis=-1).astype(jnp.uint32)

    def indexes_to_codes(self, indices: Indices) -> Codeword:
        """Inverse of `indexes_to_codes`."""
        indices = indices[..., jnp.newaxis]
        val = np.floor_divide(indices, self._basis)
        zhat_non_centered = np.mod(val, np.array([2] * self._dim))
        zhat = zhat_non_centered * 2 - 1
        return zhat


if __name__ == "__main__":
    x = jnp.array([0, 0.5, -1, 2, -3, 4])
    print(x)

    print("\nlevel2")
    level2 = FSQ(levels=[2])
    print(level2(x))

    print("\nlevel3")
    level3 = FSQ(levels=[3])
    print(level3(x))

    print("\nlevel2 my implementation")
    level2 = FSQ_Level2(dim=1)
    print(level2(x))

    print("\nlevel2_3 my implementation")
    level2_3 = FSQ_Level2(dim=3)
    for index in range(2 ** 3):
        code = level2_3.indexes_to_codes(jnp.array([index]))
        index_rev = level2_3.codes_to_indexes(jnp.array([code]))
        print(f"{index=} {code=} {index_rev=}")
