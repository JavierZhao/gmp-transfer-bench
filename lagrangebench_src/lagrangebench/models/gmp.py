"""Geometric Message Passing (GMP) block."""

from typing import Literal

import haiku as hk
import jax.numpy as jnp


class GeometricMessagePassing(hk.Module):
    """Scatter-conv-gather residual block on a fixed 2D grid.

    This is a JAX/Haiku variant of the GMP idea used in ParT:
    1) scatter point features to a quantized grid,
    2) apply depthwise 2D convolution,
    3) gather back to points + pointwise projection + residual.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        grid_bins: int = 64,
        scatter_reduce: Literal["sum", "mean"] = "sum",
        eps: float = 1e-6,
        name: str = "gmp",
    ):
        super().__init__(name=name)
        if scatter_reduce not in ("sum", "mean"):
            raise ValueError(f"Invalid scatter_reduce={scatter_reduce}.")
        if grid_bins <= 0:
            raise ValueError(f"grid_bins must be positive, got {grid_bins}.")

        self._channels = channels
        self._kernel_size = kernel_size
        self._grid_bins = grid_bins
        self._scatter_reduce = scatter_reduce
        self._eps = eps

    def __call__(self, x: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        """Apply GMP.

        Args:
            x: Node features of shape (N, C).
            coords: Point coordinates of shape (N, D), uses first 2 dims.
        """
        if coords.shape[-1] < 2:
            raise ValueError("coords must have at least 2 dimensions per point.")

        coords_2d = coords[:, :2]

        # Normalize coordinates to [0, 1] per sample and quantize to fixed bins.
        c_min = jnp.min(coords_2d, axis=0, keepdims=True)
        c_max = jnp.max(coords_2d, axis=0, keepdims=True)
        c_span = jnp.maximum(c_max - c_min, self._eps)
        c_norm = (coords_2d - c_min) / c_span

        max_bin = self._grid_bins - 1
        grid_ij = jnp.floor(c_norm * max_bin).astype(jnp.int32)
        grid_ij = jnp.clip(grid_ij, 0, max_bin)

        flat_idx = grid_ij[:, 0] * self._grid_bins + grid_ij[:, 1]
        grid_hw = self._grid_bins * self._grid_bins

        grid_flat = jnp.zeros((grid_hw, self._channels), dtype=x.dtype)
        grid_flat = grid_flat.at[flat_idx].add(x)

        if self._scatter_reduce == "mean":
            counts = jnp.zeros((grid_hw,), dtype=x.dtype)
            counts = counts.at[flat_idx].add(1.0)
            grid_flat = grid_flat / (counts[:, None] + self._eps)

        grid = jnp.reshape(grid_flat, (self._grid_bins, self._grid_bins, self._channels))

        # Depthwise spatial mixing on the quantized feature grid.
        depthwise = hk.DepthwiseConv2D(
            channel_multiplier=1,
            kernel_shape=self._kernel_size,
            stride=1,
            padding="SAME",
            with_bias=True,
            data_format="NHWC",
            name="depthwise_conv",
        )
        grid = depthwise(grid[None, ...])[0]

        gathered = grid[grid_ij[:, 0], grid_ij[:, 1], :]
        out = hk.Linear(self._channels, name="pointwise")(gathered)
        out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm")(out)

        return x + out
