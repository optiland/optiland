"""
PyTorch backend -- random-number generation operations.

Provides RandomMixin, one of the mixins composed into
TorchBackend (see optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Generator as TorchGenerator
    from torch import Tensor


class RandomMixin:
    """Random-number generation operations."""

    # ------------------------------------------------------------------
    # Random number generation
    # ------------------------------------------------------------------

    def default_rng(self, seed: int | None = None) -> TorchGenerator:
        """Return a PyTorch random number generator.

        Args:
            seed: Optional seed.

        Returns:
            Generator: PyTorch Generator.
        """
        if seed is None:
            seed = torch.initial_seed()
        return torch.Generator(device=self._device()).manual_seed(seed)

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Any = None,
        generator: TorchGenerator | None = None,
    ) -> Tensor:
        """Uniform random samples in [low, high).

        Args:
            low: Lower boundary.
            high: Upper boundary.
            size: Output shape.
            generator: Optional torch Generator.

        Returns:
            Tensor: Uniform random samples.
        """
        size = size or 1
        gen_args = {"generator": generator} if generator else {}
        return torch.empty(size, device=self._device(), dtype=self._dtype()).uniform_(
            low, high, **gen_args
        )

    def rand(self, *size: int) -> Tensor:
        """Random values from a uniform distribution on [0, 1).

        Args:
            *size: Shape of the output tensor.

        Returns:
            Tensor: Random values.
        """
        if not size:
            size = (1,)
        return torch.rand(
            size,
            device=self._device(),
            dtype=self._dtype(),
            requires_grad=self._grad(),
        )

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Any = None,
        generator: TorchGenerator | None = None,
    ) -> Tensor:
        """Random samples from a Gaussian distribution.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation.
            size: Output shape.
            generator: Optional torch Generator.

        Returns:
            Tensor: Normal random samples.
        """
        size = size or (1,)
        gen_args = {"generator": generator} if generator else {}
        return (
            torch.randn(size, device=self._device(), dtype=self._dtype(), **gen_args)
            * scale
            + loc
        )

    def sobol_sampler(
        self,
        dim: int,
        num_samples: int,
        scramble: bool = True,
        seed: int | None = None,
    ) -> Tensor:
        """Generate quasi-random samples using Sobol sequences.

        Args:
            dim: Dimension of the samples.
            num_samples: Number of samples to generate.
            scramble: Whether to scramble the sequence.
            seed: Random seed for scrambling.

        Returns:
            Tensor: Samples of shape (num_samples_pow2, dim).
        """
        if num_samples > 0:
            num_samples_pow2 = 1 << (num_samples - 1).bit_length()
        else:
            num_samples_pow2 = num_samples
        sobol_engine = torch.quasirandom.SobolEngine(
            dimension=dim, scramble=scramble, seed=seed
        )
        samples = sobol_engine.draw(num_samples_pow2)
        return samples[:num_samples].to(device=self._device(), dtype=self._dtype())

    def erfinv(self, x: Any) -> Tensor:
        """Inverse error function.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Inverse error function of x.
        """
        return torch.erfinv(self.array(x))
