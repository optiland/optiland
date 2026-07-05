"""
NumPy backend -- random-number generation operations.

Provides RandomMixin, one of the mixins composed into
NumpyBackend (see optiland/backend/numpy_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator as NpGenerator
    from numpy.typing import ArrayLike, NDArray


class RandomMixin:
    """Random-number generation operations."""

    # ------------------------------------------------------------------
    # Random number generation
    # ------------------------------------------------------------------

    def default_rng(self, seed: int | None = None) -> NpGenerator:
        """Return a NumPy random number generator.

        Args:
            seed: Optional seed.

        Returns:
            Generator: NumPy random generator.
        """
        return np.random.default_rng(seed)

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Any = None,
        generator: NpGenerator | None = None,
    ) -> NDArray:
        """Uniform random samples in [low, high).

        Args:
            low: Lower boundary.
            high: Upper boundary.
            size: Output shape.
            generator: Optional NumPy random generator.

        Returns:
            NDArray: Uniform random samples.
        """
        if generator is None:
            generator = np.random.default_rng()
        return generator.uniform(low, high, size)

    def rand(self, *size: int) -> NDArray:
        """Random values from a uniform distribution on [0, 1).

        Args:
            *size: Shape of the output array.

        Returns:
            NDArray: Random values.
        """
        return np.random.rand(*size) if size else np.random.rand()

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Any = None,
        generator: NpGenerator | None = None,
    ) -> NDArray:
        """Random samples from a Gaussian distribution.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation.
            size: Output shape.
            generator: Optional NumPy random generator.

        Returns:
            NDArray: Normal random samples.
        """
        if generator is None:
            generator = np.random.default_rng()
        return generator.normal(loc, scale, size)

    def sobol_sampler(
        self,
        dim: int,
        num_samples: int,
        scramble: bool = True,
        seed: int | None = None,
    ) -> NDArray:
        """Generate quasi-random samples using Sobol sequences.

        Args:
            dim: Dimension of the samples.
            num_samples: Number of samples to generate.
            scramble: Whether to scramble the sequence.
            seed: Random seed for scrambling.

        Returns:
            NDArray: Samples of shape (num_samples_pow2, dim).
        """
        try:
            from scipy.stats import qmc
        except ImportError as exc:
            raise ImportError(
                "scipy is required for Sobol sampling with numpy backend"
            ) from exc

        if num_samples > 0:
            num_samples_pow2 = 1 << (num_samples - 1).bit_length()
        else:
            num_samples_pow2 = num_samples

        sobol = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
        samples = sobol.random(n=num_samples_pow2)
        return samples[:num_samples].astype(self._dtype)

    def erfinv(self, x: ArrayLike) -> NDArray:
        """Inverse error function.

        Args:
            x: Input array.

        Returns:
            NDArray: Inverse error function of x.
        """
        try:
            from scipy.special import erfinv as scipy_erfinv
        except ImportError as exc:
            raise ImportError(
                "scipy is required for erfinv with numpy backend"
            ) from exc
        return scipy_erfinv(np.asarray(x))
