"""Perturbation Module

This module contains classes for generating perturbations to optical system
variables. Perturbations are used to simulate the effects of manufacturing
errors or other sources of variability in an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import optiland.backend as be
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable import Variable


class BaseSampler(ABC):
    """Abstract base class for samplers.

    This class defines the interface for samplers in the optiland.tolerancing
    module.
    """

    @abstractmethod
    def sample(self):
        pass  # pragma: no cover


class ScalarSampler(BaseSampler):
    """A sampler that always returns a fixed scalar value.

    Args:
        value (float or int): The scalar value to be returned by the sampler.

    Attributes:
        value (float or int): The scalar value to be returned by the sampler.
        size (int): The size of the sample, which is always 1 for
            ScalarSampler.

    """

    def __init__(self, value):
        self.value = value
        self.size = 1

    def sample(self):
        """Returns the fixed scalar value.

        Returns:
            float or int: The scalar value.

        """
        return self.value


class RangeSampler(BaseSampler):
    """A sampler that generates a range of samples over a linear range.

    Args:
        start (float or int): The start of the range.
        end (float or int): The end of the range.
        steps (int): The number of samples to generate.

    Attributes:
        start (float or int): The start of the range.
        end (float or int): The end of the range.
        size (int): The number of samples to generate.

    """

    def __init__(self, start, end, steps):
        self.values = be.linspace(start, end, steps)
        self.index = 0
        self.size = steps

    def sample(self):
        """Return the next value in the range.

        Returns:
            float: The next value in the range.

        """
        if self.index >= len(self.values):
            self.index = 0  # loop over values
        value = self.values[self.index]
        self.index += 1
        return value


class DistributionSampler(BaseSampler):
    """A sampler that generates random samples from a given distribution.

    Options for the distribution include 'normal' and 'uniform' and these
    require keyword arguments as follows:

    - ``"normal"``: ``loc``, ``scale`` (mean and standard deviation)
    - ``"uniform"``: ``low``, ``high``

    Args:
        distribution (str): The type of distribution to sample from. Options
            are 'normal' and 'uniform'.
        *params: Variable length arguments representing the parameters of the
            distribution.
        seed (int, optional): The seed value for random number generation.
            Defaults to None.

    Raises:
        ValueError: If the distribution type is unknown.

    """

    def __init__(self, distribution, seed=None, **params):
        self.generator = be.default_rng(seed)
        self.distribution = distribution
        self.params = params

        valid_params = ["loc", "scale", "low", "high"]
        for param in params:
            if param not in valid_params:
                raise ValueError(f"Invalid parameter: {param}")

    def sample(self):
        """Return a random value from the given distribution.

        Returns:
            float: A random value sampled from the specified distribution.

        """
        # TODO: consider vectorizing this method with 'size' parameter
        if self.distribution == "normal":
            return be.random_normal(**self.params, generator=self.generator)
        if self.distribution == "uniform":
            return be.random_uniform(**self.params, generator=self.generator)
        raise ValueError(f"Unknown distribution: {self.distribution}")


class Perturbation:
    """A class representing a perturbation to an optic variable. Perturbations
    are used to simulate the effects of manufacturing errors or other sources
    of variability in an optical system.

    Args:
        optic: The optic object to apply the perturbation to.
        variable_type: The type of variable to perturb.
        sampler: The sampler object used to generate perturbation values.
        **kwargs: Additional keyword arguments to be passed to the Variable
            object.

    Attributes:
        optic: The optic object to apply the perturbation to.
        type: The type of variable to perturb.
        sampler: The sampler object used to generate perturbation values.
        variable: The Variable object representing the perturbed variable.
        value: The value of the perturbation.

    """

    def __init__(self, optic, variable_type, sampler, **kwargs):
        self.optic = optic
        self.type = variable_type
        self.sampler = sampler
        self.variable = Variable(
            optic, variable_type, scaler=IdentityScaler(), **kwargs
        )
        self.value = None

    def apply(self):
        """Apply the perturbation to the optic."""
        self.value = self.sampler.sample()
        self.variable.update(self.value)

    def reset(self):
        """Reset the perturbation to its original value."""
        self.variable.reset()
        self.value = self.variable.variable.get_value()
