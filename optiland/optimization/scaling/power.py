"""Power Scaler Module

This module contains the PowerScaler class, which is a scaler that
performs a power transformation on the value.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.optimization.errors import ConfigurationError

from .base import Scaler


class PowerScaler(Scaler):
    """Represents a scaler that performs a power transformation on the value.

    The power map ``x → x**p`` is only single-valued and real on the positive
    domain when ``p`` is even or fractional.  Applying it to a signed value in
    those cases produces ``NaN`` (fractional ``p``) or destroys invertibility
    (even ``p``), so such combinations are rejected up front.
    """

    def __init__(self, power=1.0):
        if power == 0:
            raise ConfigurationError(
                "PowerScaler power must be non-zero; x**0 is constant and not "
                "invertible."
            )
        self.power = power

    @property
    def monotonic(self) -> bool:
        """``x**p`` is monotonic increasing on the positive domain iff ``p>0``."""
        return self.power > 0

    def scale(self, value):
        """Scale the value using a power transformation.

        Args:
            value: The value to scale

        Raises:
            ConfigurationError: If ``value`` is negative and ``power`` is even
                or fractional (the transform is then non-real or non-invertible).
        """
        arr = be.asarray(value)
        is_integer = float(self.power) == int(self.power)
        is_even = is_integer and int(self.power) % 2 == 0
        if (not is_integer or is_even) and bool(be.any(arr < 0)):
            raise ConfigurationError(
                f"PowerScaler with power={self.power} cannot be applied to "
                f"negative values (got {value!r}): the result is non-real "
                "(fractional power) or non-invertible (even power). Restrict the "
                "variable to a positive domain or use an odd integer power."
            )
        return be.power(arr, self.power)

    def inverse_scale(self, scaled_value):
        """Inverse scale the value using a power transformation.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return be.power(scaled_value, 1.0 / self.power)
