"""Unit tests for the GrinPropagation model."""

from __future__ import annotations

import pytest

from optiland.propagation.grin import GRINPropagation
from optiland.rays.real_rays import RealRays


def test_grin_propagation_raises_not_implemented_error():
    """Verify that GRINPropagation.propagate raises NotImplementedError."""
    model = GRINPropagation()
    rays = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1], wavelength=[0.5]
    )

    with pytest.raises(NotImplementedError):
        model.propagate(rays, t=10.0)
