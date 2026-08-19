"""Shared fixtures for prescription tests."""

from __future__ import annotations

import pytest

from optiland.samples.objectives import CookeTriplet


@pytest.fixture
def cooke_triplet():
    """Return a Cooke Triplet Optic instance."""
    lens = CookeTriplet()
    lens.name = "Cooke Triplet"
    return lens
