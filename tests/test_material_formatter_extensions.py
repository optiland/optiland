"""Material identity must reach displays without registering every new family."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from optiland.materials import BaseMaterial
from optiland.visualization.info.material_formatter import MaterialFormatter


class UnregisteredMaterial(BaseMaterial):
    @property
    def display_name(self):
        return "custom spectral model"

    def _calculate_n(self, wavelength, **kwargs):
        raise AssertionError("Display must not sample an unknown optical law")

    def _calculate_k(self, wavelength, **kwargs):
        raise AssertionError("Display must not sample an unknown optical law")


@pytest.fixture
def surface(monkeypatch):
    # Isolate public registration state without changing built-in formatters.
    monkeypatch.setattr(MaterialFormatter, "_formatters", MaterialFormatter._formatters.copy())
    monkeypatch.setattr(MaterialFormatter, "_default_formatter", None)
    return SimpleNamespace(
        material_post=UnregisteredMaterial(),
        interaction_model=SimpleNamespace(is_reflective=False),
    )


def test_new_material_uses_its_display_contract(surface):
    assert MaterialFormatter.format(surface) == "custom spectral model"


def test_explicit_material_formatter_takes_precedence(surface):
    MaterialFormatter.register(UnregisteredMaterial, lambda _: "registered")
    assert MaterialFormatter.format(surface) == "registered"


def test_explicit_default_takes_precedence_for_unregistered_material(surface):
    MaterialFormatter.set_default(lambda _: "configured default")
    assert MaterialFormatter.format(surface) == "configured default"


def test_mirror_identity_takes_precedence(surface):
    surface.interaction_model.is_reflective = True
    assert MaterialFormatter.format(surface) == "Mirror"


@pytest.mark.parametrize("missing", [False, True])
def test_unknown_non_material_still_requires_a_formatter(surface, missing):
    if missing:
        del surface.material_post
    else:
        surface.material_post = object()
    with pytest.raises(ValueError, match="Unknown material"):
        MaterialFormatter.format(surface)
