"""Repeated material construction reuses catalog scoring and parsed files."""

from __future__ import annotations

import pytest

import optiland.materials._material_matching as matching
import optiland.materials.material_file as material_file
from optiland.materials import Material
from optiland.materials.registry import MaterialRegistry
from optiland.materials.warnings import OptilandMaterialWarning


@pytest.fixture
def fresh_cache():
    """Isolate the process-wide registry between cache tests."""
    registry = MaterialRegistry.instance()
    registry.clear_cache()
    yield registry
    registry.clear_cache()


def _count_calls(monkeypatch, module, name):
    calls = []
    original = getattr(module, name)

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


def test_scoring_is_reused_for_repeated_lookup(monkeypatch, fresh_cache):
    calls = _count_calls(monkeypatch, matching, "_find_matches")

    first = Material("N-BK7")
    assert len(calls) == 1

    second = Material("N-BK7")
    third = Material("N-BK7")
    assert len(calls) == 1
    assert second.filename == first.filename == third.filename


def test_yaml_parse_is_reused_for_repeated_lookup(monkeypatch, fresh_cache):
    calls = []
    original = material_file.yaml.load

    def spy(stream, Loader=None, **kwargs):
        if Loader is material_file._YAML_LOADER:
            calls.append(1)
        return original(stream, Loader=Loader, **kwargs)

    monkeypatch.setattr(material_file.yaml, "load", spy)

    Material("N-BK7")
    Material("N-BK7")
    assert len(calls) == 1


def test_clear_cache_forces_fresh_scoring(monkeypatch, fresh_cache):
    calls = _count_calls(monkeypatch, matching, "_find_matches")

    Material("N-BK7")
    assert len(calls) == 1

    fresh_cache.clear_cache()
    Material("N-BK7")
    assert len(calls) == 2


def test_fuzzy_match_warning_is_not_cached_away(fresh_cache):
    with pytest.warns(OptilandMaterialWarning):
        Material("N-BK7H")
    with pytest.warns(OptilandMaterialWarning):
        Material("N-BK7H")
