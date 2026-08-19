"""Tests for AnalysisRunner."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

from optiland_gui.services.analysis_runner import AnalysisRunner


@pytest.fixture()
def runner():
    conn = MagicMock()
    conn._optic = None
    return AnalysisRunner(conn)


class TestAnalysisRunner:
    def test_registry_contains_spot_diagram(self, runner):
        registry = runner.get_analysis_registry()
        names = [name for _, name, _ in registry]
        assert any("spot" in name.lower() for name in names)

    def test_all_class_paths_importable(self, runner):
        from optiland_gui.registry import ANALYSIS_REGISTRY

        for _category, name, class_path in ANALYSIS_REGISTRY:
            module_path, class_name = class_path.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                assert cls is not None
            except (ImportError, AttributeError) as exc:
                pytest.fail(f"Could not import '{name}' at '{class_path}': {exc}")

    def test_run_does_not_raise(self, runner, minimal_optic):
        # run() is a stub — must not raise regardless of inputs
        runner.run("Spot Diagram", {}, minimal_optic)

    def test_stop_does_not_raise(self, runner):
        runner.stop()
