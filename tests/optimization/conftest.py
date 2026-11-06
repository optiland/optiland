"""Shared fixtures for the optimization tests."""

from __future__ import annotations

import pytest

from optiland import _epoch


@pytest.fixture(autouse=True)
def _epoch_check(monkeypatch):
    """Run every optimization test with the evaluation-epoch debug guard on."""
    monkeypatch.setenv(_epoch.CHECK_ENV, "1")
