"""Integration tests for ConsoleRenderer."""

from __future__ import annotations

import io

import pytest

rich = pytest.importorskip("rich")

from optiland.prescription import Prescription
from optiland.prescription.renderers.console import ConsoleRenderer


def test_console_render_returns_string(cooke_triplet):
    renderer = ConsoleRenderer()
    doc = Prescription(cooke_triplet).build()
    output = renderer.render(doc)
    assert isinstance(output, str)
    assert len(output) > 0


def test_console_render_contains_title(cooke_triplet):
    renderer = ConsoleRenderer()
    doc = Prescription(cooke_triplet).build()
    output = renderer.render(doc)
    assert "Cooke Triplet" in output or "Optical Prescription" in output


def test_console_print_does_not_raise(cooke_triplet):
    """ConsoleRenderer.print() should not raise."""
    from rich.console import Console

    renderer = ConsoleRenderer()
    doc = Prescription(cooke_triplet).build()
    buf = io.StringIO()
    console = Console(file=buf, highlight=False)
    renderer._write_to_console(console, doc)
    output = buf.getvalue()
    assert len(output) > 0


def test_console_write_to_file(cooke_triplet, tmp_path):
    out = tmp_path / "report_rich.txt"
    renderer = ConsoleRenderer()
    doc = Prescription(cooke_triplet).build()
    renderer.write(doc, out)
    assert out.exists()
    assert out.stat().st_size > 0
