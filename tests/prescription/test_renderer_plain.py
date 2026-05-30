"""Integration tests for PlainTextRenderer."""

from __future__ import annotations

from optiland.prescription import Prescription
from optiland.prescription.renderers.plain_text import PlainTextRenderer


def test_save_plain_text(cooke_triplet, tmp_path):
    out = tmp_path / "report.txt"
    Prescription(cooke_triplet).save(out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert len(content) > 100
    assert "EFL" in content or "Effective Focal" in content


def test_plain_text_contains_title(cooke_triplet):
    renderer = PlainTextRenderer()
    doc = Prescription(cooke_triplet).build()
    text = renderer.render(doc)
    assert "OPTICAL PRESCRIPTION" in text.upper()


def test_plain_text_contains_surface_section(cooke_triplet):
    renderer = PlainTextRenderer()
    doc = Prescription(cooke_triplet).build()
    text = renderer.render(doc)
    assert "Surface Data" in text


def test_plain_text_contains_infinity_symbol(cooke_triplet):
    renderer = PlainTextRenderer()
    doc = Prescription(cooke_triplet).build()
    text = renderer.render(doc)
    assert "∞" in text


def test_plain_text_extension_inferred(cooke_triplet, tmp_path):
    """Saving with .txt extension uses PlainTextRenderer by default."""
    out = tmp_path / "out.txt"
    Prescription(cooke_triplet).save(out)
    assert out.read_text(encoding="utf-8").strip() != ""
