"""Integration tests for PDFRenderer."""

from __future__ import annotations

import pytest

reportlab = pytest.importorskip("reportlab")

from optiland.prescription import Prescription
from optiland.prescription.renderers.pdf import PDFRenderer


def test_pdf_save(cooke_triplet, tmp_path):
    out = tmp_path / "report.pdf"
    Prescription(cooke_triplet).save(out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_pdf_magic_bytes(cooke_triplet, tmp_path):
    out = tmp_path / "report.pdf"
    Prescription(cooke_triplet).save(out)
    with open(out, "rb") as fh:
        header = fh.read(4)
    assert header == b"%PDF"


def test_pdf_extension_inferred(cooke_triplet, tmp_path):
    out = tmp_path / "out.pdf"
    Prescription(cooke_triplet).save(out)
    assert out.stat().st_size > 1000
