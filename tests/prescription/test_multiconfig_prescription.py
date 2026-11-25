"""Tests for MultiConfigPrescription and its multi-configuration sections."""

from __future__ import annotations

import pytest

from optiland.materials import Material
from optiland.multiconfig import MultiConfiguration
from optiland.prescription import (
    BaseMultiConfigSection,
    ConfigurationTableSection,
    FirstOrderSection,
    MultiConfigPrescription,
    PerConfigurationSection,
    Prescription,
    SystemOverviewSection,
)
from optiland.prescription.document import Document, Section, TableBlock
from optiland.prescription.renderers.console import ConsoleRenderer
from optiland.prescription.renderers.plain_text import PlainTextRenderer
from optiland.samples.objectives import CookeTriplet


@pytest.fixture
def zoom():
    """Three-configuration 'zoom' of the Cooke triplet."""
    lens = CookeTriplet()
    lens.name = "Cooke Zoom"
    mc = MultiConfiguration(lens)
    mc.add_configuration()
    mc.add_configuration()
    mc.set_thickness(2, 3.0, configurations=[1])
    mc.set_thickness(2, 1.5, configurations=[2])
    mc.set_material(3, Material("N-SF5"), configurations=[2])
    mc.configurations[2].set_aperture("EPD", 8.0)
    for optic in mc.configurations:
        optic.updater.update()
    return mc


VARYING_ROWS = ["Aperture value", "Surface 2 thickness", "Surface 3 material"]


def _assert_report_content(text: str) -> None:
    for i in range(3):
        assert f"Config {i}" in text
        assert f"Configuration {i}" in text
    for row in VARYING_ROWS:
        assert row in text
    assert "N-SF5" in text


def test_plain_text_contains_configurations_and_varying_parameters(zoom):
    text = PlainTextRenderer().render(MultiConfigPrescription(zoom).build())
    _assert_report_content(text)
    assert "MULTI-CONFIGURATION PRESCRIPTION — COOKE ZOOM (3 CONFIGURATIONS)" in text


def test_console_contains_configurations_and_varying_parameters(zoom):
    text = ConsoleRenderer().render(MultiConfigPrescription(zoom).build())
    _assert_report_content(text)


def test_default_section_order(zoom):
    doc = MultiConfigPrescription(zoom).build()
    assert isinstance(doc, Document)
    assert [s.title for s in doc.sections] == [
        "System Overview (Configuration 0)",
        "Configurations",
        "First-Order Properties by Configuration",
        "Surface Data — Geometry (Configuration 0)",
        "Surface Data — Materials (Configuration 0)",
    ]


def test_configuration_table_matches_summary(zoom):
    section = ConfigurationTableSection().build(zoom)
    (table,) = section.blocks
    assert isinstance(table, TableBlock)
    assert table.headers == ["Parameter", "Config 0", "Config 1", "Config 2"]
    frame = zoom.summary()
    assert [row[0] for row in table.rows] == list(frame.index)
    thickness = next(r for r in table.rows if r[0] == "Surface 2 thickness")
    assert thickness[1:] == ["6.0076", "3.0000", "1.5000"]


def test_configuration_table_marks_failed_values(zoom, monkeypatch):
    def _raise():
        raise ValueError("no paraxial EFL for this system")

    monkeypatch.setattr(zoom.configurations[1].paraxial, "f2", _raise)
    section = ConfigurationTableSection().build(zoom)
    efl = next(r for r in section.blocks[0].rows if r[0] == "EFL")
    assert efl[2] == "N/A"
    assert efl[1] != "N/A"


def test_per_configuration_section_labels_each_configuration(zoom):
    section = PerConfigurationSection(SystemOverviewSection()).build(zoom)
    titles = [getattr(b, "title", None) for b in section.blocks]
    for i in range(3):
        assert f"Configuration {i}" in titles
        assert f"Configuration {i} — Wavelengths" in titles
        assert f"Configuration {i} — Fields" in titles


def test_custom_sections(zoom):
    class CountSection(BaseMultiConfigSection):
        def build(self, multiconfig):
            n = len(multiconfig.configurations)
            return Section(title=f"{n} configurations")

    doc = MultiConfigPrescription(
        zoom, sections=[CountSection(), FirstOrderSection()]
    ).build()
    assert [s.title for s in doc.sections] == [
        "3 configurations",
        "First-Order Properties (Configuration 0)",
    ]


def test_save_plain_text(zoom, tmp_path):
    path = tmp_path / "zoom.txt"
    MultiConfigPrescription(zoom).save(path)
    _assert_report_content(path.read_text(encoding="utf-8"))


def test_view_prints(zoom, capsys):
    MultiConfigPrescription(zoom).view()
    assert "Configuration 2" in capsys.readouterr().out


def test_prescription_still_rejects_multiconfiguration(zoom):
    with pytest.raises(NotImplementedError, match="MultiConfigPrescription"):
        Prescription(zoom)
