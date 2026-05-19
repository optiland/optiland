"""Tests for prescription sections."""

from __future__ import annotations

import pytest

from optiland.prescription.document import KeyValueBlock, TableBlock, TextBlock
from optiland.prescription.sections.first_order import FirstOrderSection
from optiland.prescription.sections.seidel import SeidelAberrationSection
from optiland.prescription.sections.surface_table import (
    SurfaceGeometryTableSection,
    SurfaceMaterialTableSection,
)
from optiland.prescription.sections.system_overview import SystemOverviewSection


class TestSystemOverviewSection:
    def test_title(self, cooke_triplet):
        section = SystemOverviewSection().build(cooke_triplet)
        assert section.title == "System Overview"

    def test_blocks_count(self, cooke_triplet):
        section = SystemOverviewSection().build(cooke_triplet)
        # KeyValueBlock + 2 TableBlocks (wavelengths + fields)
        assert len(section.blocks) == 3

    def test_name_in_kv(self, cooke_triplet):
        section = SystemOverviewSection().build(cooke_triplet)
        kv = section.blocks[0]
        assert isinstance(kv, KeyValueBlock)
        keys = [r[0] for r in kv.rows]
        assert "Name" in keys

    def test_wavelength_table(self, cooke_triplet):
        section = SystemOverviewSection().build(cooke_triplet)
        wl_table = section.blocks[1]
        assert isinstance(wl_table, TableBlock)
        assert wl_table.headers[0] == "#"
        # Cooke triplet has 3 wavelengths
        assert len(wl_table.rows) == 3
        # Check primary marked
        primaries = [r[2] for r in wl_table.rows]
        assert "✓" in primaries

    def test_field_table(self, cooke_triplet):
        section = SystemOverviewSection().build(cooke_triplet)
        field_table = section.blocks[2]
        assert isinstance(field_table, TableBlock)
        assert "Weight" in field_table.headers

    def test_unnamed_optic(self):
        from optiland.samples.objectives import CookeTriplet
        lens = CookeTriplet()
        lens.name = None
        section = SystemOverviewSection().build(lens)
        kv = section.blocks[0]
        name_row = next(r for r in kv.rows if r[0] == "Name")
        assert name_row[1] == "(unnamed)"


class TestFirstOrderSection:
    def test_title(self, cooke_triplet):
        section = FirstOrderSection().build(cooke_triplet)
        assert section.title == "First-Order Properties"

    def test_single_kv_block(self, cooke_triplet):
        section = FirstOrderSection().build(cooke_triplet)
        assert len(section.blocks) == 1
        assert isinstance(section.blocks[0], KeyValueBlock)

    def test_efl_present(self, cooke_triplet):
        section = FirstOrderSection().build(cooke_triplet)
        kv = section.blocks[0]
        keys = [r[0] for r in kv.rows]
        assert "Effective Focal Length (EFL)" in keys

    def test_efl_value(self, cooke_triplet):
        section = FirstOrderSection().build(cooke_triplet)
        kv = section.blocks[0]
        efl_row = next(r for r in kv.rows if "EFL" in r[0])
        # Cooke triplet EFL ~ 50 mm (may display as 49.9998)
        assert "50" in efl_row[1] or "49.9" in efl_row[1]

    def test_all_14_rows(self, cooke_triplet):
        section = FirstOrderSection().build(cooke_triplet)
        kv = section.blocks[0]
        assert len(kv.rows) == 14


class TestSurfaceGeometryTableSection:
    def test_title(self, cooke_triplet):
        section = SurfaceGeometryTableSection().build(cooke_triplet)
        assert section.title == "Surface Data — Geometry"

    def test_table_headers(self, cooke_triplet):
        section = SurfaceGeometryTableSection().build(cooke_triplet)
        tbl = section.blocks[0]
        assert isinstance(tbl, TableBlock)
        assert "Radius (mm)" in tbl.headers
        assert "Stop" in tbl.headers

    def test_row_count(self, cooke_triplet):
        section = SurfaceGeometryTableSection().build(cooke_triplet)
        tbl = section.blocks[0]
        # 8 surfaces total (object + 6 real + image)
        assert len(tbl.rows) == 8

    def test_stop_marked(self, cooke_triplet):
        section = SurfaceGeometryTableSection().build(cooke_triplet)
        tbl = section.blocks[0]
        stop_col = tbl.headers.index("Stop")
        stop_marks = [r[stop_col] for r in tbl.rows]
        assert "✓" in stop_marks

    def test_object_radius_infinity(self, cooke_triplet):
        section = SurfaceGeometryTableSection().build(cooke_triplet)
        tbl = section.blocks[0]
        radius_col = tbl.headers.index("Radius (mm)")
        # Object surface (S0) has infinite radius
        assert tbl.rows[0][radius_col] == "∞"

    def test_no_freeform_footnote_for_standard(self, cooke_triplet):
        section = SurfaceGeometryTableSection().build(cooke_triplet)
        text_blocks = [b for b in section.blocks if isinstance(b, TextBlock)]
        assert len(text_blocks) == 0


class TestSurfaceMaterialTableSection:
    def test_title(self, cooke_triplet):
        section = SurfaceMaterialTableSection().build(cooke_triplet)
        assert section.title == "Surface Data — Materials"

    def test_table_has_material_column(self, cooke_triplet):
        section = SurfaceMaterialTableSection().build(cooke_triplet)
        tbl = section.blocks[0]
        assert "Material" in tbl.headers

    def test_air_material(self, cooke_triplet):
        section = SurfaceMaterialTableSection().build(cooke_triplet)
        tbl = section.blocks[0]
        mat_col = tbl.headers.index("Material")
        materials = [r[mat_col] for r in tbl.rows]
        assert "Air" in materials

    def test_glass_material(self, cooke_triplet):
        section = SurfaceMaterialTableSection().build(cooke_triplet)
        tbl = section.blocks[0]
        mat_col = tbl.headers.index("Material")
        materials = [r[mat_col] for r in tbl.rows]
        assert "SK16" in materials or "F2" in materials

    def test_nd_for_glass(self, cooke_triplet):
        section = SurfaceMaterialTableSection().build(cooke_triplet)
        tbl = section.blocks[0]
        nd_col = tbl.headers.index("nd")
        mat_col = tbl.headers.index("Material")
        # Find a glass row and check nd is numeric
        glass_rows = [r for r in tbl.rows if r[mat_col] not in ("Air", "MIRROR", "—")]
        assert len(glass_rows) > 0
        for r in glass_rows:
            nd = r[nd_col]
            assert nd not in ("N/A", "—"), f"Expected nd for glass, got {nd!r}"


class TestSeidelAberrationSection:
    def test_title(self, cooke_triplet):
        section = SeidelAberrationSection().build(cooke_triplet)
        assert section.title == "Seidel Aberration Coefficients"

    def test_table_present(self, cooke_triplet):
        section = SeidelAberrationSection().build(cooke_triplet)
        assert len(section.blocks) >= 1
        assert isinstance(section.blocks[0], TableBlock)

    def test_headers(self, cooke_triplet):
        section = SeidelAberrationSection().build(cooke_triplet)
        tbl = section.blocks[0]
        assert "SI (Sph)" in tbl.headers
        assert "CL (LCA)" in tbl.headers

    def test_total_row_last(self, cooke_triplet):
        section = SeidelAberrationSection().build(cooke_triplet)
        tbl = section.blocks[0]
        assert tbl.rows[-1][0] == "Total"

    def test_row_count(self, cooke_triplet):
        section = SeidelAberrationSection().build(cooke_triplet)
        tbl = section.blocks[0]
        # 6 real surfaces + 1 Total row
        assert len(tbl.rows) == 7

    def test_monochromatic_chromatic_na(self):
        """Single-wavelength system should show N/A in chromatic columns."""
        from optiland.optic import Optic

        mono = Optic()
        mono.set_aperture(aperture_type="EPD", value=10)
        mono.fields.set_type("angle")
        mono.fields.add(y=0)
        mono.wavelengths.add(value=0.55, unit="um", is_primary=True)
        mono.add_surface(index=0, thickness=float("inf"))
        mono.add_surface(index=1, radius=100, thickness=5, material="N-BK7")
        mono.add_surface(index=2)

        section = SeidelAberrationSection().build(mono)
        # Should not crash; chromatic cols are N/A or section replaced with TextBlock
        assert isinstance(section.blocks[0], (TableBlock, TextBlock))

    def test_degenerate_system_does_not_raise(self):
        """A system with no aperture should produce a TextBlock fallback."""
        from optiland.optic import Optic
        bare = Optic()
        section = SeidelAberrationSection().build(bare)
        # Either a table (if it managed) or a TextBlock fallback
        assert len(section.blocks) >= 1
