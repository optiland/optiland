"""SystemOverviewSection — system-level summary.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.prescription._formatting import safe_eval
from optiland.prescription.document import KeyValueBlock, Section, TableBlock
from optiland.prescription.sections.base import BaseSection

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


class SystemOverviewSection(BaseSection):
    """Produces the 'System Overview' section."""

    def build(self, optic: Optic) -> Section:
        """Build the System Overview section.

        Args:
            optic: The optical system to describe.

        Returns:
            Section with key-value block plus wavelength and field tables.
        """
        surfaces = optic.surfaces.surfaces
        n_real = len(surfaces) - 2  # exclude object and image

        obj_thick = safe_eval(lambda: surfaces[0].thickness)
        last_real_idx = len(surfaces) - 2
        img_thick = safe_eval(lambda: surfaces[last_real_idx].thickness)

        ap = optic.aperture
        ap_type = type(ap).__name__
        ap_value = safe_eval(lambda: ap.value, infinity=False)

        stop_idx = safe_eval(lambda: optic.surfaces.stop_index, infinity=False)

        kv = KeyValueBlock(
            rows=[
                ("Name", optic.name or "(unnamed)"),
                ("Surfaces", str(n_real)),
                ("Stop Surface", stop_idx),
                ("Aperture Type", ap_type),
                ("Aperture Value", ap_value),
                ("Object Distance", obj_thick),
                ("Image Distance", img_thick),
            ]
        )

        wl_table = self._wavelength_table(optic)
        field_table = self._field_table(optic)

        return Section(title="System Overview", blocks=[kv, wl_table, field_table])

    @staticmethod
    def _wavelength_table(optic: Optic) -> TableBlock:
        wl_group = optic.wavelengths
        rows = []
        for i, w in enumerate(wl_group.wavelengths):
            primary = "✓" if w.is_primary else ""
            rows.append([str(i + 1), f"{w.value:.4f}", primary])
        return TableBlock(
            headers=["#", "Wavelength (µm)", "Primary"],
            rows=rows,
            title="Wavelengths",
        )

    @staticmethod
    def _field_table(optic: Optic) -> TableBlock:
        field_group = optic.fields
        field_type = type(field_group.field_definition).__name__
        rows = []
        for i, f in enumerate(field_group.fields):
            rows.append(
                [
                    str(i + 1),
                    field_type,
                    f"{float(f.x):.4f}",
                    f"{float(f.y):.4f}",
                    f"{float(f.weight):.4f}",
                ]
            )
        return TableBlock(
            headers=["#", "Type", "X", "Y", "Weight"],
            rows=rows,
            title="Fields",
        )
