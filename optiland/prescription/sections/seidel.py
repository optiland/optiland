"""SeidelAberrationSection — per-surface third-order aberration table.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.prescription._formatting import fmt_float
from optiland.prescription.document import Section, TableBlock, TextBlock
from optiland.prescription.sections.base import BaseSection

if TYPE_CHECKING:
    from optiland.optic.optic import Optic

_HEADERS = [
    "S#",
    "SI (Sph)",
    "SII (Coma)",
    "SIII (Astig)",
    "SIV (Petz)",
    "SV (Dist)",
    "CL (LCA)",
    "CT (TCA)",
]

_DEC = 6


class SeidelAberrationSection(BaseSection):
    """Produces the 'Seidel Aberration Coefficients' section."""

    def build(self, optic: Optic) -> Section:
        """Build the Seidel Aberration Coefficients section.

        Args:
            optic: The optical system to describe.

        Returns:
            Section with a TableBlock of per-surface Seidel coefficients.
        """
        try:
            return self._build_table(optic)
        except Exception as exc:
            return Section(
                title="Seidel Aberration Coefficients",
                blocks=[TextBlock(text=f"Seidel coefficients unavailable: {exc}")],
            )

    def _build_table(self, optic: Optic) -> Section:
        import numpy as np

        from optiland.aberrations.third_order import ThirdOrderAberrations

        ta = ThirdOrderAberrations(optic)
        ta._precalculations()

        # Raw transverse contributions per surface (un-scaled)
        TSC = ta._compute_over_surfaces(ta._TSC_term)
        CC = ta._compute_over_surfaces(ta._CC_term)
        TAC = ta._compute_over_surfaces(ta._TAC_term)
        TPC = ta._compute_over_surfaces(ta._TPC_term)
        DC = ta._compute_over_surfaces(ta._DC_term)
        TAchC = ta._compute_over_surfaces(ta._TAchC_term)
        TchC = ta._compute_over_surfaces(ta._TchC_term)

        # Factor that maps raw transverse terms to wavefront Seidel coefficients.
        # Matches _sum_seidels: S = -sum(raw) * n[-1]*ua[-1]*2, so per-surface
        # contribution is s_i = -raw[i] * n[-1]*ua[-1]*2.
        factor = float(np.asarray(ta._n[-1] * ta._ua[-1] * 2).flat[0])

        def _scale(raw: object) -> list[float]:
            return [-float(np.asarray(v).flat[0]) * factor for v in raw]  # type: ignore[union-attr]

        si = _scale(TSC)
        sii = _scale(CC)
        siii = _scale(TAC)
        siv = _scale(TPC)
        sv = _scale(DC)

        n_wl = len(optic.wavelengths.wavelengths)
        mono = n_wl < 2

        cl = _scale(TAchC) if not mono else [0.0] * len(si)
        ct = _scale(TchC) if not mono else [0.0] * len(si)

        surfaces = optic.surfaces.surfaces
        # Real surfaces: skip object (index 0) and image (last)
        real_indices = list(range(1, len(surfaces) - 1))

        rows: list[list[str]] = []
        for col_i, surf_i in enumerate(real_indices):
            cl_str = "N/A" if mono else fmt_float(cl[col_i], decimals=_DEC)
            ct_str = "N/A" if mono else fmt_float(ct[col_i], decimals=_DEC)
            rows.append(
                [
                    str(surf_i),
                    fmt_float(si[col_i], decimals=_DEC),
                    fmt_float(sii[col_i], decimals=_DEC),
                    fmt_float(siii[col_i], decimals=_DEC),
                    fmt_float(siv[col_i], decimals=_DEC),
                    fmt_float(sv[col_i], decimals=_DEC),
                    cl_str,
                    ct_str,
                ]
            )

        # Total row — use seidels() for SI-SV (guaranteed to match reference values)
        totals = optic.aberrations.seidels()
        t_si = fmt_float(float(np.asarray(totals[0]).flat[0]), decimals=_DEC)
        t_sii = fmt_float(float(np.asarray(totals[1]).flat[0]), decimals=_DEC)
        t_siii = fmt_float(float(np.asarray(totals[2]).flat[0]), decimals=_DEC)
        t_siv = fmt_float(float(np.asarray(totals[3]).flat[0]), decimals=_DEC)
        t_sv = fmt_float(float(np.asarray(totals[4]).flat[0]), decimals=_DEC)
        t_cl = "N/A" if mono else fmt_float(sum(cl), decimals=_DEC)
        t_ct = "N/A" if mono else fmt_float(sum(ct), decimals=_DEC)

        rows.append(["Total", t_si, t_sii, t_siii, t_siv, t_sv, t_cl, t_ct])

        table = TableBlock(headers=_HEADERS, rows=rows)
        return Section(title="Seidel Aberration Coefficients", blocks=[table])
