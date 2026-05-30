"""FirstOrderSection — first-order / paraxial properties.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.prescription._formatting import safe_eval
from optiland.prescription.document import KeyValueBlock, Section
from optiland.prescription.sections.base import BaseSection

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


class FirstOrderSection(BaseSection):
    """Produces the 'First-Order Properties' section."""

    def build(self, optic: Optic) -> Section:
        """Build the First-Order Properties section.

        Args:
            optic: The optical system to describe.

        Returns:
            Section containing a KeyValueBlock of paraxial properties.
        """
        p = optic.paraxial
        rows = [
            ("Effective Focal Length (EFL)", safe_eval(lambda: p.f2())),
            ("Front Focal Length (FFL)", safe_eval(lambda: p.f1())),
            ("Back Focal Length (BFL)", safe_eval(lambda: p.F2())),
            ("Front Principal Plane (P1)", safe_eval(lambda: p.P1())),
            ("Back Principal Plane (P2)", safe_eval(lambda: p.P2())),
            ("Front Nodal Plane (N1)", safe_eval(lambda: p.N1())),
            ("Back Nodal Plane (N2)", safe_eval(lambda: p.N2())),
            ("Image-Space F/#", safe_eval(lambda: p.FNO())),
            ("Entrance Pupil Diameter", safe_eval(lambda: p.EPD())),
            ("Entrance Pupil Location", safe_eval(lambda: p.EPL())),
            ("Exit Pupil Diameter", safe_eval(lambda: p.XPD())),
            ("Exit Pupil Location", safe_eval(lambda: p.XPL())),
            ("Transverse Magnification", safe_eval(lambda: p.magnification())),
            ("Lagrange Invariant", safe_eval(lambda: p.invariant())),
        ]
        return Section(
            title="First-Order Properties",
            blocks=[KeyValueBlock(rows=rows)],
        )
