"""System-level diagnostics for Optiland optical systems.

`check_system` inspects an `Optic` and reports the failure modes newcomers
hit most often — a missing wavelength, an undefined aperture, a stop that
was never marked — each with the offending object and a runnable fix.

This module only *reads* an `Optic`; it adds no responsibilities to it and
is not imported by `optiland.optic`.

Example::

    from optiland.diagnostics import check_system

    report = check_system(lens)
    if not report.ok:
        print(report)

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.diagnostics.checks import CHECKS, SystemCheck
from optiland.diagnostics.report import Diagnostic, DiagnosticReport, Severity

if TYPE_CHECKING:
    from optiland.optic.optic import Optic

__all__ = [
    "CHECKS",
    "Diagnostic",
    "DiagnosticReport",
    "Severity",
    "SystemCheck",
    "check_system",
]


def check_system(lens: Optic) -> DiagnosticReport:
    """Run all registered diagnostic checks against an optical system.

    Args:
        lens: The optical system to inspect. It is only read, never
            modified.

    Returns:
        A `DiagnosticReport` collecting every finding across all checks.

    Example::

        from optiland.diagnostics import check_system
        from optiland.optic import Optic

        lens = Optic()
        report = check_system(lens)
        report.ok  # False - a fresh Optic has no wavelengths, aperture, ...
    """
    diagnostics: list[Diagnostic] = []
    for check in CHECKS:
        diagnostics.extend(check(lens))
    return DiagnosticReport(diagnostics)
