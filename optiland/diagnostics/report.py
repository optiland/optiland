"""Diagnostic report types.

Defines the `Severity`, `Diagnostic`, and `DiagnosticReport` types produced by
`optiland.diagnostics.check_system`. This module contains no logic that
inspects an `Optic` — that lives in `checks.py`.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class Severity(Enum):
    """Severity level of a diagnostic finding."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Diagnostic:
    """A single diagnostic finding produced by a system check.

    Attributes:
        severity: How serious the finding is.
        code: Stable identifier for the check that produced this finding,
            e.g. ``"OPT001"``.
        message: What is wrong, stating the offending object and value.
        fix: What to do about it, as a runnable line of code where possible.
        where: The surface index or object reference the finding concerns,
            or ``None`` if it applies to the system as a whole.
        doc_url: Deep link into the documentation for this check.
    """

    severity: Severity
    code: str
    message: str
    fix: str
    where: int | str | None = None
    doc_url: str | None = None

    def __str__(self) -> str:
        location = f" (surface {self.where})" if isinstance(self.where, int) else ""
        severity = self.severity.value.upper()
        return f"[{self.code}] {severity}{location}: {self.message} {self.fix}"


class DiagnosticReport:
    """The result of running `check_system` on an `Optic`.

    Attributes:
        diagnostics: All findings, in the order the checks ran.
    """

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        """Initialize a DiagnosticReport.

        Args:
            diagnostics: All findings, in the order the checks ran.
        """
        self.diagnostics = diagnostics

    @property
    def errors(self) -> list[Diagnostic]:
        """list[Diagnostic]: Findings with `Severity.ERROR`."""
        return [d for d in self.diagnostics if d.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Diagnostic]:
        """list[Diagnostic]: Findings with `Severity.WARNING`."""
        return [d for d in self.diagnostics if d.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """bool: True if no error-severity findings were reported.

        Warnings do not affect this — a system with only warnings is
        considered usable, if imperfect.
        """
        return len(self.errors) == 0

    def __bool__(self) -> bool:
        return self.ok

    def __len__(self) -> int:
        return len(self.diagnostics)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self.diagnostics)

    def __repr__(self) -> str:
        return (
            f"DiagnosticReport(errors={len(self.errors)}, "
            f"warnings={len(self.warnings)})"
        )

    def __str__(self) -> str:
        if not self.diagnostics:
            return "DiagnosticReport: no issues found."
        lines = [
            f"DiagnosticReport: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s)"
        ]
        for d in self.errors:
            lines.append(f"  {d}")
        for d in self.warnings:
            lines.append(f"  {d}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        if not self.diagnostics:
            return "<p><strong>DiagnosticReport:</strong> no issues found.</p>"

        rows = []
        for d in self.diagnostics:
            color = "#c0392b" if d.severity is Severity.ERROR else "#b7791f"
            location = "" if d.where is None else str(d.where)
            doc_link = f' &middot; <a href="{d.doc_url}">docs</a>' if d.doc_url else ""
            rows.append(
                "<tr>"
                f'<td style="color:{color};font-weight:bold;">{d.severity.value}</td>'
                f"<td>{d.code}</td>"
                f"<td>{location}</td>"
                f"<td>{d.message}</td>"
                f"<td><code>{d.fix}</code>{doc_link}</td>"
                "</tr>"
            )

        header = (
            "<tr>"
            "<th>Severity</th><th>Code</th><th>Where</th>"
            "<th>Message</th><th>Fix</th>"
            "</tr>"
        )
        return (
            "<table><thead>"
            + header
            + "</thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
