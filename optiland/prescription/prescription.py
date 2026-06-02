"""Prescription orchestrator.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from optiland.prescription.document import Document
from optiland.prescription.sections.first_order import FirstOrderSection
from optiland.prescription.sections.seidel import SeidelAberrationSection
from optiland.prescription.sections.surface_table import (
    SurfaceGeometryTableSection,
    SurfaceMaterialTableSection,
)
from optiland.prescription.sections.system_overview import SystemOverviewSection

if TYPE_CHECKING:
    import os

    from optiland.optic.optic import Optic
    from optiland.prescription.renderers.base import BaseRenderer
    from optiland.prescription.sections.base import BaseSection

_DEFAULT_SECTIONS: list[BaseSection] = [
    SystemOverviewSection(),
    FirstOrderSection(),
    SurfaceGeometryTableSection(),
    SurfaceMaterialTableSection(),
    SeidelAberrationSection(),
]


class Prescription:
    """Generates an optical prescription report from an Optic instance.

    Args:
        optic: The optical system to describe.
        sections: Optional list of section instances.  Defaults to the
            standard five-section set when None.

    Raises:
        NotImplementedError: If a MultiConfiguration object is passed.
    """

    def __init__(
        self,
        optic: Optic,
        sections: list[BaseSection] | None = None,
    ) -> None:
        from optiland.multiconfig import MultiConfiguration

        if isinstance(optic, MultiConfiguration):
            raise NotImplementedError(
                "Prescription does not support MultiConfiguration objects. "
                "Pass a single Optic instance.  Access individual configs via "
                "multi_config.configs[i] and generate a prescription for each."
            )
        self._optic = optic
        self._sections: list[BaseSection] = (
            list(sections) if sections is not None else list(_DEFAULT_SECTIONS)
        )

    def build(self) -> Document:
        """Build and return the Document model without rendering.

        Returns:
            Populated Document ready for a renderer to consume.
        """
        title = f"Optical Prescription — {self._optic.name or 'Unnamed System'}"
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        sections = [s.build(self._optic) for s in self._sections]
        return Document(title=title, generated_at=generated_at, sections=sections)

    def view(self) -> None:
        """Print the prescription to the console using ConsoleRenderer."""
        from optiland.prescription.renderers.console import ConsoleRenderer

        ConsoleRenderer().print(self.build())

    def save(
        self,
        path: str | os.PathLike,
        renderer: BaseRenderer | None = None,
    ) -> None:
        """Save the prescription to a file.

        Args:
            path: Output file path.  Extension determines the renderer
                when renderer is None: ``.pdf`` → PDFRenderer, anything
                else → PlainTextRenderer.
            renderer: Explicit renderer instance.  Overrides extension
                inference.
        """
        path = pathlib.Path(path)
        if renderer is None:
            if path.suffix.lower() == ".pdf":
                from optiland.prescription.renderers.pdf import PDFRenderer

                renderer = PDFRenderer()
            else:
                from optiland.prescription.renderers.plain_text import (
                    PlainTextRenderer,
                )

                renderer = PlainTextRenderer()
        renderer.write(self.build(), path)

    @staticmethod
    def _default_sections() -> list[BaseSection]:
        """Return fresh instances of the default section set."""
        return [
            SystemOverviewSection(),
            FirstOrderSection(),
            SurfaceGeometryTableSection(),
            SurfaceMaterialTableSection(),
            SeidelAberrationSection(),
        ]
