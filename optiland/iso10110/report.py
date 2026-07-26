"""ISO 10110 System Report

Batches :class:`~optiland.iso10110.drawing.ElementDrawing` instances for every
element in a :class:`~optiland.iso10110.spec.DrawingSpec` and provides
combined save/preview methods.

Bernhard Lutzer, 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.iso10110.spec import DrawingSpec

from optiland.iso10110.drawing import ElementDrawing
from optiland.iso10110.style import DrawingStyle


class ISO10110Report:
    """Generate ISO 10110 drawings for all elements in a system.

    Args:
        spec: The :class:`~optiland.iso10110.spec.DrawingSpec` that carries
            both the optic geometry and all tolerance annotations.
        style: :class:`~optiland.iso10110.style.DrawingStyle` font-size scale
            factors and border margin, applied to every element's drawing.
            Defaults to ``DrawingStyle()``, reproducing the built-in
            appearance exactly.

    Example::

        report = ISO10110Report(spec)
        report.generate()
        report.save_dxf("output/")
        report.save_pdf("output/system_drawings.pdf")
        report.show()
    """

    def __init__(
        self,
        spec: DrawingSpec,
        paper: str = "A4",
        orientation: str = "portrait",
        style: DrawingStyle | None = None,
    ) -> None:
        self.spec = spec
        self.paper = paper
        self.orientation = orientation
        self.style = style if style is not None else DrawingStyle()
        self._drawings: list[ElementDrawing] = []

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self) -> list[ElementDrawing]:
        """Build an :class:`ElementDrawing` for every element.

        Note this only constructs the :class:`ElementDrawing` instances — it
        does not eagerly build each one's DXF document, so calling this (and
        the PDF/PNG/show methods below, which call it automatically) never
        requires ``ezdxf`` to be installed. ``ElementDrawing.generate()``
        (DXF-specific) is invoked lazily by :meth:`save_dxf` instead.

        Returns:
            List of generated :class:`ElementDrawing` instances.
        """
        total = len(self.spec.elements)
        self._drawings = [
            ElementDrawing(
                element,
                self.spec,
                idx,
                paper=self.paper,
                orientation=self.orientation,
                total_sheets=total,
                style=self.style,
            )
            for idx, element in enumerate(self.spec.elements)
        ]
        return self._drawings

    # ------------------------------------------------------------------
    # DXF export
    # ------------------------------------------------------------------

    def save_dxf(self, directory: str | Path) -> list[Path]:
        """Save one DXF file per element into *directory*.

        Returns:
            List of written file paths.
        """
        if not self._drawings:
            self.generate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for d in self._drawings:
            proj = (self.spec.project_name or "element").replace(" ", "_")
            fname = directory / f"{proj}_E{d.element_index + 1:02d}.dxf"
            d.save_dxf(fname)
            paths.append(fname)
        return paths

    # ------------------------------------------------------------------
    # PDF export
    # ------------------------------------------------------------------

    def save_pdf(
        self,
        path: str | Path,
        combined: bool = True,
    ) -> Path:
        """Save drawings to PDF.

        Args:
            path: If *combined* is True, the path of the combined multi-page
                PDF.  If False, treated as a directory and one PDF per element
                is written.
            combined: When True (default) all elements go into a single
                multi-page PDF.

        Returns:
            Path to the written file (or the directory when *combined=False*).
        """
        if not self._drawings:
            self.generate()

        path = Path(path)

        if combined:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            path.parent.mkdir(parents=True, exist_ok=True)
            with PdfPages(str(path)) as pdf:
                for d in self._drawings:
                    fig = d._mpl_figure()
                    pdf.savefig(
                        fig, bbox_inches="tight", facecolor="white", edgecolor="none"
                    )
                    plt.close(fig)
            return path
        else:
            path.mkdir(parents=True, exist_ok=True)
            proj = (self.spec.project_name or "element").replace(" ", "_")
            for d in self._drawings:
                pdf_path = path / f"{proj}_E{d.element_index + 1:02d}.pdf"
                d.save_pdf(pdf_path)
            return path

    # ------------------------------------------------------------------
    # PNG export
    # ------------------------------------------------------------------

    def save_png(
        self,
        directory: str | Path,
        dpi: int = 200,
    ) -> list[Path]:
        """Save one PNG file per element into *directory*.

        Args:
            directory: Destination folder (created if needed).
            dpi:       Resolution (default 200).

        Returns:
            List of written file paths.
        """
        if not self._drawings:
            self.generate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        proj = (self.spec.project_name or "element").replace(" ", "_")
        for d in self._drawings:
            png_path = directory / f"{proj}_E{d.element_index + 1:02d}.png"
            d.save_png(png_path, dpi=dpi)
            paths.append(png_path)
        return paths

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def show(
        self,
        element_index: int | None = None,
        figsize: tuple | None = None,
    ) -> None:
        """Display drawing(s) using matplotlib.

        Args:
            element_index: If given, show only that element (0-based).
                If None, show all elements sequentially.
            figsize: Optional figure size in inches.
        """
        if not self._drawings:
            self.generate()

        if element_index is not None:
            self._drawings[element_index].show(figsize=figsize)
        else:
            for d in self._drawings:
                d.show(figsize=figsize)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def drawings(self) -> list[ElementDrawing]:
        """List of generated :class:`ElementDrawing` instances."""
        return self._drawings

    def __len__(self) -> int:
        return len(self._drawings)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ISO10110Report({len(self.spec.elements)} elements)"
