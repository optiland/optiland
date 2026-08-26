"""ISO 10110 Element Drawing Generator

Rendering architecture
----------------------
Two isolated renderer classes handle format-specific output
(HarrisonKramer's architectural feedback), sharing layout logic via
:class:`~optiland.iso10110._base_renderer._BaseRenderer`:

* :class:`~optiland.iso10110._mpl_renderer._MatplotlibRenderer` — PDF/PNG.
* :class:`~optiland.iso10110._dxf_renderer._DxfRenderer` — DXF via ezdxf.

:class:`ElementDrawing` is a thin public coordinator that delegates to
whichever renderer ``save_pdf`` / ``save_dxf`` / ``save_png`` / ``show``
require.  Neither renderer class is part of the public API.

Paper / orientation
-------------------
``paper`` : "A5","A4","A3","A2","A1","LETTER","LEGAL","TABLOID"
``orientation`` : "portrait" (default) | "landscape"

Bernhard Lutzer, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.iso10110._dxf_renderer import _DxfRenderer
from optiland.iso10110._geometry import _paper_wh
from optiland.iso10110._mpl_renderer import _MatplotlibRenderer
from optiland.iso10110.style import DrawingStyle

if TYPE_CHECKING:
    from pathlib import Path

    import ezdxf


# ezdxf (the "manufacturing" extra) is imported lazily, only where DXF output
# is actually produced (ElementDrawing.generate(), dxf_to_png()) — not here at
# module scope — so that `import optiland.iso10110` and PDF/PNG-only usage
# (save_pdf, save_png, show) work without ezdxf installed. _DxfRenderer itself
# only operates on an already-constructed ezdxf document/modelspace passed in
# by the caller, so it needs no import of its own either.


# ═══════════════════════════════════════════════════════════════════════════════
# ElementDrawing — thin public coordinator
# ═══════════════════════════════════════════════════════════════════════════════


class ElementDrawing:
    """ISO 10110 fabrication drawing for one lens element.

    Delegates all rendering to :class:`_MatplotlibRenderer` (PDF/PNG) and
    :class:`_DxfRenderer` (DXF).  Rendering logic is fully isolated in those
    classes; this class handles only file-I/O and public API.

    Args:
        element:       :class:`~optiland.iso10110.elements.LensElement`
        spec:          :class:`~optiland.iso10110.spec.DrawingSpec`
        element_index: 0-based element number.
        paper:         Paper size string (default ``"A4"``).
        orientation:   ``"portrait"`` (default) or ``"landscape"``.
        total_sheets:  Total sheet count for X/Y numbering.
        rotational_axis_y: y-offset (optical mm) of the rotational axis when
            it differs from the optical axis.  Default ``0.0`` = coincident.
        style:         :class:`~optiland.iso10110.style.DrawingStyle` font-size
            scale factors and border margin. Defaults to ``DrawingStyle()``,
            reproducing the built-in appearance exactly.
    """

    def __init__(
        self,
        element,
        spec,
        element_index: int = 0,
        paper: str = "A4",
        orientation: str = "portrait",
        total_sheets: int = 1,
        rotational_axis_y: float = 0.0,
        style: DrawingStyle | None = None,
    ) -> None:
        self.element = element
        self.spec = spec
        self.element_index = element_index
        self.paper = paper.upper()
        self.orientation = orientation.lower()
        self.total_sheets = total_sheets
        self.rotational_axis_y = rotational_axis_y
        self.style = style if style is not None else DrawingStyle()
        self._pw, self._ph = _paper_wh(paper, orientation)
        self._doc = None

    # ── public ──────────────────────────────────────────────────────────────

    def generate(self) -> ezdxf.document.Drawing:
        """Build DXF document (for save_dxf)."""
        try:
            import ezdxf
            from ezdxf.math import Vec3 as _DxfVec3
        except ImportError as exc:
            raise ImportError(
                "ezdxf is required for DXF output. Install it with the "
                "'manufacturing' extra: pip install optiland[manufacturing]"
            ) from exc

        doc = ezdxf.new("R2010", units=4)
        doc.header["$INSUNITS"] = 4
        renderer = _DxfRenderer(
            self.element,
            self.spec,
            self.element_index,
            self.total_sheets,
            self._pw,
            self._ph,
            rotational_axis_y=self.rotational_axis_y,
            style=self.style,
        )
        renderer.render(doc)
        # ezdxf.write() calls update_extents() which overwrites $EXTMIN/$EXTMAX
        # from modelspace.dxf.extmin/extmax — and Vec3(0,0,0) is falsy, so the
        # origin corner is never written.  Override the method on this instance
        # so the paper-boundary extents always survive to the saved file.
        _pw, _ph = self._pw, self._ph

        def _paper_extents():
            doc.header["$EXTMIN"] = _DxfVec3(0.0, 0.0, 0.0)
            doc.header["$EXTMAX"] = _DxfVec3(_pw, _ph, 0.0)
            active = doc.active_layout()
            doc.header["$PEXTMIN"] = active.dxf.extmin
            doc.header["$PEXTMAX"] = active.dxf.extmax

        doc.update_extents = _paper_extents
        self._doc = doc
        return doc

    def save_dxf(self, path: str | Path) -> None:
        """Save native DXF file."""
        if self._doc is None:
            self.generate()
        self._doc.saveas(str(path))

    def save_pdf(self, path: str | Path) -> None:
        """Save clean PDF via matplotlib."""
        import matplotlib.pyplot as plt

        fig = self._mpl_figure()
        fig.savefig(
            str(path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.close(fig)

    def save_png(self, path: str | Path, dpi: int = 200) -> None:
        """Save PNG via matplotlib."""
        import matplotlib.pyplot as plt

        fig = self._mpl_figure()
        fig.savefig(str(path), dpi=dpi, facecolor="white", edgecolor="none")
        plt.close(fig)

    def show(self, figsize: tuple | None = None) -> None:
        """Display inline."""
        import matplotlib.pyplot as plt

        fig = self._mpl_figure(figsize=figsize)
        plt.show()
        plt.close(fig)

    def _mpl_figure(self, figsize: tuple | None = None):
        """Delegate to _MatplotlibRenderer (kept for backward compatibility)."""
        renderer = _MatplotlibRenderer(
            self.element,
            self.spec,
            self.element_index,
            self.total_sheets,
            self._pw,
            self._ph,
            rotational_axis_y=self.rotational_axis_y,
            style=self.style,
        )
        return renderer.render(figsize)


# ── DXF → PNG converter ───────────────────────────────────────────────────────


def dxf_to_png(
    dxf_path: str | Path,
    png_path: str | Path,
    dpi: int = 200,
) -> None:
    """Render a DXF file to PNG using ezdxf with a white background."""
    import ezdxf
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    doc = ezdxf.readfile(str(dxf_path))
    pw, ph = 210.0, 297.0
    try:
        extmin = doc.header["$EXTMIN"]
        extmax = doc.header["$EXTMAX"]
        pw = abs(extmax[0] - extmin[0]) or pw
        ph = abs(extmax[1] - extmin[1]) or ph
    except Exception:
        pass

    fig = plt.figure(figsize=(pw / 25.4, ph / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")

    cfg = Configuration.defaults().with_changes(
        background_policy=BackgroundPolicy.WHITE
    )
    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend, config=cfg).draw_layout(doc.modelspace(), finalize=True)
    ax.set_axis_off()
    fig.savefig(
        str(png_path), dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)


# ── convenience factory ───────────────────────────────────────────────────────


def draw_element(
    element,
    spec,
    element_index: int = 0,
    paper: str = "A4",
    orientation: str = "portrait",
    style: DrawingStyle | None = None,
) -> ElementDrawing:
    """Build and generate an :class:`ElementDrawing` for *element* in one call."""
    d = ElementDrawing(
        element,
        spec,
        element_index,
        paper=paper,
        orientation=orientation,
        style=style,
    )
    d.generate()
    return d
