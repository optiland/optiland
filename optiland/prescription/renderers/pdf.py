"""PDFRenderer — reportlab-based PDF output.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.prescription.renderers.base import BaseRenderer

if TYPE_CHECKING:
    import os

    from optiland.prescription.document import Document

import optiland as _optiland_pkg

_VERSION = getattr(_optiland_pkg, "__version__", "")

_MARGIN_H = 25 * 2.8346  # 25 mm in points
_MARGIN_V = 20 * 2.8346  # 20 mm in points
_HEADER_Y_OFFSET = 10  # points above top margin
_FOOTER_Y_OFFSET = 10  # points below bottom margin


class PDFRenderer(BaseRenderer):
    """Renders a Document to PDF using the 'reportlab' library."""

    def __init__(self) -> None:
        try:
            from reportlab.lib import colors  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PDFRenderer requires the 'reportlab' package. "
                "Install it with: pip install reportlab"
            ) from exc

    def render(self, document: Document) -> str:
        """Not meaningful for PDF; returns a description string."""
        return f"<PDFDocument: {document.title}>"

    def write(self, document: Document, path: str | os.PathLike) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate

        path_str = str(path)
        page_w, _ = A4
        content_w = page_w - 2 * _MARGIN_H

        doc = SimpleDocTemplate(
            path_str,
            pagesize=A4,
            leftMargin=_MARGIN_H,
            rightMargin=_MARGIN_H,
            topMargin=_MARGIN_V + 18,
            bottomMargin=_MARGIN_V + 18,
        )

        story = self._build_story(document, content_w)

        _meta = {"title": document.title, "generated_at": document.generated_at}

        def _on_page(canvas: object, doc: object) -> None:  # type: ignore[override]
            self._draw_header_footer(canvas, doc, _meta)  # type: ignore[arg-type]

        doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)

    def _build_story(self, document: Document, content_w: float) -> list:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import HRFlowable, Paragraph, Spacer

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "PrescriptionTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceAfter=6,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
        )
        italic_style = ParagraphStyle(
            "Italic",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=colors.grey,
        )

        story: list = []
        story.append(Paragraph(document.title, title_style))
        story.append(Paragraph(f"Generated: {document.generated_at}", body_style))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", color=colors.darkblue, thickness=1.5))
        story.append(Spacer(1, 8))

        for section in document.sections:
            story.append(Paragraph(section.title, heading_style))
            story.append(
                HRFlowable(width="100%", color=colors.lightgrey, thickness=0.5)
            )
            story.append(Spacer(1, 4))

            for block in section.blocks:
                story.extend(
                    self._render_block(
                        block, content_w, body_style, italic_style, colors
                    )
                )

        return story

    def _render_block(
        self,
        block: object,
        content_w: float,
        body_style: object,
        italic_style: object,
        colors: object,
    ) -> list:
        from reportlab.platypus import HRFlowable, Paragraph, Spacer

        from optiland.prescription.document import (
            KeyValueBlock,
            SeparatorBlock,
            TableBlock,
            TextBlock,
        )

        result: list = []
        if isinstance(block, KeyValueBlock):
            result.append(self._kv_table(block, content_w))
            result.append(Spacer(1, 4))
        elif isinstance(block, TableBlock):
            result.append(self._data_table(block, content_w))
            result.append(Spacer(1, 6))
        elif isinstance(block, TextBlock):
            result.append(Paragraph(block.text, italic_style))  # type: ignore[arg-type]
            result.append(Spacer(1, 4))
        elif isinstance(block, SeparatorBlock):
            result.append(
                HRFlowable(width="100%", color=colors.lightgrey)  # type: ignore[union-attr]
            )
            result.append(Spacer(1, 4))
        return result

    @staticmethod
    def _kv_table(block: object, content_w: float) -> object:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, Table, TableStyle

        from optiland.prescription.document import KeyValueBlock

        assert isinstance(block, KeyValueBlock)

        key_style = ParagraphStyle(
            "KVKey", fontName="Helvetica", fontSize=9, textColor=colors.grey
        )
        val_style = ParagraphStyle("KVVal", fontName="Courier", fontSize=8)

        data = [
            [Paragraph(k, key_style), Paragraph(v, val_style)] for k, v in block.rows
        ]
        col_w = [content_w * 0.45, content_w * 0.55]
        tbl = Table(data, colWidths=col_w)
        tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        return tbl

    @staticmethod
    def _data_table(block: object, content_w: float) -> object:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, Table, TableStyle

        from optiland.prescription.document import TableBlock

        assert isinstance(block, TableBlock)

        hdr_style = ParagraphStyle(
            "TblHdr",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.white,
        )
        cell_style = ParagraphStyle("TblCell", fontName="Courier", fontSize=7)

        n_cols = len(block.headers)
        col_w = [content_w / n_cols] * n_cols

        data: list[list] = [[Paragraph(h, hdr_style) for h in block.headers]]
        for row in block.rows:
            data.append([Paragraph(cell, cell_style) for cell in row])

        tbl = Table(data, colWidths=col_w, repeatRows=1)

        row_commands = []
        for i in range(1, len(data)):
            if i % 2 == 0:
                row_commands.append(("BACKGROUND", (0, i), (-1, i), colors.whitesmoke))

        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    *row_commands,
                ]
            )
        )
        return tbl

    @staticmethod
    def _draw_header_footer(canvas: object, doc: object, meta: dict) -> None:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4

        c = canvas  # type: ignore[assignment]
        page_w, page_h = A4

        c.saveState()  # type: ignore[union-attr]
        c.setFont("Helvetica", 7)  # type: ignore[union-attr]
        c.setFillColor(colors.grey)  # type: ignore[union-attr]

        header_y = page_h - _MARGIN_V + _HEADER_Y_OFFSET
        c.drawString(_MARGIN_H, header_y, meta["title"])  # type: ignore[union-attr]
        c.drawRightString(  # type: ignore[union-attr]
            page_w - _MARGIN_H,
            header_y,
            f"Generated: {meta['generated_at']}",
        )
        c.line(  # type: ignore[union-attr]
            _MARGIN_H, header_y - 2, page_w - _MARGIN_H, header_y - 2
        )

        footer_y = _MARGIN_V - _FOOTER_Y_OFFSET
        c.drawString(  # type: ignore[union-attr]
            _MARGIN_H, footer_y, f"Optiland v{_VERSION}"
        )
        page_num = getattr(doc, "page", "?")  # type: ignore[union-attr]
        c.drawRightString(  # type: ignore[union-attr]
            page_w - _MARGIN_H, footer_y, f"Page {page_num}"
        )
        c.line(  # type: ignore[union-attr]
            _MARGIN_H, footer_y + 8, page_w - _MARGIN_H, footer_y + 8
        )

        c.restoreState()  # type: ignore[union-attr]
