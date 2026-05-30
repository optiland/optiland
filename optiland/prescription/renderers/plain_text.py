"""PlainTextRenderer — pure-Python plain text output.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.prescription.renderers.base import BaseRenderer

if TYPE_CHECKING:
    import os

    from optiland.prescription.document import (
        Block,
        Document,
        KeyValueBlock,
        Section,
        TableBlock,
    )

_PAGE_WIDTH = 70
_SECTION_BAR = "─" * (_PAGE_WIDTH - 2)


def _col_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    return widths


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = _col_widths(headers, rows)
    gap = 3

    def _row_str(cells: list[str]) -> str:
        parts = [cell.ljust(w) for cell, w in zip(cells, widths, strict=False)]
        return "  " + (" " * gap).join(parts)

    def _sep_str() -> str:
        parts = ["─" * w for w in widths]
        return "  " + (" " * gap).join(parts)

    lines = [_row_str(headers), _sep_str()]
    for row in rows:
        lines.append(_row_str(row))
    return lines


class PlainTextRenderer(BaseRenderer):
    """Renders a Document to plain text with Unicode box-drawing characters."""

    def render(self, document: Document) -> str:
        horiz = "═" * _PAGE_WIDTH
        lines: list[str] = []
        lines.append(horiz)
        lines.append(f"  {document.title.upper()}")
        lines.append(f"  Generated: {document.generated_at}")
        lines.append(horiz)

        for section in document.sections:
            lines.append("")
            lines.extend(self._render_section(section))

        lines.append("")
        return "\n".join(lines)

    def write(self, document: Document, path: str | os.PathLike) -> None:
        content = self.render(document)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def _render_section(self, section: Section) -> list[str]:
        lines: list[str] = []
        fill = max(0, _PAGE_WIDTH - len(section.title) - 5)
        title_bar = f"┌─ {section.title} " + "─" * fill + "┐"
        lines.append(title_bar)

        for block in section.blocks:
            lines.extend(self._render_block(block))

        lines.append("└" + "─" * (_PAGE_WIDTH - 2) + "┘")
        return lines

    def _render_block(self, block: Block) -> list[str]:
        from optiland.prescription.document import (
            KeyValueBlock,
            SeparatorBlock,
            TableBlock,
            TextBlock,
        )

        if isinstance(block, KeyValueBlock):
            return self._render_kv(block)
        if isinstance(block, TableBlock):
            return self._render_table_block(block)
        if isinstance(block, TextBlock):
            return [f"  {block.text}", ""]
        if isinstance(block, SeparatorBlock):
            return ["  " + _SECTION_BAR]
        return []

    @staticmethod
    def _render_kv(block: KeyValueBlock) -> list[str]:
        if not block.rows:
            return []
        key_w = max(len(k) for k, _ in block.rows)
        lines = []
        if block.title:
            lines.append(f"  {block.title}")
        for key, val in block.rows:
            lines.append(f"  {key.ljust(key_w)}   {val}")
        lines.append("")
        return lines

    @staticmethod
    def _render_table_block(block: TableBlock) -> list[str]:
        lines = []
        if block.title:
            lines.append(f"  {block.title}")
        for line in _render_table(block.headers, block.rows):
            lines.append(line)
        lines.append("")
        return lines
