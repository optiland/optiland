"""ConsoleRenderer — rich terminal output.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.prescription.renderers.base import BaseRenderer

if TYPE_CHECKING:
    import os

    from optiland.prescription.document import Block, Document, Section


class ConsoleRenderer(BaseRenderer):
    """Renders a Document to the terminal using the 'rich' library."""

    def __init__(self) -> None:
        try:
            import rich  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ConsoleRenderer requires the 'rich' package. "
                "Install it with: pip install rich"
            ) from exc

    def render(self, document: Document) -> str:
        import io

        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        self._write_to_console(console, document)
        return buf.getvalue()

    def print(self, document: Document) -> None:
        from rich.console import Console

        console = Console(highlight=False)
        self._write_to_console(console, document)

    def write(self, document: Document, path: str | os.PathLike) -> None:
        content = self.render(document)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def _write_to_console(self, console: object, document: Document) -> None:
        from rich.panel import Panel
        from rich.text import Text

        c = console  # type: ignore[assignment]
        title_text = Text(document.title, style="bold white")
        subtitle = f"Generated: {document.generated_at}"
        panel = Panel(title_text, subtitle=subtitle, style="on dark_blue")
        c.print(panel)  # type: ignore[union-attr]

        for section in document.sections:
            self._render_section(c, section)

    def _render_section(self, console: object, section: Section) -> None:
        from rich.rule import Rule

        c = console  # type: ignore[assignment]
        c.print(Rule(f"[bold]{section.title}[/bold]"))  # type: ignore[union-attr]

        for block in section.blocks:
            self._render_block(c, block)

    def _render_block(self, console: object, block: Block) -> None:
        from optiland.prescription.document import (
            KeyValueBlock,
            SeparatorBlock,
            TableBlock,
            TextBlock,
        )

        c = console  # type: ignore[assignment]
        if isinstance(block, KeyValueBlock):
            self._render_kv(c, block)
        elif isinstance(block, TableBlock):
            self._render_table(c, block)
        elif isinstance(block, TextBlock):
            from rich.text import Text

            c.print(Text(block.text, style="italic dim"))  # type: ignore[union-attr]
        elif isinstance(block, SeparatorBlock):
            from rich.rule import Rule

            c.print(Rule())  # type: ignore[union-attr]

    @staticmethod
    def _render_kv(console: object, block: object) -> None:
        import rich.box
        from rich.table import Table

        from optiland.prescription.document import KeyValueBlock

        assert isinstance(block, KeyValueBlock)
        c = console  # type: ignore[assignment]
        if block.title:
            from rich.text import Text

            c.print(Text(f" {block.title}", style="bold"))  # type: ignore[union-attr]

        tbl = Table(show_header=False, box=rich.box.SIMPLE, padding=(0, 1))
        tbl.add_column("key", style="dim cyan")
        tbl.add_column("value", style="white")
        for key, val in block.rows:
            tbl.add_row(key, val)
        c.print(tbl)  # type: ignore[union-attr]

    @staticmethod
    def _render_table(console: object, block: object) -> None:
        import rich.box
        from rich.table import Table

        from optiland.prescription.document import TableBlock

        assert isinstance(block, TableBlock)
        c = console  # type: ignore[assignment]
        if block.title:
            from rich.text import Text

            c.print(Text(f" {block.title}", style="bold"))  # type: ignore[union-attr]

        tbl = Table(
            box=rich.box.SIMPLE_HEAVY,
            header_style="bold magenta",
            row_styles=["", "dim"],
        )
        for h in block.headers:
            tbl.add_column(h)
        for row in block.rows:
            tbl.add_row(*row)
        c.print(tbl)  # type: ignore[union-attr]
