"""Sphinx extension: highlight function and method calls in Python code.

Pygments' Python lexer only marks the names in ``def`` and ``class``
statements as functions or classes. Every other identifier, including calls
such as ``lens.add_surface(...)`` or ``Optic()``, is a plain ``Name`` token and
is rendered in the text colour, so a typical Optiland snippet shows colour
only on keywords, strings, numbers and built-in types.

This extension adds a token filter that retags an identifier immediately
followed by ``(`` as ``Name.Function`` (or ``Name.Class`` for CapWords names,
which are almost always constructors) and installs it for every Python
flavoured language used in the documentation: ``python`` (also ``py``,
``python3`` and Sphinx's ``default``), ``pycon`` doctest blocks and the
``ipython3`` cells rendered by nbsphinx. Nothing else changes: attribute
access, keyword arguments, annotations and imported names keep the plain text
colour, and the colours themselves still come from the Pygments styles set in
``conf.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pygments.filter import Filter
from pygments.lexers import PythonConsoleLexer, PythonLexer
from pygments.token import Name, Punctuation, Text, Whitespace

try:
    from ipython_pygments_lexers import IPython3Lexer
except ImportError:  # IPython < 9
    from IPython.lib.lexers import IPython3Lexer

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from pygments.lexer import Lexer
    from pygments.token import _TokenType
    from sphinx.application import Sphinx


def _is_inline_space(ttype: _TokenType, value: str) -> bool:
    """Whitespace on the same line (spaces or tabs, no newline)."""
    if ttype not in Whitespace and ttype is not Text:
        return False
    return value.isspace() and "\n" not in value


class CallHighlightFilter(Filter):
    """Retag ``Name`` tokens that are immediately called as functions."""

    def filter(
        self, lexer: Lexer, stream: Iterable[tuple[_TokenType, str]]
    ) -> Iterator[tuple[_TokenType, str]]:
        pending: list[tuple[_TokenType, str]] = []  # a Name plus trailing spaces
        for ttype, value in stream:
            if pending:
                if _is_inline_space(ttype, value):
                    pending.append((ttype, value))
                    continue
                if ttype is Punctuation and value == "(":
                    name = pending[0][1]
                    called = Name.Class if name[:1].isupper() else Name.Function
                    pending[0] = (called, name)
                yield from pending
                pending = []
            if ttype is Name:
                pending.append((ttype, value))
            else:
                yield ttype, value
        yield from pending


class _CallHighlightMixin:
    """Add :class:`CallHighlightFilter` and keep Sphinx's ``stripnl=False``."""

    def __init__(self, **options) -> None:
        options.setdefault("stripnl", False)
        super().__init__(**options)
        self.add_filter(CallHighlightFilter())


class OptilandPythonLexer(_CallHighlightMixin, PythonLexer):
    """``python`` / ``py`` / ``python3`` / ``default`` code blocks."""


class OptilandPythonConsoleLexer(_CallHighlightMixin, PythonConsoleLexer):
    """``pycon`` doctest blocks (``>>>`` prompts)."""


class OptilandIPython3Lexer(_CallHighlightMixin, IPython3Lexer):
    """``ipython3`` notebook cells rendered by nbsphinx."""


def setup(app: Sphinx) -> dict[str, bool | str]:
    app.add_lexer("python", OptilandPythonLexer)
    app.add_lexer("pycon", OptilandPythonConsoleLexer)
    app.add_lexer("ipython", OptilandIPython3Lexer)
    app.add_lexer("ipython3", OptilandIPython3Lexer)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
