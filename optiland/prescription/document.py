"""Prescription Document Model

Intermediate representation that decouples section data collection
from rendering.  Sections write only to this model; renderers read
only from this model.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KeyValueBlock:
    """A sequence of key-value pairs."""

    rows: list[tuple[str, str]]
    title: str | None = None


@dataclass
class TableBlock:
    """A structured table with pre-formatted string cells."""

    headers: list[str]
    rows: list[list[str]]
    title: str | None = None


@dataclass
class TextBlock:
    """A free-form paragraph or note."""

    text: str


@dataclass
class SeparatorBlock:
    """Visual separator (horizontal rule / blank line)."""


Block = KeyValueBlock | TableBlock | TextBlock | SeparatorBlock


@dataclass
class Section:
    """One labelled section of the document."""

    title: str
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Document:
    """Top-level document model."""

    title: str
    generated_at: str
    sections: list[Section] = field(default_factory=list)
