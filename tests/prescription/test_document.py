"""Tests for the prescription document model."""

from __future__ import annotations

from optiland.prescription.document import (
    Document,
    KeyValueBlock,
    Section,
    SeparatorBlock,
    TableBlock,
    TextBlock,
)


def test_key_value_block():
    block = KeyValueBlock(rows=[("EFL", "50.0000"), ("FNO", "4.0000")])
    assert block.rows[0] == ("EFL", "50.0000")
    assert block.title is None


def test_key_value_block_with_title():
    block = KeyValueBlock(rows=[("Name", "Test")], title="Meta")
    assert block.title == "Meta"


def test_table_block():
    block = TableBlock(headers=["A", "B"], rows=[["1", "2"], ["3", "4"]])
    assert block.headers == ["A", "B"]
    assert block.rows[1] == ["3", "4"]


def test_text_block():
    block = TextBlock(text="Some note.")
    assert block.text == "Some note."


def test_separator_block():
    block = SeparatorBlock()
    assert isinstance(block, SeparatorBlock)


def test_section():
    s = Section(title="Test Section", blocks=[TextBlock("hello")])
    assert s.title == "Test Section"
    assert len(s.blocks) == 1


def test_document():
    doc = Document(
        title="My Report",
        generated_at="2026-05-17 12:00 UTC",
        sections=[Section(title="S1", blocks=[])],
    )
    assert doc.title == "My Report"
    assert len(doc.sections) == 1
