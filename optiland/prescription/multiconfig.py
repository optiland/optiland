"""Prescription report for multi-configuration systems.

Kramer Harrison, 2026
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import TYPE_CHECKING, Any

from optiland.prescription._formatting import fmt_infinity
from optiland.prescription._output import print_document, save_document, timestamp
from optiland.prescription.document import Document, Section, TableBlock, TextBlock
from optiland.prescription.sections.first_order import FirstOrderSection
from optiland.prescription.sections.surface_table import (
    SurfaceGeometryTableSection,
    SurfaceMaterialTableSection,
)
from optiland.prescription.sections.system_overview import SystemOverviewSection

if TYPE_CHECKING:
    import os

    from optiland.multiconfig import MultiConfiguration
    from optiland.prescription.renderers.base import BaseRenderer
    from optiland.prescription.sections.base import BaseSection


class BaseMultiConfigSection(abc.ABC):
    """Generates one Section of the report from a whole MultiConfiguration."""

    @abc.abstractmethod
    def build(self, multiconfig: MultiConfiguration) -> Section:
        """Build and return a document Section for the given system.

        Args:
            multiconfig: The multi-configuration system to describe.

        Returns:
            A populated Section ready for rendering.
        """
        ...


def _label_blocks(blocks: list[Any], label: str) -> list[Any]:
    """Prefix the title of every titled block with a configuration label.

    If no block carries a title, a TextBlock with the label is put first.
    """
    if not any(hasattr(block, "title") for block in blocks):
        return [TextBlock(text=label), *blocks]
    labelled = []
    for block in blocks:
        if hasattr(block, "title"):
            title = f"{label} — {block.title}" if block.title else label
            block = dataclasses.replace(block, title=title)
        labelled.append(block)
    return labelled


class PerConfigurationSection(BaseMultiConfigSection):
    """Renders a single-optic section once per configuration.

    The blocks of every configuration are put in one Section, each block
    title prefixed with ``"Configuration i"``.

    Args:
        section: The single-optic section to repeat, e.g.
            ``FirstOrderSection()``.
    """

    def __init__(self, section: BaseSection):
        self.section = section

    def build(self, multiconfig: MultiConfiguration) -> Section:
        """Build the section for every configuration.

        Args:
            multiconfig: The multi-configuration system to describe.

        Returns:
            One Section holding the blocks of all configurations.
        """
        title = None
        blocks = []
        for i, optic in enumerate(multiconfig.configurations):
            inner = self.section.build(optic)
            title = title or inner.title
            blocks.extend(_label_blocks(inner.blocks, f"Configuration {i}"))
        return Section(title=f"{title} by Configuration", blocks=blocks)


def _format_cell(value: Any) -> str:
    """Format one configuration-table cell."""
    if isinstance(value, str):
        return value
    try:
        if math.isnan(float(value)):
            return "N/A"
    except (TypeError, ValueError):
        return str(value)
    return fmt_infinity(value)


class ConfigurationTableSection(BaseMultiConfigSection):
    """Tabulates the configurations side by side.

    The rows are those of :meth:`MultiConfiguration.summary`: the first-order
    properties of every configuration, then every property whose value
    differs between configurations.
    """

    def build(self, multiconfig: MultiConfiguration) -> Section:
        """Build the configuration table section.

        Args:
            multiconfig: The multi-configuration system to describe.

        Returns:
            Section containing one TableBlock, one column per configuration.
        """
        frame = multiconfig.summary()
        rows = [
            [str(label), *(_format_cell(v) for v in values)]
            for label, values in zip(frame.index, frame.to_numpy(), strict=True)
        ]
        table = TableBlock(headers=["Parameter", *map(str, frame.columns)], rows=rows)
        return Section(title="Configurations", blocks=[table])


class _MasterConfigurationSection(BaseMultiConfigSection):
    """Builds a single-optic section on configuration 0 (the master)."""

    def __init__(self, section: BaseSection):
        self.section = section

    def build(self, multiconfig: MultiConfiguration) -> Section:
        inner = self.section.build(multiconfig.configurations[0])
        return dataclasses.replace(inner, title=f"{inner.title} (Configuration 0)")


class MultiConfigPrescription:
    """Generates a prescription report for a multi-configuration system.

    The default report has the system overview of configuration 0, the
    configuration table, the first-order properties of every configuration
    and the surface tables of configuration 0.

    Args:
        multiconfig: The multi-configuration system to describe.
        sections: Optional list of sections. A
            :class:`BaseMultiConfigSection` is built from the whole system;
            a single-optic :class:`BaseSection` is built on configuration 0
            (wrap it in :class:`PerConfigurationSection` to repeat it for
            every configuration). Defaults to the set above when None.
    """

    def __init__(
        self,
        multiconfig: MultiConfiguration,
        sections: list[BaseMultiConfigSection | BaseSection] | None = None,
    ) -> None:
        self._multiconfig = multiconfig
        if sections is None:
            sections = self._default_sections()
        self._sections: list[BaseMultiConfigSection] = [
            s
            if isinstance(s, BaseMultiConfigSection)
            else _MasterConfigurationSection(s)
            for s in sections
        ]

    @staticmethod
    def _default_sections() -> list[BaseMultiConfigSection | BaseSection]:
        """Return fresh instances of the default section set."""
        return [
            SystemOverviewSection(),
            ConfigurationTableSection(),
            PerConfigurationSection(FirstOrderSection()),
            SurfaceGeometryTableSection(),
            SurfaceMaterialTableSection(),
        ]

    def build(self) -> Document:
        """Build and return the Document model without rendering.

        Returns:
            Populated Document ready for a renderer to consume.
        """
        configs = self._multiconfig.configurations
        name = configs[0].name or "Unnamed System"
        title = (
            f"Multi-Configuration Prescription — {name} ({len(configs)} configurations)"
        )
        sections = [s.build(self._multiconfig) for s in self._sections]
        return Document(title=title, generated_at=timestamp(), sections=sections)

    def view(self) -> None:
        """Print the prescription to the console using ConsoleRenderer."""
        print_document(self.build())

    def save(
        self,
        path: str | os.PathLike,
        renderer: BaseRenderer | None = None,
    ) -> None:
        """Save the prescription to a file.

        Args:
            path: Output file path. Extension determines the renderer when
                renderer is None: ``.pdf`` → PDFRenderer, anything else →
                PlainTextRenderer.
            renderer: Explicit renderer instance. Overrides extension
                inference.
        """
        save_document(self.build(), path, renderer)
