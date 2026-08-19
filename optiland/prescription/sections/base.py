"""BaseSection ABC for prescription sections.

Kramer Harrison, 2026
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optic.optic import Optic
    from optiland.prescription.document import Section


class BaseSection(abc.ABC):
    """Generates one Section of the prescription document from an Optic.

    Sections are stateless with respect to the optic — build() receives
    the optic fresh each call, making sections trivially testable.
    """

    @abc.abstractmethod
    def build(self, optic: Optic) -> Section:
        """Build and return a document Section for the given optic.

        Args:
            optic: The optical system to describe.

        Returns:
            A populated Section ready for rendering.
        """
        ...
