"""MaterialSpec and MatchPolicy for the Optiland materials system.

Provides the canonical type-safe spec for surface material assignment and a
three-value enum controlling fuzzy-match behavior.

Kramer Harrison, 2025
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.materials.material import Material


class MatchPolicy(str, enum.Enum):
    """Controls how Material resolves ambiguous name matches.

    Attributes:
        BEST: Silent best-match; no warning emitted.
        WARN: Warn when fuzzy match is used (edit distance > 0). Default.
        STRICT: Raise ValueError on any non-exact or ambiguous match.
    """

    BEST = "best"
    WARN = "warn"
    STRICT = "strict"


@dataclasses.dataclass(frozen=True)
class MaterialSpec:
    """Canonical, type-safe specification for a surface material.

    Instances are frozen (hashable) and safe to cache.  Pass to
    ``MaterialFactory.create()`` or call ``.to_material()`` directly.

    Args:
        name: Glass or material name (e.g. ``"N-BK7"``).
        catalog: Manufacturer catalog to restrict lookup to (e.g. ``"schott"``).
        reference: Citation string (passed through to ``Material``).
        match_policy: Controls fuzzy-match warnings/errors.
        min_wavelength: Minimum wavelength filter in microns.
        max_wavelength: Maximum wavelength filter in microns.
    """

    name: str
    catalog: str | None = None
    reference: str | None = None
    match_policy: MatchPolicy = MatchPolicy.WARN
    min_wavelength: float | None = None
    max_wavelength: float | None = None

    def to_material(self) -> Material:
        """Resolve this spec to a concrete Material instance."""
        from optiland.materials.material import Material

        return Material(
            self.name,
            reference=self.reference,
            min_wavelength=self.min_wavelength,
            max_wavelength=self.max_wavelength,
            catalog=self.catalog,
            match_policy=self.match_policy,
        )

    def to_dict(self) -> dict:
        """Serialize this spec to a plain dictionary."""
        return {
            "name": self.name,
            "catalog": self.catalog,
            "reference": self.reference,
            "match_policy": self.match_policy.value,
            "min_wavelength": self.min_wavelength,
            "max_wavelength": self.max_wavelength,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MaterialSpec:
        """Deserialize a MaterialSpec from a plain dictionary.

        Args:
            data: Dictionary with at least a ``"name"`` key.

        Returns:
            MaterialSpec instance.
        """
        return cls(
            name=data["name"],
            catalog=data.get("catalog"),
            reference=data.get("reference"),
            match_policy=MatchPolicy(data.get("match_policy", MatchPolicy.WARN.value)),
            min_wavelength=data.get("min_wavelength"),
            max_wavelength=data.get("max_wavelength"),
        )
