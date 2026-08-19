"""Base System Aperture Module

Defines the abstract base class for all system aperture types.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.paraxial import Paraxial


class BaseSystemAperture(ABC):
    """Abstract base class for system aperture type definitions.

    Each concrete subclass encapsulates the logic for computing the entrance
    pupil diameter (EPD) for one aperture specification style (e.g. EPD,
    image-space F-number, object-space NA, or float-by-stop-size).

    Subclasses must declare a class-level ``ap_type`` string (e.g. ``"EPD"``)
    that matches the legacy type key used in serialized lens files.  This
    string is used both for serialization round-trips and for the
    ``__init_subclass__`` auto-registration that powers
    :meth:`from_dict`.

    Attributes:
        _registry (dict[str, type[BaseSystemAperture]]): Class-level registry
            mapping ``ap_type`` strings to concrete subclass types.  Populated
            automatically via ``__init_subclass__``.

    """

    _registry: dict[str, type[BaseSystemAperture]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Register subclasses that declare a _ap_type_key class variable.
        # This is separate from the ap_type property to avoid descriptor conflicts.
        key = cls.__dict__.get("_ap_type_key")
        if isinstance(key, str):
            BaseSystemAperture._registry[key] = cls

    # ── Identity ───────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def ap_type(self) -> str:
        """String identifier matching the legacy type key (e.g. ``'EPD'``)."""

    @property
    @abstractmethod
    def value(self) -> float:
        """Raw aperture value as supplied by the user."""

    # ── Capability flags ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def supports_telecentric(self) -> bool:
        """True if this aperture type is compatible with telecentric object space."""

    @property
    @abstractmethod
    def is_scalable(self) -> bool:
        """True if the stored value should be scaled during ``optic.scale()``."""

    # ── Core computation ───────────────────────────────────────────────────

    @abstractmethod
    def compute_epd(self, paraxial: Paraxial, wavelength: float | None = None) -> float:
        """Return the entrance pupil diameter using paraxial context.

        Args:
            paraxial: The paraxial engine for the current optical system.
            wavelength: Primary wavelength in micrometers.  When ``None``,
                implementations may fall back to
                ``paraxial.optic.primary_wavelength``.

        Returns:
            Entrance pupil diameter in lens units.

        """

    def direct_fno(self) -> float | None:
        """Return the F-number directly if this type stores it, else ``None``.

        Overridden only by :class:`~optiland.aperture.image_fno.ImageFNOAperture`
        to avoid a redundant EPD computation in :meth:`~optiland.paraxial.Paraxial.FNO`.

        Returns:
            F-number value, or ``None`` if this type does not store one
            directly.

        """
        return None

    # ── System scaling ─────────────────────────────────────────────────────

    @abstractmethod
    def scale(self, factor: float) -> BaseSystemAperture:
        """Return a new aperture with the value scaled by *factor*.

        Non-scalable types (where :attr:`is_scalable` is ``False``) return
        ``self`` unchanged since they are immutable and scaling has no effect.

        Args:
            factor: Multiplicative scale factor (e.g. 0.001 to convert mm to m).

        Returns:
            A new :class:`BaseSystemAperture` instance, or ``self`` for
            non-scalable types.

        """

    # ── Serialization ──────────────────────────────────────────────────────

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            A dict with at least ``"type"`` and ``"value"`` keys.

        """

    @classmethod
    def from_dict(cls, data: dict | None) -> BaseSystemAperture | None:
        """Deserialize from a dictionary, dispatching by the ``"type"`` field.

        Args:
            data: Dictionary produced by :meth:`to_dict`.  Passing ``None``
                returns ``None`` (mirrors legacy ``Aperture.from_dict`` behaviour).

        Returns:
            A concrete :class:`BaseSystemAperture` instance, or ``None`` if
            *data* is ``None``.

        Raises:
            ValueError: If required keys are missing or the type is not
                registered.

        """
        if data is None:
            return None

        required_keys = {"type", "value"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys in aperture data: {missing}")

        type_str = data["type"]
        if type_str not in cls._registry:
            raise ValueError(
                f"Unknown aperture type '{type_str}'. Available: {list(cls._registry)}"
            )
        return cls._registry[type_str]._from_dict(data)

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: dict) -> BaseSystemAperture:
        """Construct an instance from a raw dict (called by :meth:`from_dict`).

        Args:
            data: The validated raw dictionary.

        Returns:
            A concrete instance of this subclass.

        """
