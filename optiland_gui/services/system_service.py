"""System-level service for the Optiland GUI.

Manages aperture, fields, and wavelength data retrieval for the System
Properties panel.
"""

from __future__ import annotations

import math

import optiland.aperture  # noqa: F401 — ensures all subclasses are registered
from optiland.aperture import BaseSystemAperture
from optiland.rays import PolarizationState


class SystemService:
    """Provides field and wavelength option lists for the GUI.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service.
    """

    def __init__(self, connector: object) -> None:
        self._connector = connector

    def get_field_options(self) -> list[tuple[str, str]]:
        """Return field selector options for the analysis panel.

        Each option is a ``(display_name, value_str)`` tuple where
        *value_str* is suitable for passing directly to an analysis class.

        Returns:
            A list of tuples starting with ``("all", "'all'")`` followed by
            one entry per field.
        """
        optic = self._connector._optic
        if not optic or optic.fields.num_fields == 0:
            return [("all", "'all'")]
        options = [("all", "'all'")]
        field_coords = optic.fields.get_field_coords()
        for i, field_tuple in enumerate(field_coords):
            hx, hy = field_tuple[0], field_tuple[1]
            display_name = f"Field {i + 1}: ({hx:.3f}, {hy:.3f})"
            value_str = f"[({hx}, {hy})]"
            options.append((display_name, value_str))
        return options

    def get_wavelength_options(self) -> list[tuple[str, str]]:
        """Return wavelength selector options for the analysis panel.

        Each option is a ``(display_name, value_str)`` tuple.

        Returns:
            A list of tuples starting with ``("all", "'all'")`` and
            ``("primary", "'primary'")`` followed by one entry per wavelength.
        """
        optic = self._connector._optic
        if not optic or optic.wavelengths.num_wavelengths == 0:
            return [("all", "'all'"), ("primary", "'primary'")]
        options = [("all", "'all'"), ("primary", "'primary'")]
        wavelength_values = optic.wavelengths.get_wavelengths()
        for wl_value in wavelength_values:
            display_name = f"{wl_value:.4f} µm"
            value_str = f"[{wl_value}]"
            options.append((display_name, value_str))
        return options

    def get_aperture_types(self) -> list[str]:
        """Return all aperture type keys registered with :class:`BaseSystemAperture`.

        Enumerates the ``_registry`` class attribute that is populated when
        each :class:`~optiland.aperture.base.BaseSystemAperture` subclass is
        imported.

        Returns:
            A sorted list of aperture type identifier strings (e.g.
            ``["EPD", "imageFNO", "objectNA", ...]``).
        """
        return sorted(BaseSystemAperture._registry.keys())

    def set_polarization_state(
        self,
        mode: str,
        Ex: float | None = None,
        Ey: float | None = None,
        phase_x_deg: float | None = None,
        phase_y_deg: float | None = None,
    ) -> None:
        """Set or clear the polarization state on the active optic.

        When *mode* is ``"ignore"`` the optic's polarization is set to
        ``"ignore"``.  When ``"unpolarized"``, an unpolarized state is created.
        When ``"polarized"``, the four field values are required;
        phase angles are provided in **degrees** and converted to radians
        before constructing :class:`~optiland.rays.PolarizationState`.

        Args:
            mode: "ignore", "unpolarized", or "polarized".
            Ex: Electric field x-component (required when "polarized").
            Ey: Electric field y-component (required when "polarized").
            phase_x_deg: Phase of Ex in degrees (required when "polarized").
            phase_y_deg: Phase of Ey in degrees (required when "polarized").

        Raises:
            ValueError: If *mode* is ``"polarized"`` but any field value is
                ``None``, or if :class:`~optiland.rays.PolarizationState` raises.
        """
        optic = self._connector._optic
        if optic is None:
            return
        if mode == "ignore":
            optic.updater.set_polarization("ignore")
        elif mode == "unpolarized":
            state = PolarizationState(is_polarized=False)
            optic.updater.set_polarization(state)
        elif mode == "polarized":
            if None in (Ex, Ey, phase_x_deg, phase_y_deg):
                raise ValueError(
                    "All polarization fields are required when mode is 'polarized'."
                )
            phase_x = math.radians(phase_x_deg)
            phase_y = math.radians(phase_y_deg)
            state = PolarizationState(
                is_polarized=True,
                Ex=Ex,
                Ey=Ey,
                phase_x=phase_x,
                phase_y=phase_y,
            )
            optic.updater.set_polarization(state)
        else:
            raise ValueError(f"Unknown polarization mode: {mode}")
        self._connector.opticChanged.emit()

    def get_field_types(self) -> list[tuple[str, str]]:
        """Return all four supported field types.

        Returns:
            A list of ``(display_name, type_key)`` tuples for every field
            type exposed by the GUI:

            - ``("Angle", "angle")``
            - ``("Object Height", "object_height")``
            - ``("Paraxial Image Height", "paraxial_image_height")``
            - ``("Real Image Height", "real_image_height")``
        """
        return [
            ("Angle", "angle"),
            ("Object Height", "object_height"),
            ("Paraxial Image Height", "paraxial_image_height"),
            ("Real Image Height", "real_image_height"),
        ]
