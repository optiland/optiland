"""Zemax to Optic Converter

Converts a ZemaxDataModel into an Optiland Optic object. This module also
provides the ``load_zemax_file`` entry point for the reader path.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import Any

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.fileio.base import BaseOpticReader
from optiland.fileio.zemax.model import ZemaxDataModel
from optiland.fileio.zemax.reader.parser import ZemaxDataParser
from optiland.fileio.zemax.reader.source import ZemaxFileSourceHandler
from optiland.optic import Optic


class ZemaxToOpticConverter(BaseOpticReader):
    """Converts a ZemaxDataModel into an Optic object.

    Also implements BaseOpticReader so that the full pipeline (source
    resolution -> parsing -> conversion) can be triggered via ``read()``.

    Args:
        zemax_data: A plain dict (legacy) or ZemaxDataModel containing the
            Zemax optical system data.

    Attributes:
        data: The Zemax data as a plain dict.
        optic: The Optic instance built by :py:meth:`convert`.
        current_cs: Running cumulative CoordinateSystem used when processing
            coordinate-break surfaces.
    """

    def __init__(self, zemax_data: dict[str, Any] | ZemaxDataModel):
        if isinstance(zemax_data, ZemaxDataModel):
            self.data = zemax_data.to_dict()
        else:
            self.data = zemax_data
        self.optic: Optic | None = None
        self.current_cs = CoordinateSystem()

    # ------------------------------------------------------------------
    # BaseOpticReader
    # ------------------------------------------------------------------

    def read(self, source: str) -> Optic:
        """Read a Zemax file and return a fully-configured Optic.

        Args:
            source: Local file path or URL to a .zmx file.

        Returns:
            A configured Optic instance.
        """
        src_handler = ZemaxFileSourceHandler(source)
        filename = src_handler.get_local_file()
        try:
            data_model = ZemaxDataParser(filename).parse()
            self.data = data_model.to_dict()
            self.current_cs = CoordinateSystem()
            return self.convert()
        finally:
            src_handler.cleanup()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(self) -> Optic:
        """Convert the stored Zemax data dict into an Optic object.

        Returns:
            The fully-configured Optic instance.
        """
        self.optic = Optic(self.data.get("name"))
        self._configure_surfaces()
        self._configure_aperture()
        self._configure_fields()
        self._configure_wavelengths()
        return self.optic

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _configure_surfaces(self) -> None:
        """Configure all surfaces on the optic."""
        has_cb = any(
            sd.get("type") == "coordinate_break"
            for sd in self.data["surfaces"].values()
        )

        if not has_cb:
            for idx, surf_data in self.data["surfaces"].items():
                self._configure_surface(idx, surf_data)
            return

        # Coordinate-break path: accumulate a running CoordinateSystem and
        # apply it as the geometry CS for each non-CB surface.
        surf_idx = 0
        for idx in sorted(self.data["surfaces"].keys(), key=int):
            surf = self.data["surfaces"][idx]

            if surf.get("type") == "coordinate_break":
                # Consume CB: update cumulative CS only
                dx = float(surf.get("param_0", 0.0))
                dy = float(surf.get("param_1", 0.0))
                dz = float(surf.get("thickness", 0.0))
                rx = be.deg2rad(surf.get("param_2", 0.0))
                ry = be.deg2rad(surf.get("param_3", 0.0))
                rz = be.deg2rad(surf.get("param_4", 0.0))

                # Chain: first apply rotations/decenters, then thickness (Z)
                cs_rot = CoordinateSystem(
                    x=dx,
                    y=dy,
                    z=0.0,
                    rx=rx,
                    ry=ry,
                    rz=rz,
                    reference_cs=self.current_cs,
                )
                self.current_cs = CoordinateSystem(
                    x=0.0,
                    y=0.0,
                    z=dz,
                    reference_cs=cs_rot,
                )
                continue

            # Resolve effective global position and orientation
            translation, _ = self.current_cs.get_effective_transform()
            rx_, ry_, rz_ = self.current_cs.get_effective_rotation_euler()
            coeffs = self._configure_surface_coefficients(surf)

            surface_params: dict[str, Any] = {
                "index": surf_idx,
                "surface_type": surf["type"],
                "conic": surf.get("conic"),
                "is_stop": surf.get("is_stop", False),
                "material": surf.get("material"),
                "coefficients": coeffs,
            }

            if surf.get("aperture") is not None:
                surface_params["aperture"] = surf["aperture"]

            if surf["type"] == "paraxial":
                surface_params["f"] = float(surf.get("param_0", 0.0))

            if surf["type"] == "toroidal":
                surface_params["radius_y"] = surf["radius"]
                surface_params["toroidal_coeffs_poly_y"] = coeffs
                radius_x = surf.get("param_1", 0.0)
                if radius_x == 0.0:
                    radius_x = be.inf
                surface_params["radius_x"] = radius_x
            else:
                surface_params["radius"] = surf["radius"]

            thickness = surf.get("thickness", 0.0)
            if be.isinf(float(thickness)):
                surface_params["thickness"] = thickness
                surface_params.update(
                    {"rx": float(rx_), "ry": float(ry_), "rz": float(rz_)}
                )
            else:
                surface_params.update(
                    {
                        "x": float(translation[0]),
                        "y": float(translation[1]),
                        "z": float(translation[2]),
                        "rx": float(rx_),
                        "ry": float(ry_),
                        "rz": float(rz_),
                    }
                )

            self.optic.surfaces.add(**surface_params)
            surf_idx += 1

            dt = surf.get("thickness", 0.0)
            if not be.isinf(dt):
                self.current_cs = CoordinateSystem(
                    x=0.0,
                    y=0.0,
                    z=dt,
                    reference_cs=self.current_cs,
                )

    def _configure_surface(self, index: int, data: dict[str, Any]) -> None:
        """Configure a single surface without coordinate-break logic.

        Args:
            index: The surface index.
            data: The raw surface dict from the parser.
        """
        coefficients = self._configure_surface_coefficients(data)

        surface_params: dict[str, Any] = {
            "index": index,
            "surface_type": data["type"],
            "conic": data.get("conic"),
            "thickness": data.get("thickness"),
            "is_stop": data.get("is_stop", False),
            "material": data.get("material"),
        }

        if data.get("aperture") is not None:
            surface_params["aperture"] = data["aperture"]

        if data["type"] == "paraxial":
            surface_params["f"] = float(data.get("param_0", 0.0))

        if data["type"] == "toroidal":
            surface_params["toroidal_coeffs_poly_y"] = coefficients
        else:
            surface_params["coefficients"] = coefficients

        if data["type"] == "coordinate_break":
            surface_params["dx"] = data.get("param_0", 0.0)
            surface_params["dy"] = data.get("param_1", 0.0)
            surface_params["rx"] = be.deg2rad(be.array(data.get("param_2", 0.0)))
            surface_params["ry"] = be.deg2rad(be.array(data.get("param_3", 0.0)))
            surface_params["rz"] = be.deg2rad(be.array(data.get("param_4", 0.0)))
            surface_params["order_flag"] = data.get("param_5", 0.0)

        if data["type"] == "toroidal":
            radius_x = data.get("param_1", 0.0)
            if radius_x == 0.0:
                radius_x = be.inf
            surface_params["radius_y"] = data["radius"]
            surface_params["radius_x"] = radius_x
        else:
            surface_params["radius"] = data["radius"]

        self.optic.surfaces.add(**surface_params)

    def _configure_surface_coefficients(
        self, data: dict[str, Any]
    ) -> list[float] | None:
        """Extract aspheric coefficients from a raw surface dict.

        Args:
            data: The surface data dict.

        Returns:
            A list of coefficient values, or None for surface types without
            coefficients.

        Raises:
            ValueError: If the surface type is not recognised.
        """
        surf_type = data["type"]
        if surf_type in ("standard", "coordinate_break", "paraxial"):
            return None

        if surf_type in ("even_asphere", "odd_asphere", "toroidal"):
            start = 2 if surf_type == "toroidal" else 0
            return [data.get(f"param_{k}", 0.0) for k in range(start, start + 8)]

        raise ValueError(f"Unsupported Zemax surface type: {surf_type}")

    def _configure_aperture(self) -> None:
        """Configure the system aperture on the optic."""
        aperture_data = self.data["aperture"]

        if aperture_data.get("floating_stop"):
            stop_diameter = None
            for surf_data in self.data["surfaces"].values():
                if surf_data.get("is_stop") and "diameter" in surf_data:
                    stop_diameter = surf_data["diameter"]
                    break
            if stop_diameter is None:
                raise ValueError(
                    "Floating stop aperture specified but no stop diameter found"
                )
            self.optic.set_aperture(
                aperture_type="float_by_stop_size", value=stop_diameter
            )
        else:
            for key, value in aperture_data.items():
                if key != "floating_stop":
                    self.optic.set_aperture(aperture_type=key, value=value)
                    break
            else:
                raise ValueError("No valid aperture type found in aperture_data.")

    def _configure_fields(self) -> None:
        """Configure the field group on the optic."""
        self.optic.fields.set_type(field_type=self.data["fields"]["type"])

        field_x = self.data["fields"]["x"]
        field_y = self.data["fields"]["y"]

        try:
            vig_x = self.data["fields"]["vignette_compress_x"]
            vig_y = self.data["fields"]["vignette_compress_y"]
        except KeyError:
            vig_x = [0.0] * len(field_x)
            vig_y = [0.0] * len(field_y)

        try:
            dx = self.data["fields"]["vignette_decenter_x"]
            dy = self.data["fields"]["vignette_decenter_y"]
            if any(dx) or any(dy):
                print("Warning: Vignette decentering is not supported.")
        except KeyError:
            pass

        weights = self.data["fields"].get("weights", [1.0] * len(field_x))
        for k in range(len(field_x)):
            self.optic.fields.add(
                x=field_x[k],
                y=field_y[k],
                vx=vig_x[k],
                vy=vig_y[k],
                weight=weights[k],
            )

    def _configure_wavelengths(self) -> None:
        """Configure the wavelength group on the optic."""
        primary_idx = self.data["wavelengths"]["primary_index"]
        wl_data = self.data["wavelengths"]["data"]
        wl_weights = self.data["wavelengths"].get("weights", [1.0] * len(wl_data))
        for idx, value in enumerate(wl_data):
            self.optic.wavelengths.add(
                value=value,
                is_primary=(idx == primary_idx),
                weight=wl_weights[idx],
            )
