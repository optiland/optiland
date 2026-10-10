"""Pure document normalization shared by GUI initialization and owned file jobs."""

from __future__ import annotations


def ensure_valid_structure(optic, default_wavelength=0.550):
    """Supply the minimal structure expected by existing GUI model services."""
    if optic.surfaces.num_surfaces < 2:
        optic.surfaces.clear()
        optic.surfaces.add(
            index=0,
            surface_type="standard",
            radius=float("inf"),
            thickness=10.0,
            comment="Object",
            material="Air",
        )
        optic.surfaces.add(
            index=1,
            surface_type="standard",
            radius=float("inf"),
            thickness=0.0,
            comment="Image",
            material="Air",
        )
    if optic.wavelengths.num_wavelengths == 0:
        optic.wavelengths.add(default_wavelength, is_primary=True, unit="um")
    elif optic.wavelengths.primary_index is None:
        optic.wavelengths.wavelengths[0].is_primary = True
    if optic.aperture is None:
        optic.set_aperture("EPD", 10.0)


def initialize_loaded_optic(optic):
    """Validate/normalize only the owned candidate before it can be published."""
    ensure_valid_structure(optic)
    optic.updater.update()
    return optic
