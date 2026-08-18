"""Optic Serializer Module

Handles serialization and deserialization of Optic instances to/from
dictionaries, keeping serialization concerns separate from optical physics.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optiland.aberrations import Aberrations
from optiland.aperture import BaseSystemAperture
from optiland.apodization import BaseApodization
from optiland.fields import BaseFieldDefinition, FieldGroup
from optiland.paraxial import Paraxial
from optiland.pickup import PickupManager
from optiland.raytrace.real_ray_tracer import RealRayTracer
from optiland.solves import SolveManager
from optiland.surfaces import SurfaceGroup
from optiland.wavelength import WavelengthGroup

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


class OpticSerializer:
    """Handles serialization and deserialization of Optic instances."""

    @staticmethod
    def to_dict(optic: Optic) -> dict:
        """Convert the optical system to a dictionary.

        Args:
            optic: The optical system to serialize.

        Returns:
            The dictionary representation of the optical system.

        """
        data = {
            "version": 1.0,
            "name": optic.name,
            "aperture": optic.aperture.to_dict() if optic.aperture else None,
            "fields": optic.fields.to_dict(),
            "wavelengths": optic.wavelengths.to_dict(),
            "apodization": optic.apodization.to_dict() if optic.apodization else None,
            "pickups": optic.pickups.to_dict(),
            "solves": optic.solves.to_dict(),
            "surface_group": optic.surfaces.to_dict(),
        }

        data["wavelengths"]["polarization"] = optic.polarization
        data["ray_tracer"] = {"ray_aiming_config": optic.ray_tracer.ray_aiming_config}

        if getattr(optic, "sequences", None):
            data["sequences"] = {
                name: seq.raw_steps for name, seq in optic.sequences.items()
            }

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optic:
        """Create an optical system from a dictionary.

        Args:
            data: The dictionary representation of the optical system.

        Returns:
            The optical system.

        """
        from optiland.optic.optic import Optic

        optic = Optic()
        optic.name = data.get("name")
        optic.aperture = BaseSystemAperture.from_dict(data["aperture"])
        optic.surfaces = SurfaceGroup.from_dict(data["surface_group"])
        optic.fields = FieldGroup.from_dict(data["fields"])
        optic.wavelengths = WavelengthGroup.from_dict(data["wavelengths"])

        apodization_data = data.get("apodization")
        if apodization_data:
            optic.apodization = BaseApodization.from_dict(apodization_data)

        optic.pickups = PickupManager.from_dict(optic, data["pickups"])
        optic.solves = SolveManager.from_dict(optic, data["solves"])

        optic.polarization = data["wavelengths"]["polarization"]
        if data["fields"].get("field_definition"):
            optic.fields.field_definition = BaseFieldDefinition.from_dict(
                data["fields"]["field_definition"]
            )
        elif data["fields"].get("field_type"):
            optic.fields.set_type(data["fields"]["field_type"])
        else:
            optic.fields.field_definition = None
        optic.obj_space_telecentric = data["fields"]["telecentric"]

        optic.paraxial = Paraxial(optic)
        optic.aberrations = Aberrations(optic)
        optic.ray_tracer = RealRayTracer(optic)

        if data.get("ray_tracer"):
            optic.ray_tracer.ray_aiming_config = data["ray_tracer"].get(
                "ray_aiming_config", optic.ray_tracer.ray_aiming_config
            )

        for name, steps in data.get("sequences", {}).items():
            optic.add_sequence(name, steps)

        return optic
