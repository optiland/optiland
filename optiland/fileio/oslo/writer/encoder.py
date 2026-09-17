"""OSLO File Encoder

Converts an Optic object into an OsloDataModel.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.fileio.common import (
    FIELD_CLASS_TO_TYPE,
    reject_unsupported_propagation,
    validate_material_propagation,
)
from optiland.fileio.oslo.constants import (
    DEFAULT_WAVELENGTHS_UM,
    OBJECT_INFINITY_THRESHOLD,
    THICKNESS_INFINITY_THRESHOLD,
)
from optiland.fileio.oslo.model import OsloDataModel
from optiland.fileio.oslo.surfaces import get_handler_for_optiland_type
from optiland.fileio.oslo.validation import validate_object_na
from optiland.interactions import RefractiveReflectiveModel, ThinLensInteractionModel
from optiland.materials import AbbeMaterial, DataMaterial, IdealMaterial, Material
from optiland.materials.definition import IndexTable
from optiland.physical_apertures import RadialAperture

if TYPE_CHECKING:
    from optiland.optic import Optic


class OpticToOsloEncoder:
    """Encodes an Optic object into an OsloDataModel.

    Args:
        optic: The Optic instance to encode.
    """

    def __init__(self, optic: Optic):
        self.optic = optic
        self.data_model = OsloDataModel()

    def encode(self) -> OsloDataModel:
        """Perform the encoding.

        Returns:
            The populated OsloDataModel.
        """
        validate_material_propagation(self.optic)
        self.data_model = OsloDataModel()
        self.data_model.name = self.optic.name or "LENS"
        self.data_model.num_surfaces = len(self.optic.surfaces) - 1  # Excluding object
        self.data_model.settings["telecentric"] = self.optic.obj_space_telecentric

        self._encode_aperture()
        self._encode_fields()
        self._encode_wavelengths()
        self._encode_surfaces()  # surfaces must be configured before EFL is valid

        # Scaling: use actual EFL (f2) after surfaces are configured
        try:
            self.data_model.scaling = float(self.optic.paraxial.f2())
        except Exception:
            self.data_model.scaling = 1.0

        return self.data_model

    def _encode_aperture(self) -> None:
        if self.optic.aperture is None:
            return
        ap_type = self.optic.aperture.ap_type
        value = self.optic.aperture.value
        if ap_type == "objectNA":
            validate_object_na(self.optic, value)
            self.data_model.aperture["NAO"] = value
            return
        if ap_type not in {"EPD", "imageFNO", "float_by_stop_size"}:
            raise NotImplementedError("OSLO writer cannot export this system aperture")
        infinite = self.optic.object_surface.is_infinite
        try:
            # OSLO EBR is the axial beam radius at surface 1, not at the
            # entrance pupil. The shared model stores this diameter as EPD.
            diameter = (
                float(self.optic.paraxial.EPD())
                if infinite
                else 2 * abs(float(self.optic.paraxial.marginal_ray()[0][1].item()))
            )
        except ValueError:
            if ap_type != "imageFNO" or not infinite:
                raise
            # Preserve the historical no-wavelength export for infinite objects.
            self.data_model.aperture["FNO"] = value
            return
        if not math.isfinite(diameter) or diameter <= 0:
            raise ValueError("OSLO export requires a finite positive entrance beam")
        self.data_model.aperture["EPD"] = diameter

    def _encode_fields(self) -> None:
        if self.optic.fields:
            fd = self.optic.fields.field_definition
            f_type = FIELD_CLASS_TO_TYPE.get(type(fd).__name__) if fd else "angle"
            self.data_model.fields["type"] = f_type
            if f_type not in {"angle", "object_height"}:
                raise NotImplementedError(
                    "OSLO writer cannot export this field definition; use native JSON"
                )
            # ANG/OBH sets the normalization envelope. Explicit RST points
            # below preserve each field's position, weight and vignetting.
            y_values = [v for f in self.optic.fields for v in (f.x, f.y)]
            max_y = max((abs(y) for y in y_values), default=0.0)
            self.data_model.fields["y"] = [max_y]
            if f_type == "angle" and max_y >= 90:
                raise NotImplementedError(
                    "OSLO writer cannot export wide-angle field tables"
                )
            points = {}
            for index, field in enumerate(self.optic.fields, 1):
                point = {"weight": field.weight, "vx": field.vx, "vy": field.vy}
                for axis in ("x", "y"):
                    value = getattr(field, axis)
                    point[axis] = (
                        (
                            math.tan(math.radians(value))
                            / math.tan(math.radians(max_y))
                            if f_type == "angle"
                            else value / max_y
                        )
                        if max_y
                        else 0.0
                    )
                points[index] = point
            self.data_model.fields["points"] = points

    def _encode_wavelengths(self) -> None:
        if self.optic.wavelengths:
            values = [w.value for w in self.optic.wavelengths]
            weights = [w.weight for w in self.optic.wavelengths]
            primary_idx = self.optic.wavelengths.primary_index

            # OSLO convention: primary wavelength must be listed first.
            if primary_idx != 0 and primary_idx < len(values):
                values = (
                    [values[primary_idx]]
                    + values[:primary_idx]
                    + values[primary_idx + 1 :]
                )
                weights = (
                    [weights[primary_idx]]
                    + weights[:primary_idx]
                    + weights[primary_idx + 1 :]
                )

            # OSLO requires wavelength 1 to carry nonzero weight (Program
            # Reference p. 123). Check after moving the native primary first.
            if weights[0] <= 0:
                raise ValueError("OSLO primary wavelength weight must be positive")
            self.data_model.wavelengths["values"] = values
            self.data_model.wavelengths["weights"] = weights
            self.data_model.wavelengths["primary_index"] = 0

    def _encode_surfaces(self) -> None:
        positions = self.optic.surfaces.global_z_positions
        for idx, surface in enumerate(self.optic.surfaces):
            interaction = surface.interaction_model
            position, rotation = surface.geometry.cs.get_effective_transform()
            if (
                abs(float(position[0])) > 1e-12
                or abs(float(position[1])) > 1e-12
                or not be.allclose(rotation, be.eye(3))
            ):
                raise NotImplementedError(
                    "OSLO writer cannot export transformed surfaces; use native JSON"
                )
            if getattr(interaction, "phase_profile", None) is not None:
                raise NotImplementedError(
                    "OSLO writer cannot export phase profiles; use native JSON"
                )
            if interaction.coating is not None or interaction.bsdf is not None:
                raise NotImplementedError(
                    "OSLO writer cannot export coatings or scattering; use native JSON"
                )
            if isinstance(interaction, ThinLensInteractionModel):
                # PFL transforms rays exactly between principal planes, unlike
                # the native thin-lens phase (OSLO Program Reference p. 68).
                # https://lambdares.com/hubfs/Support/support/oslo/oslo_releases/OSLOProgramReference.pdf#page=82
                raise NotImplementedError(
                    "OSLO writer cannot export native thin-lens interactions as "
                    "perfect-imaging PFL surfaces; use native JSON"
                )
            # Subclasses can change the ray physics while inheriting a supported
            # interaction_type name. Only the explicitly mapped model is safe.
            if type(interaction) is not RefractiveReflectiveModel:
                raise NotImplementedError(
                    "OSLO writer cannot export this interaction model; use native JSON"
                )
            s_type = getattr(surface, "surface_type", "standard") or "standard"
            handler = get_handler_for_optiland_type(s_type)
            surf_data = handler.format(surface)

            # Coordinates define the traced geometry. The construction-time
            # thickness attribute can be zero for absolute-coordinate systems
            # or stale after direct coordinate edits.
            # Native final-surface thickness does not move the detector. OSLO
            # image TH would add defocus to the preceding physical gap instead.
            th = (
                0.0
                if idx == self.data_model.num_surfaces
                else float((positions[idx + 1] - positions[idx]).item())
            )
            if math.isnan(th):
                raise ValueError("OSLO export requires defined axial surface spacings")
            if idx == 0 and math.isfinite(th) and abs(th) >= OBJECT_INFINITY_THRESHOLD:
                raise NotImplementedError(
                    "OSLO cannot represent this finite object distance; use native JSON"
                )
            if (
                idx > 0
                and math.isfinite(th)
                and abs(th) >= THICKNESS_INFINITY_THRESHOLD
            ):
                raise NotImplementedError(
                    "OSLO cannot represent this finite surface spacing; use native JSON"
                )
            if math.isinf(th):
                th = math.copysign(1e10, th)
            surf_data["TH"] = th

            # OSLO assumes surface 1 is the stop if no stop is explicitly marked.
            # We omit the AST command for surface 1 to follow this convention.
            if surface.is_stop and idx != 1:
                surf_data["AST"] = True

            # Material — detect mirror via interaction_model.is_reflective
            material_to_encode = (
                "mirror" if interaction.is_reflective else surface.material_post
            )
            if isinstance(material_to_encode, DataMaterial) and isinstance(
                material_to_encode.definition.dispersion, IndexTable
            ):
                surf_data["glass_wavelengths"] = (
                    material_to_encode.definition.dispersion.wavelengths_um
                )
            elif isinstance(material_to_encode, AbbeMaterial):
                surf_data["glass_wavelengths"] = self.data_model.wavelengths.get(
                    "values"
                ) or list(DEFAULT_WAVELENGTHS_UM)
            surf_data["material"] = self._encode_material(material_to_encode)

            # Aperture
            aperture = surface.aperture
            checked = True
            if aperture is not None and (
                type(aperture) is not RadialAperture or aperture.r_min != 0
            ):
                raise NotImplementedError(
                    "OSLO writer cannot export this aperture shape; use native JSON"
                )
            if aperture is not None and (
                not math.isfinite(aperture.r_max) or aperture.r_max <= 0
            ):
                # AP=0 does not retain a native zero-radius clipping boundary;
                # infinite radii would be formatted as finite OSLO sentinels.
                raise ValueError(
                    "OSLO export requires a finite positive surface aperture radius"
                )
            if idx == 0 and surface.is_infinite:
                # Object surface with infinite conjugate: emit a large AP sentinel
                # matching OSLO EDU convention: AP = tan(max_field_angle) * 1e10
                max_y = max((abs(f.y) for f in self.optic.fields), default=0.0)
                if max_y > 0:
                    sentinel = math.tan(math.radians(max_y)) * 1e10
                else:
                    sentinel = 1e10
                surf_data["AP"] = sentinel
            elif isinstance(aperture, RadialAperture):
                surf_data["AP"] = float(aperture.r_max)
                surf_data["aperture_checked"] = checked
            elif surface.is_stop:
                # No explicit physical aperture on stop: derive from paraxial EPD.
                # OSLO requires AP on the stop to draw full ray bundles for off-axis
                # fields; without it only the chief ray is plotted.
                with contextlib.suppress(Exception):
                    surf_data["AP"] = float(self.optic.paraxial.EPD()) / 2.0

            self.data_model.surfaces[idx] = surf_data

    def _encode_material(self, material: Any) -> str:
        if material == "air" or material is None:
            return "  AIR"
        if material == "mirror":
            return "  RFL"

        if type(material) not in {
            Material,
            DataMaterial,
            IdealMaterial,
            AbbeMaterial,
        }:
            raise NotImplementedError(
                "OSLO writer cannot export this material model; use native JSON"
            )
        reject_unsupported_propagation(material)

        if isinstance(material, Material):
            return f"  GLA {material.name}"

        if isinstance(material, DataMaterial):
            definition = material.definition
            if (
                not isinstance(definition.dispersion, IndexTable)
                or definition.extinction is not None
                or material.bounds != "raise"
            ):
                raise NotImplementedError(
                    "OSLO writer supports only sampled n data without extinction "
                    "and with bounds='raise'; "
                    "use native JSON"
                )
            if any(n <= 0 for n in definition.dispersion.indices):
                raise ValueError("OSLO glass indices must be positive")
            return "  GLA " + " ".join(str(n) for n in definition.dispersion.indices)

        if isinstance(material, IdealMaterial):
            if float(material.absorp.item()) != 0:
                raise NotImplementedError(
                    "OSLO writer cannot export ideal-material absorption; "
                    "use native JSON"
                )
            n = float(material.index.item())
            # Small index differences still carry optical path and power.
            if n == 1.0:
                return "  AIR"
            ns = str(n)
            return f"  GLA {ns} {ns} {ns}"

        # The remaining supported type is AbbeMaterial, sampled at the design
        # wavelengths so export does not invent a catalog-glass identity.
        wavelengths = (
            self.data_model.wavelengths.get("values") or DEFAULT_WAVELENGTHS_UM
        )
        return "  GLA " + " ".join(
            str(float(material.n(w).item())) for w in wavelengths
        )
