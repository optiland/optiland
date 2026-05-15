"""OSLO File Encoder

Converts an Optic object into an OsloDataModel.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.fileio.oslo.model import OsloDataModel
from optiland.fileio.oslo.surfaces import get_handler_for_optiland_type
from optiland.materials import AbbeMaterial, IdealMaterial, Material
from optiland.physical_apertures import RadialAperture

if TYPE_CHECKING:
    from optiland.optic import Optic


# Map from field definition class name to Optiland field type string
_FIELD_CLASS_TO_TYPE: dict[str, str] = {
    "AngleField": "angle",
    "ObjectHeightField": "object_height",
    "ParaxialImageHeightField": "paraxial_image_height",
    "RealImageHeightField": "real_image_height",
}


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
        self.data_model.name = self.optic.name or "LENS"
        self.data_model.num_surfaces = len(self.optic.surfaces) - 1  # Excluding object

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
        if self.optic.aperture:
            ap_type = self.optic.aperture.ap_type
            value = self.optic.aperture.value
            if ap_type == "EPD":
                self.data_model.aperture["EPD"] = value
            elif ap_type == "imageFNO":
                # OSLO always uses EBR (entrance beam radius), never FNO.
                # Convert: EPD = EFL / FNO, EBR = EPD / 2.
                try:
                    efl = float(self.optic.paraxial.f2())
                    self.data_model.aperture["EPD"] = efl / value
                except Exception:
                    self.data_model.aperture["FNO"] = value
            elif ap_type == "objectNA":
                self.data_model.aperture["NAO"] = value

    def _encode_fields(self) -> None:
        if self.optic.fields:
            fd = self.optic.fields.field_definition
            f_type = (
                _FIELD_CLASS_TO_TYPE.get(type(fd).__name__, "angle") if fd else "angle"
            )
            self.data_model.fields["type"] = f_type
            # OSLO convention: store only the maximum absolute field value.
            # The reader expands this to [0, 0.7*max, max] on load.
            y_values = [f.y for f in self.optic.fields]
            max_y = max((abs(y) for y in y_values), default=0.0)
            self.data_model.fields["y"] = [max_y]

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

            self.data_model.wavelengths["values"] = values
            self.data_model.wavelengths["weights"] = weights
            self.data_model.wavelengths["primary_index"] = 0

    def _encode_surfaces(self) -> None:
        for idx, surface in enumerate(self.optic.surfaces):
            s_type = getattr(surface, "surface_type", "standard") or "standard"
            handler = get_handler_for_optiland_type(s_type)
            surf_data = handler.format(surface)

            # Common properties
            th = float(surface.thickness)
            if be.isinf(th):
                th = 1e10
            surf_data["TH"] = th

            # OSLO assumes surface 1 is the stop if no stop is explicitly marked.
            # We omit the AST command for surface 1 to follow this convention.
            if surface.is_stop and idx != 1:
                surf_data["AST"] = True

            # Material — detect mirror via interaction_model.is_reflective
            im = getattr(surface, "interaction_model", None)
            is_mirror = bool(getattr(im, "is_reflective", False))
            material_to_encode = "mirror" if is_mirror else surface.material_post
            surf_data["material"] = self._encode_material(material_to_encode)

            # Aperture
            # th >= 9.9e9 covers both be.inf (converted to 1e10) and 1e10 as-stored
            if idx == 0 and th >= 9.9e9:
                # Object surface with infinite conjugate: emit a large AP sentinel
                # matching OSLO EDU convention: AP = tan(max_field_angle) * 1e10
                max_y = max((abs(f.y) for f in self.optic.fields), default=0.0)
                if max_y > 0:
                    sentinel = math.tan(math.radians(max_y)) * 1e10
                else:
                    sentinel = 1e10
                surf_data["AP"] = sentinel
            elif surface.aperture and isinstance(surface.aperture, RadialAperture):
                surf_data["AP"] = float(surface.aperture.r_max)
            elif surface.is_stop:
                # No explicit physical aperture on stop: derive from paraxial EPD.
                # OSLO requires AP on the stop to draw full ray bundles for off-axis
                # fields; without it only the chief ray is plotted.
                with contextlib.suppress(Exception):
                    surf_data["AP"] = float(self.optic.paraxial.EPD()) / 2.0

            # Decenter/Tilt
            for k in ["DCX", "DCY", "DCZ", "TLA", "TLB", "TLC"]:
                val = getattr(surface, k.lower(), 0.0)
                if val != 0:
                    surf_data[k] = float(val)

            self.data_model.surfaces[idx] = surf_data

    def _encode_material(self, material: Any) -> str:
        if material == "air" or material is None:
            return "AIR"
        if material == "mirror":
            return "RFL"

        if isinstance(material, Material):
            return f"GLA {material.name}"

        if isinstance(material, IdealMaterial):
            n = float(material.index.item())
            # IdealMaterial with n≈1.0 is air - write AIR, not GLA 1.0 1.0 1.0
            if abs(n - 1.0) < 1e-6:
                return "AIR"
            ns = f"{n:.7g}"
            return f"GLA {ns} {ns} {ns}"

        if isinstance(material, AbbeMaterial):
            nd = float(material.index.item())
            # OSLO direct index format: GLA <n_d> <n_F> <n_C>
            try:
                w_d, w_F, w_C = 0.58756, 0.48613, 0.65627
                n_d = f"{float(material.n(w_d).item()):.7g}"
                n_F = f"{float(material.n(w_F).item()):.7g}"
                n_C = f"{float(material.n(w_C).item()):.7g}"
                return f"GLA {n_d} {n_F} {n_C}"
            except Exception:
                ns = f"{nd:.7g}"
                return f"GLA {ns} {ns} {ns}"

        # Fallback
        try:
            name = getattr(material, "name", str(material))
            return f"GLA {name}"
        except Exception:
            return "AIR"
