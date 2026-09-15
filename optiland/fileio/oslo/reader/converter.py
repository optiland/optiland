"""OSLO to Optic Converter

Converts an OsloDataModel into an Optiland Optic object.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.fields.field_types import ObjectHeightField
from optiland.fileio.base import BaseOpticReader
from optiland.fileio.oslo.constants import (
    DEFAULT_WAVELENGTHS_UM,
    OBJECT_INFINITY_THRESHOLD,
    THICKNESS_INFINITY_THRESHOLD,
)
from optiland.fileio.oslo.reader.configurations import select_configuration
from optiland.fileio.oslo.reader.coordinates import surface_coordinates
from optiland.fileio.oslo.reader.geometry import surface_geometry
from optiland.fileio.oslo.reader.parser import OsloDataParser
from optiland.fileio.oslo.reader.pickups import resolve_pickups
from optiland.fileio.oslo.reader.solves import SOLVES, apply_solve, check_solve
from optiland.fileio.oslo.syntax import decode_text, tokenize
from optiland.fileio.oslo.validation import validate_object_na
from optiland.materials import (
    AbbeMaterial,
    BaseMaterial,
    DataMaterial,
    IdealMaterial,
    MatchPolicy,
    Material,
)
from optiland.optic import Optic
from optiland.phase import LinearGratingPhaseProfile
from optiland.physical_apertures import RadialAperture

# ---------------------------------------------------------------------------
# Fallback glass catalog for OSLO glass names not in the refractiveindex.info
# database.  Values are (nd, Vd) sourced from Schott, Ohara, and CDGM
# historical datasheets.  Used only when Material() lookup fails entirely.
# ---------------------------------------------------------------------------
_OSLO_GLASS_FALLBACK: dict[str, tuple[float, float]] = {
    # --- Barium crown / flint ---
    "BAF13": (1.6670, 48.64),
    "BAF53": (1.7012, 41.47),
    "BAFN10": (1.6700, 47.21),
    "BAFN11": (1.6670, 48.45),
    "BALF51": (1.6035, 60.56),
    "BALKN3": (1.5180, 58.90),
    # --- Barium dense flint ---
    "BASF10": (1.7076, 30.05),
    "BASF51": (1.6692, 45.00),
    "BASF52": (1.7432, 39.17),
    "BASF56": (1.7847, 43.97),
    # --- Borosilicate crown ---
    "BK3": (1.4970, 66.10),
    # --- Fluorite crown ---
    "FK1": (1.4678, 67.81),
    # --- Crown flint ---
    "KF3": (1.5143, 56.47),
    "KZF6": (1.5193, 41.87),
    "KZFN1": (1.5688, 45.43),
    "KZFSN5": (1.5979, 42.08),
    # --- Lanthanum flint ---
    "LAF13": (1.7880, 47.46),
    "LAFN24": (1.7040, 39.10),
    "LAFN28": (1.8052, 25.43),
    # --- Lanthanum crown ---
    "LAK28": (1.7880, 47.35),
    "LAK31": (1.6935, 53.20),
    "LAKN6": (1.6400, 60.10),
    "LAKN12": (1.6779, 55.30),
    "LAKN13": (1.6935, 53.20),
    "LAKN16": (1.7130, 53.83),
    # --- Lanthanum dense flint ---
    "LASF8": (1.7847, 26.09),
    "LASFN31": (1.8830, 40.78),
    # --- Light flint ---
    "LLF7": (1.5750, 41.47),
    # --- Dense flint ---
    "SF16": (1.6200, 36.35),
    "SF17": (1.6517, 33.82),
    "SF18": (1.7215, 29.24),
    "SF53": (1.7280, 28.68),
    "SF62": (1.5163, 64.06),
    "SF63": (1.5677, 42.84),
    # --- Crown glasses ---
    "SK19": (1.6667, 41.99),
    "SKN18": (1.6385, 55.45),
    # --- Dense crown ---
    "SSK51": (1.6030, 38.03),
    "SSKN5": (1.6582, 44.84),
    # --- Special / thallium-free ---
    "TIF1": (1.5860, 41.45),
    # --- Short-flint crown ---
    "ZKN7": (1.5082, 65.49),
}

# Manufacturer prefixes used in OSLO glass names that should be stripped before
# retrying the lookup (e.g. H_LAF2 → LAF2, O_PBH1 → PBH1).
_MANUFACTURER_PREFIXES = ("H_", "O_", "P_", "J_", "E_", "K_")

if TYPE_CHECKING:
    from collections.abc import Mapping

    from optiland.fileio.oslo.model import OsloDataModel


class OsloToOpticConverter(BaseOpticReader):
    """Converts an OsloDataModel into an Optic object.

    Args:
        oslo_data: OsloDataModel containing the OSLO optical system data.
    """

    def __init__(
        self,
        oslo_data: OsloDataModel | None = None,
        *,
        strict: bool = False,
        configuration: int = 1,
        material_overrides: Mapping[str, BaseMaterial] | None = None,
    ):
        self.data = oslo_data
        self.strict = strict
        self.configuration = configuration
        self.material_overrides = {}
        for name, material in (material_overrides or {}).items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(material, BaseMaterial)
                or name.casefold() in self.material_overrides
            ):
                raise ValueError(
                    "material_overrides requires unique nonempty names "
                    "and material objects"
                )
            self.material_overrides[name.casefold()] = deepcopy(material)
        self._warned_missing_catalogs: set[str] = set()
        self.optic: Optic | None = None

    def read(self, source: str) -> Optic:
        """Read an OSLO file and return a fully-configured Optic.

        Args:
            source: Local file path to a .len file.

        Returns:
            A configured Optic instance.
        """
        self.data = OsloDataParser(source, strict=self.strict).parse()
        return self.convert()

    def convert(self) -> Optic:
        """Convert the stored OSLO data model into an Optic object.

        Returns:
            The fully-configured Optic instance.
        """
        if self.data is None:
            raise ValueError("No OSLO data to convert.")
        if self.strict and self.data.diagnostics:
            diagnostic = self.data.diagnostics[0]
            raise ValueError(
                f"OSLO {diagnostic.command} at line {diagnostic.line}, surface "
                f"{diagnostic.surface}: {diagnostic.message}; strict conversion "
                "requires a prescription without unsupported commands"
            )

        prescription = self.data
        try:
            self.data = select_configuration(prescription, self.configuration)
            self.data.surfaces = resolve_pickups(self.data.surfaces)
            self._validate_catalog_references()
            self._build_optic()
            self._apply_solves()
            self._apply_image_focus()
            return self.optic
        finally:
            # Overrides and solved values belong to this conversion only. The
            # next selection must start from the original base, even on failure.
            self.data = prescription

    def _apply_image_focus(self) -> None:
        """Move the detector by OSLO's defocus after nominal solves and pickups."""
        image_index = max(self.data.surfaces)
        shift = self.data.surfaces[image_index].get("TH", 0.0)
        image = self.optic.surfaces[image_index]
        image.thickness = 0.0
        if not shift:
            return
        if image_index < 2:
            raise ValueError("OSLO image focus shift requires an interior surface")
        if "GC" in self.data.surfaces[image_index]:
            raise ValueError(
                "OSLO image focus shift with a global reference is not mapped"
            )

        # Image TH adds to the preceding nominal gap; it does not advance a
        # nonexistent next surface. Apply it after solves so PY=0 may retain
        # deliberate defocus. Program Reference pp. 46 and 122:
        # https://lambdares.com/hubfs/Support/support/oslo/oslo_releases/OSLOProgramReference.pdf#page=60
        if self._coordinates:
            # Reuse the coordinate convention, including bends and returns,
            # rather than moving a tilted image leg along global z.
            surfaces = deepcopy(self.data.surfaces)
            previous = surfaces[image_index - 1]
            previous["TH"] = previous.get("TH", 0.0) + shift
            position = surface_coordinates(surfaces, self.data.units)[image_index]
            for axis in ("x", "y", "z"):
                setattr(image.geometry.cs, axis, be.array(position[axis]))
        else:
            image.geometry.cs.z = image.geometry.cs.z + shift * self.data.units
        self.optic.surfaces[image_index - 1].thickness += shift * self.data.units
        if image.is_stop:
            # Moving the stop moves its entrance pupil. Preserve OSLO's beam
            # specification with the new pupil while keeping nominal GIH fields.
            self._configure_aperture()
            self._configure_telecentric_launch()

    def _build_optic(self) -> None:
        """Build from the current resolved prescription without applying solves."""
        self.optic = Optic(self.data.name)
        self.optic.obj_space_telecentric = self.data.settings.get("telecentric", False)
        self.optic.fields.set_telecentric(self.optic.obj_space_telecentric)
        self._configure_surfaces()
        self._configure_wavelengths()
        self._configure_aperture()
        self._configure_fields()
        self._configure_telecentric_launch()

    def _configure_telecentric_launch(self) -> None:
        """Adapt supported TELE prescriptions to the native real-ray launcher."""
        if not self.optic.obj_space_telecentric:
            return
        if (
            self.optic.object_surface.is_infinite
            or not isinstance(self.optic.fields.field_definition, ObjectHeightField)
            or float(
                self.optic.object_surface.material_post.n(
                    self.optic.primary_wavelength
                ).item()
            )
            != 1
            or not self.optic.surfaces.build_paraxial_path().entry_is_positive_z
        ):
            message = (
                "OSLO TELE real-ray launch currently requires finite object-height "
                "fields in air and entry along +z"
            )
            if self.strict:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=3)
            return
        if self.optic.aperture.ap_type != "objectNA":
            # Keep the axial cone fixed while allowing every field point's
            # chief ray to launch parallel to the axis. The native TELE aimer
            # expects the cone's sine as an object-NA aperture in air.
            slope = abs(float(self.optic.paraxial.marginal_ray()[1][0].item()))
            self.optic.set_aperture("objectNA", math.sin(math.atan(slope)))
        if not 0 < self.optic.aperture.value < 1:
            raise ValueError(
                "OSLO TELE launch requires object NA strictly between 0 and 1"
            )

    def _configure_surfaces(self) -> None:
        """Configure all surfaces on the optic."""
        coordinate_commands = {
            "DCX",
            "DCY",
            "DCZ",
            "TLA",
            "TLB",
            "TLC",
            "GC",
            "RCO",
            "BEN",
            "TOX",
            "TOY",
            "TOZ",
        }
        has_coord_transform = any(
            coordinate_commands.intersection(sd) for sd in self.data.surfaces.values()
        )

        # Determine if any surface is explicitly marked as the stop
        has_stop = any(sd.get("AST", False) for sd in self.data.surfaces.values())
        self._coordinates = (
            surface_coordinates(self.data.surfaces, self.data.units)
            if has_coord_transform
            else {}
        )

        for idx in sorted(self.data.surfaces.keys()):
            surf_data = self.data.surfaces[idx]

            # OSLO convention: if no stop is specified, it's surface 1
            if not has_stop and idx == 1:
                surf_data["AST"] = True

            self._configure_surface(idx, surf_data, has_coord_transform)

    def _configure_surface(
        self, index: int, data: dict[str, Any], has_coord_transform: bool
    ) -> None:
        scale = self.data.units
        surface_params = surface_geometry(data, scale)
        # Native scalar traces use the base radius only. Constant, linear and
        # quadratic sag terms can change the vertex, normal or paraxial power.
        low_order = {"ASR": (1,), "ARA": (1, 2), "ASX": tuple(range(6))}
        if any(data.get(f"AS{i}", 0) for i in low_order.get(data.get("ASP"), ())):
            message = (
                f"OSLO asphere at surface {index} has low-order sag terms ignored "
                "by native paraxial analysis; real-ray geometry is retained, but "
                "paraxial pupils, fields and solves may be inaccurate"
            )
            if self.strict:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=3)
        surface_params["index"] = index
        surface_params["is_stop"] = data.get("AST", False)

        th = data.get("TH", 0.0)
        # Object conjugates have a documented cutoff below the large sentinel
        # used for other distances (sometimes saved as 9.9999999996e+09).
        infinity_threshold = (
            OBJECT_INFINITY_THRESHOLD if index == 0 else THICKNESS_INFINITY_THRESHOLD
        )
        if abs(th) >= infinity_threshold:
            th = be.inf if th > 0 else -be.inf
        surface_params["thickness"] = th * scale

        # Is paraxial?
        if "PFL" in data:
            message = "OSLO PFL perfect imagery is approximated by a paraxial thin lens"
            if self.strict:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=3)
            surface_params["surface_type"] = "paraxial"
            surface_params["f"] = data["PFL"] * scale

        if "GSP" in data or "GOR" in data:
            spacing, order = data.get("GSP", 0.0), data.get("GOR", 1)
            if int(order) != order or spacing < 0 or (spacing == 0 and order != 0):
                raise ValueError(
                    "OSLO GSP requires positive spacing and GOR an integer order"
                )
            if spacing and order:
                surface_params["phase_profile"] = LinearGratingPhaseProfile(
                    spacing * scale, angle=math.pi / 2, order=int(order)
                )

        # Handle material
        material_raw = data.get("material", "AIR")
        surface_params["material"] = self._resolve_material(
            material_raw, data.get("glass_wavelengths")
        )

        if (
            data.get("aperture_checked")
            and self.data.settings.get("aperture_check", True)
            and data.get("AP", 0.0) > 0
        ):
            surface_params["aperture"] = RadialAperture(r_max=data["AP"] * scale)

        if has_coord_transform:
            surface_params.pop("thickness")
            surface_params.update(self._coordinates[index])

        self.optic.surfaces.add(**surface_params)
        if has_coord_transform:
            self.optic.surfaces[index].thickness = th * scale

    def _resolve_material(
        self, material_raw: str, wavelengths: list[float] | None = None
    ) -> Any:
        if material_raw == "AIR":
            return "air"
        if material_raw == "RFL":
            return "mirror"

        if material_raw.startswith("GLA "):
            rest = material_raw[4:].strip()
            parts = tokenize(rest)
            if not parts:
                return "air"

            name = ""
            # Modeled glass (e.g. GLA MOD G1 1.6489 1.662...).
            modeled = parts[0].upper() == "MOD"
            if modeled:
                parts = parts[1:]
            if not parts:
                raise ValueError("GLA MOD requires refractive-index data")
            try:
                float(parts[0])
            except ValueError:
                name, parts = decode_text(parts[0]), parts[1:]
            if not parts:
                if modeled:
                    raise ValueError("GLA MOD requires refractive-index data")
                # Catalog glass (e.g. GLA BK7).
                return self._resolve_catalog_glass(name)
            # Direct indices (e.g. GLA 1.573 1.573 1.573), or index data
            # remaining after a named/model-glass prefix.
            indices = [float(value) for value in parts]
            if any(value <= 0 for value in indices):
                raise ValueError(
                    f"OSLO glass {name!r} requires positive refractive indices"
                )
            if modeled and len(indices) == 2:
                # Interactive model glass (e.g. GLA MOD 1.6 50) specifies
                # refractive index and Abbe number.
                # Saved legacy MOD records instead contain explicit index samples.
                if self.strict:
                    raise ValueError("GLA MOD index/Abbe dispersion is approximate")
                warnings.warn(
                    "OSLO GLA MOD index/Abbe uses Optiland's Buchdahl model",
                    UserWarning,
                    stacklevel=3,
                )
                return AbbeMaterial(*indices, model="buchdahl")
            if len(set(indices)) == 1:
                return IdealMaterial(indices[0])
            wavelengths = wavelengths or list(DEFAULT_WAVELENGTHS_UM)
            if len(indices) != len(wavelengths):
                raise ValueError(f"OSLO glass {name!r} index/wavelength counts differ")
            return DataMaterial.from_samples(wavelengths, indices, name=name)

        return "air"

    def _validate_catalog_references(self) -> None:
        """Report all missing names together before constructing the optic."""
        missing = {}
        for surface in self.data.surfaces.values():
            raw = surface.get("material", "AIR")
            parts = tokenize(raw)
            if len(parts) != 2 or parts[0] != "GLA" or parts[1].upper() == "MOD":
                continue
            try:
                float(parts[1])
                continue
            except ValueError:
                name = decode_text(parts[1])
            key = name.casefold()
            if key not in self.material_overrides:
                missing[key] = name
        if self.strict and missing:
            raise ValueError(
                "OSLO glasses could not be resolved from material bindings: "
                + ", ".join(sorted(missing.values(), key=str.casefold))
                + ". Provide material_overrides."
            )

    def _resolve_catalog_glass(self, name: str) -> Any:
        """Prefer caller bindings and material data before permissive fallback."""
        if name.casefold() in self.material_overrides:
            return deepcopy(self.material_overrides[name.casefold()])
        if name.casefold() not in self._warned_missing_catalogs:
            warnings.warn(
                f"OSLO glass {name!r} has no definition in material bindings; "
                "using database/approximation fallback",
                UserWarning,
                stacklevel=3,
            )
            self._warned_missing_catalogs.add(name.casefold())

        # OSLO requires unique glass names across its installed catalogs. The
        # refractiveindex.info catalog has broader family names and author data;
        # strict import cannot silently accept a fuzzy or ambiguous identity.
        # https://lambdares.com/support-posts/editing-glass-catalogs
        # Step 1 – direct DB lookup.
        try:
            return Material(name, match_policy=MatchPolicy.WARN)
        except ValueError:
            pass

        # Step 2 – strip manufacturer prefix and retry.
        for prefix in _MANUFACTURER_PREFIXES:
            if name.upper().startswith(prefix):
                base = name[len(prefix) :]
                try:
                    material = Material(base)
                    message = (
                        f"OSLO glass {name!r} requires a manufacturer-prefix "
                        "substitution; "
                        "provide material_overrides to specify its catalog identity"
                    )
                    warnings.warn(message, UserWarning, stacklevel=3)
                    return material
                except ValueError:
                    pass
                # Also check the fallback table for the base name.
                if base.upper() in _OSLO_GLASS_FALLBACK:
                    nd, vd = _OSLO_GLASS_FALLBACK[base.upper()]
                    return self._fallback_material(name, nd, vd)
                break

        # Step 3 – built-in OSLO fallback table.
        entry = _OSLO_GLASS_FALLBACK.get(name.upper())
        if entry is not None:
            nd, vd = entry
            return self._fallback_material(name, nd, vd)

        # Step 4 – cannot resolve; warn and fall back to air.
        warnings.warn(
            f"OSLO glass '{name}' could not be resolved and will be treated as "
            "air. Provide material_overrides with a verified catalog definition.",
            UserWarning,
            stacklevel=4,
        )
        return "air"

    def _fallback_material(self, name: str, nd: float, vd: float) -> AbbeMaterial:
        message = f"OSLO glass {name!r} uses approximate historical Abbe dispersion"
        warnings.warn(message, UserWarning, stacklevel=4)
        return AbbeMaterial(nd, vd, model="buchdahl")

    def _configure_aperture(self) -> None:
        aperture_data = self.data.aperture or {"EPD": 2.0}
        if (
            "FNO" in aperture_data
            and self.optic.object_surface.is_infinite
            and float(
                self.optic.surfaces[-1]
                .material_pre.n(self.optic.primary_wavelength)
                .item()
            )
            == 1.0
            and 0 < float(self.optic.paraxial.f2()) < math.inf
        ):
            self.optic.set_aperture("imageFNO", aperture_data["FNO"])
            return
        if "EPD" in aperture_data:
            # The shared model's EPD is twice OSLO's EBR, the radius at surface 1
            # (Program Reference p. 120). For finite objects, this plane need
            # not coincide with the entrance pupil.
            diameter = aperture_data["EPD"] * self.data.units
            self.optic.set_aperture("EPD", 1.0)
            if not self.optic.object_surface.is_infinite:
                height = abs(float(self.optic.paraxial.marginal_ray()[0][1].item()))
                if not math.isfinite(height) or height == 0:
                    raise ValueError("EBR requires a finite nonzero beam at surface 1")
                diameter /= 2 * height
            self.optic.set_aperture("EPD", diameter)

        if "NAO" in aperture_data:
            value = aperture_data["NAO"]
            validate_object_na(self.optic, value)
            self.optic.set_aperture("objectNA", value)
        if any(key in aperture_data for key in ("NAP", "FNO", "PUK")):
            # OSLO's image NA is an aplanatic paraxial specification. Scale a
            # unit pupil using its image-space reduced slope (n * u).
            self.optic.set_aperture("EPD", 1.0)
            _, slopes = self.optic.paraxial.marginal_ray()
            n_image = self.optic.surfaces[-1].material_pre.n(
                self.optic.primary_wavelength
            )
            reduced_slope = abs(float((n_image * slopes[-2]).item()))
            target = aperture_data.get("NAP")
            if "FNO" in aperture_data:
                target = 1 / (2 * aperture_data["FNO"])
            elif "PUK" in aperture_data:
                target = aperture_data["PUK"] * float(n_image.item())
            if reduced_slope == 0:
                raise ValueError("NAP cannot define an aperture for an afocal system")
            self.optic.set_aperture("EPD", target / reduced_slope)

    def _configure_fields(self) -> None:
        field_data = self.data.fields
        field_type = field_data.get("type", "angle")
        y_coords = field_data.get("y", [0.0])

        if field_type == "gaussian_image_height":
            height = y_coords[0] * self.data.units
            if self.optic.object_surface.is_infinite:
                field_type = "angle"
                y_coords = [
                    math.degrees(math.atan(height / float(self.optic.paraxial.f2())))
                ]
            else:
                field_type = "object_height"
                y_coords = [
                    height
                    / float(self.optic.paraxial.magnification())
                    / self.data.units
                ]

        # If object is at infinity, ObjectHeightField is invalid in Optiland.
        # OSLO often uses OBH even for infinite objects, encoding the angle.
        # OSLO OBH sign convention: negative means below axis - take abs().
        if field_type == "object_height" and self.optic.object_surface.is_infinite:
            field_type = "angle"
            distance = self.data.surfaces[0].get("TH", 1e10)
            y_coords = [
                abs(float(be.degrees(be.arctan(y / distance)))) for y in y_coords
            ]
        elif field_type == "angle":
            # ANG is always positive in OSLO for the max half-angle.
            y_coords = [abs(y) for y in y_coords]
        else:
            y_coords = [y * self.data.units for y in y_coords]

        # The reference field must stay below 90 degrees even in OSLO's
        # separate WARM mode (Program Reference p. 215). Our ordinary field
        # mapping uses tan/atan, which would fold 100 degrees onto -80 degrees.
        # https://lambdares.com/hubfs/Support/support/oslo/oslo_releases/OSLOProgramReference.pdf#page=229
        if field_type == "angle" and any(abs(y) >= 90 for y in y_coords):
            raise ValueError("OSLO angular reference must be less than 90 degrees")

        self.optic.fields.set_type(field_type)

        if "points" in field_data:
            maximum = y_coords[0]
            for point in field_data["points"].values():
                coords = dict(point)
                for axis in ("x", "y"):
                    fraction = point[axis]
                    coords[axis] = (
                        math.degrees(
                            math.atan(fraction * math.tan(math.radians(maximum)))
                        )
                        if field_type == "angle"
                        else fraction * maximum
                    )
                self.optic.fields.add(**coords)
            if self.optic.fields.num_fields:
                return

        # OSLO stores only the maximum field value.  Expand to three standard
        # field points (on-axis, 0.7×max, full field) for usable analysis.
        max_y = max((abs(y) for y in y_coords), default=0.0)
        fields_to_add = [0.0, round(0.7 * max_y, 8), max_y] if max_y > 0.0 else [0.0]

        for y in fields_to_add:
            self.optic.fields.add(y=y, x=0.0)

    def _apply_solves(self) -> None:
        """Apply surface-specific solves, retaining saved values on warned failures."""
        applied = []
        for index in sorted(self.data.surfaces):
            for command in SOLVES:
                data = self.data.surfaces[index]
                if command not in data:
                    continue
                saved = self.optic.to_dict()
                saved_surfaces = deepcopy(self.data.surfaces)
                target = data[command]
                try:
                    apply_solve(self.optic, index, command, target, self.data.units)
                    key = "TH" if command in {"PY", "PYC", "EC"} else "RD"
                    surface = self.optic.surfaces[index]
                    value = (
                        surface.thickness if key == "TH" else surface.geometry.radius
                    )
                    data[key] = float(be.asarray(value).item()) / self.data.units
                    self.data.surfaces = resolve_pickups(self.data.surfaces)
                    self._build_optic()
                    # Recomputing dependent geometry, NAP/FNO/PUK or GIH may
                    # change the rays used by this solve or an earlier one.
                    # Accept only a prescription that still meets every target.
                    pending = (index, command, target)
                    for solve_index, solve_command, solve_target in [*applied, pending]:
                        check_solve(
                            self.optic,
                            solve_index,
                            solve_command,
                            solve_target,
                            self.data.units,
                        )
                    applied.append(pending)
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    self.data.surfaces = saved_surfaces
                    self.optic = Optic.from_dict(saved)
                    message = f"OSLO {command} at surface {index}: {exc}"
                    if self.strict:
                        raise ValueError(message) from exc
                    warnings.warn(
                        message + "; retained saved prescription",
                        UserWarning,
                        stacklevel=3,
                    )

    def _configure_wavelengths(self) -> None:
        wl_data = self.data.wavelengths
        values = wl_data.get("values", [0.58756])
        weights = wl_data.get("weights", [1.0] * len(values))
        primary_idx = wl_data.get("primary_index", 0)

        # OSLO wavelengths are in microns.
        for idx, val in enumerate(values):
            is_primary = idx == primary_idx
            w = weights[idx] if idx < len(weights) else 1.0
            self.optic.wavelengths.add(value=val, is_primary=is_primary, weight=w)
