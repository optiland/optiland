"""Bounded scalar phase-screen adapter for sequential Optiland systems."""

from __future__ import annotations

import math
from copy import copy
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import optiland.backend as be
from optiland.backend.utils import is_torch_tensor
from optiland.geometries.even_asphere import EvenAsphere
from optiland.geometries.plane import Plane
from optiland.geometries.standard import StandardGeometry
from optiland.interactions.phase_interaction_model import PhaseInteractionModel
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel
from optiland.materials.abbe import AbbeMaterial, AbbeMaterialE
from optiland.materials.base import BaseMaterial
from optiland.phase.constant import ConstantPhaseProfile
from optiland.phase.radial import RadialPhaseProfile
from optiland.physical_apertures import (
    EllipticalAperture,
    RadialAperture,
    RectangularAperture,
)
from optiland.physical_optics.field import ScalarField, _cast_real_like
from optiland.physical_optics.propagation import _phase_precision
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.surfaces.image_surface import ImageSurface
from optiland.surfaces.standard_surface import Surface

if TYPE_CHECKING:
    from optiland._types import BEArrayT
    from optiland.optic import Optic


def _real_value(value, name: str, *, allow_infinite: bool = False) -> float:
    """Read scalar metadata, never a sampled field, for preflight validation."""
    if isinstance(value, Real):
        result = float(value)
    elif isinstance(value, be.ndarray):
        if (be.get_backend() == "torch") != is_torch_tensor(value):
            raise ValueError(f"{name} must belong to the active backend.")
        complex_value = (
            value.is_complex() if is_torch_tensor(value) else value.dtype.kind == "c"
        )
        if complex_value or be.size(value) != 1:
            raise ValueError(f"{name} must be a real scalar (no multi-configuration).")
        scalar = value.detach() if is_torch_tensor(value) else value
        result = float(scalar.item())
    else:
        raise ValueError(f"{name} must be a real scalar.")
    if math.isnan(result) or (not allow_infinite and not math.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


def _like(value, data):
    """Move scalar geometry parameters without detaching their gradient graph."""
    if isinstance(value, be.ndarray):
        return _cast_real_like(value, data).reshape(())
    return value


def _validate_aperture(aperture, label: str) -> None:
    if aperture is None:
        return
    if type(aperture) is RadialAperture:
        lower = _real_value(aperture.r_min, f"{label} aperture r_min")
        upper = _real_value(aperture.r_max, f"{label} aperture r_max")
        if not 0 <= lower <= upper:
            raise ValueError(f"{label}: aperture requires 0 <= r_min <= r_max.")
    elif type(aperture) is RectangularAperture:
        for axis in ("x", "y"):
            lower = _real_value(getattr(aperture, f"{axis}_min"), label)
            upper = _real_value(getattr(aperture, f"{axis}_max"), label)
            if lower > upper:
                raise ValueError(f"{label}: aperture minimum exceeds maximum.")
    elif type(aperture) is EllipticalAperture:
        for name in ("a", "b", "offset_x", "offset_y"):
            value = _real_value(getattr(aperture, name), f"{label} aperture {name}")
            if name in ("a", "b") and value <= 0:
                raise ValueError(f"{label}: ellipse semi-axes must be positive.")
    else:
        raise ValueError(
            f"{label}: unsupported physical aperture {type(aperture).__name__}."
        )


def _index(
    material: BaseMaterial,
    wavelength_um: float,
    label: str,
    absorption: Literal["reject", "axial"],
) -> tuple[float, float]:
    """Evaluate on a shallow material view to leave source caches untouched."""
    for property_name in ("n", "k"):
        bounds = material.spectral_range(property_name)
        if bounds is not None and not bounds[0] <= wavelength_um <= bounds[1]:
            raise ValueError(
                f"{label}: wavelength is outside the material {property_name} range."
            )
    view = copy(material)
    if isinstance(material, (AbbeMaterial, AbbeMaterialE)):
        # Native predictive models refresh derived coefficients on evaluation.
        # Isolate attribute replacement while keeping live parameter tensors
        # shared; deep-copying non-leaf Torch tensors would break autograd.
        view.model = copy(material.model)
    view._n_cache = {}
    view._k_cache = {}
    view._cache_context = None
    n = view.n(wavelength_um)
    k = view.k(wavelength_um)
    index = _real_value(n, f"{label} refractive index")
    extinction = _real_value(k, f"{label} extinction coefficient")
    if absorption == "reject" and (index <= 0 or extinction != 0):
        raise ValueError(
            f"{label}: only positive, real, lossless indices are supported."
        )
    if index <= 0 or extinction < 0:
        raise ValueError(
            f"{label}: axial absorption requires a positive real index and "
            "nonnegative passive extinction coefficient (no gain)."
        )
    return index, extinction


def _validate_profile(profile, label: str) -> None:
    """Whitelist native, audited radian-valued analytic profiles only."""
    if type(profile) is ConstantPhaseProfile:
        _real_value(profile.phase, f"{label} constant phase")
    elif type(profile) is RadialPhaseProfile:
        coefficients = profile.coefficients
        if not isinstance(coefficients, (list, tuple)) and not (
            isinstance(coefficients, be.ndarray) and coefficients.ndim == 1
        ):
            raise ValueError(f"{label}: radial coefficients must be a scalar sequence.")
        for coefficient in coefficients:
            _real_value(coefficient, f"{label} radial phase coefficient")
    else:
        raise ValueError(
            f"{label}: unsupported phase profile {type(profile).__name__}; "
            "only native ConstantPhaseProfile and RadialPhaseProfile are audited."
        )
    efficiency = _real_value(profile.efficiency, f"{label} phase efficiency")
    if not 0 <= efficiency <= 1:
        raise ValueError(f"{label}: phase efficiency must be between 0 and 1.")


def _profile_screen(profile, x, y, wavelength_um: float, data):
    """Evaluate a read-only native profile view in field coordinates and units."""
    view = copy(profile)
    if type(profile) is ConstantPhaseProfile:
        view.phase = _like(profile.phase, data)
    else:
        view.coefficients = [
            _like(coefficient, data) for coefficient in profile.coefficients
        ]

    bits = (
        data.real.element_size() * 8
        if is_torch_tensor(data)
        else data.real.dtype.itemsize * 8
    )
    precision_matches = be.get_precision() == bits
    device_matches = not is_torch_tensor(data) or str(data.device) == str(
        be.get_device()
    )
    if precision_matches and device_matches:
        phase = view.get_phase(x, y, wavelength_um)
    else:
        # Native *_like constructors use backend-default dtype/device, not
        # those of x/y. These two wavelength-independent analytic profiles have
        # exact equivalents that avoid downcasting or moving a sampled grid.
        phase = x * 0
        if type(view) is ConstantPhaseProfile:
            phase = phase + view.phase
        else:
            r2 = x**2 + y**2
            for power, coefficient in enumerate(view.coefficients, start=1):
                phase = phase + coefficient * r2**power
    phase = _cast_real_like(phase, data)
    efficiency = _like(profile.efficiency, data)
    amplitude = (
        math.sqrt(efficiency) if isinstance(efficiency, Real) else be.sqrt(efficiency)
    )
    return phase, amplitude


class ScalarOpticalTrain:
    r"""Propagate arbitrary sampled fields through a bounded Optiland train.

    This is a **scalar paraxial phase-screen approximation**, not exact curved
    interface remapping, vector optics, or a high-NA/general ray-train solver.
    Native ``Plane``, ``StandardGeometry`` (sphere/conic), and ``EvenAsphere``
    (conic plus even radial polynomial) surfaces must be
    coaxial with global +z, transmissive, homogeneous, and uncoated.
    Materials must be lossless unless ``absorption="axial"`` is selected.
    Radial, rectangular, and elliptical physical apertures are supported.
    Only explicit ``surface.aperture`` objects clip the supplied field.
    ``Optic.aperture`` (EPD/imageFNO/objectNA) controls ray-launch/pupil sampling;
    it does not define a physical mask here. No stop radius is inferred through
    ``optic.paraxial``; callers must provide physical stop apertures explicitly.
    Native planar ``ThinLensInteractionModel`` and ``PhaseInteractionModel``
    with exact native ``ConstantPhaseProfile`` / ``RadialPhaseProfile`` types
    are also supported. Other phase profiles, cylindrical thin lenses, and
    curved thin-lens/phase interactions have not been audited and are rejected.
    All rotations, decenters, reference frames, mirrors, negative vertex gaps,
    GRIN media, scattering, coatings, unsupported interactions/phase profiles,
    custom surface/geometry subclasses, and other apertures are rejected.

    The input plane is immediately **before** the start surface vertex; the
    output is immediately **after** the end surface vertex. Surface indices
    follow ``optic.surface_group``: index 0 is the object and is never included.
    By default the last listed surface is included, including its aperture and
    material transition. An actual ``ImageSurface`` is supported only as a
    planar marker without a refractive-index or extinction change.
    No object-to-start distance or trailing end-surface thickness is propagated,
    even for an object at infinity. Vertex gaps are taken from geometry
    coordinates, not thickness.

    Field spacings, vacuum wavelength, sag, and vertex distances are all in
    **millimeters**. Material lookup alone converts wavelength to micrometers.
    Native even-asphere ``coefficients[j]`` multiply ``r**(2*(j+1))`` in sag;
    their units are ``mm**(1-2*(j+1))``. Only exact native ``EvenAsphere`` types
    are supported, with finite scalar coefficients and conic constant and a
    nonzero real radius. Infinite radius in either native conic geometry means
    a flat conic base. Its radius and conic are constant metadata, not
    differentiable parameters in that limit. Polynomial coefficient gradients
    remain supported on a flat base.
    With positive ``exp(+ikz)`` propagation, each surface multiplies the field
    by ``exp(1j * 2*pi/wavelength * (n_before-n_after) * sag(x,y))`` on its
    grid returned by ``field.coordinates()`` (including its center offset), then
    ASM propagates the vertex gap in the outgoing medium.

    The default ``absorption="reject"`` rejects every nonzero extinction
    coefficient, including arbitrarily small catalog values. The explicit
    ``"axial"`` opt-in permits passive extinction ``kappa >= 0`` and applies
    homogeneous Beer--Lambert amplitude ``exp(-2*pi*kappa*gap/wavelength)``
    once per outgoing vertex gap, using that surface's ``material_post`` at
    the field's vacuum wavelength. This uniform axial-length approximation
    has no extra refractive-index factor. It does not model curved/sag-dependent
    or angle/secant path lengths, interface flux, Fresnel reflection, coatings,
    or complex-index ASM. Zero gaps and the unpropagated trailing medium add
    no attenuation; a differentiable zero gap retains the positive one-sided
    absorption derivative. Absorbed power is not renormalized.
    A native thin lens adds the **paraxial quadratic** phase
    ``-2*pi/wavelength * (x*x+y*y)/(2*f)``: its ``f`` is inverse reduced
    optical power, so a collimated input focuses at ``n_after*f`` rather than
    ``f`` in a non-air output medium. No extra index multiplies this phase.
    This is not the native real-ray hyperbolic OPD expression. Finite nonzero
    positive/negative ``f`` and infinite (identity) ``f`` are supported.
    Supported phase profiles return radians; their ``get_phase(x,y,wavelength)``
    is evaluated with millimeter coordinates and micrometer vacuum wavelength,
    without another wavelength/index factor. Transmission is
    ``sqrt(profile.efficiency) * exp(+1j*phase)`` with scalar efficiency in [0,1].
    The two native profiles are wavelength-independent and currently have unit
    efficiency; the efficiency contract is nevertheless validated and consumed.
    If native array creation would use a different dtype/device from the field,
    the same two audited analytic formulas are evaluated directly on its grid.
    The scalar amplitude is power-normalized (square root of power density,
    not volts/meter): lossless interfaces do not change integrated
    ``abs(data)**2``. Fresnel reflection, obliquity, and polarization
    factors are deliberately omitted; nontrivial coatings are not approximated.

    The caller must resolve phase gradients and aperture edges and provide
    sufficient window/padding for periodic FFT propagation. Propagating spectral
    content is retained; evanescent content is discarded by ASM at nonzero gaps.
    Zero gaps apply screens without FFT filtering (a differentiable zero gap
    uses the identity-preserving ASM decay policy). Sampling does not establish
    physical validity of the paraxial surface approximation.

    This read-only view retains references to selected surfaces. Subsequent
    parameter edits are revalidated before propagation. Field and geometry
    Torch gradients, including lens f and phase coefficients, dtype, and device
    are retained; material-index and extinction-coefficient gradients
    are unsupported because ``ScalarField`` uses scalar medium metadata.
    Native material caches and Abbe-model coefficient updates are isolated.
    Custom material callbacks must not mutate external/shared state; this view
    is not a transaction around arbitrary user code.
    Aperture edges and an efficiency square root at zero are nondifferentiable.

    Args:
        surfaces: Selected surfaces in sequential order. Prefer ``from_optic``.
        start_surface: Original index of the first selected surface.
        absorption: ``"reject"`` (default) requires exactly lossless materials;
            ``"axial"`` opts into uniform vertex-gap absorption only.
    """

    def __init__(
        self,
        surfaces: tuple[Surface, ...],
        start_surface: int = 1,
        *,
        absorption: Literal["reject", "axial"] = "reject",
    ) -> None:
        if not isinstance(absorption, str) or absorption not in ("reject", "axial"):
            raise ValueError("absorption must be either 'reject' or 'axial'.")
        self._absorption = absorption
        self._surfaces = tuple(surfaces)
        self.start_surface = start_surface
        self.end_surface = start_surface + len(self._surfaces) - 1
        self._backend = be.get_backend()
        self._validate_structure()

    @classmethod
    def from_optic(
        cls,
        optic: Optic,
        start_surface: int = 1,
        end_surface: int | None = None,
        *,
        absorption: Literal["reject", "axial"] = "reject",
    ) -> ScalarOpticalTrain:
        """Select an inclusive range, excluding the object at index zero.

        Args:
            optic: Sequential Optiland optic with a native surface group.
            start_surface: First included surface index. Defaults to 1.
            end_surface: Last included surface index, or the last listed surface.
            absorption: ``"reject"`` (default) or the bounded uniform axial
                absorption approximation ``"axial"``; see the class docstring.

        Returns:
            ScalarOpticalTrain: Validated read-only view of the selected train.

        Raises:
            ValueError: If indices or selected surface physics are unsupported.
            TypeError: If the input is not an Optiland optic.
        """
        from optiland.optic import Optic

        if not isinstance(optic, Optic):
            raise TypeError("optic must be an Optiland Optic.")
        surfaces = tuple(optic.surfaces)
        end_surface = len(surfaces) - 1 if end_surface is None else end_surface
        if (
            not isinstance(start_surface, Integral)
            or isinstance(start_surface, bool)
            or not isinstance(end_surface, Integral)
            or isinstance(end_surface, bool)
            or not 1 <= start_surface <= end_surface < len(surfaces)
        ):
            raise ValueError(
                "require 1 <= start_surface <= end_surface < surface count."
            )
        return cls(
            surfaces[start_surface : end_surface + 1],
            int(start_surface),
            absorption=absorption,
        )

    def _validate_structure(self) -> None:
        if be.get_backend() != self._backend:
            raise RuntimeError(
                "the active backend changed after this train was created"
            )
        if not self._surfaces:
            raise ValueError("the train must contain at least one surface.")
        previous_z = None
        for offset, surface in enumerate(self._surfaces):
            label = f"surface {self.start_surface + offset}"
            if type(surface) not in (Surface, ImageSurface):
                raise ValueError(f"{label}: unsupported surface type.")
            if offset and surface.previous_surface is not self._surfaces[offset - 1]:
                raise ValueError(
                    f"{label}: nonsequential material links are unsupported."
                )
            geometry = surface.geometry
            if type(geometry) not in (Plane, StandardGeometry, EvenAsphere):
                raise ValueError(
                    f"{label}: unsupported geometry {type(geometry).__name__}."
                )
            cs = geometry.cs
            if cs.reference_cs is not None:
                raise ValueError(
                    f"{label}: reference coordinate systems are unsupported."
                )
            for name in ("x", "y", "rx", "ry", "rz"):
                if _real_value(getattr(cs, name), f"{label} {name}") != 0:
                    raise ValueError(
                        f"{label}: tilted/decentered/rotated surfaces are unsupported."
                    )
            z = _real_value(cs.z, f"{label} vertex z")
            if previous_z is not None and z < previous_z:
                raise ValueError(
                    f"{label}: negative vertex gaps/folded trains are unsupported."
                )
            previous_z = z
            if type(geometry) in (StandardGeometry, EvenAsphere):
                radius = _real_value(
                    geometry.radius, f"{label} radius", allow_infinite=True
                )
                if radius == 0:
                    raise ValueError(f"{label}: radius must be nonzero.")
                _real_value(geometry.k, f"{label} conic constant")
            if type(geometry) is EvenAsphere:
                coefficients = geometry.coefficients
                if not isinstance(coefficients, (list, tuple)) and not (
                    isinstance(coefficients, be.ndarray) and coefficients.ndim == 1
                ):
                    raise ValueError(
                        f"{label}: asphere coefficients must be a scalar sequence."
                    )
                for coefficient in coefficients:
                    _real_value(coefficient, f"{label} asphere coefficient")
            if type(surface) is ImageSurface and type(geometry) is not Plane:
                raise ValueError(f"{label}: ImageSurface must be a planar marker.")
            model = surface.interaction_model
            if type(model) not in (
                RefractiveReflectiveModel,
                ThinLensInteractionModel,
                PhaseInteractionModel,
            ):
                raise ValueError(f"{label}: unsupported interaction/phase profile.")
            if model.is_reflective:
                raise ValueError(f"{label}: reflective surfaces are unsupported.")
            if model.coating is not None or model.bsdf is not None:
                raise ValueError(f"{label}: coatings and scattering are unsupported.")
            if type(model) in (ThinLensInteractionModel, PhaseInteractionModel):
                if type(geometry) is not Plane or type(surface) is ImageSurface:
                    raise ValueError(
                        f"{label}: thin-lens/phase interactions require a native "
                        "planar non-image surface."
                    )
                if type(model) is ThinLensInteractionModel:
                    focal_length = _real_value(
                        model.f, f"{label} lens f", allow_infinite=True
                    )
                    if focal_length == 0:
                        raise ValueError(f"{label}: lens f must be nonzero.")
                else:
                    _validate_profile(model.phase_profile, label)
            for material in (surface.material_pre, surface.material_post):
                if (
                    not isinstance(material, BaseMaterial)
                    or type(material.propagation_model) is not HomogeneousPropagation
                ):
                    raise ValueError(
                        f"{label}: only homogeneous materials are supported (no GRIN)."
                    )
            _validate_aperture(surface.aperture, label)

    def propagate(self, field: ScalarField[BEArrayT]) -> ScalarField[BEArrayT]:
        """Apply all screens and outgoing-medium vertex-gap propagation.

        Input finiteness and all selected surfaces, material indices/extinction, gaps,
        aperture/profile parameters, and sampled screen phases are checked
        before any propagation is run. Sag outside an
        aperture is not evaluated (blocked coordinates are replaced by zero).

        Args:
            field: Arbitrary complex field immediately before the start vertex,
                using millimeters and the incident material's refractive index.

        Returns:
            ScalarField: New field immediately after the end vertex on the same
            grid, with the outgoing refractive index.

        Raises:
            ValueError: If the field medium or any selected physics is invalid.
            TypeError: If ``field`` is not a ScalarField.
            RuntimeError: If the active backend changed.
        """
        if not isinstance(field, ScalarField):
            raise TypeError("field must be a ScalarField.")
        field._ensure_active_backend()
        if not bool(be.all(be.isfinite(field.data))):
            raise ValueError("field data must be finite (no NaN or Inf).")
        self._validate_structure()
        x, y = field.coordinates()
        x_grid, y_grid = be.meshgrid(x, y)
        screens = []
        indices = {}
        for offset, surface in enumerate(self._surfaces):
            label = f"surface {self.start_surface + offset}"
            media = []
            for material in (surface.material_pre, surface.material_post):
                if id(material) not in indices:
                    indices[id(material)] = _index(
                        material, field.wavelength * 1000, label, self._absorption
                    )
                media.append(indices[id(material)])
            n_before, extinction_before = media[0]
            n_after, extinction = media[1]
            if offset == 0 and not math.isclose(
                field.refractive_index, n_before, rel_tol=1e-7, abs_tol=1e-12
            ):
                raise ValueError(
                    "field refractive_index does not match the incident material."
                )
            if type(surface) is ImageSurface and (
                n_before != n_after or extinction_before != extinction
            ):
                raise ValueError(
                    f"{label}: ImageSurface cannot change the material index "
                    "or extinction coefficient."
                )
            aperture = surface.aperture
            if aperture is None:
                mask = x_grid == x_grid
            else:
                aperture = copy(aperture)
                for name, value in vars(aperture).items():
                    if isinstance(value, be.ndarray):
                        setattr(aperture, name, _like(value, field.data))
                mask = aperture.contains(x_grid, y_grid)
            geometry = copy(surface.geometry)
            if type(geometry) in (StandardGeometry, EvenAsphere) and math.isinf(
                _real_value(geometry.radius, f"{label} radius", allow_infinite=True)
            ):
                # Native sag is flat or polynomial on a flat conic base. Constant
                # base metadata avoids inf*0 in native Torch radius/conic
                # backward paths without duplicating its sag implementation.
                geometry.radius = math.inf
                geometry.k = 0.0
            elif type(geometry) in (StandardGeometry, EvenAsphere):
                geometry.radius = _like(geometry.radius, field.data)
                geometry.k = _like(geometry.k, field.data)
                if type(geometry) is EvenAsphere:
                    if (
                        _real_value(geometry.radius, f"{label} field-precision radius")
                        == 0
                    ):
                        raise ValueError(f"{label}: radius is zero in field precision.")
                    _real_value(geometry.k, f"{label} field-precision conic")
            if type(geometry) is EvenAsphere:
                geometry.coefficients = [
                    _like(coefficient, field.data)
                    for coefficient in geometry.coefficients
                ]
                for coefficient in geometry.coefficients:
                    _real_value(
                        coefficient, f"{label} field-precision asphere coefficient"
                    )
            x_sample = be.where(mask, x_grid, 0.0)
            y_sample = be.where(mask, y_grid, 0.0)
            sag = geometry.sag(x_sample, y_sample)
            # Some native geometries create arrays in backend-default precision
            # (notably Plane.zeros_like), rather than the field's precision.
            sag = _cast_real_like(sag, field.data)
            if not bool(be.all(be.isfinite(sag))):
                raise ValueError(f"{label}: sag is not finite on the transmitted grid.")
            phase = 2 * be.pi / field.wavelength * (n_before - n_after) * sag
            amplitude = 1.0
            model = surface.interaction_model
            if type(model) is ThinLensInteractionModel:
                phase = phase - 2 * be.pi / field.wavelength * (
                    x_sample**2 + y_sample**2
                ) / (2 * _like(model.f, field.data))
            elif type(model) is PhaseInteractionModel:
                extra_phase, amplitude = _profile_screen(
                    model.phase_profile,
                    x_sample,
                    y_sample,
                    field.wavelength * 1000,
                    field.data,
                )
                phase = phase + extra_phase
            if not bool(be.all(be.isfinite(phase))):
                raise ValueError(f"{label}: screen phase is not finite.")
            screen = be.where(mask, amplitude * be.exp(1j * phase), 0.0)
            gap = None
            if offset < len(self._surfaces) - 1:
                z_next = self._surfaces[offset + 1].geometry.cs.z
                # Subtract in source precision, and retain that precision for
                # the ASM carrier phase even when the sampled field is complex64.
                gap = z_next - surface.geometry.cs.z
                if is_torch_tensor(gap):
                    gap = gap.to(device=field.data.device).reshape(())
                elif isinstance(gap, be.ndarray):
                    gap = gap.reshape(())
                if _real_value(gap, f"{label} outgoing vertex gap") < 0:
                    raise ValueError(
                        f"{label}: outgoing vertex gap must be nonnegative."
                    )
                if self._absorption == "axial" and extinction != 0:
                    # No abs(gap): at zero the forward absorption derivative
                    # remains -2*pi*kappa/wavelength. Material metadata is scalar;
                    # the live gap retains its geometry gradient graph.
                    attenuation_gap = (
                        _phase_precision(gap, field.data)
                        if isinstance(gap, be.ndarray)
                        else float(gap)
                    )
                    exponent = (
                        -2 * be.pi * (extinction * attenuation_gap) / field.wavelength
                    )
                    if isinstance(exponent, Real) and is_torch_tensor(field.data):
                        exponent = field.data.real.new_tensor(exponent)
                    attenuation = _cast_real_like(be.exp(exponent), field.data)
                    if not bool(be.all(be.isfinite(attenuation))):
                        raise ValueError(f"{label}: axial attenuation must be finite.")
                    screen = screen * attenuation
            screens.append((screen, n_after, gap))

        result = field
        grid_metadata = {"center": field.center} if hasattr(field, "center") else {}
        for screen, n_after, gap in screens:
            result = ScalarField(
                result.data * screen,
                dx=field.dx,
                dy=field.dy,
                wavelength=field.wavelength,
                refractive_index=n_after,
                **grid_metadata,
            )
            if gap is not None:
                if _real_value(gap, "vertex gap") != 0:
                    result = result.propagate(gap)
                elif getattr(gap, "requires_grad", False):
                    result = result.propagate(gap, evanescent="decay")
        return result
