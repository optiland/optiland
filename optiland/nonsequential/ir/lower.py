"""``lower()`` -- NSQScene -> SceneIR.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.ir.bsdf_ir import BsdfIR
from optiland.nonsequential.ir.medium_ir import MediumIR
from optiland.nonsequential.ir.scene_ir import (
    EmitterIR,
    PrimitiveIR,
    RngContract,
    SamplingPolicy,
    SceneIR,
    SensorIR,
)

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.scene import NSQScene


def _to_world_matrix(cs: CoordinateSystem) -> np.ndarray:
    """Build a (4, 4) homogeneous local -> global transform from a CS.

    Args:
        cs: Coordinate system.

    Returns:
        (4, 4) float64 array. Position (translation) is not currently a
        differentiable NSQ parameter, so this is a detached numpy array --
        consistent with :func:`~optiland.nonsequential.components.base._get_transform`,
        which every existing intersection routine already relies on.
    """
    translation, rotation = _get_transform(cs)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rotation
    m[:3, 3] = translation
    return m


class _MediumRegistry:
    """Deduplicates materials into a flat, id-addressable ``MediumIR`` list."""

    def __init__(self) -> None:
        self._index: dict[tuple, int] = {}
        self.media: list[MediumIR] = []

    def get_id(self, material: NSQMaterial | None, *, strict: bool = True) -> int:
        """Return the ``MediumIR`` id for ``material``, adding it if new.

        Args:
            material: An ``NSQMaterial`` (or ``None``, treated as vacuum).
            strict: If True (the default -- used when the IR must be
                losslessly serializable, e.g. for JSON export or the
                translatability checklist tests), a material that is
                neither vacuum nor catalog-backed raises. If False (used by
                the backends' internal ``lower(scene)`` call before every
                trace, where the IR only needs to drive dispatch, not
                round-trip through JSON), such a material is still assigned
                a distinct id -- keyed by Python object identity, tagged
                ``n_model={"kind": "opaque"}`` -- rather than blocking the
                trace. Its ``n(wavelength)`` is never evaluated from the IR
                (the interpreter reads it from the live component instead;
                see :mod:`optiland.nonsequential.ir.interpreter`), so this
                only affects introspection/serialization, never physics.

        Returns:
            Index into ``self.media``.

        Raises:
            ValueError: If ``strict`` and ``material`` is neither vacuum nor
                catalog-backed -- mirrors the same limitation in
                :func:`optiland.nonsequential.serialization._serialize_material`.
        """
        underlying = None if material is None else material.optiland_material
        if underlying is None:
            key: tuple = ("vacuum",)
        else:
            glass_name = getattr(underlying, "name", None) or getattr(
                underlying, "_name", None
            )
            if glass_name is None:
                if strict:
                    raise ValueError(
                        f"Cannot lower material {underlying!r} to the scene IR: "
                        "only vacuum and catalog-backed materials (exposing a "
                        "'name' attribute) round-trip losslessly. This mirrors "
                        "NSQScene.to_json()'s existing material limitation."
                    )
                key = ("opaque", id(underlying))
            else:
                key = ("catalog", glass_name)

        if key not in self._index:
            idx = len(self.media)
            if key[0] == "vacuum":
                medium = MediumIR(
                    id=idx,
                    name="vacuum",
                    n_model={"kind": "constant", "n": 1.0},
                    k_model={"kind": "constant", "k": 0.0},
                )
            elif key[0] == "opaque":
                medium = MediumIR(
                    id=idx,
                    name=f"opaque_{idx}",
                    n_model={"kind": "opaque"},
                    k_model={"kind": "opaque"},
                )
            else:
                medium = MediumIR(
                    id=idx,
                    name=key[1],
                    n_model={"kind": "catalog", "name": key[1]},
                    k_model={"kind": "catalog", "name": key[1]},
                )
            self.media.append(medium)
            self._index[key] = idx
        return self._index[key]


def _lower_geometry(geometry: object) -> tuple[str, dict[str, Any]]:
    """Map a live ``ComponentGeometry`` to an (IR kind, params) pair.

    Args:
        geometry: Any geometry object exposed by a scene surface.

    Returns:
        ``(kind, params)`` per
        :data:`~optiland.nonsequential.ir.scene_ir.PrimitiveKind`.

    Raises:
        TypeError: If no lowering is registered for this geometry type.
    """
    from optiland.nonsequential.components.geometry.analytic.annulus import (  # noqa: PLC0415
        AnnularPlaneGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
        ConicGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.frustum import (  # noqa: PLC0415
        CylindricalFrustumGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
        FinitePlaneGeometry,
        PlaneGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.sphere import (  # noqa: PLC0415
        SphereGeometry,
    )
    from optiland.nonsequential.components.geometry.mesh.mesh_geometry import (  # noqa: PLC0415
        MeshGeometry,
    )

    # ConicGeometry check also covers ParaboloidGeometry (a subclass that
    # only fixes conic=-1 at construction time).
    if isinstance(geometry, ConicGeometry):
        return "conic", {
            "radius": geometry.radius,
            "conic": geometry.conic,
            "aperture_radius": geometry.aperture_radius,
        }
    if isinstance(geometry, FinitePlaneGeometry):
        return "plane", {
            "width": geometry.width,
            "height": geometry.height,
            "aperture_radius": geometry.aperture_radius,
        }
    if isinstance(geometry, PlaneGeometry):
        # Infinite plane: no width/height/aperture limit.
        return "plane", {"width": None, "height": None, "aperture_radius": None}
    if isinstance(geometry, AnnularPlaneGeometry):
        return "annulus", {
            "inner_radius": geometry.inner_radius,
            "outer_radius": geometry.outer_radius,
            "z_offset": geometry.z_offset,
        }
    if isinstance(geometry, CylindricalFrustumGeometry):
        return "frustum", {
            "r_front": geometry.r_front,
            "r_back": geometry.r_back,
            "z_front": geometry.z_front,
            "z_back": geometry.z_back,
        }
    if isinstance(geometry, SphereGeometry):
        return "sphere", {
            "radius": geometry.radius,
            "aperture_radius": geometry.aperture_radius,
        }
    if isinstance(geometry, MeshGeometry):
        mesh = geometry.mesh
        return "mesh", {
            "vertices": np.asarray(mesh.vertices, dtype=np.float64).tolist(),
            "faces": np.asarray(mesh.faces, dtype=np.int64).tolist(),
        }
    raise TypeError(
        f"No scene-IR lowering registered for geometry type {type(geometry).__name__}."
    )


def _lower_bsdf(bsdf: object | None) -> BsdfIR:
    """Map a live ``BaseBSDF`` (or ``None``) to a :class:`BsdfIR`.

    Args:
        bsdf: A BSDF instance, or ``None`` for no attached scatter lobe.

    Returns:
        The corresponding :class:`BsdfIR`.

    Raises:
        TypeError: If no lowering is registered for this BSDF type.
    """
    from optiland.nonsequential.bsdf.harvey_shack import (
        HarveyShackBSDF,  # noqa: PLC0415
    )
    from optiland.nonsequential.bsdf.lambertian import LambertianBSDF  # noqa: PLC0415
    from optiland.nonsequential.bsdf.specular import SpecularBRDF  # noqa: PLC0415
    from optiland.nonsequential.bsdf.tabulated import TabulatedBSDF  # noqa: PLC0415

    if bsdf is None:
        return BsdfIR(kind="none")
    if isinstance(bsdf, LambertianBSDF):
        return BsdfIR(
            kind="lambertian",
            params={
                "reflectance_value": bsdf.reflectance_value,
                "transmissive_fraction": bsdf.transmissive_fraction,
            },
        )
    if isinstance(bsdf, HarveyShackBSDF):
        return BsdfIR(
            kind="harvey_shack",
            params={
                "b0": bsdf.b0,
                "l0": bsdf.l0,
                "s": bsdf.s,
                "transmissive_fraction": bsdf.transmissive_fraction,
            },
        )
    if isinstance(bsdf, TabulatedBSDF):
        return BsdfIR(
            kind="tabulated",
            params={
                "path": str(bsdf.path),
                "transmissive_fraction": bsdf.transmissive_fraction,
            },
        )
    if isinstance(bsdf, SpecularBRDF):
        return BsdfIR(kind="specular")
    raise TypeError(
        f"No scene-IR lowering registered for BSDF type {type(bsdf).__name__}."
    )


def _component_kind(component: BaseComponent) -> str:
    """Return the :data:`ComponentKind` for a live component instance.

    Args:
        component: A scene surface.

    Returns:
        One of ``"refractive"``, ``"reflective"``, ``"absorbing"``.

    Raises:
        TypeError: If the component type is not one of the three known
            interaction kinds.
    """
    from optiland.nonsequential.components.absorbing import (
        AbsorbingComponent,  # noqa: PLC0415
    )
    from optiland.nonsequential.components.reflective import (
        ReflectiveComponent,  # noqa: PLC0415
    )
    from optiland.nonsequential.components.refractive import (
        RefractiveComponent,  # noqa: PLC0415
    )

    if isinstance(component, RefractiveComponent):
        return "refractive"
    if isinstance(component, ReflectiveComponent):
        return "reflective"
    if isinstance(component, AbsorbingComponent):
        return "absorbing"
    raise TypeError(
        f"No scene-IR lowering registered for component type "
        f"{type(component).__name__}."
    )


def _lower_spectrum(spectrum: object) -> dict[str, Any]:
    """Lower a ``Spectrum`` to a plain dict.

    Args:
        spectrum: A ``Spectrum`` instance.

    Returns:
        ``{"wavelengths": [...], "weights": [...]}``.
    """
    return {
        "wavelengths": np.asarray(spectrum.wavelengths, dtype=np.float64).tolist(),
        "weights": np.asarray(spectrum.weights, dtype=np.float64).tolist(),
    }


def _lower_source(
    idx: int, source: object, media: _MediumRegistry, *, strict: bool = True
) -> EmitterIR:
    """Map a live source to an :class:`EmitterIR`.

    Args:
        idx: Index to assign in ``SceneIR.emitters``.
        source: A ``BaseNSQSource`` instance.
        media: Shared medium registry to resolve/add the source's medium.
        strict: Forwarded to :meth:`_MediumRegistry.get_id`.

    Returns:
        The corresponding :class:`EmitterIR`.

    Raises:
        TypeError: If no lowering is registered for this source type.
    """
    from optiland.nonsequential.sources.collimated import (
        CollimatedSource,  # noqa: PLC0415
    )
    from optiland.nonsequential.sources.extended import ExtendedSource  # noqa: PLC0415
    from optiland.nonsequential.sources.point import PointSource  # noqa: PLC0415

    common: dict[str, Any] = {
        "total_flux": source.total_flux,
        "spectrum": _lower_spectrum(source.spectrum),
    }
    if isinstance(source, PointSource):
        kind = "point"
        params = {**common, "half_angle_deg": source.half_angle_deg}
    elif isinstance(source, CollimatedSource):
        kind = "collimated"
        params = {
            **common,
            "aperture_radius": source.aperture_radius,
            "profile": source.profile,
            "gaussian_sigma": source.gaussian_sigma,
        }
    elif isinstance(source, ExtendedSource):
        kind = "extended"
        params = {
            **common,
            "width": source.width,
            "height": source.height,
            "aperture_radius": source.aperture_radius,
            "half_angle_deg": source.half_angle_deg,
        }
    else:
        raise TypeError(
            f"No scene-IR lowering registered for source type {type(source).__name__}."
        )

    medium = getattr(source, "medium", None)
    medium_id = media.get_id(medium, strict=strict) if medium is not None else None
    return EmitterIR(
        id=idx,
        kind=kind,
        to_world=_to_world_matrix(source.cs),
        params=params,
        medium_id=medium_id,
        name=getattr(source, "name", "") or f"source_{idx}",
    )


def _lower_detector(idx: int, detector: object) -> SensorIR:
    """Map a live detector to a :class:`SensorIR`.

    Args:
        idx: Index to assign in ``SceneIR.sensors``.
        detector: A ``BaseDetector`` instance.

    Returns:
        The corresponding :class:`SensorIR`.

    Raises:
        TypeError: If no lowering is registered for this detector type.
    """
    from optiland.nonsequential.detectors.far_field import (
        FarFieldDetector,  # noqa: PLC0415
    )
    from optiland.nonsequential.detectors.irradiance import (
        IrradianceDetector,  # noqa: PLC0415
    )
    from optiland.nonsequential.detectors.ray_database import (  # noqa: PLC0415
        RayDatabaseDetector,
    )
    from optiland.nonsequential.detectors.spectral import (
        SpectralDetector,  # noqa: PLC0415
    )

    if isinstance(detector, SpectralDetector):
        kind = "spectral"
        params = {
            "width": detector.width,
            "height": detector.height,
            "num_pixels_x": detector.num_pixels_x,
            "num_pixels_y": detector.num_pixels_y,
            "wavelength_bins": np.asarray(
                detector.wavelength_bins, dtype=np.float64
            ).tolist(),
            "splat": detector.splat,
            "splat_sigma": detector.splat_sigma,
        }
    elif isinstance(detector, IrradianceDetector):
        kind = "irradiance"
        params = {
            "width": detector.width,
            "height": detector.height,
            "num_pixels_x": detector.num_pixels_x,
            "num_pixels_y": detector.num_pixels_y,
            "splat": detector.splat,
            "splat_sigma": detector.splat_sigma,
        }
    elif isinstance(detector, FarFieldDetector):
        kind = "far_field"
        params = {
            "num_bins_theta": detector.num_bins_theta,
            "num_bins_phi": detector.num_bins_phi,
        }
    elif isinstance(detector, RayDatabaseDetector):
        kind = "ray_database"
        geom = detector.geometry
        params = {
            "width": getattr(geom, "width", 10.0),
            "height": getattr(geom, "height", 10.0),
        }
    else:
        raise TypeError(
            f"No scene-IR lowering registered for detector type "
            f"{type(detector).__name__}."
        )

    return SensorIR(
        id=idx,
        kind=kind,
        to_world=_to_world_matrix(detector.cs),
        params=params,
        primitive_id=None,  # still not literally in SceneIR.primitives
        absorb=detector.absorb,
        name=getattr(detector, "name", "") or f"detector_{idx}",
    )


def lower(scene: NSQScene, *, strict: bool = True) -> SceneIR:
    """Lower a live :class:`NSQScene` to a data-only :class:`SceneIR`.

    Args:
        scene: The scene to lower. Not mutated.
        strict: Forwarded to :meth:`_MediumRegistry.get_id` for every
            material referenced by a surface or a source. Defaults to True
            (every ``MediumIR`` must be losslessly identifiable -- vacuum or
            catalog-backed), which is what JSON export and the
            translatability checklist require. The backends pass
            ``strict=False`` for the ``lower()`` call they make before every
            ``trace()``: that IR only needs to drive dispatch (D-1
            sidedness is resolved from geometry, not from a medium id --
            see :mod:`optiland.nonsequential.components.refractive`), so a
            custom, non-catalog material must not block tracing the way it
            would block serialization.

    Returns:
        The scene, described as plain data.

    Raises:
        TypeError: If the scene contains a component, BSDF, source, or
            detector type this revamp does not yet know how to lower.
        ValueError: If ``strict`` and a material cannot be losslessly
            identified (mirrors :meth:`NSQScene.to_json`'s existing
            limitation).
    """
    media = _MediumRegistry()

    primitives = []
    for i, component in enumerate(scene.surfaces):
        kind, params = _lower_geometry(component.geometry)
        primitives.append(
            PrimitiveIR(
                id=i,
                kind=kind,
                to_world=_to_world_matrix(component.cs),
                params=params,
                bsdf=_lower_bsdf(component.bsdf),
                # Descriptive metadata only -- not consumed by the
                # interpreter. The actual D-1 fix (geometric sidedness) is
                # in each geometry's n_geom, not in these ids; see
                # RefractiveComponent.interact().
                exterior_medium_id=media.get_id(
                    component.material_front, strict=strict
                ),
                interior_medium_id=media.get_id(component.material_back, strict=strict),
                volume_id=None,
                component_kind=_component_kind(component),
                scatter_fraction=component.scatter_fraction,
                name=component.name or f"component_{i}",
            )
        )

    emitters = [
        _lower_source(i, source, media, strict=strict)
        for i, source in enumerate(scene.sources)
    ]
    sensors = [
        _lower_detector(i, detector) for i, detector in enumerate(scene.detectors)
    ]

    return SceneIR(
        primitives=tuple(primitives),
        volumes=(),
        media=tuple(media.media),
        emitters=tuple(emitters),
        sensors=tuple(sensors),
        rng=RngContract(),
        sampling=getattr(scene, "sampling_policy", None) or SamplingPolicy(),
    )
