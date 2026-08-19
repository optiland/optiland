"""Versioned JSON serialization for NSQ scenes.

Provides :func:`scene_to_dict` and :func:`scene_from_dict` which convert an
:class:`~optiland.nonsequential.scene.NSQScene` to and from a plain
JSON-serializable :class:`dict`.

Schema version
--------------
The top-level key ``"nsq_schema_version"`` is ``1``. NSQ has never been
officially released, so there is exactly one schema and no compatibility or
migration machinery for an earlier one: a file whose ``nsq_schema_version``
does not match the current loader is refused with a generic mismatch error
naming both versions. Since NSQ is still pre-release, the physics and the
schema can both change without notice; a scene built against an older
checkout should be rebuilt from its original construction code (or
converted again via
:func:`~optiland.nonsequential.convert.sequential_to_nonsequential`) against
the current API.

Tensor handling
---------------
All PyTorch tensors are detached and serialized as plain Python floats or
lists before writing.  ``requires_grad`` is **not** persisted.  A scene loaded
from JSON is plain-valued; users must re-wrap parameters in
``torch.tensor(..., requires_grad=True)`` to enable differentiation after
loading.

Coordinate systems
------------------
Only the local (x, y, z, rx, ry, rz) components are serialized.  Nested
``reference_cs`` chains are serialized recursively.

Materials
---------
String catalog names (e.g. ``'N-BK7'``) round-trip as strings.
:class:`~optiland.nonsequential.materials.nsq_material.NSQMaterial` instances
with an underlying optiland material round-trip via the catalog name stored on
the material object.  NSQMaterial vacuum (``optiland_material=None``) is
serialized as ``null``.

Not serialized
--------------
- :class:`~optiland.nonsequential.tracer.SimulationResult` / detector data
- Ray databases
- Mesh geometry file content (only the file path is stored)

Kramer Harrison, 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import os

    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.materials.nsq_material import NSQMaterial
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.sources.base import Spectrum

NSQ_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float:
    """Convert a scalar (float, numpy scalar, or torch Tensor) to Python float.

    Args:
        value: Scalar value to convert.

    Returns:
        Plain Python float.
    """
    try:
        # torch.Tensor
        return float(value.detach().cpu().item())
    except AttributeError:
        return float(value)


def _to_list(value: Any) -> list:
    """Convert an array-like (numpy array or torch Tensor) to a Python list.

    Args:
        value: Array-like to convert.

    Returns:
        Plain Python list of floats.
    """
    try:
        # torch.Tensor
        return value.detach().cpu().tolist()
    except AttributeError:
        return np.asarray(value, dtype=float).tolist()


def _serialize_cs(cs: CoordinateSystem) -> dict:
    """Serialize a :class:`CoordinateSystem` to a JSON-safe dict.

    Recursively serializes any chained ``reference_cs``.

    Args:
        cs: Coordinate system to serialize.

    Returns:
        Dict with keys x, y, z, rx, ry, rz, and optionally reference_cs.
    """
    d: dict[str, Any] = {
        "x": _to_float(cs.x),
        "y": _to_float(cs.y),
        "z": _to_float(cs.z),
        "rx": _to_float(cs.rx),
        "ry": _to_float(cs.ry),
        "rz": _to_float(cs.rz),
        "reference_cs": _serialize_cs(cs.reference_cs) if cs.reference_cs else None,
    }
    return d


def _deserialize_cs(d: dict) -> CoordinateSystem:
    """Reconstruct a :class:`CoordinateSystem` from a serialized dict.

    Args:
        d: Dict previously produced by :func:`_serialize_cs`.

    Returns:
        Reconstructed :class:`CoordinateSystem`.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415

    ref = _deserialize_cs(d["reference_cs"]) if d.get("reference_cs") else None
    return CoordinateSystem(
        x=d.get("x", 0.0),
        y=d.get("y", 0.0),
        z=d.get("z", 0.0),
        rx=d.get("rx", 0.0),
        ry=d.get("ry", 0.0),
        rz=d.get("rz", 0.0),
        reference_cs=ref,
    )


def _serialize_spectrum(spectrum: Spectrum) -> dict:
    """Serialize a :class:`Spectrum` to a JSON-safe dict.

    Args:
        spectrum: Spectrum object with ``wavelengths`` and ``weights`` arrays.

    Returns:
        Dict with keys ``wavelengths`` and ``weights`` as plain float lists.
    """
    return {
        "wavelengths": _to_list(spectrum.wavelengths),
        "weights": _to_list(spectrum.weights),
    }


def _deserialize_spectrum(d: dict) -> Spectrum:
    """Reconstruct a :class:`Spectrum` from a serialized dict.

    Args:
        d: Dict previously produced by :func:`_serialize_spectrum`.

    Returns:
        Reconstructed :class:`Spectrum`.
    """
    from optiland.nonsequential.sources.base import Spectrum  # noqa: PLC0415

    return Spectrum(
        wavelengths=np.array(d["wavelengths"], dtype=np.float64),
        weights=np.array(d["weights"], dtype=np.float64),
    )


def _serialize_material(mat: str | NSQMaterial | None) -> Any:
    """Serialize a material reference to a JSON-safe value.

    - ``None`` or vacuum NSQMaterial -> ``null``
    - string catalog name -> that string
    - NSQMaterial with optiland_material -> ``{"type": "catalog", "name": ...}``

    Args:
        mat: Material to serialize; may be a catalog name string, an
            :class:`~optiland.nonsequential.materials.nsq_material.NSQMaterial`,
            or ``None``.

    Returns:
        JSON-serializable representation.

    Raises:
        ValueError: If the NSQMaterial cannot be round-tripped (no catalog name
            is available on the underlying material).
    """
    if mat is None:
        return None
    if isinstance(mat, str):
        return mat

    # NSQMaterial
    from optiland.nonsequential.materials.nsq_material import (  # noqa: PLC0415
        NSQMaterial,
    )

    if isinstance(mat, NSQMaterial):
        if mat.optiland_material is None:
            return None  # vacuum
        # Try to recover the catalog name from the underlying material
        underlying = mat.optiland_material
        glass_name = getattr(underlying, "name", None) or getattr(
            underlying, "_name", None
        )
        if glass_name is None:
            raise ValueError(
                f"Cannot serialize NSQMaterial: the underlying material "
                f"{underlying!r} does not expose a 'name' attribute. "
                "Only catalog-name materials can be round-tripped."
            )
        return {"type": "catalog", "name": glass_name}

    raise TypeError(f"Unrecognised material type: {type(mat).__name__}")


def _deserialize_material(d: Any) -> str | None:
    """Reconstruct a material from a serialized value.

    Args:
        d: Value produced by :func:`_serialize_material`.

    Returns:
        String catalog name (which ``add_lens`` etc. resolve at build time),
        or ``None`` for vacuum.
    """
    if d is None:
        return None
    if isinstance(d, str):
        return d
    if isinstance(d, dict) and d.get("type") == "catalog":
        return d["name"]  # scene builder resolves via NSQMaterial.from_glass
    raise ValueError(f"Cannot deserialize material: {d!r}")


# ---------------------------------------------------------------------------
# Component serialization
# ---------------------------------------------------------------------------


def _serialize_component(name: str, compound: Any) -> dict:
    """Serialize a named compound component to a JSON-safe dict.

    Supports :class:`~optiland.nonsequential.components.lens.Lens`,
    :class:`~optiland.nonsequential.components.mirror.Mirror`, and
    :class:`~optiland.nonsequential.components.doublet.Doublet` by reading
    the ``_config`` and ``_cs`` attributes stored on every compound.

    Args:
        name: Registry name of the component.
        compound: Compound component object.

    Returns:
        Dict describing the component type and configuration.

    Raises:
        TypeError: If the component type cannot be serialized.
    """
    from optiland.nonsequential.components.doublet import Doublet  # noqa: PLC0415
    from optiland.nonsequential.components.lens import Lens  # noqa: PLC0415
    from optiland.nonsequential.components.mirror import Mirror  # noqa: PLC0415

    cs = compound._cs
    config = compound._config

    if isinstance(compound, Lens):
        return {
            "type": "lens",
            "name": name,
            "cs": _serialize_cs(cs),
            "config": {
                "r1": _to_float(config.r1),
                "r2": _to_float(config.r2),
                "thickness": _to_float(config.thickness),
                "material": _serialize_material(config.material),
                "front_aperture_radius": _to_float(config.front_aperture_radius),
                "back_aperture_radius": (
                    _to_float(config.back_aperture_radius)
                    if config.back_aperture_radius is not None
                    else None
                ),
                "conic1": _to_float(config.conic1),
                "conic2": _to_float(config.conic2),
            },
        }

    if isinstance(compound, Mirror):
        if not isinstance(config.reflectance, int | float) and not hasattr(
            config.reflectance, "numpy"
        ):
            raise TypeError(
                f"Cannot serialize mirror '{name}': reflectance is "
                f"{type(config.reflectance).__name__}, not a constant. "
                "Only a scalar reflectance round-trips through JSON "
                "serialization; a callable or coating reflectance must be "
                "re-attached after loading."
            )
        return {
            "type": "mirror",
            "name": name,
            "cs": _serialize_cs(cs),
            "config": {
                "radius": _to_float(config.radius),
                "reflectance": _to_float(config.reflectance),
                "conic": _to_float(config.conic),
                "aperture_radius": _to_float(config.aperture_radius),
            },
        }

    if isinstance(compound, Doublet):
        return {
            "type": "doublet",
            "name": name,
            "cs": _serialize_cs(cs),
            "config": {
                "r1": _to_float(config.r1),
                "r2": _to_float(config.r2),
                "r3": _to_float(config.r3),
                "thickness1": _to_float(config.thickness1),
                "thickness2": _to_float(config.thickness2),
                "material1": _serialize_material(config.material1),
                "material2": _serialize_material(config.material2),
                "aperture_radius": _to_float(config.aperture_radius),
                "conic1": _to_float(config.conic1),
                "conic2": _to_float(config.conic2),
                "conic3": _to_float(config.conic3),
            },
        }

    raise TypeError(
        f"Cannot serialize component '{name}' of type "
        f"'{type(compound).__name__}'. Only Lens, Mirror, and Doublet are "
        "supported for round-trip serialization."
    )


def _deserialize_component(d: dict, scene: NSQScene) -> None:
    """Reconstruct a compound component and add it to the scene.

    Args:
        d: Dict produced by :func:`_serialize_component`.
        scene: Target :class:`NSQScene` to populate.

    Raises:
        ValueError: If the component type is unknown.
    """
    from optiland.nonsequential.components.configs import (  # noqa: PLC0415
        DoubletConfig,
        LensConfig,
        MirrorConfig,
    )

    ctype = d["type"]
    name = d["name"]
    cs = _deserialize_cs(d["cs"])
    cfg_d = d["config"]

    if ctype == "lens":
        config = LensConfig(
            r1=cfg_d["r1"],
            r2=cfg_d["r2"],
            thickness=cfg_d["thickness"],
            material=_deserialize_material(cfg_d["material"]),
            front_aperture_radius=cfg_d["front_aperture_radius"],
            back_aperture_radius=cfg_d.get("back_aperture_radius"),
            conic1=cfg_d.get("conic1", 0.0),
            conic2=cfg_d.get("conic2", 0.0),
        )
        scene.add_lens(name, cs, config)

    elif ctype == "mirror":
        config = MirrorConfig(
            radius=cfg_d["radius"],
            reflectance=cfg_d["reflectance"],
            conic=cfg_d.get("conic", 0.0),
            aperture_radius=cfg_d["aperture_radius"],
        )
        scene.add_mirror(name, cs, config)

    elif ctype == "doublet":
        config = DoubletConfig(
            r1=cfg_d["r1"],
            r2=cfg_d["r2"],
            r3=cfg_d["r3"],
            thickness1=cfg_d["thickness1"],
            thickness2=cfg_d["thickness2"],
            material1=_deserialize_material(cfg_d["material1"]),
            material2=_deserialize_material(cfg_d["material2"]),
            aperture_radius=cfg_d["aperture_radius"],
            conic1=cfg_d.get("conic1", 0.0),
            conic2=cfg_d.get("conic2", 0.0),
            conic3=cfg_d.get("conic3", 0.0),
        )
        scene.add_doublet(name, cs, config)

    else:
        raise ValueError(
            f"Unknown component type '{ctype}' in NSQ JSON. "
            "Expected 'lens', 'mirror', or 'doublet'."
        )


# ---------------------------------------------------------------------------
# Source serialization
# ---------------------------------------------------------------------------


def _serialize_source(name: str, source: Any) -> dict:
    """Serialize a named source to a JSON-safe dict.

    Args:
        name: Registry name of the source.
        source: Source object (PointSource, CollimatedSource, or ExtendedSource).

    Returns:
        Dict describing the source type and parameters.

    Raises:
        TypeError: If the source type is not supported.
    """
    from optiland.nonsequential.sources.collimated import (  # noqa: PLC0415
        CollimatedSource,
    )
    from optiland.nonsequential.sources.extended import (  # noqa: PLC0415
        ExtendedSource,
    )
    from optiland.nonsequential.sources.point import PointSource  # noqa: PLC0415

    cs_d = _serialize_cs(source.cs)
    spectrum_d = _serialize_spectrum(source.spectrum)
    total_flux = _to_float(source.total_flux)
    medium = _serialize_material(getattr(source, "medium", None))

    if isinstance(source, PointSource):
        return {
            "type": "point",
            "name": name,
            "cs": cs_d,
            "spectrum": spectrum_d,
            "total_flux": total_flux,
            "half_angle_deg": _to_float(source.half_angle_deg),
            "medium": medium,
        }

    if isinstance(source, CollimatedSource):
        return {
            "type": "collimated",
            "name": name,
            "cs": cs_d,
            "spectrum": spectrum_d,
            "total_flux": total_flux,
            "aperture_radius": _to_float(source.aperture_radius),
            "profile": source.profile,
            "gaussian_sigma": _to_float(source.gaussian_sigma),
            "medium": medium,
        }

    if isinstance(source, ExtendedSource):
        return {
            "type": "extended",
            "name": name,
            "cs": cs_d,
            "spectrum": spectrum_d,
            "total_flux": total_flux,
            "width": _to_float(source.width),
            "height": _to_float(source.height),
            "aperture_radius": (
                _to_float(source.aperture_radius)
                if source.aperture_radius is not None
                else None
            ),
            "half_angle_deg": _to_float(source.half_angle_deg),
            "medium": medium,
        }

    raise TypeError(
        f"Cannot serialize source '{name}' of type '{type(source).__name__}'. "
        "Only PointSource, CollimatedSource, and ExtendedSource are supported."
    )


def _deserialize_source(d: dict, scene: NSQScene) -> None:
    """Reconstruct a source and add it to the scene.

    Args:
        d: Dict produced by :func:`_serialize_source`.
        scene: Target :class:`NSQScene` to populate.

    Raises:
        ValueError: If the source type is unknown.
    """
    from optiland.nonsequential.sources.configs import (  # noqa: PLC0415
        CollimatedSourceConfig,
        ExtendedSourceConfig,
        PointSourceConfig,
    )

    stype = d["type"]
    name = d["name"]
    cs = _deserialize_cs(d["cs"])
    spectrum = _deserialize_spectrum(d["spectrum"])
    total_flux = d["total_flux"]
    medium = _deserialize_material(d.get("medium"))

    if stype == "point":
        config = PointSourceConfig(
            spectrum=spectrum,
            total_flux=total_flux,
            half_angle_deg=d.get("half_angle_deg", 90.0),
            medium=medium,
        )
        scene.add_source(name, cs, config)

    elif stype == "collimated":
        config = CollimatedSourceConfig(
            spectrum=spectrum,
            total_flux=total_flux,
            aperture_radius=d.get("aperture_radius", 1.0),
            profile=d.get("profile", "tophat"),
            gaussian_sigma=d.get("gaussian_sigma"),
            medium=medium,
        )
        scene.add_source(name, cs, config)

    elif stype == "extended":
        config = ExtendedSourceConfig(
            spectrum=spectrum,
            total_flux=total_flux,
            width=d.get("width", 1.0),
            height=d.get("height", 1.0),
            aperture_radius=d.get("aperture_radius"),
            half_angle_deg=d.get("half_angle_deg", 90.0),
            medium=medium,
        )
        scene.add_source(name, cs, config)

    else:
        raise ValueError(
            f"Unknown source type '{stype}' in NSQ JSON. "
            "Expected 'point', 'collimated', or 'extended'."
        )


# ---------------------------------------------------------------------------
# Detector serialization
# ---------------------------------------------------------------------------


def _serialize_detector(name: str, detector: Any) -> dict:
    """Serialize a named detector to a JSON-safe dict.

    Args:
        name: Registry name of the detector.
        detector: Detector object.

    Returns:
        Dict describing the detector type and parameters.

    Raises:
        TypeError: If the detector type is not supported.
    """
    from optiland.nonsequential.detectors.far_field import (  # noqa: PLC0415
        FarFieldDetector,
    )
    from optiland.nonsequential.detectors.irradiance import (  # noqa: PLC0415
        IrradianceDetector,
    )
    from optiland.nonsequential.detectors.ray_database import (  # noqa: PLC0415
        RayDatabaseDetector,
    )
    from optiland.nonsequential.detectors.spectral import (  # noqa: PLC0415
        SpectralDetector,
    )

    cs_d = _serialize_cs(detector.cs)

    if isinstance(detector, IrradianceDetector):
        return {
            "type": "irradiance",
            "name": name,
            "cs": cs_d,
            "width": _to_float(detector.width),
            "height": _to_float(detector.height),
            "num_pixels_x": int(detector.num_pixels_x),
            "num_pixels_y": int(detector.num_pixels_y),
            "splat": detector.splat,
            "splat_sigma": _to_float(detector.splat_sigma),
            "absorb": bool(detector.absorb),
        }

    if isinstance(detector, SpectralDetector):
        wl_bins = np.asarray(detector.wavelength_bins, dtype=float)
        return {
            "type": "spectral",
            "name": name,
            "cs": cs_d,
            "width": _to_float(detector.width),
            "height": _to_float(detector.height),
            "num_pixels_x": int(detector.num_pixels_x),
            "num_pixels_y": int(detector.num_pixels_y),
            "wl_min": float(wl_bins[0]),
            "wl_max": float(wl_bins[-1]),
            "num_bins": int(len(wl_bins) - 1),
            "splat": detector.splat,
            "splat_sigma": _to_float(detector.splat_sigma),
            "absorb": bool(detector.absorb),
        }

    if isinstance(detector, FarFieldDetector):
        return {
            "type": "far_field",
            "name": name,
            "cs": cs_d,
            "num_bins_theta": int(detector.num_bins_theta),
            "num_bins_phi": int(detector.num_bins_phi),
            "absorb": bool(detector.absorb),
        }

    if isinstance(detector, RayDatabaseDetector):
        # RayDatabaseDetector holds a geometry object; extract width/height.
        geom = detector.geometry
        return {
            "type": "ray_database",
            "name": name,
            "cs": cs_d,
            "width": float(getattr(geom, "width", 10.0)),
            "height": float(getattr(geom, "height", 10.0)),
            "absorb": bool(detector.absorb),
        }

    raise TypeError(
        f"Cannot serialize detector '{name}' of type '{type(detector).__name__}'. "
        "Only IrradianceDetector, SpectralDetector, FarFieldDetector, and "
        "RayDatabaseDetector are supported."
    )


def _deserialize_detector(d: dict, scene: NSQScene) -> None:
    """Reconstruct a detector and add it to the scene.

    Args:
        d: Dict produced by :func:`_serialize_detector`.
        scene: Target :class:`NSQScene` to populate.

    Raises:
        ValueError: If the detector type is unknown.
    """
    from optiland.nonsequential.detectors.configs import (  # noqa: PLC0415
        FarFieldDetectorConfig,
        IrradianceDetectorConfig,
        RayDatabaseConfig,
        SpectralDetectorConfig,
    )

    dtype = d["type"]
    name = d["name"]
    cs = _deserialize_cs(d["cs"])

    if dtype == "irradiance":
        config = IrradianceDetectorConfig(
            width=d["width"],
            height=d["height"],
            num_pixels_x=d.get("num_pixels_x", 256),
            num_pixels_y=d.get("num_pixels_y", 256),
            splat=d.get("splat", "bilinear"),
            splat_sigma=d.get("splat_sigma", 0.5),
            absorb=d.get("absorb", True),
        )
        scene.add_detector(name, cs, config)

    elif dtype == "spectral":
        config = SpectralDetectorConfig(
            width=d["width"],
            height=d["height"],
            num_pixels_x=d.get("num_pixels_x", 256),
            num_pixels_y=d.get("num_pixels_y", 256),
            wl_min=d.get("wl_min", 0.4),
            wl_max=d.get("wl_max", 0.7),
            num_bins=d.get("num_bins", 100),
            splat=d.get("splat", "bilinear"),
            splat_sigma=d.get("splat_sigma", 0.5),
            absorb=d.get("absorb", True),
        )
        scene.add_detector(name, cs, config)

    elif dtype == "far_field":
        config = FarFieldDetectorConfig(
            num_theta=d.get("num_bins_theta", 90),
            num_phi=d.get("num_bins_phi", 360),
            absorb=d.get("absorb", True),
        )
        scene.add_detector(name, cs, config)

    elif dtype == "ray_database":
        config = RayDatabaseConfig(
            width=d["width"],
            height=d["height"],
            absorb=d.get("absorb", True),
        )
        scene.add_detector(name, cs, config)

    else:
        raise ValueError(
            f"Unknown detector type '{dtype}' in NSQ JSON. "
            "Expected 'irradiance', 'spectral', 'far_field', or 'ray_database'."
        )


# ---------------------------------------------------------------------------
# Top-level scene serialization
# ---------------------------------------------------------------------------


def scene_to_dict(scene: NSQScene) -> dict:
    """Convert an :class:`NSQScene` to a JSON-serializable dict.

    The returned dict includes a top-level ``"nsq_schema_version"`` key.
    Simulation results and detector data are **not** included.

    Args:
        scene: The scene to serialize.

    Returns:
        JSON-serializable dict representing the scene structure.

    Raises:
        TypeError: If any component, source, or detector type is not supported.
        ValueError: If any material cannot be round-tripped.
    """
    components = []
    for name, compound in scene.component_registry._registry.items():
        components.append(_serialize_component(name, compound))

    sources = []
    for name, source in scene.source_registry._registry.items():
        sources.append(_serialize_source(name, source))

    detectors = []
    for name, detector in scene.detector_registry._registry.items():
        detectors.append(_serialize_detector(name, detector))

    return {
        "nsq_schema_version": NSQ_SCHEMA_VERSION,
        "components": components,
        "sources": sources,
        "detectors": detectors,
    }


def scene_from_dict(d: dict) -> NSQScene:
    """Reconstruct an :class:`NSQScene` from a serialized dict.

    Validates ``"nsq_schema_version"`` before loading.  A loaded scene is
    plain-valued; all parameters are plain Python floats, not tensors.

    Args:
        d: Dict previously produced by :func:`scene_to_dict` or read from a
            JSON file written by :meth:`NSQScene.to_json`.

    Returns:
        Reconstructed :class:`NSQScene`.

    Raises:
        ValueError: If ``"nsq_schema_version"`` is missing or does not match
            :data:`NSQ_SCHEMA_VERSION`. NSQ has never been officially
            released, so there is no compatibility mode or auto-migration
            for an older schema -- a mismatched file must be rebuilt against
            the current API.
        ValueError: If any component/source/detector type is unknown.
    """
    from optiland.nonsequential.scene import NSQScene  # noqa: PLC0415

    version = d.get("nsq_schema_version")
    if version is None:
        raise ValueError(
            "The JSON file is missing the required 'nsq_schema_version' key. "
            "This file may not be an Optiland NSQ scene file."
        )
    if version != NSQ_SCHEMA_VERSION:
        raise ValueError(
            f"NSQ schema version mismatch: the file uses version {version!r}, "
            f"but this version of Optiland only supports version "
            f"{NSQ_SCHEMA_VERSION}. "
            "Please update Optiland or re-export the scene."
        )

    scene = NSQScene()

    for comp_d in d.get("components", []):
        _deserialize_component(comp_d, scene)

    for src_d in d.get("sources", []):
        _deserialize_source(src_d, scene)

    for det_d in d.get("detectors", []):
        _deserialize_detector(det_d, scene)

    return scene


def scene_to_json(scene: NSQScene, path: str | os.PathLike) -> None:
    """Serialize an :class:`NSQScene` to a versioned JSON file.

    This is the low-level implementation called by
    :meth:`NSQScene.to_json`.

    Args:
        scene: The scene to serialize.
        path: Destination file path (created or overwritten).
    """
    import json  # noqa: PLC0415

    d = scene_to_dict(scene)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def scene_from_json(path: str | os.PathLike) -> NSQScene:
    """Load an :class:`NSQScene` from a versioned JSON file.

    This is the low-level implementation called by
    :meth:`NSQScene.from_json`.

    Args:
        path: Path to the JSON file previously written by
            :func:`scene_to_json` or :meth:`NSQScene.to_json`.

    Returns:
        Reconstructed :class:`NSQScene`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the schema version is missing or does not match.
    """
    import json  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"NSQ scene file not found: {path}. "
            "Check the path and ensure the file has not been moved or deleted."
        )
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return scene_from_dict(d)
