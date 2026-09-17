"""Drift detection for the Python sources the fused trace kernel mirrors.

The fused kernel reproduces, expression by expression, a set of Python
functions ("mirror, never improve", plan 0.2.1).  Nothing in the language
stops an upstream merge from changing one of them, so every mirrored function
is fingerprinted here and the fingerprint is re-checked at runtime.

Two classes of row (plan 3.7):

``MIRRORED``
    Physics the MSL reproduces.  Hashed from the live source and checked on
    every process that uses the fused path.  A mismatch emits one
    ``FusedTraceDriftWarning`` and refuses every candidate with
    ``FusedTraceSkip.MIRROR_DRIFT`` until the row is re-verified with
    ``--update --verified``.

``CONTRACT``
    Host-consumed helpers whose *values* the adapter tests compare.  Not
    hashed (``sha256 is None``) so they cannot cause alarm fatigue; each row
    names the value test that guards it instead.

The hash is ``sha256(ast.dump(parsed_source_without_docstring))``.  ``ast.dump``
omits line and column attributes by default, so reformatting, comment edits
and docstring edits do not flap; any change to an expression does.
Non-callable constants are hashed from ``repr(value)``.

A harness that *wraps* a mirrored function to count calls replaces the live
class attribute without changing the physics -- every census of plan 1.3 / 8.3
wraps ``SurfaceGroup.trace``, and one gate test wraps ``Surface.trace`` too.
``check_all()`` therefore follows ``__wrapped__`` and closure cells from the
live attribute to the function it wraps (:func:`live_digests`) and accepts the
row when the wrapped original still matches.  A replacement that does not hold
the original -- round 0's ``exec``-compiled divergence injections, an upstream
edit of the source file -- still drifts, and so does an upstream edit *under* a
wrapper.

CLI::

    python -m optiland.backend.torch_backend.metal.trace_mirror --check
    python -m optiland.backend.torch_backend.metal.trace_mirror --update \\
        --verified "optiland.geometries.plane:Plane.distance=re-mirrored 2026-09-17"

``--update`` refuses to touch a ``MIRRORED`` row without a matching
``--verified`` note: re-baselining mirrored physics is a decision a human
records, not a command that silences a warning.

Nothing here imports torch.  ``check_all()`` costs ~60 ``inspect.getsource`` +
``ast.parse`` + sha256 calls (tens of milliseconds) and the driver caches it
once per process.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "FINGERPRINTS",
    "Fingerprint",
    "STRUCTURAL_IDENTITIES",
    "check_all",
    "live_digests",
    "source_digest",
    "table_hash",
]

MIRRORED = "MIRRORED"
CONTRACT = "CONTRACT"


@dataclass(frozen=True)
class Fingerprint:
    """One mirrored (or contracted) Python source and the MSL that copies it.

    Attributes:
        qualname: ``"<module>:<dotted attribute path>"``.
        msl_function: the function in ``kernels/trace.metal`` that mirrors it,
            or the value test that guards it for ``CONTRACT`` rows.
        klass: ``MIRRORED`` or ``CONTRACT``.
        sha256: the recorded digest; ``None`` for every ``CONTRACT`` row.
        verified_note: why the current digest is trusted.
        source_sha: the fork commit the digest was taken on.
    """

    qualname: str
    msl_function: str
    klass: str
    sha256: str | None
    verified_note: str
    source_sha: str


def _resolve(qualname: str):
    """Return the live object named by ``"<module>:<dotted path>"``."""
    module_name, sep, path = qualname.partition(":")
    if not sep:
        raise ValueError(f"qualname must be '<module>:<path>', got {qualname!r}")
    obj = importlib.import_module(module_name)
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _strip_docstring(tree: ast.AST) -> ast.AST:
    """Drop the docstring node of every def/class in ``tree`` (in place)."""
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _digest_object(obj) -> str:
    """Digest one live object; see :func:`source_digest` for the rules."""
    if inspect.isroutine(obj) or inspect.isclass(obj):
        src = textwrap.dedent(inspect.getsource(obj))
        payload = "ast:" + ast.dump(_strip_docstring(ast.parse(src)))
    else:
        payload = "const:" + repr(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_digest(qualname: str) -> str:
    """Digest the live source (or value) of ``qualname``.

    Callables and classes are hashed from their AST with docstrings stripped;
    anything else (a module-level constant) from ``repr(value)``.  This is the
    *live attribute*, wrapper and all; :func:`check_all` uses
    :func:`live_digests`, because an instrumentation wrapper is not drift.
    """
    return _digest_object(_resolve(qualname))


#: How far :func:`live_digests` follows a wrapper towards the function it wraps.
_MAX_WRAPPER_DEPTH = 4


def _delegates(obj, path: str) -> list:
    """The routines ``obj`` may be a transparent wrapper around.

    Three ways a wrapper can hold the function it wraps, all followed here:
    ``functools.wraps`` records it in ``__wrapped__``; a wrapper built inside a
    function (every census in this plan) keeps it in a closure cell; a wrapper
    built at module level reads it from a module global, which is followed only
    when the global is the very function the row names (``__qualname__`` equals
    ``path``), so following a global can never wander off into the module.

    Wrapping a mirrored function for instrumentation -- which is what the
    censuses of plan 1.3 / 8.3 do to ``SurfaceGroup.trace`` and ``Surface.trace``
    -- is therefore not reported as drift, at any nesting depth.

    The relaxation is deliberate and bounded: a wrapper that *holds* the
    mirrored original is accepted without proof that it *calls* it unchanged.
    What the table exists to catch is an upstream merge editing a mirrored
    source (note 10), which moves the original's own digest and is still
    caught through the wrapper; the alternative -- refusing every wrapped
    attribute -- turns the fused path off in every harness that counts traces.
    """
    inner: list = []
    wrapped = getattr(obj, "__wrapped__", None)
    if inspect.isroutine(wrapped):
        inner.append(wrapped)
    func = obj.__func__ if inspect.ismethod(obj) else obj
    for cell in getattr(func, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:  # an empty cell of a closure still being built
            continue
        if inspect.isroutine(value) and value is not obj:
            inner.append(value)
    code = getattr(func, "__code__", None)
    module_globals = getattr(func, "__globals__", None)
    if code is not None and module_globals:
        for name in code.co_names:
            value = module_globals.get(name)
            if (
                inspect.isroutine(value)
                and value is not obj
                and getattr(value, "__qualname__", None) == path
            ):
                inner.append(value)
    return inner


def live_digests(qualname: str):
    """Yield ``(digest, error)`` for the live object and what it wraps.

    The live attribute comes first, so an unwrapped function still costs
    exactly one ``getsource`` + ``ast.parse`` + sha256: the caller stops at the
    first match.  Exactly one of ``digest`` / ``error`` is set per entry; an
    entry whose source cannot be retrieved (an ``exec``-compiled replacement,
    ``OSError``) reports the error and the walk continues.
    """
    obj = _resolve(qualname)
    path = qualname.partition(":")[2]
    # ``alive`` holds a reference to every object walked so that ``seen`` (keyed
    # by ``id``) cannot be fooled by an id reused after a candidate is freed.
    alive, seen, queue = [obj], {id(obj)}, [(obj, 0)]
    while queue:
        item, depth = queue.pop(0)
        try:
            yield _digest_object(item), None
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            yield None, f"{type(exc).__name__}: {exc}"
        if depth >= _MAX_WRAPPER_DEPTH:
            continue
        for candidate in _delegates(item, path):
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            alive.append(candidate)
            queue.append((candidate, depth + 1))


# --------------------------------------------------------------------------
# Structural identities: facts about the class graph that the record compiler
# relies on but that no single function body expresses.  Each returns True
# when the fact still holds.  Names are the test-parametrization ids.
# --------------------------------------------------------------------------

#: The interaction-model registry as measured at I0 on fork HEAD 096ccfc8.
FROZEN_INTERACTION_KEYS = frozenset(
    {"diffractive", "phase", "refractive_reflective", "thin_lens"}
)

#: Aperture classes the adapters whitelist; the registry must keep containing
#: them (it may grow -- a new class is refused with ``aperture_type``).
WHITELISTED_APERTURE_KEYS = frozenset(
    {
        "RadialAperture",
        "OffsetRadialAperture",
        "RectangularAperture",
        "EllipticalAperture",
    }
)

#: ``_SURFACE_INTERACTION_OVERRIDES`` as measured at I0.  Plan 3.7 predicted
#: ``{}``; the checkout has two entries (see day1-decisions.md, Q/deviation
#: D3).  Freezing the measured value keeps the identity meaningful: any new
#: override would silently redirect a surface type away from
#: ``RefractiveReflectiveModel``.
FROZEN_SURFACE_INTERACTION_OVERRIDES = {
    "paraxial": "thin_lens",
    "grating": "diffractive",
}


def image_surface_reuses_surface_trace_real() -> bool:
    from optiland.surfaces.image_surface import ImageSurface
    from optiland.surfaces.standard_surface import Surface

    return ImageSurface._trace_real is Surface._trace_real


def image_surface_reuses_surface_trace() -> bool:
    from optiland.surfaces.image_surface import ImageSurface
    from optiland.surfaces.standard_surface import Surface

    return ImageSurface.trace is Surface.trace


def object_surface_reuses_surface_reset() -> bool:
    from optiland.surfaces.object_surface import ObjectSurface
    from optiland.surfaces.standard_surface import Surface

    return ObjectSurface.reset is Surface.reset


def default_interaction_model_is_refractive_reflective() -> bool:
    from optiland.interactions.refractive_reflective_model import (
        RefractiveReflectiveModel,
    )
    from optiland.optic import Optic

    optic = Optic()
    optic.surfaces.add(index=0, radius=float("inf"), thickness=float("inf"))
    optic.surfaces.add(
        index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    return (
        type(optic.surfaces.surfaces[1].interaction_model) is RefractiveReflectiveModel
    )


def surface_interaction_overrides_unchanged() -> bool:
    from optiland.surfaces.factories.surface_factory import (
        _SURFACE_INTERACTION_OVERRIDES,
    )

    return dict(_SURFACE_INTERACTION_OVERRIDES) == FROZEN_SURFACE_INTERACTION_OVERRIDES


def interaction_registry_keys_unchanged() -> bool:
    from optiland.surfaces.factories.interaction_model_factory import (
        _INTERACTION_REGISTRY,
    )

    return frozenset(_INTERACTION_REGISTRY) == FROZEN_INTERACTION_KEYS


def aperture_registry_contains_whitelist() -> bool:
    from optiland.physical_apertures.base import BaseAperture

    return frozenset(BaseAperture._registry) >= WHITELISTED_APERTURE_KEYS


def distance_capability_attribute_name_unchanged() -> bool:
    """``_aperture_aware_distance`` caches on ``surface._distance_capability``."""
    from optiland.surfaces import standard_surface

    src = inspect.getsource(standard_surface._aperture_aware_distance)
    return "_distance_capability" in src


STRUCTURAL_IDENTITIES: tuple = (
    image_surface_reuses_surface_trace_real,
    image_surface_reuses_surface_trace,
    object_surface_reuses_surface_reset,
    default_interaction_model_is_refractive_reflective,
    surface_interaction_overrides_unchanged,
    interaction_registry_keys_unchanged,
    aperture_registry_contains_whitelist,
    distance_capability_attribute_name_unchanged,
)


# --------------------------------------------------------------------------
# The table.  Rewritten in place by ``--update``; edit the rows by hand only
# to add or remove a qualname.
# --------------------------------------------------------------------------
# --- BEGIN FINGERPRINT TABLE ---
_ROWS: tuple[tuple[str, str, str, str | None, str, str], ...] = (
    (
        "optiland.surfaces.surface_group:SurfaceGroup.trace",
        "trace_body",
        MIRRORED,
        "5d0af3c85c9933e6399393dcffdc2f945eeef6ccbb61e3b0d10d00c49b375e60",
        "WP4 hook (plan 3.4): mirrored loop unchanged, only the branch above it",
        "2f9f2912",
    ),
    (
        "optiland.surfaces.standard_surface:_TracingCoordinator.trace",
        "trace_body",
        MIRRORED,
        "90780d003ac2ec0e49ca2964c02577be26410bf77e111f4ce7c7212a85dd80ac",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.surfaces.standard_surface:_aperture_aware_distance",
        "select_distance",
        MIRRORED,
        "5a962b275f869bff4288e2353254572bad9799f480371b4cdb45a4e268483c7c",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.surfaces.standard_surface:Surface._trace_real",
        "trace_body",
        MIRRORED,
        "30e9a1e3fda3d5c7b5de6b41c1a306bbf37912c486f58ab8361c09065f429656",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.surfaces.standard_surface:Surface._record_real",
        "trace_body",
        MIRRORED,
        "258f648be0bdea9c32157ccb55a84fd8e6f67310c63b49d840da583662b82f4f",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.surfaces.standard_surface:Surface.reset",
        "trace_body",
        MIRRORED,
        "d0306bcd649f9d0f4930b9397601a0ce6183dba5002eb04fe6f0081c4a88ea5c",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.surfaces.standard_surface:Surface.trace",
        "trace_body",
        MIRRORED,
        "5bf0ada3d8b8cd050bbed0479d283433ba7b8883886e0da44127541eb2f14941",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.surfaces.standard_surface:Surface.__init__",
        "host:compile_records",
        MIRRORED,
        "3d3354edc5d5cfe944e2d097fb53a9f89ce6632406d3a4b66d3bee113569c6f5",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.surfaces.object_surface:ObjectSurface.trace",
        "trace_body",
        MIRRORED,
        "c87ac1637e8e0c3ee9c895d0bd6e360c4726354f26fbd5c2c3080846b40a6236",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.coordinate_system:CoordinateSystem.localize",
        "localize",
        MIRRORED,
        "d0dff34653c464d3f1a705a446debe304abae45f3c5ae46d636e27ed723f94c9",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.coordinate_system:CoordinateSystem.globalize",
        "globalize",
        MIRRORED,
        "d4d64fdb01001c723946dafa78991856d3e75994186f2922cca27b6f027c52c9",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.base:BaseRays.translate",
        "translate",
        MIRRORED,
        "9d5d2bf5431d47338709add02b51974bfb2173aba505684346a3dd8d48851f12",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays.rotate_x",
        "rotate_x",
        MIRRORED,
        "14c6516c3109857167fea955ba70ee6f75af88fbd3d5917e12d5931c867535d9",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays.rotate_y",
        "rotate_y",
        MIRRORED,
        "65591274c3ff19c8b17b7b4700cd4c566df0e1eede3d620e881853033c184e07",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays.rotate_z",
        "rotate_z",
        MIRRORED,
        "a0bfa16bbd2bbd60e8e591cfdd16f648f099f86e5679e3a2b34e114ee57e2a1a",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays.clip",
        "clip_intensity",
        MIRRORED,
        "1798d1b4fd2fa73d0c2214446de79317316765f5e62a2237c61e823cdf0fdc08",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays.refract",
        "refract",
        MIRRORED,
        "a5136321a4e3c29cdba537480e274d8ded6942a1501bd9a3e725efd489d45465",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays.reflect",
        "reflect",
        MIRRORED,
        "942bdfba17584291ffbea997e5310c1f0ba8e41890589f81ca41f4e2392c59ed",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays._align_surface_normal",
        "align_surface_normal",
        MIRRORED,
        "9c93e2a530ef5e034a815cca1e272e81aaccc858f44633a992dda223815213eb",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays.update",
        "interact",
        MIRRORED,
        "bd27daeaf1b94c29ed858fb7ac6589551baad435fdc12e2212b14e09132787a9",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays.trace_on_surface",
        "trace_body",
        MIRRORED,
        "89bbed24e3fa91f3595c6eb2cdfacb13cbf107b4b7325bf38381fa4c68807e26",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.rays.real_rays:RealRays.record_on_surface",
        "trace_body",
        MIRRORED,
        "d5c18a580f4ce95fd5b7eb8a10c56f53cf418d44e62c286928a1b217b738ebe6",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.plane:Plane.distance",
        "plane_distance",
        MIRRORED,
        "d76208987d03913d4fba866005f2440245dc7faccafa6db1040a53961ab9dbf4",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.plane:Plane.surface_normal",
        "plane_normal",
        MIRRORED,
        "acf67be3aee3ce70701000a72287247f10cd28e34345fcc088cc4b4bac129c97",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.standard:_is_radius_infinite",
        "host:compile_records",
        MIRRORED,
        "0c9b6f08c9cf03aa0377d3d465fc3ab5732cd69256004567e80bb1d6fc79dfbc",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.standard:_conic_intersection_distance",
        "conic_distance",
        MIRRORED,
        "cbae5dd545ad29d0183e4508a59bf54072a081f3e0f1d80023f5a9a0d682de6b",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.standard:StandardGeometry.sag",
        "std_sag",
        MIRRORED,
        "3155eeb028ba3dd875474816a47bfa80da15d2f6f6e0ce1db2dad863ee70bc83",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.standard:StandardGeometry._normal_components",
        "std_normal",
        MIRRORED,
        "be720885d5f434aeb5e75e7bfe94ba542b9dbc0d484d37748cd8d8b89b677eef",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.standard:StandardGeometry.distance",
        "conic_distance",
        MIRRORED,
        "ae26db23582f4d0f6b18959f411a2c4fa090649379eaa4b5d887994560af1e24",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.backend._conic:_conic_candidates",
        "conic_candidates",
        MIRRORED,
        "d530bfaa16a707ae7f2b442b747d03b019de96a5cf3178e4c9f62d5241347dec",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.backend._conic:_select_distance",
        "select_distance",
        MIRRORED,
        "73510240bb8ba8dc76e5999e89e692c31f608861c988b76ea4a51dad25ed3c7c",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:_nz_threshold",
        "nz_threshold",
        MIRRORED,
        "acfafb8cef378ef00016545cf4f32bcb2c97e1f0af1e8d1818231782756ff344",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:_sign_preserving_floor",
        "sign_preserving_floor",
        MIRRORED,
        "a8b7c6406ffe46414eef88a6a3c113bc28f7a3089a2c73bd65da22077f27f17e",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:_denominator_threshold",
        "denominator_threshold",
        MIRRORED,
        "017080ab19e8aadf56b3c1710e375d67e825dc6060f697e0cd84df10e504b12a",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:_regularize_signed",
        "regularize_signed",
        MIRRORED,
        "191865f618e7aae5f5741fdfdd55d7536aa0ac9c80ff231ec31762cc7eaa8784",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:_effective_tolerance",
        "effective_tolerance",
        MIRRORED,
        "565e32bc4cf9faede48b195e10437e8a24d670a5c08e8481a4bcac2b252f76ab",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:NewtonRaphsonGeometry._surface_residual",
        "surface_residual",
        MIRRORED,
        "518896f59d5febc2e38c92dc443872090623d831805dabf7e23f6ba3b9131ac8",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:NewtonRaphsonGeometry._surface_residual_dt",
        "surface_residual_dt",
        MIRRORED,
        "967582b065a83f1282f1d899c58e5f3e62be1d08e73cb28b8a1f573b4f246e31",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:NewtonRaphsonGeometry._solve_distance_primal",
        "newton_distance",
        MIRRORED,
        "21d4c7e92aa12beec523a1f7faa01c1e6ddbc65561da4bf6a5c76e9a2bfe07a9",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:NewtonRaphsonGeometry.distance",
        "newton_distance",
        MIRRORED,
        "48ebb668115caf527f47619194d4f8a3ca62564d8d2df7cdea7506ecf196f372",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:_DENOM_EPS_MULTIPLIER",
        "denominator_threshold",
        MIRRORED,
        "85497c22356c2cdcb163b1a3e7fa8bcab026503644cbacec92b84ce0a788c7bb",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.newton_raphson:_CONV_EPS_MULTIPLIER",
        "newton_distance",
        MIRRORED,
        "867ac4548aab3227263c53fb6d8788b140c7fb38a4011a3d0a025be0590deb10",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.even_asphere:EvenAsphere.sag",
        "even_sag",
        MIRRORED,
        "c60228ea2f5ac1bcbedb2adc9ff6fcaf2991836318513e389c0351fda3959ee0",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.even_asphere:EvenAsphere._surface_normal",
        "even_normal",
        MIRRORED,
        "09f7d18de3998092f21c3d48722c985ae4faebaf90a8bd400b04fe7463cd86c1",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.even_asphere:EvenAsphere.__init__",
        "host:compile_records",
        MIRRORED,
        "5885796157bb9458fdacb9e0320bfe3d845b384a1f804044024c3d1a9451c570",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.even_asphere:EvenAsphere.scale",
        "host:compile_records",
        MIRRORED,
        "0859f5a59260a3cebe9146b520c2a9585b17f981db04fe7e53219cb37ce882f7",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.odd_asphere:OddAsphere.sag",
        "odd_sag",
        MIRRORED,
        "fb1a82f70bdc93344d5659223c4a6ba514577020e4eb7fddf4a86ce4cc57f932",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.odd_asphere:OddAsphere._surface_normal",
        "odd_normal",
        MIRRORED,
        "5f3338daf95c8eccdb39314387aca3771fb8d3790fe564b0ce82fafa3a4d66ca",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.odd_asphere:OddAsphere.__init__",
        "host:compile_records",
        MIRRORED,
        "45d50d6bd31f55d3aa0a2b9e5af95e5513d694c0e4a2c7945dfba5e0d349d77a",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.geometries.odd_asphere:OddAsphere.scale",
        "host:compile_records",
        MIRRORED,
        "765e6f6ae88e34444bdfeee244d7a40082ab3ac03bab55ebe85b7c8ab9f71c63",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.propagation.homogeneous:HomogeneousPropagation.propagate",
        "propagate_absorb",
        MIRRORED,
        "5303219bf141a84ddaf60c2053fa89e3cfb85be7506ceb56f9f57df75d20e8fd",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.physical_apertures.base:BaseAperture.clip",
        "ap_clip",
        MIRRORED,
        "f801fe88cf687665a5bbe68ed5c29da8eb0de9b2f806d23d787d83c62296b789",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.physical_apertures.radial:RadialAperture.contains",
        "ap_contains[AP_RADIAL]",
        MIRRORED,
        "2a1366e75171400b6218395e8acc8235f393824bd3ecc58cf5dbdedeb99a6dd1",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.physical_apertures.offset_radial:OffsetRadialAperture.contains",
        "ap_contains[AP_OFFSET_RADIAL]",
        MIRRORED,
        "060aa9c76e91b6d0b08c101ce82bedbddf1e14ea6e8f242034c506aea790e87b",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.physical_apertures.rectangular:RectangularAperture.contains",
        "ap_contains[AP_RECT]",
        MIRRORED,
        "49a839989d457f6f71b3750d0d929cece5116c6d8e2612b086556c0bd0197a3b",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.physical_apertures.elliptical:EllipticalAperture.contains",
        "ap_contains[AP_ELLIPSE]",
        MIRRORED,
        "8af8103e50ff826571156f1ca7259159be3a3b71c928e25a2e8e1ba231331123",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.interactions.refractive_reflective_model:RefractiveReflectiveModel.interact_real_rays",
        "interact",
        MIRRORED,
        "3d3444b009b3324eb8c83f0bc1ce4faeed3bb3ed4dc15c4636c33b75014ba704",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.interactions.base:BaseInteractionModel._apply_coating_and_bsdf",
        "interact",
        MIRRORED,
        "f31511dadccd9f2a4790e2b1eb68eb7b7b1dcf48038e52528dbcb448b7cce786",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.utils:machine_eps",
        "consts[C_EPS]",
        MIRRORED,
        "e1ce7cb9765b7ae86f4ec518045136a0ee3ba8632f82246ded1c10cf1629f0b6",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.backend.torch_backend.conic:_epsilon",
        "consts[C_EPS]",
        MIRRORED,
        "334f99b0adefd54df3c951038eaac03db9de614b0470546a9eb95b8ea3055aec",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.backend.torch_backend.metal.conic:conic_intersection_metal",
        "conic_distance",
        MIRRORED,
        "4e0baa15a8ebd8ff9972ef56644a3f724a8d12bb7ef481812ba06e9471cccb55",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.backend.torch_backend.metal.tensor:MACHINE_EPS",
        "consts[C_EPS]",
        MIRRORED,
        "05b1f8e72d83fd6e7f96a048780810fd47021b14faf2245d0e4a94dac79b208d",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.backend.torch_backend.metal.tensor:DEFAULT_HOST_THRESHOLD",
        "gate:host_resident",
        MIRRORED,
        "47fc791089e38a0111dfd37ba5a66ad8247fe4ea124de88d7cac3069a5798d72",
        "I0 baseline: MSL mirrored from this source",
        "096ccfc8",
    ),
    (
        "optiland.materials.base:BaseMaterial._evaluate_property",
        "test_w0_canonical",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.materials.base:BaseMaterial._uniform_representative",
        "test_w0_canonical",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.materials.base:BaseMaterial._create_cache_key",
        "test_w0_canonical",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.materials.base:BaseMaterial._MAX_VALUE_KEY_ARRAY_SIZE",
        "test_uniform_key_boundary",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.backend.torch_backend.metal.ops_elementwise:_binary",
        "test_host_scalar_values",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.backend.torch_backend.metal.ops_elementwise:_host_scalar_op",
        "test_host_scalar_values",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.backend.torch_backend.metal.ops_elementwise:_pow_tensor_scalar",
        "test_pow_tensor_scalar_values",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.optic.optic_updater:OpticUpdater.set_radius",
        "test_updater_applies_scaled_values",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.optic.optic_updater:OpticUpdater.set_conic",
        "test_updater_applies_scaled_values",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.optic.optic_updater:OpticUpdater.set_thickness",
        "test_updater_applies_scaled_values",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.optic.optic_updater:OpticUpdater.set_asphere_coeff",
        "test_updater_applies_scaled_values",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.optic.optic_updater:OpticUpdater.set_index",
        "test_updater_applies_scaled_values",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.optic.optic_updater:OpticUpdater.update",
        "test_updater_applies_scaled_values",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.surfaces.surface_group:SurfaceGroup._update_coordinate_systems",
        "test_pose_refresh_after_update",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.optimization.variable.variable:Variable.update",
        "test_variable_update_units",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
    (
        "optiland.optimization.variable.variable:Variable.reset",
        "test_variable_update_units",
        CONTRACT,
        None,
        "contract row: values compared by the named test; never hashed",
        "096ccfc8",
    ),
)
# --- END FINGERPRINT TABLE ---

FINGERPRINTS: tuple[Fingerprint, ...] = tuple(Fingerprint(*row) for row in _ROWS)

_BY_QUALNAME: dict[str, Fingerprint] = {fp.qualname: fp for fp in FINGERPRINTS}


def table_hash() -> str:
    """Digest of the whole table; stamped into oracle/suite/benchmark JSONs."""
    payload = "\n".join(
        f"{fp.qualname}|{fp.msl_function}|{fp.klass}|{fp.sha256}" for fp in FINGERPRINTS
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_all() -> list[str]:
    """Return one message per drifted MIRRORED row or failed identity.

    An empty list means the kernel's mirrors are still faithful to the Python
    sources they were verified against.
    """
    problems: list[str] = []
    for fp in FINGERPRINTS:
        if fp.klass != MIRRORED:
            continue
        digests: list[str] = []
        errors: list[str] = []
        matched = False
        try:
            for digest, error in live_digests(fp.qualname):
                if error is not None:
                    errors.append(error)
                    continue
                digests.append(digest)
                if digest == fp.sha256:
                    matched = True
                    break
        except Exception as exc:  # noqa: BLE001 - a rename is drift, not a crash
            problems.append(
                f"{fp.qualname}: cannot be resolved ({type(exc).__name__}: {exc}); "
                f"MSL {fp.msl_function} in kernels/trace.metal is unverified"
            )
            continue
        if matched:
            continue
        if not digests:
            why = "; ".join(errors) or "no source"
            problems.append(
                f"{fp.qualname}: cannot be resolved ({why}); "
                f"MSL {fp.msl_function} in kernels/trace.metal is unverified"
            )
            continue
        problems.append(
            f"Python `{fp.qualname}` changed; re-verify MSL `{fp.msl_function}` in "
            "kernels/trace.metal, then run `python -m "
            "optiland.backend.torch_backend.metal.trace_mirror --update --verified "
            f'"{fp.qualname}=<why it is still mirrored>"`'
        )
    for identity in STRUCTURAL_IDENTITIES:
        try:
            ok = bool(identity())
        except Exception as exc:  # noqa: BLE001
            problems.append(
                f"structural identity {identity.__name__} raised "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if not ok:
            problems.append(f"structural identity {identity.__name__} no longer holds")
    return problems


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_TABLE_PATH = Path(__file__).resolve()
_BEGIN = "# --- BEGIN FINGERPRINT TABLE ---"
_END = "# --- END FINGERPRINT TABLE ---"


def _fork_sha() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=str(_TABLE_PATH.parent),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"
    return out.stdout.strip() or "unknown"


#: ``ruff``'s line-length gate (``pyproject.toml``), which every rendered line
#: of the table has to fit: E501 in this file blocks the commit (plan 10.1).
_LINE_LENGTH = 88


def _escape(value: str) -> str:
    """``value`` as the body of a double-quoted Python string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _literal_lines(value: str, indent: str, suffix: str) -> list[str]:
    """``value`` as adjacent string literals, every line within the gate.

    A ``--verified`` note is free text a human writes (plan 0.2.8), so it can be
    any length; Python concatenates adjacent literals, so the note round-trips
    exactly while each rendered line stays inside ``_LINE_LENGTH``.  Splitting is
    on the raw string and each piece is escaped afterwards, so a split can never
    land inside an escape sequence.  A value with no space in it is left on one
    line: there is nothing to split, and ruff exempts a single-word line from
    E501 (which is what the over-long qualnames in the table already rely on).
    """
    if " " not in value:
        # No split point -- and ruff exempts a line that is a single long word
        # from E501, which is what every over-long qualname in the table uses.
        return [f'{indent}"{_escape(value)}"{suffix}']
    budget = _LINE_LENGTH - len(indent) - 2 - len(suffix)  # quotes + comma
    pieces: list[str] = []
    current = ""
    for index, word in enumerate(value.split(" ")):
        piece = word if index == 0 else " " + word
        if current and len(_escape(current + piece)) > budget:
            pieces.append(current)
            current = ""
        while len(_escape(piece)) > budget:  # one word wider than a whole line
            head = piece[:budget]
            while len(_escape(head)) > budget:
                head = head[:-1]
            pieces.append(head)
            piece = piece[len(head) :]
        current += piece
    pieces.append(current)
    lines = [f'{indent}"{_escape(piece)}"' for piece in pieces]
    lines[-1] += suffix
    return lines


def _render_table(rows) -> str:
    lines = ["_ROWS: tuple[tuple[str, str, str, str | None, str, str], ...] = ("]
    for qualname, msl, klass, sha, note, sha_src in rows:
        sha_lit = "None" if sha is None else f'"{sha}"'
        lines.append("    (")
        lines.extend(_literal_lines(qualname, "        ", ","))
        lines.extend(_literal_lines(msl, "        ", ","))
        lines.append(f"        {klass},")
        lines.append(f"        {sha_lit},")
        lines.extend(_literal_lines(note, "        ", ","))
        lines.extend(_literal_lines(sha_src, "        ", ","))
        lines.append("    ),")
    lines.append(")")
    return "\n".join(lines)


def _write_table(rows) -> None:
    text = _TABLE_PATH.read_text(encoding="utf-8")
    start = text.index(_BEGIN) + len(_BEGIN) + 1
    end = text.index(_END)
    _TABLE_PATH.write_text(
        text[:start] + _render_table(rows) + "\n" + text[end:], encoding="utf-8"
    )


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="fused-trace mirror fingerprints")
    parser.add_argument("--check", action="store_true", help="exit 1 on any drift")
    parser.add_argument(
        "--update", action="store_true", help="rewrite the recorded digests"
    )
    parser.add_argument(
        "--verified",
        action="append",
        default=[],
        metavar="QUALNAME=NOTE",
        help="authorise updating one MIRRORED row (repeatable)",
    )
    parser.add_argument("--list", action="store_true", help="print the table")
    args = parser.parse_args(argv)

    if args.list:
        for fp in FINGERPRINTS:
            digest = fp.sha256 or "-"
            print(f"{fp.klass:9s} {fp.qualname}  ->  {fp.msl_function}  {digest}")
        print(f"table_hash = {table_hash()}")
        return 0

    if args.update:
        notes: dict[str, str] = {}
        for item in args.verified:
            key, sep, note = item.partition("=")
            if not sep:
                print(f"--verified needs QUALNAME=NOTE, got {item!r}")
                return 2
            notes[key.strip()] = note.strip()
        sha_src = _fork_sha()
        rows: list[tuple] = []
        refused: list[str] = []
        for fp in FINGERPRINTS:
            if fp.klass == CONTRACT:
                rows.append(
                    (
                        fp.qualname,
                        fp.msl_function,
                        CONTRACT,
                        None,
                        fp.verified_note,
                        fp.source_sha,
                    )
                )
                continue
            live = source_digest(fp.qualname)
            if live == fp.sha256:
                rows.append(
                    (
                        fp.qualname,
                        fp.msl_function,
                        MIRRORED,
                        fp.sha256,
                        fp.verified_note,
                        fp.source_sha,
                    )
                )
                continue
            if fp.qualname not in notes:
                refused.append(fp.qualname)
                rows.append(
                    (
                        fp.qualname,
                        fp.msl_function,
                        MIRRORED,
                        fp.sha256,
                        fp.verified_note,
                        fp.source_sha,
                    )
                )
                continue
            rows.append(
                (
                    fp.qualname,
                    fp.msl_function,
                    MIRRORED,
                    live,
                    notes[fp.qualname],
                    sha_src,
                )
            )
        if refused:
            print("refusing to update MIRRORED rows without --verified:")
            for qualname in refused:
                print(f"  {qualname}")
            print("re-verify the MSL first, then pass --verified 'QUALNAME=why'.")
            return 1
        _write_table(rows)
        print(f"updated {len(rows)} rows at {sha_src}")
        return 0

    problems = check_all()
    if problems:
        print(f"{len(problems)} mirror problem(s):")
        for message in problems:
            print(f"  - {message}")
        return 1
    print(f"ok: {len(FINGERPRINTS)} rows, table_hash = {table_hash()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())
