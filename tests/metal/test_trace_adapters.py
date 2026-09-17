"""WP2: the fused-trace eligibility gate and the record compiler.

Every assertion predicts an exact result (plan 0.2.6): a counter value, a
refusal reason, or a float64 bit pattern.  The mirror target is the *per-op
Metal path*, never NumPy: day-1 measurements repeated here show the NumPy and
torch host paths disagree by 1 ulp in ``be.cos`` and in a catalogue glass's
extinction coefficient, so NumPy cross-checks are restricted to the slots whose
expressions are exact (translations, radius, conic, ``1 + k``, ``R**2``,
aperture parameters, coefficients).

Run: ``pytest tests/metal/test_trace_adapters.py -q -p no:cacheprovider -o addopts=``
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("Metal GPU required", allow_module_level=True)

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend.metal import trace_adapters as A  # noqa: E402
from optiland.backend.torch_backend.metal import trace_layout as LY  # noqa: E402
from optiland.backend.torch_backend.metal import trace_record as R  # noqa: E402
from optiland.backend.torch_backend.metal.trace_adapters import (  # noqa: E402
    FusedTraceSkip,
)
from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.geometries.even_asphere import EvenAsphere  # noqa: E402
from optiland.geometries.odd_asphere import OddAsphere  # noqa: E402
from optiland.geometries.plane import Plane  # noqa: E402
from optiland.geometries.standard import StandardGeometry  # noqa: E402
from optiland.geometries.toroidal import ToroidalGeometry  # noqa: E402
from optiland.materials import IdealMaterial, Material  # noqa: E402
from optiland.optic import Optic  # noqa: E402
from optiland.physical_apertures.elliptical import EllipticalAperture  # noqa: E402
from optiland.physical_apertures.offset_radial import (  # noqa: E402
    OffsetRadialAperture,
)
from optiland.physical_apertures.radial import RadialAperture  # noqa: E402
from optiland.physical_apertures.rectangular import RectangularAperture  # noqa: E402
from optiland.rays.real_rays import RealRays  # noqa: E402

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import trace_fixtures  # noqa: E402

MODES = ("df64", "sf64")

#: Day-1 Q13: tier-A comparisons use a bundle above ``_MAX_VALUE_KEY_ARRAY_SIZE``
#: so the per-op path takes the ``_uniform_representative`` branch.
N_RAYS = 4096

#: Day-1 P12 measured these as *not* df64-representable, so they are where the
#: canonical-wavelength rule earns its keep.
LOSSY_WAVELENGTHS = (0.543, 0.5875618, 0.65)

#: Day-1 P5: 941 us per 32-slot row on this machine.
P5_ROW_SECONDS = 941e-6

#: Day-1 P4 measured these two on CookeTriplet surface 1 under mps.
P4_RADIUS_SQUARED = 484.59814468810004
P4_ONE_PLUS_K = 1.0

#: Slots whose expressions are exact in every backend (no libm, no dispersion),
#: so the NumPy backend is a valid bit-for-bit oracle for them.
EXACT_SLOTS = (
    (LY.SR_TX, LY.SR_TY, LY.SR_TZ, LY.SR_NTX, LY.SR_NTY, LY.SR_NTZ)
    + (LY.SR_R, LY.SR_K, LY.SR_K1, LY.SR_R2, LY.SR_TOL)
    + (LY.SR_AP0, LY.SR_AP1, LY.SR_AP2, LY.SR_AP3)
)

if hasattr(trace_fixtures, "CATALOG"):
    CATALOG = dict(trace_fixtures.CATALOG)
    KNOWN_INELIGIBLE = dict(trace_fixtures.KNOWN_INELIGIBLE)
    CATALOG_SOURCE = "scripts/trace_fixtures.py"
else:  # pragma: no cover - only while WP5's rewrite of the module is in flight
    import optiland.samples as _samples
    from optiland.samples import objectives as _objectives

    CATALOG = {name: getattr(_samples, name) for name in _samples.__all__}
    for _name in (
        "WideAngle100FOV",
        "ProjectionLens120FOV",
        "ProjectionLens160FOV",
        "WideAngle170FOV",
    ):
        CATALOG[_name] = getattr(_objectives, _name)
    KNOWN_INELIGIBLE = {}
    CATALOG_SOURCE = "optiland.samples (trace_fixtures has no CATALOG yet)"

CATALOG_NAMES = sorted(CATALOG)


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def mps_df64():
    """The torch-mps backend in df64, grad off."""
    yield from _backend("df64")


@pytest.fixture(params=MODES, ids=lambda m: f"mode={m}")
def mps_mode(request):
    """The torch-mps backend in each representation, grad off."""
    yield from _backend(request.param)
    return


def _backend(mode: str):
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    previous = be.metal_mode()
    be.set_metal_mode(mode)
    be.metal_reset_stats()
    try:
        yield mode
    finally:
        be.set_metal_mode(previous)
        be.grad_mode.disable()
        be.set_backend("numpy")


_OPTIC_CACHE: dict[tuple[str, str], object] = {}


def _catalog_optic(name: str, mode: str):
    """A catalog system built under ``mode`` (cached; never mutated by a test)."""
    key = (name, mode)
    if key not in _OPTIC_CACHE:
        _OPTIC_CACHE[key] = CATALOG[name]()
    return _OPTIC_CACHE[key]


def _bundle(n: int = N_RAYS, wavelength: float = 0.55, seed: int = 5) -> RealRays:
    """A GPU-resident ``RealRays`` bundle of ``n`` rays at one wavelength."""
    rng = np.random.default_rng(seed)
    x = be.array(rng.uniform(-4.0, 4.0, n))
    y = be.array(rng.uniform(-4.0, 4.0, n))
    z = be.array(np.full(n, -12.0))
    zeros = be.array(np.zeros(n))
    ones = be.array(np.ones(n))
    return RealRays(
        x,
        y,
        z,
        zeros,
        be.array(np.zeros(n)),
        ones,
        be.array(np.ones(n)),
        be.array(np.full(n, wavelength)),
    )


def _simple_optic(
    *,
    geometry: str = "conic",
    aperture=None,
    max_iter: int = 100,
    coefficients=(-2.2e-4, -4.7e-6),
    reflective: bool = False,
) -> Optic:
    """A hand-built three-surface optic exercising one adapter at a time."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    common = {
        "index": 1,
        "thickness": 5.0,
        "is_stop": True,
        "material": "mirror" if reflective else "N-BK7",
    }
    if geometry in ("plane", "std_inf"):
        optic.surfaces.add(radius=np.inf, **common)
    elif geometry == "conic":
        optic.surfaces.add(radius=25.0, conic=-0.5, **common)
    elif geometry in ("even", "odd"):
        optic.surfaces.add(
            radius=25.0,
            conic=0.0,
            surface_type=f"{geometry}_asphere",
            coefficients=list(coefficients),
            tol=1e-10,
            max_iter=max_iter,
            **common,
        )
    else:  # pragma: no cover - programming error
        raise ValueError(geometry)
    optic.surfaces.add(index=2, thickness=20.0)
    optic.surfaces.add(index=3)

    surface = optic.surfaces.surfaces[1]
    if geometry == "std_inf":
        # ``Plane`` and an infinite-radius ``StandardGeometry`` are different
        # codes and different Python code paths (``standard.py`` guards N).
        surface.geometry = StandardGeometry(surface.geometry.cs, be.array(np.inf), 0.0)
    if aperture is not None:
        surface.aperture = aperture

    optic.set_aperture(aperture_type="EPD", value=6.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    return optic


def _group(optic) -> object:
    return optic.surfaces


def _gate(optic, rays, skip: int = 0, **kwargs) -> R.GateResult:
    return R.can_fuse_trace(_group(optic), rays, skip, **kwargs)


def _float(value) -> float:
    return float(np.ravel(be.to_numpy(value))[0])


class _CopyCounter:
    """Counts device->host transfers (day-1 probe P10)."""

    NAMES = ("cpu", "item", "tolist", "numpy")

    def __enter__(self):
        self.counts = dict.fromkeys(self.NAMES, 0)
        self._orig = {}
        for name in self.NAMES:
            orig = getattr(torch.Tensor, name)
            self._orig[name] = orig

            def make(name=name, orig=orig):
                def wrapper(tensor, *a, **kw):
                    if tensor.device.type != "cpu":
                        self.counts[name] += 1
                    return orig(tensor, *a, **kw)

                return wrapper

            setattr(torch.Tensor, name, make())
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(torch.Tensor, name, orig)
        return False

    @property
    def total(self) -> int:
        return sum(self.counts.values())


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_gate_accepts_catalog(mps_df64, name):
    """Every shipped sample is eligible unless WP5 recorded it as not."""
    optic = _catalog_optic(name, mps_df64)
    result = _gate(optic, _bundle())
    expected = KNOWN_INELIGIBLE.get(name)
    if expected is None:
        assert result.ok is True, (
            f"{name} was refused with {result.reason}; if that is correct it "
            "belongs in KNOWN_INELIGIBLE (scripts/trace_fixtures.py, WP5) and "
            "must be requested through NOTES/fused-trace-research/status.md"
        )
        assert result.reason is None
        assert result.n == N_RAYS
        assert result.s == len(optic.surfaces.surfaces)
        assert result.mode == "df64"
        assert result.w0 == R.canonical_w0(0.55, "df64")
    else:
        assert result.ok is False
        assert result.reason.value == expected


def test_catalog_invariants(mps_df64):
    """``KNOWN_INELIGIBLE`` names catalog systems and reasons the gate returns."""
    assert len(CATALOG) >= 29, (CATALOG_SOURCE, len(CATALOG))
    assert set(KNOWN_INELIGIBLE) <= set(CATALOG), set(KNOWN_INELIGIBLE) - set(CATALOG)
    reasons = {reason.value for reason in FusedTraceSkip}
    for name, expected in KNOWN_INELIGIBLE.items():
        assert expected in reasons, (name, expected)
        # Re-evaluated, never copied from the table (plan fix L3.11).
        result = _gate(_catalog_optic(name, mps_df64), _bundle())
        assert result.ok is False and result.reason.value == expected, (name, result)


def test_gate_reason_order(mps_mode):
    """The first reason in design 2.2's order wins; structural beats feature."""
    optic = _simple_optic()
    optic.surfaces.surfaces[1].geometry = ToroidalGeometry(
        optic.surfaces.surfaces[1].geometry.cs, 25.0, 30.0
    )
    # geometry_type alone
    assert _gate(optic, _bundle()).reason is FusedTraceSkip.GEOMETRY_TYPE
    # skip (check 1) beats geometry_type (check 10)
    assert _gate(optic, _bundle(), skip=1).reason is FusedTraceSkip.SKIP
    # host residency (structural, check 5) beats geometry_type (feature)
    small = _gate(optic, _bundle(n=256))
    assert small.reason is FusedTraceSkip.HOST_RESIDENT
    assert small.structural is True
    # ``skip`` is check 1, so it wins even over a non-``RealRays`` bundle.
    assert _gate(optic, object(), skip=3).reason is FusedTraceSkip.SKIP


def test_gate_one_readback(mps_mode):
    """One device copy, and exactly one counter, over an eligible 8-surface trace."""
    optic = _catalog_optic("CookeTriplet", mps_mode)
    assert len(optic.surfaces.surfaces) == 8
    rays = _bundle()
    _gate(optic, rays)  # warm the material caches, as any second trace would

    be.metal_reset_stats()
    with _CopyCounter() as counter:
        result = _gate(optic, rays)
    assert result.ok is True
    assert counter.total == 1, counter.counts
    assert dict(be.metal_stats()) == {"fused_trace:readback": 1}

    # With the wavelength supplied there is no readback at all.
    be.metal_reset_stats()
    with _CopyCounter() as counter2:
        supplied = _gate(optic, rays, wavelength=result.w0)
    assert supplied.ok is True and supplied.w0 == result.w0
    assert counter2.total == 0, counter2.counts
    assert dict(be.metal_stats()) == {}


def test_gate_refuses_host_resident(mps_mode):
    """N = 256 is the dual-residency path; N = 257 is the kernel's."""
    optic = _catalog_optic("CookeTriplet", mps_mode)
    refused = _gate(optic, _bundle(n=256))
    assert refused.ok is False
    assert refused.reason is FusedTraceSkip.HOST_RESIDENT
    assert refused.structural is True
    assert _gate(optic, _bundle(n=257)).ok is True


def test_gate_refuses_mixed_and_nonfinite_wavelength(mps_mode):
    """A mixed bundle and an all-NaN bundle are both refused by the readback."""
    optic = _catalog_optic("CookeTriplet", mps_mode)
    mixed = _bundle()
    values = np.full(N_RAYS, 0.55)
    values[7] = 0.6328
    mixed.w = be.array(values)
    assert _gate(optic, mixed).reason is FusedTraceSkip.MIXED_WAVELENGTH

    # sf64 stores bit patterns, so an all-NaN bundle is *uniform*: the explicit
    # finiteness test is what refuses it (day-1 probe P10).
    nan_rays = _bundle()
    nan_rays.w = be.array(np.full(N_RAYS, np.nan))
    assert _gate(optic, nan_rays).reason is FusedTraceSkip.MIXED_WAVELENGTH


def test_gate_requires_grad(mps_df64):
    """Grad is structural: rays, and any tensor an adapter reads."""
    be.grad_mode.enable()
    try:
        optic = _simple_optic()
        rays = _bundle()
        result = _gate(optic, rays)
        assert result.ok is False
        assert result.reason is FusedTraceSkip.REQUIRES_GRAD
        assert result.structural is True
        with torch.no_grad():
            assert _gate(optic, rays).ok is True
    finally:
        be.grad_mode.disable()

    # A tensor coefficient that requires grad refuses on its own (plan 3.9).
    optic = _simple_optic(geometry="even")
    rays = _bundle()
    assert _gate(optic, rays).ok is True
    geometry = optic.surfaces.surfaces[1].geometry
    be.grad_mode.enable()
    try:
        geometry.coefficients[0] = be.array(-2.2e-4)
    finally:
        be.grad_mode.disable()
    assert geometry.coefficients[0].requires_grad is True
    assert _gate(optic, rays).reason is FusedTraceSkip.REQUIRES_GRAD
    with torch.no_grad():
        assert _gate(optic, rays).ok is True


def test_gate_newton_params(mps_mode):
    """``max_iter`` must fit under the ``ITERS_UNWRITTEN`` sentinel."""
    assert LY.ITERS_UNWRITTEN == 255
    optic = _simple_optic(geometry="even", max_iter=254)
    assert _gate(optic, _bundle()).ok is True
    optic.surfaces.surfaces[1].geometry.max_iter = 255
    assert _gate(optic, _bundle()).reason is FusedTraceSkip.NEWTON_PARAMS
    optic.surfaces.surfaces[1].geometry.max_iter = -1
    assert _gate(optic, _bundle()).reason is FusedTraceSkip.NEWTON_PARAMS
    optic.surfaces.surfaces[1].geometry.max_iter = 100
    optic.surfaces.surfaces[1].geometry.tol = "tight"
    assert _gate(optic, _bundle()).reason is FusedTraceSkip.NEWTON_PARAMS


def test_gate_structural_split_is_the_frozen_six(mps_df64):
    """The structural set is exactly the six reasons that never raise."""
    assert {r.value for r in A.STRUCTURAL_REASONS} == {
        "group_type",
        "rays_type",
        "rays_shape",
        "host_resident",
        "requires_grad",
        "skip",
    }
    optic = _catalog_optic("CookeTriplet", mps_df64)
    assert _gate(optic, _bundle(), skip=2).structural is True
    assert R.can_fuse_trace(object(), _bundle(), 0).reason is FusedTraceSkip.GROUP_TYPE
    assert _gate(optic, object()).reason is FusedTraceSkip.RAYS_TYPE
    ragged = _bundle()
    ragged.opd = be.array(np.zeros(N_RAYS - 1))
    assert _gate(optic, ragged).reason is FusedTraceSkip.RAYS_SHAPE


def test_gate_refuses_reference_cs_and_nonfinite_pose(mps_df64):
    """Nested frames and an infinite pose scalar are feature refusals."""
    optic = _simple_optic()
    surface = optic.surfaces.surfaces[1]
    surface.geometry.cs.reference_cs = CoordinateSystem(z=1.0)
    assert _gate(optic, _bundle()).reason is FusedTraceSkip.REFERENCE_CS
    surface.geometry.cs.reference_cs = None
    surface.geometry.cs.rx = np.inf
    assert _gate(optic, _bundle()).reason is FusedTraceSkip.POSE_NONFINITE


def test_gate_memory_budget(mps_df64, monkeypatch):
    """The budget refuses before anything is allocated."""
    optic = _catalog_optic("CookeTriplet", mps_df64)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION", "1e-12")
    assert _gate(optic, _bundle()).reason is FusedTraceSkip.MEMORY
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION", "0.25")
    assert _gate(optic, _bundle()).ok is True


def test_gate_min_rays(mps_df64, monkeypatch):
    """``MIN_RAYS`` defaults to 0 (day-1 Q7) and refuses below itself when set."""
    optic = _catalog_optic("CookeTriplet", mps_df64)
    assert _gate(optic, _bundle(n=512)).ok is True
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE_MIN_RAYS", "1024")
    assert _gate(optic, _bundle(n=512)).reason is FusedTraceSkip.MIN_RAYS
    assert _gate(optic, _bundle(n=4096)).ok is True


# ---------------------------------------------------------------------------
# canonical wavelength and material scalars
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "material_name", ["N-BK7", "SF11", "N-SK16", "ideal"], ids=lambda n: f"mat={n}"
)
@pytest.mark.parametrize("wavelength", LOSSY_WAVELENGTHS + (0.55,))
def test_w0_canonical(mps_mode, material_name, wavelength):
    """The hook-path readback IS ``canonical_w0``, and both give one index."""
    mode = mps_mode
    if material_name == "ideal":
        material = IdealMaterial(1.5, 1e-7)
    else:
        material = Material(material_name)
    w0 = R.canonical_w0(wavelength, mode)
    if mode == "sf64":
        assert w0 == wavelength  # the round trip is the identity
    rays = _bundle(wavelength=wavelength)

    optic = _catalog_optic("CookeTriplet", mode)
    readback = _gate(optic, rays).w0
    assert readback == w0, (readback, w0)

    from_w0 = _float(material.n(be.array([w0])))
    from_bundle = _float(material.n(rays.w))
    assert from_w0 == from_bundle, (from_w0, from_bundle)

    k_from_w0 = _float(material.k(be.array([w0])))
    k_from_bundle = _float(material.k(rays.w))
    assert k_from_w0 == k_from_bundle

    alpha_w0 = _float((4 * be.pi) * material.k(be.array([w0])))
    alpha_bundle = _float((4 * be.pi) * material.k(rays.w))
    assert alpha_w0 == alpha_bundle


def test_uniform_key_boundary(mps_df64):
    """``_MAX_VALUE_KEY_ARRAY_SIZE`` and the behaviour change at 1025 (day-1 Q13)."""
    from optiland.materials.base import BaseMaterial

    assert BaseMaterial._MAX_VALUE_KEY_ARRAY_SIZE == 1024

    w = 0.5499999999999998
    values = {}
    keys = {}
    for n in (1024, 1025, N_RAYS):
        material = Material("N-BK7")
        array = be.array(np.full(n, w))
        keys[n] = material._create_cache_key(array)[0][0]
        values[n] = _float(material.n(array))
    assert keys[1024] == "array-values"
    assert keys[1025] == "array-uniform"
    assert keys[N_RAYS] == "array-uniform"
    # The uniform branch is the one the record compiler mirrors, and it is a
    # different value from the full-array branch: that is tier-B site 1.
    assert values[1025] == values[N_RAYS]
    assert values[1024] != values[1025]
    assert _float(Material("N-BK7").n(be.array([w]))) == values[N_RAYS]


@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_material_scalars_match_per_op_representative(mps_df64, name):
    """Slots 22-26 equal the values the per-op path computes for the bundle."""
    optic = _catalog_optic(name, mps_df64)
    rays = _bundle()
    w0 = R.canonical_w0(0.55, "df64")
    records = R.compile_records(_group(optic), w0, "df64")
    for index, surface in enumerate(optic.surfaces.surfaces[1:], start=1):
        row = records.surf_real[0, index]
        n_pre = _float(surface.material_pre.n(rays.w))
        n_post = _float(surface.material_post.n(rays.w))
        k_pre = _float(surface.material_pre.k(rays.w))
        assert row[LY.SR_NPRE] == n_pre, (name, index)
        assert row[LY.SR_NPOST] == n_post, (name, index)
        assert row[LY.SR_U] == n_pre / n_post, (name, index)
        assert row[LY.SR_U2] == (n_pre / n_post) ** 2, (name, index)
        assert row[LY.SR_ALPHA] == 4 * np.pi * k_pre, (name, index)


# ---------------------------------------------------------------------------
# surf_real / surf_int
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_surf_real_matches_host_scalar_ops(mps_df64, name):
    """Each slot equals the documented host expression, bit for bit.

    The exact slots are additionally pinned against the NumPy backend and
    against ``to_dict()``; the trig and material slots are not, because the two
    backends' libm differ by 1 ulp (measured: ``be.cos(0.1)`` and ``Material
    ('SF2').k``), and the mirror target is the mps path.
    """
    w0 = R.canonical_w0(0.55, "df64")
    optic = _catalog_optic(name, mps_df64)
    records = R.compile_records(_group(optic), w0, "df64")

    for index, surface in enumerate(optic.surfaces.surfaces[1:], start=1):
        row = records.surf_real[0, index]
        geometry = surface.geometry
        cs = geometry.cs
        expected = {
            LY.SR_TX: float(cs.x),
            LY.SR_TY: float(cs.y),
            LY.SR_TZ: float(cs.z),
            LY.SR_NTX: float(-cs.x),
            LY.SR_NTY: float(-cs.y),
            LY.SR_NTZ: float(-cs.z),
            LY.SR_CNRZ: float(be.cos(-cs.rz)),
            LY.SR_SNRZ: float(be.sin(-cs.rz)),
            LY.SR_CNRY: float(be.cos(-cs.ry)),
            LY.SR_SNRY: float(be.sin(-cs.ry)),
            LY.SR_CNRX: float(be.cos(-cs.rx)),
            LY.SR_SNRX: float(be.sin(-cs.rx)),
            LY.SR_CRX: float(be.cos(cs.rx)),
            LY.SR_SRX: float(be.sin(cs.rx)),
            LY.SR_CRY: float(be.cos(cs.ry)),
            LY.SR_SRY: float(be.sin(cs.ry)),
            LY.SR_CRZ: float(be.cos(cs.rz)),
            LY.SR_SRZ: float(be.sin(cs.rz)),
            LY.SR_R: float(geometry.radius),
            LY.SR_K: float(getattr(geometry, "k", 0.0)),
        }
        for slot, value in expected.items():
            assert row[slot] == value or (np.isnan(row[slot]) and np.isnan(value)), (
                name,
                index,
                slot,
            )

        as_dict = geometry.to_dict()
        assert row[LY.SR_R] == float(as_dict["radius"]), (name, index)
        if "conic" in as_dict:
            assert row[LY.SR_K] == float(as_dict["conic"]), (name, index)
            assert row[LY.SR_K1] == float(1 + geometry.k), (name, index)
            assert row[LY.SR_R2] == float(geometry.radius**2), (name, index)
        else:
            assert row[LY.SR_K1] == 1.0 and np.isinf(row[LY.SR_R2])

    # NumPy backend oracle for the exact slots only.
    try:
        be.set_backend("numpy")
        numpy_records = R.compile_records(_group(CATALOG[name]()), w0, "df64")
    finally:
        be.set_backend("torch")
        be.set_device("mps")
        be.set_precision("float64")
    for slot in EXACT_SLOTS:
        assert np.array_equal(
            records.surf_real[:, :, slot],
            numpy_records.surf_real[:, :, slot],
            equal_nan=True,
        ), (name, slot)
    assert np.array_equal(records.surf_int, numpy_records.surf_int), name
    assert np.array_equal(records.coef, numpy_records.coef, equal_nan=True), name


def test_host_scalar_values(mps_df64):
    """Day-1 P4: the record compiler's reads are host ops with exact values."""
    optic = _catalog_optic("CookeTriplet", mps_df64)
    surface = optic.surfaces.surfaces[1]
    geometry = surface.geometry
    cs = geometry.cs
    # warm the material cache so the delta below is only this expression's
    surface.material_pre.n(be.array([0.55]))
    surface.material_post.n(be.array([0.55]))

    be.metal_reset_stats()
    cos_value = float(be.cos(-cs.rz))
    one_plus_k = float(1 + geometry.k)
    radius_squared = float(geometry.radius**2)
    n_pre = _float(surface.material_pre.n(be.array([0.55])))
    n_post = _float(surface.material_post.n(be.array([0.55])))
    ratio = n_pre / n_post
    stats = dict(be.metal_stats())

    assert not [key for key in stats if key.startswith("gpu:")], stats
    assert cos_value == 1.0
    assert one_plus_k == P4_ONE_PLUS_K
    assert radius_squared == P4_RADIUS_SQUARED
    assert ratio == n_pre / n_post

    records = R.compile_records(_group(optic), R.canonical_w0(0.55, "df64"), "df64")
    row = records.surf_real[0, 1]
    assert row[LY.SR_CNRZ] == cos_value
    assert row[LY.SR_K1] == one_plus_k
    assert row[LY.SR_R2] == radius_squared


def test_pow_tensor_scalar_values(mps_df64):
    """``x**e`` on a host-resident 0-d tensor equals the Python float64 result."""
    optic = _catalog_optic("CookeTriplet", mps_df64)
    geometry = optic.surfaces.surfaces[1].geometry
    be.metal_reset_stats()
    for exponent in (2, 3, 0.5, -1, -2):
        tensor_result = float(geometry.radius**exponent)
        python_result = float(geometry.radius) ** exponent
        assert tensor_result == python_result, exponent
    assert not [key for key in be.metal_stats() if key.startswith("gpu:")]
    assert float(geometry.radius**2) == P4_RADIUS_SQUARED


@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_flags_per_surface(mps_df64, name):
    """Every flag bit equals the Python predicate it mirrors."""
    optic = _catalog_optic(name, mps_df64)
    w0 = R.canonical_w0(0.55, "df64")
    records = R.compile_records(_group(optic), w0, "df64")
    from optiland.geometries.standard import _is_radius_infinite

    for index, surface in enumerate(optic.surfaces.surfaces[1:], start=1):
        flags = int(records.surf_int[0, index, LY.SI_FLAGS])
        cs = surface.geometry.cs
        assert bool(flags & LY.FL_HAS_RX) == bool(cs.rx), (name, index)
        assert bool(flags & LY.FL_HAS_RY) == bool(cs.ry), (name, index)
        assert bool(flags & LY.FL_HAS_RZ) == bool(cs.rz), (name, index)
        assert bool(flags & LY.FL_REFLECTIVE) == bool(
            surface.interaction_model.is_reflective
        ), (name, index)
        assert bool(flags & LY.FL_HAS_APERTURE) == (surface.aperture is not None)
        assert bool(flags & LY.FL_RADIUS_INF) == bool(
            _is_radius_infinite(surface.geometry.radius)
        ), (name, index)
        k_pre = _float(surface.material_pre.k(be.array([w0])))
        assert bool(flags & LY.FL_ABSORBING) == (k_pre > 0), (name, index)


def test_ap_in_root_matches_distance_capability(mps_df64):
    """``FL_AP_IN_ROOT`` equals what ``_aperture_aware_distance`` would cache."""
    optic = _simple_optic(aperture=RadialAperture(r_max=4.0))
    surface = optic.surfaces.surfaces[1]
    assert getattr(surface, "_distance_capability", None) is None
    records = R.compile_records(_group(optic), R.canonical_w0(0.55, "df64"), "df64")
    flags = int(records.surf_int[0, 1, LY.SI_FLAGS])
    # The compiler must not write the cache the per-op path keeps (design 2.6).
    assert getattr(surface, "_distance_capability", None) is None
    assert flags & LY.FL_AP_IN_ROOT

    optic.trace(Hx=0.0, Hy=0.0, wavelength=0.55, num_rays=8, distribution="line_y")
    cached = surface._distance_capability
    assert cached[0] is surface.geometry
    assert bool(flags & LY.FL_AP_IN_ROOT) == cached[1]

    # A plane short-circuits before the aperture, so the bit stays clear.
    plane_optic = _simple_optic(geometry="plane", aperture=RadialAperture(r_max=4.0))
    plane_records = R.compile_records(
        _group(plane_optic), R.canonical_w0(0.55, "df64"), "df64"
    )
    plane_flags = int(plane_records.surf_int[0, 1, LY.SI_FLAGS])
    assert plane_flags & LY.FL_HAS_APERTURE
    assert not plane_flags & LY.FL_AP_IN_ROOT


# ---------------------------------------------------------------------------
# apertures
# ---------------------------------------------------------------------------
APERTURE_CASES = {
    "radial": (
        RadialAperture(r_max=5.0, r_min=1.0),
        LY.AP_RADIAL,
        (25.0, 1.0, 0.0, 0.0),
    ),
    "radial_inf": (
        RadialAperture(r_max=np.inf, r_min=0.0),
        LY.AP_RADIAL,
        (np.inf, 0.0, 0.0, 0.0),
    ),
    "radial_zero": (
        RadialAperture(r_max=0.0, r_min=0.0),
        LY.AP_RADIAL,
        (0.0, 0.0, 0.0, 0.0),
    ),
    "offset_radial": (
        OffsetRadialAperture(r_max=5.0, r_min=1.0, offset_x=0.25, offset_y=-0.5),
        LY.AP_OFFSET_RADIAL,
        (25.0, 1.0, 0.25, -0.5),
    ),
    "rect": (
        RectangularAperture(-1.0, 2.0, -3.0, 4.0),
        LY.AP_RECT,
        (-1.0, 2.0, -3.0, 4.0),
    ),
    "ellipse": (
        EllipticalAperture(a=3.0, b=1.5, offset_x=0.5, offset_y=0.25),
        LY.AP_ELLIPSE,
        (9.0, 2.25, 0.5, 0.25),
    ),
}


@pytest.mark.parametrize("case", sorted(APERTURE_CASES), ids=lambda c: f"ap={c}")
def test_aperture_encoding(mps_df64, case):
    """Each aperture's four slots are the squares/bounds its ``contains`` uses."""
    aperture, code, params = APERTURE_CASES[case]
    optic = _simple_optic(aperture=aperture)
    records = R.compile_records(_group(optic), R.canonical_w0(0.55, "df64"), "df64")
    row_int = records.surf_int[0, 1]
    row = records.surf_real[0, 1]
    assert int(row_int[LY.SI_APCODE]) == code
    assert int(row_int[LY.SI_FLAGS]) & LY.FL_HAS_APERTURE
    assert tuple(row[LY.SR_AP0 : LY.SR_AP3 + 1]) == params
    # inf survives the encoder with lo = 0 (design 3.3).
    from optiland.backend.torch_backend.metal import encode

    hi, lo = encode.encode_df64(np.asarray(row[LY.SR_AP0 : LY.SR_AP3 + 1]))
    decoded = encode.decode_df64(hi, lo)
    assert np.array_equal(decoded, np.asarray(params), equal_nan=True)
    if np.isinf(params[0]):
        assert lo[0] == 0.0


def test_aperture_denormal_band_is_df64_only(mps_mode):
    """A squared parameter the float32 encoder would flush refuses in df64."""
    optic = _simple_optic(aperture=RadialAperture(r_max=5.0, r_min=1e-20))
    assert (1e-20) ** 2 < 2.0**-126
    result = _gate(optic, _bundle())
    if mps_mode == "df64":
        assert result.reason is FusedTraceSkip.APERTURE_PARAMS
    else:
        assert result.ok is True
    # A representable r_min is accepted in both modes.
    ok_optic = _simple_optic(aperture=RadialAperture(r_max=5.0, r_min=1.0))
    assert _gate(ok_optic, _bundle()).ok is True


def test_gate_refuses_unknown_aperture(mps_df64):
    """A whitelisted-by-exact-type registry refuses anything else."""
    from optiland.physical_apertures.polygon import PolygonAperture

    polygon = PolygonAperture([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    optic = _simple_optic(aperture=polygon)
    assert _gate(optic, _bundle()).reason is FusedTraceSkip.APERTURE_TYPE


# ---------------------------------------------------------------------------
# records: rows, cost, coefficients
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "record,expected_rows,expected_n",
    [
        (True, [0, 1, 2, 3, 4, 5, 6, 7], 8),
        (False, [-1] * 8, 0),
        ("image", [-1, -1, -1, -1, -1, -1, -1, 0], 1),
        ("stop", [-1, -1, -1, -1, 0, -1, -1, -1], 1),
        ([1, 3], [-1, 0, -1, 1, -1, -1, -1, -1], 2),
    ],
    ids=["true", "false", "image", "stop", "explicit"],
)
def test_snap_rows_policy(mps_df64, record, expected_rows, expected_n):
    """The ``record`` policy maps onto one snapshot row per recorded surface."""
    optic = _catalog_optic("CookeTriplet", mps_df64)
    if record == "stop":
        assert optic.surfaces.stop_index == 4
    records = R.compile_records(
        _group(optic), R.canonical_w0(0.55, "df64"), "df64", record=record
    )
    assert records.n_rows == expected_n
    assert list(records.snap_rows[0]) == expected_rows
    assert list(records.surf_int[0, :, LY.SI_SNAPROW]) == expected_rows


def test_step_cost(mps_df64):
    """1 per surface; ``1 + max_iter`` on a Newton row (plan fix L4.5)."""
    cooke = R.compile_records(
        _group(_catalog_optic("CookeTriplet", mps_df64)),
        R.canonical_w0(0.55, "df64"),
        "df64",
    )
    assert list(cooke.step_cost) == [1] * 8
    assert cooke.weighted_steps == 7
    assert cooke.has_newton is False
    assert list(cooke.surf_int[0, :, LY.SI_STEPCOST]) == [1] * 8

    asphere = R.compile_records(
        _group(_simple_optic(geometry="even", max_iter=37)),
        R.canonical_w0(0.55, "df64"),
        "df64",
    )
    assert list(asphere.step_cost) == [1, 38, 1, 1]
    assert asphere.weighted_steps == 40
    assert asphere.has_newton is True
    assert int(asphere.surf_int[0, 1, LY.SI_MAXITER]) == 37

    odd = R.compile_records(
        _group(_simple_optic(geometry="odd", max_iter=12)),
        R.canonical_w0(0.55, "df64"),
        "df64",
    )
    assert int(odd.surf_int[0, 1, LY.SI_GEOM]) == LY.GEOM_ODD
    assert list(odd.step_cost) == [1, 13, 1, 1]


def test_geometry_codes(mps_df64):
    """The five traced codes, chosen by exact type and the radius test."""
    w0 = R.canonical_w0(0.55, "df64")
    codes = {}
    for kind in ("plane", "std_inf", "conic", "even", "odd"):
        records = R.compile_records(_group(_simple_optic(geometry=kind)), w0, "df64")
        codes[kind] = int(records.surf_int[0, 1, LY.SI_GEOM])
    assert codes == {
        "plane": LY.GEOM_PLANE,
        "std_inf": LY.GEOM_STD_INF,
        "conic": LY.GEOM_CONIC,
        "even": LY.GEOM_EVEN,
        "odd": LY.GEOM_ODD,
    }
    assert (
        int(
            R.compile_records(
                _group(_simple_optic(geometry="plane")), w0, "df64"
            ).surf_int[0, 0, LY.SI_GEOM]
        )
        == LY.GEOM_OBJECT
    )


def test_coefficients_after_variable_update(mps_df64):
    """A coefficient list holding 0-d tensors compiles to the same floats."""
    optic = _simple_optic(geometry="even", coefficients=(-2.2e-4, -4.7e-6, -6.4e-8))
    w0 = R.canonical_w0(0.55, "df64")
    before = R.compile_records(_group(optic), w0, "df64")
    assert list(before.coef[0, 1]) == [-2.2e-4, -4.7e-6, -6.4e-8]
    assert int(before.surf_int[0, 1, LY.SI_NCOEFF]) == 3
    assert before.C == 3

    with torch.no_grad():
        optic.updater.set_asphere_coeff(be.array(-3.5e-4), 1, 0)
    coefficients = optic.surfaces.surfaces[1].geometry.coefficients
    assert hasattr(coefficients[0], "ndim")  # a 0-d tensor, not a float
    assert isinstance(coefficients[1], float)

    after = R.compile_records(_group(optic), w0, "df64")
    assert after.coef[0, 1, 0] == _float(coefficients[0])
    assert after.coef[0, 1, 0] == -3.5e-4
    assert list(after.coef[0, 1, 1:]) == [-4.7e-6, -6.4e-8]
    assert _gate(optic, _bundle()).ok is True


def test_compile_records_designs_axis(mps_df64):
    """``designs > 1`` allocates the leading axis and replicates the rows."""
    optic = _catalog_optic("CookeTriplet", mps_df64)
    w0 = R.canonical_w0(0.55, "df64")
    records = R.compile_records(_group(optic), w0, "df64", designs=4)
    assert records.B == 4
    assert records.surf_int.shape == (4, 8, LY.SI_STRIDE)
    assert records.surf_real.shape == (4, 8, LY.SR_STRIDE)
    assert records.snap_rows.shape == (4, 8)
    for b in range(1, 4):
        assert np.array_equal(records.surf_int[b], records.surf_int[0])
        assert np.array_equal(
            records.surf_real[b], records.surf_real[0], equal_nan=True
        )
        assert np.array_equal(records.coef[b], records.coef[0], equal_nan=True)


def test_compile_records_under_strict(mps_mode, monkeypatch):
    """STRICT turns CPU fallbacks into errors; the compiler produces none (1.4)."""
    monkeypatch.setenv("OPTILAND_METAL_STRICT", "1")
    from optiland.backend.torch_backend.metal import tensor as T

    assert T.strict_mode() is True
    optic = _catalog_optic("CookeTriplet", mps_mode)
    rays = _bundle()
    _gate(optic, rays)  # warm
    be.metal_reset_stats()
    result = _gate(optic, rays)
    records = R.compile_records(_group(optic), result.w0, mps_mode)
    stats = dict(be.metal_stats())
    assert records.S == 8
    assert not [key for key in stats if key.startswith("cpu_")], stats


def test_compile_records_cost_per_row(mps_df64):
    """Compilation stays launch-free and near the day-1 P5 measurement."""
    import statistics
    import time

    optic = _catalog_optic("CookeTriplet", mps_df64)
    w0 = R.canonical_w0(0.55, "df64")
    R.compile_records(_group(optic), w0, "df64")  # warm
    be.metal_reset_stats()
    samples = []
    for _ in range(200):
        start = time.perf_counter()
        records = R.compile_records(_group(optic), w0, "df64")
        samples.append(time.perf_counter() - start)
    stats = dict(be.metal_stats())
    assert not [key for key in stats if key.startswith("gpu:")], stats
    per_row = statistics.median(samples) / records.S
    assert per_row <= 3 * P5_ROW_SECONDS, (per_row, P5_ROW_SECONDS)


def test_compile_records_rejects_unknown_geometry(mps_df64):
    """A geometry with no adapter is a gate refusal, not a silent table."""
    optic = _simple_optic()
    surface = optic.surfaces.surfaces[1]
    surface.geometry = ToroidalGeometry(surface.geometry.cs, 25.0, 30.0)
    assert _gate(optic, _bundle()).reason is FusedTraceSkip.GEOMETRY_TYPE
    with pytest.raises(ValueError, match="no geometry adapter"):
        R.compile_records(_group(optic), R.canonical_w0(0.55, "df64"), "df64")


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
def test_memory_bytes_matches_the_plan_table(mps_df64):
    """Plan 3.6's Cooke row, to the byte (day-1 probe P6 measured 665 MiB)."""
    optic = _catalog_optic("CookeTriplet", mps_df64)
    records = R.compile_records(_group(optic), R.canonical_w0(0.55, "df64"), "df64")
    n = 1_000_000
    total = R.memory_bytes(records, n, 1, True)
    expected = (
        9 * 1 * n * 8  # launch, shared
        + 8 * 1 * 8 * n * 8  # snap, all eight rows
        + 11 * 1 * n * 8  # final
        + 2 * 1 * 8 * n  # status + iters
        + 1 * 8 * (32 * 8 + 8 * 4 + 1 * 8)  # tables
    )
    assert total == expected
    assert total / 2**20 == pytest.approx(656.13, abs=0.01)

    # record="image" drops the snap buffer to one row; write_final=False drops
    # the final buffer entirely.
    image = R.compile_records(
        _group(optic), R.canonical_w0(0.55, "df64"), "df64", record="image"
    )
    assert R.memory_bytes(image, n, 1, False) == (
        9 * n * 8 + 8 * 1 * n * 8 + 2 * 8 * n + 8 * (32 * 8 + 8 * 4 + 8)
    )
    # A per-design launch buffer scales with B.
    batch = R.compile_records(
        _group(optic), R.canonical_w0(0.55, "df64"), "df64", record="image", designs=100
    )
    shared = R.memory_bytes(batch, 100_000, 1, False)
    per_design = R.memory_bytes(batch, 100_000, 100, False)
    assert per_design - shared == 9 * 99 * 100_000 * 8


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
def test_registries_are_keyed_by_exact_type():
    """Exact-type keying: a subclass must not inherit its parent's adapter."""
    assert set(A.GEOMETRY_ADAPTERS) == {
        Plane,
        StandardGeometry,
        EvenAsphere,
        OddAsphere,
    }
    for registry in (
        A.GEOMETRY_ADAPTERS,
        A.APERTURE_ADAPTERS,
        A.INTERACTION_ADAPTERS,
    ):
        for cls, adapter in registry.items():
            assert adapter.cls is cls
    # NewtonRaphsonGeometry subclasses StandardGeometry and must not be keyed.
    from optiland.geometries.newton_raphson import NewtonRaphsonGeometry

    assert NewtonRaphsonGeometry not in A.GEOMETRY_ADAPTERS
    assert ToroidalGeometry not in A.GEOMETRY_ADAPTERS


def test_adapter_codes_are_the_frozen_layout_codes():
    """Each adapter carries the layout code the kernel switches on."""
    assert A.GEOMETRY_ADAPTERS[Plane].code == LY.GEOM_PLANE
    assert A.GEOMETRY_ADAPTERS[StandardGeometry].code == LY.GEOM_CONIC
    assert A.GEOMETRY_ADAPTERS[EvenAsphere].code == LY.GEOM_EVEN
    assert {a.code for a in A.APERTURE_ADAPTERS.values()} == {
        LY.AP_RADIAL,
        LY.AP_OFFSET_RADIAL,
        LY.AP_RECT,
        LY.AP_ELLIPSE,
    }


def test_adapters_have_fixtures():
    """Every registered adapter ships at least one conformance fixture.

    ``trace_fixtures.register`` (WP5) replaces each frozen registry entry with
    a copy carrying its fixtures; it is idempotent, so calling it here is safe
    whatever else has already called it.
    """
    filled = trace_fixtures.register(strict=True)
    assert filled == {"geometry": 4, "aperture": 4, "interaction": 1}
    for registry in (
        A.GEOMETRY_ADAPTERS,
        A.APERTURE_ADAPTERS,
        A.INTERACTION_ADAPTERS,
    ):
        for cls, adapter in registry.items():
            assert len(adapter.fixtures) >= 1, cls.__name__
            for builder in adapter.fixtures:
                assert callable(builder), (cls.__name__, builder)


# ---------------------------------------------------------------------------
# coverage the shipped catalog does not provide
# ---------------------------------------------------------------------------
TILT_POSES = {
    "rx": {"rx": 0.1},
    "ry": {"ry": -0.3},
    "rz": {"rz": 1.234567},
    "decenter": {"x": 1.5, "y": -2.25},
    "rxryrz": {"rx": 0.1, "ry": -0.3, "rz": 0.017453292519943295},
}


@pytest.mark.parametrize("pose", sorted(TILT_POSES), ids=lambda p: f"pose={p}")
def test_pose_slots_under_tilt(mps_mode, pose):
    """The 18 pose slots and the three ``HAS_R`` bits, on a tilted surface.

    No shipped sample carries a tilt or a decenter (measured: zero systems in
    the catalog set any of ``rx``, ``ry``, ``rz``), so without this case the
    ``cos(-a)`` / ``cos(+a)`` slot pairs are only ever filled with
    ``cos(0) == 1`` and the localize/globalize distinction is untested.
    """
    values = TILT_POSES[pose]
    optic = _simple_optic()
    cs = optic.surfaces.surfaces[1].geometry.cs
    for name, value in values.items():
        setattr(cs, name, value)

    records = R.compile_records(_group(optic), R.canonical_w0(0.55, mps_mode), mps_mode)
    row = records.surf_real[0, 1]
    flags = int(records.surf_int[0, 1, LY.SI_FLAGS])

    assert row[LY.SR_TX] == float(cs.x) and row[LY.SR_NTX] == float(-cs.x)
    assert row[LY.SR_TY] == float(cs.y) and row[LY.SR_NTY] == float(-cs.y)
    assert row[LY.SR_TZ] == float(cs.z) and row[LY.SR_NTZ] == float(-cs.z)

    # localize: rotate_z(-rz), rotate_y(-ry), rotate_x(-rx)
    assert row[LY.SR_CNRZ] == float(be.cos(-cs.rz))
    assert row[LY.SR_SNRZ] == float(be.sin(-cs.rz))
    assert row[LY.SR_CNRY] == float(be.cos(-cs.ry))
    assert row[LY.SR_SNRY] == float(be.sin(-cs.ry))
    assert row[LY.SR_CNRX] == float(be.cos(-cs.rx))
    assert row[LY.SR_SNRX] == float(be.sin(-cs.rx))
    # globalize: rotate_x(+rx), rotate_y(+ry), rotate_z(+rz)
    assert row[LY.SR_CRX] == float(be.cos(cs.rx))
    assert row[LY.SR_SRX] == float(be.sin(cs.rx))
    assert row[LY.SR_CRY] == float(be.cos(cs.ry))
    assert row[LY.SR_SRY] == float(be.sin(cs.ry))
    assert row[LY.SR_CRZ] == float(be.cos(cs.rz))
    assert row[LY.SR_SRZ] == float(be.sin(cs.rz))

    # The sine slots differ in sign whenever the angle is non-zero, so a slot
    # pair filled from the wrong side of the rotation is visible here.  The
    # cosine pair cannot be: cosine is even, and this libm is exactly even.
    for negated, positive, angle in (
        (LY.SR_SNRX, LY.SR_SRX, cs.rx),
        (LY.SR_SNRY, LY.SR_SRY, cs.ry),
        (LY.SR_SNRZ, LY.SR_SRZ, cs.rz),
    ):
        if float(angle) != 0.0:
            assert row[negated] != row[positive]
        else:
            assert row[negated] == row[positive] == 0.0

    assert bool(flags & LY.FL_HAS_RX) == bool(cs.rx)
    assert bool(flags & LY.FL_HAS_RY) == bool(cs.ry)
    assert bool(flags & LY.FL_HAS_RZ) == bool(cs.rz)
    assert _gate(optic, _bundle()).ok is True


def test_reflective_flag(mps_df64):
    """``FL_REFLECTIVE`` mirrors ``interaction_model.is_reflective``.

    No shipped sample in ``CATALOG`` sets the bit on a surface the record
    compiler reaches, so the flag needs its own fixture.
    """
    optic = _simple_optic(reflective=True)
    surface = optic.surfaces.surfaces[1]
    assert surface.interaction_model.is_reflective is True
    records = R.compile_records(_group(optic), R.canonical_w0(0.55, "df64"), "df64")
    assert int(records.surf_int[0, 1, LY.SI_FLAGS]) & LY.FL_REFLECTIVE
    assert not int(records.surf_int[0, 2, LY.SI_FLAGS]) & LY.FL_REFLECTIVE

    refracting = _simple_optic()
    plain = R.compile_records(_group(refracting), R.canonical_w0(0.55, "df64"), "df64")
    assert not int(plain.surf_int[0, 1, LY.SI_FLAGS]) & LY.FL_REFLECTIVE


def test_absorbing_flag_and_alpha(mps_df64):
    """``FL_ABSORBING`` is ``k_pre > 0`` and ``SR_ALPHA`` is ``(4 pi) k``."""
    optic = _simple_optic()
    optic.surfaces.surfaces[1].material_post = IdealMaterial(1.5, 2.5e-7)
    w0 = R.canonical_w0(0.55, "df64")
    records = R.compile_records(_group(optic), w0, "df64")
    # surface 2 sees the absorbing glass as its material_pre
    row_int = records.surf_int[0, 2]
    row = records.surf_real[0, 2]
    assert int(row_int[LY.SI_FLAGS]) & LY.FL_ABSORBING
    k_pre = _float(optic.surfaces.surfaces[2].material_pre.k(be.array([w0])))
    assert k_pre == 2.5e-7
    assert row[LY.SR_ALPHA] == (4 * np.pi) * k_pre
    # the first surface sits in air: no absorption, alpha is exactly zero
    assert not int(records.surf_int[0, 1, LY.SI_FLAGS]) & LY.FL_ABSORBING
    assert records.surf_real[0, 1, LY.SR_ALPHA] == 0.0


def test_gate_refuses_small_gpu_resident_bundle(mps_mode):
    """A GPU-resident bundle at or under the threshold is still refused.

    ``be.array`` makes anything at or under ``HOST_THRESHOLD`` host-resident, so
    the residency probe alone would never exercise the count rule.  A slice of a
    large GPU-resident bundle stays on the device (a view must alias its base),
    which is the case the ``N <= HOST_THRESHOLD`` clause exists for.
    """
    from optiland.backend.torch_backend.metal.tensor import HOST_THRESHOLD

    assert HOST_THRESHOLD == 256
    optic = _catalog_optic("CookeTriplet", mps_mode)
    big = _bundle()
    sliced = RealRays(
        big.x[:200],
        big.y[:200],
        big.z[:200],
        big.L[:200],
        big.M[:200],
        big.N[:200],
        big.i[:200],
        big.w[:200],
    )
    # ``RealRays.__init__`` seeds ``opd`` with ``zeros_like``, which lands on
    # the host at this size; slice it too so every one of the nine planes is
    # genuinely device-resident and only the count rule can refuse.
    sliced.opd = big.opd[:200]
    assert not any(getattr(sliced, name).is_host_resident for name in R.RAY_ATTRS)
    result = _gate(optic, sliced)
    assert result.ok is False
    assert result.reason is FusedTraceSkip.HOST_RESIDENT
    assert result.n == 200
