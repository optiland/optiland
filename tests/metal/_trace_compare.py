"""WP6 conformance helpers: captures, the tier rules and the status oracle.

This module holds everything the two conformance test files share and nothing
that launches a kernel.  Three groups:

**Captures.**  A :class:`Capture` is a host copy of one trace's result -- the
eight recorded planes of every surface row plus the eleven final planes of the
returned bundle -- kept as *raw components* (the df64 ``hi``/``lo`` float32
words, or the sf64 int64 bit patterns) and never decoded.  Tier A compares
those words; only the external (NumPy) rule decodes.

**Tier rules (plan 7.1).**

* :func:`assert_tier_a` -- ``np.array_equal(..., equal_nan=True)`` on every raw
  component of every plane of every row.  No tolerance exists on this path: a
  single differing word is a mirror bug.  Valid only for ``N > 1024`` in df64,
  which :func:`assert_tier_a` enforces itself (below
  ``BaseMaterial._MAX_VALUE_KEY_ARRAY_SIZE`` the per-op path evaluates the
  dispersion on the whole wavelength array instead of the one-element view the
  kernel's ``w0`` mirrors, so the two paths legitimately use different indices
  -- WP4 finding 2).
* :func:`assert_tier_b` -- site 1 only (``256 < N <= 1024``):
  ``|delta| <= 64 * eps_mode * scale`` per quantity, NaN masks equal, ``i == 0``
  masks equal.
* :func:`assert_vs_numpy` -- the external rule against a NumPy float64 trace of
  the same optic, with the per-aperture rim band on the intensity mask.

**The status oracle.**  :func:`predict_status` is a pure-NumPy interpreter of
the *record tables* that predicts, per ``(surface, ray)``, the exact
``status``/``iters`` planes the kernel must produce.  It is anchored to the R1
recorded rows -- every surface step starts from row ``s - 1`` of the reference
trace, so a prediction error cannot accumulate down the surface list -- and it
re-derives every branch (the conic root selection, the Newton iteration with
its per-ray freeze, the ``nz``/``dF/dt`` floors, the aperture ``contains``, the
refraction radicand) from the mirrored Python sources rather than from the
kernel.  It launches nothing, so it is unit-testable on hand-built records
(``test_trace_kernel.py::test_predict_status_*``) [fix: L2.16].

Deviation from the one-line sketch of plan 7.1, stated here because it changes
what the counts mean.  The sketch defines ``MISS[s]`` as "the *first* row where
position is NaN with finite direction at ``s-1``" and ``TIR[s]`` likewise.  The
kernel raises ``ST_MISS`` on *every* row whose distance is NaN
(``trace.metal::surface_distance``), so a ray that misses -- or that totally
internally reflects, which makes its direction NaN and therefore the next
distance NaN -- carries the bit on every downstream row too.  Predicting only
the first row would under-count every miss fixture by ``S - s`` per ray.
:func:`predict_status` therefore predicts the per-row bit the kernel defines,
and :func:`first_event_rows` exposes the sketch's first-occurrence view for
tests that want it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

from optiland.backend.torch_backend.metal import trace_layout as L
from optiland.backend.torch_backend.metal.tensor import MACHINE_EPS

__all__ = [
    "BIT_NAMES",
    "Capture",
    "EXTERNAL_COS_FACTOR",
    "EXTERNAL_POS_ABS",
    "MODES",
    "RIM_RTOL",
    "StatusPrediction",
    "TIER_A_MIN_RAYS",
    "TIER_B_MAX_RAYS",
    "TIER_B_MIN_RAYS",
    "TIER_FACTOR",
    "assert_raw_equal",
    "assert_tier_a",
    "assert_tier_b",
    "assert_vs_numpy",
    "bit_counts",
    "capture",
    "copy_rays",
    "decode",
    "decoded_rows",
    "first_event_rows",
    "predict_status",
    "raw",
    "rim_band_mask",
    "system_scale",
]

#: The two representations every conformance row runs in (plan 7.2).
MODES: tuple[str, ...] = ("df64", "sf64")

#: Tier A is only defined above ``BaseMaterial._MAX_VALUE_KEY_ARRAY_SIZE``
#: (materials/base.py:107) -- see the module docstring.
TIER_A_MIN_RAYS = 1024

#: Tier-B site 1 of plan 7.1: ``256 < N <= 1024``.
TIER_B_MIN_RAYS = 256
TIER_B_MAX_RAYS = 1024

#: The eps multiplier of the tier-B and external rules (plan 7.1).
TIER_FACTOR = 64.0

#: The external rule's absolute position floor, in mm (plan 7.1).
EXTERNAL_POS_ABS = 1e-11

#: Direction cosines are dimensionless: their external bound is
#: ``64 * eps_mode`` with no scale.
EXTERNAL_COS_FACTOR = 64.0

#: Half-width of the per-aperture rim band, relative to the edge (plan 7.1).
RIM_RTOL = 1e-13

#: The eight recorded planes, in ``S_*`` order (plan 3.2).
SNAP_ATTRS: tuple[str, ...] = ("x", "y", "z", "L", "M", "N", "intensity", "opd")

#: The eleven final planes, in ``F_*`` order (plan 3.2).
FINAL_ATTRS: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "L",
    "M",
    "N",
    "i",
    "opd",
    "L0",
    "M0",
    "N0",
)

#: Status bit -> the name used in a histogram and in a counter suffix.  Same
#: order and same spelling as ``trace._DIAG_BITS``.
BIT_NAMES: tuple[tuple[int, str], ...] = (
    (L.ST_MISS, "miss"),
    (L.ST_TIR, "tir"),
    (L.ST_CLIPPED, "clipped"),
    (L.ST_NEWTON_NOT_CONVERGED, "newton_not_converged"),
    (L.ST_TOL_CROSSOVER, "tol_crossover"),
    (L.ST_NZ_FLOORED, "nz_floored"),
    (L.ST_DF_FLOORED, "df_floored"),
    (L.ST_NONUNIFORM_W, "nonuniform_w"),
)

#: ``std_inf_distance``'s |N| floor (``geometries/standard.py:90``).
N_FLOOR = 1e-14

#: ``_DENOM_EPS_MULTIPLIER`` / ``_CONV_EPS_MULTIPLIER``
#: (``geometries/newton_raphson.py:47,52``), mirrored so the oracle does not
#: import the module it predicts.
DENOM_EPS_MULTIPLIER = 32.0
CONV_EPS_MULTIPLIER = 8.0


# ---------------------------------------------------------------------------
# Raw access
# ---------------------------------------------------------------------------


def raw(value: Any) -> list[np.ndarray]:
    """Host copies of ``value``'s raw components, never decoded.

    A ``MetalFloat64`` yields its ``hi``/``lo`` float32 words (df64) or its
    single int64 bit-pattern buffer (sf64); anything else yields one float64
    array, so the same helper works on a NumPy reference trace.
    """
    components = getattr(value, "components", None)
    if components is None:
        return [np.asarray(value, dtype=np.float64)]
    return [np.asarray(c.detach().cpu().numpy()) for c in components]


def decode(value: Any) -> np.ndarray:
    """``value`` as a float64 host array (decoded; never used by tier A)."""
    import optiland.backend as be

    return np.asarray(be.to_numpy(value), dtype=np.float64)


def copy_rays(rays: Any) -> Any:
    """A second bundle holding the SAME encoded words as ``rays``.

    Component-level ``be.copy``: a ``to_numpy``/``be.array`` round trip
    re-encodes and shifts the df64 low word, which would silently make a
    tier-A comparison compare two different launch bundles (WP1-c finding 4).
    """
    import optiland.backend as be
    from optiland.rays import RealRays

    out = RealRays(
        *(be.copy(getattr(rays, a)) for a in ("x", "y", "z", "L", "M", "N", "i", "w"))
    )
    out.opd = be.copy(rays.opd)
    return out


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capture:
    """One trace's result, as host copies.

    Attributes:
        rows: One dict per surface, ``attr -> list of raw components``.  A
            surface whose snapshot was not recorded contributes an empty dict.
        final: The returned bundle's eleven planes, same encoding.
        n: The bundle size.
        mode: ``'df64'``, ``'sf64'`` or ``'numpy'``.
    """

    rows: tuple[dict[str, list[np.ndarray]], ...]
    final: dict[str, list[np.ndarray]]
    n: int
    mode: str


def capture(group: Any, rays: Any, mode: str) -> Capture:
    """Snapshot ``group``'s recorded rows and ``rays``' final planes."""
    rows: list[dict[str, list[np.ndarray]]] = []
    for surface in group.surfaces:
        row: dict[str, list[np.ndarray]] = {}
        for attr in SNAP_ATTRS:
            value = getattr(surface, attr, None)
            if value is None:
                continue
            components = raw(value)
            if components[0].size == 0:
                continue
            row[attr] = components
        rows.append(row)
    final = {attr: raw(getattr(rays, attr)) for attr in FINAL_ATTRS}
    n = int(final["x"][0].size)
    return Capture(rows=tuple(rows), final=final, n=n, mode=mode)


def decoded_rows(group: Any) -> list[dict[str, np.ndarray]]:
    """``group``'s recorded rows as float64 host arrays, for the oracle."""
    out: list[dict[str, np.ndarray]] = []
    for surface in group.surfaces:
        row: dict[str, np.ndarray] = {}
        for attr in SNAP_ATTRS:
            value = getattr(surface, attr, None)
            if value is None:
                continue
            arr = decode(value)
            if arr.size == 0:
                continue
            row[attr] = arr
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Tier A
# ---------------------------------------------------------------------------


def assert_raw_equal(
    got: Sequence[np.ndarray], ref: Sequence[np.ndarray], what: str
) -> None:
    """Raw-component equality, with a failure message that names the rays."""
    assert len(got) == len(ref), f"{what}: {len(got)} components vs {len(ref)}"
    for c, (g, r) in enumerate(zip(got, ref, strict=True)):
        assert g.shape == r.shape, f"{what}: component {c} {g.shape} vs {r.shape}"
        if np.array_equal(g, r, equal_nan=True):
            continue
        same = (np.isnan(g) & np.isnan(r)) | (g == r)
        bad = np.flatnonzero(~same.reshape(-1))
        raise AssertionError(
            f"{what}: component {c} differs on {bad.size}/{g.size} entries; "
            f"first indices {bad[:8].tolist()}; "
            f"got {g.reshape(-1)[bad[:4]].tolist()} "
            f"vs {r.reshape(-1)[bad[:4]].tolist()}"
        )


def assert_tier_a(got: Capture, ref: Capture, what: str) -> None:
    """Plan 7.1 tier A: raw-component equality on every row and plane."""
    assert got.mode == ref.mode, f"{what}: mode {got.mode} vs {ref.mode}"
    assert got.n == ref.n, f"{what}: {got.n} rays vs {ref.n}"
    if got.mode == "df64":
        assert got.n > TIER_A_MIN_RAYS, (
            f"{what}: tier A needs N > {TIER_A_MIN_RAYS} in df64 "
            f"(BaseMaterial._MAX_VALUE_KEY_ARRAY_SIZE), got {got.n}"
        )
    assert len(got.rows) == len(ref.rows), (
        f"{what}: {len(got.rows)} surface rows vs {len(ref.rows)}"
    )
    for s, (g_row, r_row) in enumerate(zip(got.rows, ref.rows, strict=True)):
        assert set(g_row) == set(r_row), (
            f"{what}: surface {s} recorded {sorted(g_row)} vs {sorted(r_row)}"
        )
        for attr in sorted(g_row):
            assert_raw_equal(g_row[attr], r_row[attr], f"{what}: surface {s}.{attr}")
    for attr in FINAL_ATTRS:
        assert_raw_equal(got.final[attr], ref.final[attr], f"{what}: rays.{attr}")


# ---------------------------------------------------------------------------
# Tier B and the external rule
# ---------------------------------------------------------------------------


def system_scale(optic: Any, rows: Sequence[dict[str, np.ndarray]]) -> float:
    """``max|finite positions| + max|x, y|``, as ``metal_oracle_e2e`` computes it."""
    positions = []
    transverse = []
    for row in rows:
        for attr in ("x", "y", "z"):
            arr = row.get(attr)
            if arr is None:
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size:
                positions.append(float(np.max(np.abs(finite))))
                if attr in ("x", "y"):
                    transverse.append(float(np.max(np.abs(finite))))
    del optic
    return (max(positions) if positions else 0.0) + (
        max(transverse) if transverse else 0.0
    )


def _abs_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``|a - b|`` where both are finite, 0 where both are NaN/inf alike."""
    both = np.isfinite(a) & np.isfinite(b)
    out = np.zeros(a.shape, dtype=np.float64)
    out[both] = np.abs(a[both] - b[both])
    return out


def _assert_nan_masks(a: np.ndarray, b: np.ndarray, what: str) -> None:
    bad = np.flatnonzero(np.isnan(a) != np.isnan(b))
    assert bad.size == 0, (
        f"{what}: NaN patterns differ on {bad.size} rays; first {bad[:8].tolist()}"
    )


def assert_tier_b(
    got: Sequence[dict[str, np.ndarray]],
    ref: Sequence[dict[str, np.ndarray]],
    *,
    mode: str,
    scale: float,
    what: str,
) -> None:
    """Plan 7.1 tier B (site 1 only): an eps-derived bound, never a widened one.

    ``got``/``ref`` are decoded rows.  The bound is
    ``64 * MACHINE_EPS[mode] * scale`` for positions and OPD and
    ``64 * MACHINE_EPS[mode]`` for the direction cosines; the NaN masks and the
    ``i == 0`` masks must be equal exactly.
    """
    eps = MACHINE_EPS[mode]
    pos_bound = TIER_FACTOR * eps * scale
    cos_bound = TIER_FACTOR * eps
    for s, (g_row, r_row) in enumerate(zip(got, ref, strict=True)):
        for attr in sorted(set(g_row) & set(r_row)):
            g, r = g_row[attr], r_row[attr]
            _assert_nan_masks(g, r, f"{what}: surface {s}.{attr}")
            if attr == "intensity":
                bad = np.flatnonzero((g == 0.0) != (r == 0.0))
                assert bad.size == 0, (
                    f"{what}: surface {s}: i == 0 masks differ on {bad.size} "
                    f"rays; first {bad[:8].tolist()}"
                )
                continue
            bound = cos_bound if attr in ("L", "M", "N") else pos_bound
            delta = _abs_delta(g, r)
            worst = float(delta.max()) if delta.size else 0.0
            assert worst <= bound, (
                f"{what}: surface {s}.{attr}: max |delta| {worst:.3e} > "
                f"{bound:.3e} (= {TIER_FACTOR} * eps[{mode}] * scale)"
            )


def rim_band_mask(
    aperture: Any, x: np.ndarray, y: np.ndarray, *, rtol: float = RIM_RTOL
) -> np.ndarray:
    """Rays within ``rtol`` of a finite aperture edge (plan 7.1).

    The only place the external rule allows an intensity-mask difference: a
    ray whose distance to the nearest *finite* edge is ``<= rtol * edge`` may
    land on either side of the inclusive bound after two independently rounded
    traces.  Infinite edges are skipped, so an annulus with ``r_max = inf``
    contributes only its ``r_min`` band.
    """
    band = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    if aperture is None:
        return band
    name = type(aperture).__name__
    if name in ("RadialAperture", "OffsetRadialAperture"):
        dx = x - float(getattr(aperture, "offset_x", 0.0) or 0.0)
        dy = y - float(getattr(aperture, "offset_y", 0.0) or 0.0)
        r = np.sqrt(dx * dx + dy * dy)
        for edge in (float(aperture.r_max), float(aperture.r_min)):
            if np.isfinite(edge) and edge != 0.0:
                band |= np.abs(r - edge) <= rtol * abs(edge)
    elif name == "RectangularAperture":
        for value, edge in (
            (x, float(aperture.x_min)),
            (x, float(aperture.x_max)),
            (y, float(aperture.y_min)),
            (y, float(aperture.y_max)),
        ):
            if np.isfinite(edge) and edge != 0.0:
                band |= np.abs(value - edge) <= rtol * abs(edge)
    elif name == "EllipticalAperture":
        a = float(aperture.a)
        b = float(aperture.b)
        dx = x - float(getattr(aperture, "offset_x", 0.0) or 0.0)
        dy = y - float(getattr(aperture, "offset_y", 0.0) or 0.0)
        if np.isfinite(a) and np.isfinite(b) and a != 0.0 and b != 0.0:
            rho = np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
            band |= np.abs(rho - 1.0) <= rtol
    return band


def assert_vs_numpy(
    got: Sequence[dict[str, np.ndarray]],
    ref: Sequence[dict[str, np.ndarray]],
    *,
    mode: str,
    scale: float,
    opl_scale: float,
    rim: Sequence[np.ndarray] | None,
    what: str,
) -> None:
    """Plan 7.1's external rule, against a NumPy float64 trace (R2).

    ``rim`` is one boolean mask per surface marking the rays inside that
    surface's aperture rim band; the intensity-mask comparison skips them and
    only them.
    """
    eps = MACHINE_EPS[mode]
    pos_bound = max(EXTERNAL_POS_ABS, TIER_FACTOR * eps * scale)
    opd_bound = max(EXTERNAL_POS_ABS, TIER_FACTOR * eps * opl_scale)
    cos_bound = EXTERNAL_COS_FACTOR * eps
    for s, (g_row, r_row) in enumerate(zip(got, ref, strict=True)):
        for attr in sorted(set(g_row) & set(r_row)):
            g, r = g_row[attr], r_row[attr]
            _assert_nan_masks(g, r, f"{what}: surface {s}.{attr}")
            if attr == "intensity":
                differ = (g == 0.0) != (r == 0.0)
                if rim is not None:
                    differ &= ~rim[s]
                bad = np.flatnonzero(differ)
                assert bad.size == 0, (
                    f"{what}: surface {s}: i == 0 masks differ outside the rim "
                    f"band on {bad.size} rays; first {bad[:8].tolist()}"
                )
                continue
            if attr == "opd":
                bound = opd_bound
            elif attr in ("L", "M", "N"):
                bound = cos_bound
            else:
                bound = pos_bound
            delta = _abs_delta(g, r)
            worst = float(delta.max()) if delta.size else 0.0
            assert worst <= bound, (
                f"{what}: surface {s}.{attr}: max |delta| {worst:.3e} > {bound:.3e}"
            )


# ---------------------------------------------------------------------------
# The status oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusPrediction:
    """The predicted ``status``/``iters`` planes for one design.

    Attributes:
        bits: ``uint8[S][N]`` -- the status word per surface row and ray.
        iters: ``uint8[S][N]`` -- the Newton iteration count per row and ray.
        clip_uncertain: ``bool[S][N]`` -- rays inside an aperture rim band,
            whose ``ST_CLIPPED`` bit the float64 oracle cannot decide (plan
            7.1); excluded from the exact ``clipped`` count.
        counts: bit name -> number of ``(s, i)`` entries carrying it.
    """

    bits: np.ndarray
    iters: np.ndarray
    clip_uncertain: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)


def bit_counts(status: np.ndarray) -> dict[str, int]:
    """The per-bit histogram of a ``status`` plane, in ``BIT_NAMES`` order."""
    arr = np.asarray(status, dtype=np.uint8)
    return {name: int(np.count_nonzero(arr & bit)) for bit, name in BIT_NAMES}


def first_event_rows(rows: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Plan 7.1's first-occurrence view of the miss / TIR events.

    ``miss[i]`` is the first surface row whose position is NaN while the
    direction at ``s - 1`` is finite, ``tir[i]`` the first row whose direction
    is NaN while its position is finite, and ``-1`` where the ray has neither.
    This is the sketch of plan 7.1; the per-row bit the kernel writes is
    :func:`predict_status`'s (see the module docstring).
    """
    n = int(rows[0]["x"].size)
    miss = np.full(n, -1, dtype=np.int64)
    tir = np.full(n, -1, dtype=np.int64)
    for s in range(1, len(rows)):
        pos_nan = np.isnan(rows[s]["x"]) | np.isnan(rows[s]["y"])
        dir_prev = (
            np.isfinite(rows[s - 1]["L"])
            & np.isfinite(rows[s - 1]["M"])
            & np.isfinite(rows[s - 1]["N"])
        )
        hit = pos_nan & dir_prev & (miss < 0)
        miss[hit] = s
        dir_nan = np.isnan(rows[s]["L"]) | np.isnan(rows[s]["M"])
        pos_ok = np.isfinite(rows[s]["x"]) & np.isfinite(rows[s]["y"])
        hit = dir_nan & pos_ok & (tir < 0)
        tir[hit] = s
    return {"miss": miss, "tir": tir}


def _sign(a: np.ndarray) -> np.ndarray:
    """``be.sign`` with ``sign(0) == 0`` and NaN preserved (real_rays.py:571)."""
    out = np.sign(a)
    out[np.isnan(a)] = np.nan
    return out


def _pow_scalar(x: np.ndarray, e: float) -> np.ndarray:
    """``pow_scalar`` of the kernel: the exact fast paths, then ``pow``."""
    if e == 0.0:
        return np.ones_like(x)
    if e == 1.0:
        return x
    if e == 2.0:
        return x * x
    if e == 3.0:
        return (x * x) * x
    if e == 0.5:
        return np.sqrt(x)
    if e == -1.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / x
    if e == -2.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / (x * x)
    if e == -0.5:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.sqrt(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(x, e)


def _rot_z(x, y, ell, m, c, s):
    return (x * c - y * s, x * s + y * c, ell * c - m * s, ell * s + m * c)


def _rot_y(x, z, ell, n, c, s):
    return (x * c + z * s, -x * s + z * c, ell * c + n * s, -ell * s + n * c)


def _rot_x(y, z, m, n, c, s):
    return (y * c - z * s, y * s + z * c, m * c - n * s, m * s + n * c)


def _localize(state: dict[str, np.ndarray], p: np.ndarray, flags: int) -> dict:
    """``CoordinateSystem.localize`` (coordinate_system.py:127-143)."""
    r = dict(state)
    r["x"] = r["x"] + p[L.SR_NTX]
    r["y"] = r["y"] + p[L.SR_NTY]
    r["z"] = r["z"] + p[L.SR_NTZ]
    if flags & L.FL_HAS_RZ:
        r["x"], r["y"], r["L"], r["M"] = _rot_z(
            r["x"], r["y"], r["L"], r["M"], p[L.SR_CNRZ], p[L.SR_SNRZ]
        )
    if flags & L.FL_HAS_RY:
        r["x"], r["z"], r["L"], r["N"] = _rot_y(
            r["x"], r["z"], r["L"], r["N"], p[L.SR_CNRY], p[L.SR_SNRY]
        )
    if flags & L.FL_HAS_RX:
        r["y"], r["z"], r["M"], r["N"] = _rot_x(
            r["y"], r["z"], r["M"], r["N"], p[L.SR_CNRX], p[L.SR_SNRX]
        )
    return r


def _conic_radicand(r2: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - (p[L.SR_K1] * r2) / p[L.SR_R2]


def _conic_sag(r2: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(_conic_radicand(r2, p))
        return r2 / (p[L.SR_R] * (s + 1.0))


def _sag_of(geom, x, y, p, coef, ncoef):
    """``geometry.sag`` in the local frame (design 4.7)."""
    if geom == L.GEOM_PLANE:
        return np.zeros_like(y)
    r2 = x * x + y * y
    if geom == L.GEOM_ODD:
        r = np.sqrt(r2)
        z = _conic_sag(r2, p)
        for i in range(ncoef):
            z = z + _pow_scalar(r, float(i + 1)) * coef[i]
        return z
    z = _conic_sag(r2, p)
    if geom == L.GEOM_EVEN:
        for i in range(ncoef):
            z = z + _pow_scalar(r2, float(i + 1)) * coef[i]
    return z


def _normal_of(geom, x, y, p, coef, ncoef):
    """The normalised surface normal at a local ``(x, y)`` on the surface."""
    if geom == L.GEOM_PLANE:
        zero = np.zeros_like(x)
        return zero, zero, np.ones_like(x)
    r2 = x * x + y * y
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = p[L.SR_R] * np.sqrt(_conic_radicand(r2, p))
        dfdx = x / denom
        dfdy = y / denom
    if geom == L.GEOM_EVEN:
        for i in range(ncoef):
            ci = coef[i]
            pw = _pow_scalar(r2, float(i))
            m = float(2 * (i + 1))
            dfdx = dfdx + ((x * m) * ci) * pw
            dfdy = dfdy + ((y * m) * ci) * pw
    elif geom == L.GEOM_ODD:
        r = np.sqrt(r2)
        for i in range(ncoef):
            ci = coef[i]
            pw = _pow_scalar(r, float(i - 1))
            m = float(i + 1)
            xt = ((x * m) * ci) * pw
            yt = ((y * m) * ci) * pw
            dfdx = dfdx + np.where(np.isfinite(xt), xt, 0.0)
            dfdy = dfdy + np.where(np.isfinite(yt), yt, 0.0)
    mag = np.sqrt((dfdx * dfdx + dfdy * dfdy) + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dfdx / mag, dfdy / mag, (1.0 / mag) * -1.0


def _ap_contains(code, p, x, y):
    """``aperture.contains`` in the local frame (design 4.9), NaN-false."""
    with np.errstate(invalid="ignore"):
        if code == L.AP_RADIAL:
            r2 = x * x + y * y
            return (r2 <= p[L.SR_AP0]) & (r2 >= p[L.SR_AP1])
        if code == L.AP_OFFSET_RADIAL:
            dx = x - p[L.SR_AP2]
            dy = y - p[L.SR_AP3]
            r2 = dx * dx + dy * dy
            return (r2 <= p[L.SR_AP0]) & (r2 >= p[L.SR_AP1])
        if code == L.AP_RECT:
            return (
                (x >= p[L.SR_AP0])
                & (x <= p[L.SR_AP1])
                & (y >= p[L.SR_AP2])
                & (y <= p[L.SR_AP3])
            )
        if code == L.AP_ELLIPSE:
            dx = x - p[L.SR_AP2]
            dy = y - p[L.SR_AP3]
            return ((dx * dx) / p[L.SR_AP0] + (dy * dy) / p[L.SR_AP1]) <= 1.0
    return np.ones(x.shape, dtype=bool)


def _conic_candidates(x, y, z, ell, m, n, radius, conic, eps):
    """``conic_candidates_body`` (kernels/conic.metal:55-96) in NumPy."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        k1 = 1.0 + conic
        kz = k1 * z
        transverse = x * x + y * y
        transverse_direction = ell * ell + m * m
        n2 = n * n
        a = transverse_direction + k1 * n2
        b = 2.0 * (ell * x + m * y + n * (kz - radius))
        c = transverse + z * (kz - 2.0 * radius)
        residual_scale = transverse + np.abs(z) * (np.abs(kz) + 2.0 * abs(radius))
        roundoff = 4.0 * eps
        resolved_c = np.abs(c) > roundoff * residual_scale
        d = b * b - 4.0 * a * c
        d_ok = d >= 0.0
        positive_d = d > 0.0
        sqrt_d = np.where(positive_d, np.sqrt(np.where(positive_d, d, 0.0)), 0.0)
        q = -0.5 * (b + np.copysign(sqrt_d, b))
        a_ok = (a != 0.0) & np.isfinite(a)
        q_ok = (q != 0.0) & np.isfinite(q)
        t1 = q / np.where(a_ok, a, 1.0)
        t2 = c / np.where(q_ok, q, 1.0)
        resolved_step = t2 * t2 * (transverse_direction + n2) > (
            roundoff * roundoff * (transverse + z * z)
        )
        solvable1 = d_ok & a_ok & np.isfinite(t1)
        solvable2 = d_ok & q_ok & np.isfinite(t2)
        z1 = z + t1 * n
        z2 = z + t2 * n
        valid1 = solvable1 & (t1 > 0.0) & ((1.0 - k1 * z1 / radius) >= 0.0)
        valid2 = (
            solvable2
            & (resolved_c | resolved_step)
            & (t2 > 0.0)
            & ((1.0 - k1 * z2 / radius) >= 0.0)
        )
        az2 = np.where(solvable2, np.abs(z2), np.inf)
        az1 = np.where(solvable1, np.abs(z1), np.inf)
        vertex2 = az2 < az1
        pick2 = valid2 | np.where(valid1, False, vertex2)
        solvable = solvable1 | solvable2
    return t1, t2, valid1, valid2, pick2, solvable


def _select_distance(x, y, z, ell, m, n, p, eps, flags, apcode):
    """``_select_distance`` (backend/_conic.py:113-126) with the aperture rule."""
    t1, t2, valid1, valid2, pick2, solvable = _conic_candidates(
        x, y, z, ell, m, n, p[L.SR_R], p[L.SR_K], eps
    )
    if flags & L.FL_AP_IN_ROOT:
        pref1 = valid1 & _ap_contains(apcode, p, x + t1 * ell, y + t1 * m)
        pref2 = valid2 & _ap_contains(apcode, p, x + t2 * ell, y + t2 * m)
        pick2 = pref2 | np.where(pref1, False, pick2)
    return np.where(solvable, np.where(pick2, t2, t1), np.nan)


def _std_inf_distance(z, n, bits):
    """``standard.py:88-90``: a POSITIVE floor for both signs of ``N``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ns = np.where(np.abs(n) > N_FLOOR, n, N_FLOOR)
        bits |= np.where(ns != n, L.ST_NZ_FLOORED, 0).astype(np.uint8)
        return -z / ns


def _newton_distance(
    x, y, z, ell, m, n, p, coef, geom, ncoef, max_iter, flags, apcode, eps, bits, iters
):
    """``_solve_distance_primal`` (newton_raphson.py:317-375), per ray.

    The batch ``if be.all(converged): break`` plus the ``be.where(converged, 0,
    step)`` freeze is, per ray, "stop as soon as I converge", so ``iters`` is
    the ray's own count (design 4.12).  A non-finite ``t`` stops the ray too.
    """
    if flags & L.FL_RADIUS_INF:
        t = _std_inf_distance(z, n, bits)
    else:
        t = _select_distance(x, y, z, ell, m, n, p, eps, flags, apcode)

    with np.errstate(invalid="ignore"):
        atol = np.abs(t)
        floor_tol = (8.0 * eps) * np.where(atol > 1.0, atol, 1.0)
        crossed = floor_tol > p[L.SR_TOL]
    bits |= np.where(crossed, L.ST_TOL_CROSSOVER, 0).astype(np.uint8)
    tol = np.where(crossed, floor_tol, p[L.SR_TOL])

    with np.errstate(invalid="ignore"):
        f_t = _sag_of(geom, x + t * ell, y + t * m, p, coef, ncoef) - (z + t * n)
        conv = np.abs(f_t) < tol
    tau = DENOM_EPS_MULTIPLIER * eps
    active = ~conv & np.isfinite(t)
    for _ in range(max_iter):
        if not active.any():
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            nx, ny, nz = _normal_of(geom, x + t * ell, y + t * m, p, coef, ncoef)
            floored = ~(np.abs(nz) > tau)
            nzs = np.where(floored, np.where(nz >= 0.0, tau, -tau), nz)
            bits |= np.where(active & (nzs != nz), L.ST_NZ_FLOORED, 0).astype(np.uint8)
            fx = -nx / nzs
            fy = -ny / nzs
            fxl = fx * ell
            fym = fy * m
            df = (fxl + fym) - n
            scale = (np.abs(fxl) + np.abs(fym)) + np.abs(n)
            tau_s = np.maximum(scale, 1.0) * tau
            near = np.abs(df) <= tau_s
            bits |= np.where(active & near, L.ST_DF_FLOORED, 0).astype(np.uint8)
            safe = np.where(near, np.where(df >= 0.0, tau_s, -tau_s), df)
            step = f_t / safe
            t = np.where(active, t - step, t)
            new_f = _sag_of(geom, x + t * ell, y + t * m, p, coef, ncoef) - (z + t * n)
            f_t = np.where(active, new_f, f_t)
            conv = np.abs(f_t) < tol
        iters += active.astype(np.uint8)
        active = active & ~conv & np.isfinite(t)
    with np.errstate(invalid="ignore"):
        bits |= np.where(~conv & np.isfinite(t), L.ST_NEWTON_NOT_CONVERGED, 0).astype(
            np.uint8
        )
    return t


def _surface_distance(
    geom, r, p, coef, ncoef, max_iter, flags, apcode, eps, bits, iters
):
    """The ``SI_GEOM`` switch of design 4.6, then the ``MISS`` bit."""
    x, y, z = r["x"], r["y"], r["z"]
    ell, m, n = r["L"], r["M"], r["N"]
    if geom == L.GEOM_PLANE:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -z / n
    elif geom == L.GEOM_STD_INF:
        t = _std_inf_distance(z, n, bits)
    elif geom == L.GEOM_CONIC:
        t = _select_distance(x, y, z, ell, m, n, p, eps, flags, apcode)
    else:
        t = _newton_distance(
            x,
            y,
            z,
            ell,
            m,
            n,
            p,
            coef,
            geom,
            ncoef,
            max_iter,
            flags,
            apcode,
            eps,
            bits,
            iters,
        )
    bits |= np.where(np.isnan(t), L.ST_MISS, 0).astype(np.uint8)
    return t


def _interact_bits(ell0, m0, n0, nx, ny, nz, reflective, u2, bits):
    """The TIR bit of ``interact`` (design 4.8): a NaN radicand at finite dot."""
    if reflective:
        return
    with np.errstate(invalid="ignore"):
        dot = np.abs((ell0 * nx + m0 * ny) + n0 * nz)
        root = np.sqrt(1.0 - u2 * (1.0 - dot * dot))
        tir = np.isnan(root) & np.isfinite(dot)
    bits |= np.where(tir, L.ST_TIR, 0).astype(np.uint8)


def predict_status(
    rows: Sequence[dict[str, np.ndarray]],
    optic: Any,
    *,
    mode: str,
    records: Any,
    design: int = 0,
    rim_rtol: float = RIM_RTOL,
) -> StatusPrediction:
    """Predict the ``status``/``iters`` planes of a fused trace, in pure NumPy.

    Args:
        rows: The R1 recorded rows (decoded float64 host arrays, global
            frame), one dict per surface, as :func:`decoded_rows` returns.
            Every surface must be recorded: step ``s`` starts from row
            ``s - 1``, which is what keeps the prediction anchored.
        optic: The optic whose surface group produced ``rows``.  Used only to
            compile the record tables and to read the apertures for the rim
            band.
        mode: ``'df64'`` or ``'sf64'`` -- selects ``MACHINE_EPS``.
        records: A pre-compiled ``TraceRecords`` to reuse; compiled from
            ``optic`` when None.
        design: Which design row of ``records`` to read.
        rim_rtol: Half-width of the aperture rim band.

    Returns:
        StatusPrediction: per-``(s, i)`` bits, Newton iteration counts, the
        rim-band mask and the histogram.
    """
    group = optic.surfaces
    surfaces = list(group.surfaces)
    s_count = len(surfaces)
    assert len(rows) == s_count, f"{len(rows)} rows for {s_count} surfaces"
    n = int(rows[0]["x"].size)

    si = np.asarray(records.surf_int)[design]
    sr = np.asarray(records.surf_real)[design]
    cf = np.asarray(records.coef)[design]

    eps = MACHINE_EPS[mode]
    bits = np.zeros((s_count, n), dtype=np.uint8)
    iters = np.zeros((s_count, n), dtype=np.uint8)
    uncertain = np.zeros((s_count, n), dtype=bool)

    for s in range(1, s_count):
        geom = int(si[s, L.SI_GEOM])
        flags = int(si[s, L.SI_FLAGS])
        ncoef = int(si[s, L.SI_NCOEFF])
        max_iter = int(si[s, L.SI_MAXITER])
        apcode = int(si[s, L.SI_APCODE])
        p = sr[s]
        coef = cf[s]

        prev = rows[s - 1]
        state = {k: np.asarray(prev[k], dtype=np.float64).copy() for k in "xyzLMN"}
        r = _localize(state, p, flags)

        row_bits = bits[s]
        row_iters = iters[s]
        t = _surface_distance(
            geom, r, p, coef, ncoef, max_iter, flags, apcode, eps, row_bits, row_iters
        )
        with np.errstate(invalid="ignore"):
            hx = r["x"] + t * r["L"]
            hy = r["y"] + t * r["M"]

        if flags & L.FL_HAS_APERTURE:
            inside = _ap_contains(apcode, p, hx, hy)
            row_bits |= np.where(~inside, L.ST_CLIPPED, 0).astype(np.uint8)
            uncertain[s] = rim_band_mask(surfaces[s].aperture, hx, hy, rtol=rim_rtol)

        nx, ny, nz = _normal_of(geom, hx, hy, p, coef, ncoef)
        _interact_bits(
            r["L"],
            r["M"],
            r["N"],
            nx,
            ny,
            nz,
            bool(flags & L.FL_REFLECTIVE),
            p[L.SR_U2],
            row_bits,
        )

    return StatusPrediction(
        bits=bits,
        iters=iters,
        clip_uncertain=uncertain,
        counts=bit_counts(bits),
    )


def compile_tables(optic: Any, mode: str, w0: float, *, record: bool = True) -> Any:
    """``compile_records`` for ``optic``'s group, for :func:`predict_status`."""
    from optiland.backend.torch_backend.metal import trace_record as tr

    return tr.compile_records(optic.surfaces, w0, mode, record=record)


def env(**values: str | None) -> dict[str, str | None]:
    """Snapshot ``values``' current environment and apply them (test helper)."""
    previous = {k: os.environ.get(k) for k in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def restore(previous: dict[str, str | None]) -> None:
    """Undo :func:`env`."""
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
