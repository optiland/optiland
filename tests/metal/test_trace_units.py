"""WP1 unit tests for ``kernels/trace.metal``, per surface-pipeline function.

Every comparison is on RAW COMPONENTS (``hi``/``lo`` float32 words in df64,
int64 bit patterns in sf64) with ``np.array_equal(..., equal_nan=True)``: plan
7.1 tier A is component equality, and day-1 probe P1 showed that comparing
decoded float64 "fails" at the 2**-48 level for inputs that round-tripped
perfectly.

Three layers, in order of what they can catch:

1. per-function probes against the live Python object (localize/globalize,
   ``contains``, ``interact``, ``pow_scalar``, the propagate/absorb/OPD chain,
   sag/normal, the distance switch and the Newton solver);
2. the assembled ``trace_body`` against the per-op path over the WP5 fixture
   set -- record for record, plane for plane, both modes (plan 7.1 tier A);
3. the contract layers around them: the object-row / sentinel rules of plan
   3.2, the slab addressing of design 4.4, the amalgamation include order, the
   ``lit()`` / bare-literal discipline of design 4.2, the measured
   scalar-operand side of the per-op kernels, and the round-0 divergence hooks
   of plan section 6 (each certified to be able to break the comparison).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - CPU-only machine
    pytest.skip("Metal GPU required", allow_module_level=True)

import contextlib  # noqa: E402

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend.metal import encode, trace_layout  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import MetalFloat64  # noqa: E402
from optiland.backend.torch_backend.metal.trace_record import (  # noqa: E402
    compile_records,
)
from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.materials import IdealMaterial  # noqa: E402
from optiland.physical_apertures import (  # noqa: E402
    EllipticalAperture,
    OffsetRadialAperture,
    RadialAperture,
    RectangularAperture,
)
from optiland.propagation import HomogeneousPropagation  # noqa: E402
from optiland.rays.real_rays import RealRays  # noqa: E402
from tests.metal import _trace_probe_lib as tp  # noqa: E402

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import trace_fixtures  # noqa: E402

MODES = list(tp.MODES)

#: ``(RealRays/Surface attribute, snapshot plane)`` in record order.
_SNAP_PLANES: tuple[tuple[str, int], ...] = (
    ("x", trace_layout.S_X),
    ("y", trace_layout.S_Y),
    ("z", trace_layout.S_Z),
    ("L", trace_layout.S_L),
    ("M", trace_layout.S_M),
    ("N", trace_layout.S_N),
    ("intensity", trace_layout.S_I),
    ("opd", trace_layout.S_OPD),
)

#: ``(RealRays attribute, final plane)``; the last three are the local-frame
#: pre-interaction direction the last surface left on the bundle.
_FINAL_PLANES: tuple[tuple[str, int], ...] = (
    ("x", trace_layout.F_X),
    ("y", trace_layout.F_Y),
    ("z", trace_layout.F_Z),
    ("L", trace_layout.F_L),
    ("M", trace_layout.F_M),
    ("N", trace_layout.F_N),
    ("i", trace_layout.F_I),
    ("opd", trace_layout.F_OPD),
    ("L0", trace_layout.F_L0),
    ("M0", trace_layout.F_M0),
    ("N0", trace_layout.F_N0),
)


# ---------------------------------------------------------------------------
# Object-row contract (plan 3.2) and the launch -> snap/final copy of the stub
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tables", ["blank", "cooke"])
def test_object_row_written(mode: str, tables: str) -> None:
    """Every ``(b, s, i)`` is visited, object row included; no sentinel survives.

    The host pre-fills ``iters`` with ``ITERS_UNWRITTEN`` (255).  Plan 3.2
    makes the kernel responsible for writing ``status = 0`` and ``iters = 0``
    at ``s = 0`` explicitly -- ``ObjectSurface.trace`` is reset + record with
    no physics (object_surface.py:66-83) -- so a surviving 255 anywhere is an
    unvisited thread, exactly what P9 showed a silently aborted command buffer
    produces.  Run against the placeholder-free body twice: on hand-built
    tables and on a REAL compiled record set, where the loop actually refracts
    through eight surfaces.
    """
    B, S, N = 2, 4, 1024
    lib = tp.trace_library(mode)
    if tables == "blank":
        surf_int, surf_real, coef, n_rows = tp.blank_tables(B, S)
        out = tp.run_trace(
            lib,
            mode,
            launch=tp.random_launch(N),
            surf_int=surf_int,
            surf_real=surf_real,
            coef=coef,
            n_rows=n_rows,
            write_final=True,
        )
    else:
        with metal_backend(mode):
            optic, rays = _build_fixture("cooke", N)
            n = int(np.asarray(be.to_numpy(rays.x)).reshape(-1).size)
            records = compile_records(
                optic.surfaces, _scalar(rays.w), mode, record=True
            )
            B, S = records.B, records.S
            N = n
            out = tp.run_trace(
                lib,
                mode,
                launch_bufs=tp.pack_rays(rays, mode, n),
                surf_int=records.surf_int,
                surf_real=records.surf_real,
                coef=records.coef,
                n_rows=records.n_rows,
                write_final=True,
            )
    status, iters = out["status"], out["iters"]
    assert status.shape == (B, S, N)
    assert iters.shape == (B, S, N)
    assert np.array_equal(status[:, 0, :], np.zeros((B, N), dtype=np.uint8))
    assert np.array_equal(iters[:, 0, :], np.zeros((B, N), dtype=np.uint8))
    assert int((iters == trace_layout.ITERS_UNWRITTEN).sum()) == 0
    if tables == "blank":
        # Zeroed planes: no miss, no clip, no TIR, no Newton row.
        assert int(status.sum()) == 0
        assert int(iters.sum()) == 0


@pytest.mark.parametrize("mode", MODES)
def test_unvisited_sentinel_survives_a_short_dispatch(mode: str) -> None:
    """The sentinel really detects unvisited threads (the check is not vacuous).

    Dispatching half the rays must leave exactly ``B * S * N / 2`` entries at
    ``ITERS_UNWRITTEN``.  Without this, ``test_object_row_written`` would pass
    even if the kernel wrote nothing and the host pre-fill were broken.
    """
    B, S, N = 2, 4, 1024
    lib = tp.trace_library(mode)
    surf_int, surf_real, coef, n_rows = tp.blank_tables(B, S)
    out = tp.run_trace(
        lib,
        mode,
        launch=tp.random_launch(N),
        surf_int=surf_int,
        surf_real=surf_real,
        coef=coef,
        n_rows=n_rows,
        write_final=False,
        threads=(N // 2, B),
    )
    iters = out["iters"]
    assert int((iters == trace_layout.ITERS_UNWRITTEN).sum()) == B * S * (N // 2)
    assert int((iters[:, :, : N // 2] == 0).sum()) == B * S * (N // 2)


def _negate_raw(component: np.ndarray, mode: str) -> np.ndarray:
    """The raw component of ``-v``: df64 negates both words, sf64 flips bit 63."""
    if mode == "df64":
        return -component
    return (component.view(np.uint64) ^ np.uint64(1 << 63)).view(np.int64)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("Lb", [1, 3])
def test_mirror_tables_reach_every_recorded_row(mode: str, Lb: int) -> None:
    """Offset arithmetic of design 4.4, pinned on an EXACTLY predicted trace.

    ``mirror_tables`` + ``mirror_launch`` make every surface row a reflective
    plane the rays already sit on: ``t = -0.0``, so positions, intensity and
    OPD are untouched and ``reflect`` flips ``N`` exactly (see the helper's
    docstring).  Row ``k`` therefore holds the launch state with
    ``N * (-1)**k`` -- a per-row DISTINGUISHABLE value, so unlike the day-1
    stub assertion this cannot pass by writing the same thing everywhere, and
    it checks both launch layouts (shared, ``launch_stride = 0``; per design,
    ``launch_stride = 1``) and both modes on raw components.
    """
    B, S, N = 3, 4, 512
    lib = tp.trace_library(mode)
    surf_int, surf_real, coef, n_rows = tp.mirror_tables(B, S)
    launch = tp.mirror_launch(N, Lb=Lb, seed=5)
    out = tp.run_trace(
        lib,
        mode,
        launch=launch,
        surf_int=surf_int,
        surf_real=surf_real,
        coef=coef,
        n_rows=n_rows,
        write_final=True,
    )
    src = out["launch_raw"]  # [comp][Q, Lb, N]
    snap = out["snap"]  # [comp][S_PLANES, B, n_rows, N]
    fin = out["final"]  # [comp][F_PLANES, B, N]
    assert n_rows == S
    assert int(out["status"].sum()) == 0
    assert int(out["iters"].sum()) == 0
    for b in range(B):
        lb = 0 if Lb == 1 else b
        for plane in range(trace_layout.S_PLANES):  # x,y,z,L,M,N,i,opd
            for row in range(n_rows):
                flip = plane == trace_layout.S_N and row % 2 == 1
                for c, s_raw in enumerate(src):
                    want = s_raw[plane, lb]
                    if flip:
                        want = _negate_raw(want, mode)
                    assert np.array_equal(
                        snap[c][plane, b, row], want, equal_nan=True
                    ), f"snap plane {plane} design {b} row {row} component {c}"
        # final: the same state after S - 1 flips, and L0/M0/N0 = the last
        # surface's pre-interaction direction, i.e. one flip earlier.
        for fplane, qplane, flips in (
            (trace_layout.F_X, trace_layout.Q_X, 0),
            (trace_layout.F_Y, trace_layout.Q_Y, 0),
            (trace_layout.F_Z, trace_layout.Q_Z, 0),
            (trace_layout.F_L, trace_layout.Q_L, 0),
            (trace_layout.F_M, trace_layout.Q_M, 0),
            (trace_layout.F_N, trace_layout.Q_N, S - 1),
            (trace_layout.F_I, trace_layout.Q_I, 0),
            (trace_layout.F_OPD, trace_layout.Q_OPD, 0),
            (trace_layout.F_L0, trace_layout.Q_L, 0),
            (trace_layout.F_M0, trace_layout.Q_M, 0),
            (trace_layout.F_N0, trace_layout.Q_N, S - 2),
        ):
            for c, s_raw in enumerate(src):
                want = s_raw[qplane, lb]
                if flips % 2:
                    want = _negate_raw(want, mode)
                assert np.array_equal(fin[c][fplane, b], want, equal_nan=True), (
                    f"final plane {fplane} design {b} component {c}"
                )


@pytest.mark.parametrize("mode", MODES)
def test_unrecorded_rows_are_not_written(mode: str) -> None:
    """``SI_SNAPROW == -1`` records nothing; the object row is still visited."""
    B, S, N = 1, 3, 256
    lib = tp.trace_library(mode)
    surf_int, surf_real, coef, n_rows = tp.blank_tables(B, S, snap_rows=False)
    assert n_rows == 0
    out = tp.run_trace(
        lib,
        mode,
        launch=tp.random_launch(N),
        surf_int=surf_int,
        surf_real=surf_real,
        coef=coef,
        n_rows=n_rows,
        write_final=False,
    )
    assert out["snap"] is None
    assert out["final"] is None
    assert int((out["iters"] == trace_layout.ITERS_UNWRITTEN).sum()) == 0


@pytest.mark.parametrize("mode", MODES)
def test_ray_and_design_bases_offset_the_slab(mode: str) -> None:
    """``D_RAY_BASE`` / ``D_DESIGN_BASE`` select the slab, as design 4.4 says.

    A slab covering rays ``[N/2, N)`` of design 1 only must leave every other
    ``(b, s, i)`` at the ``ITERS_UNWRITTEN`` sentinel -- an exact count, not a
    spot check.
    """
    B, S, N = 3, 2, 256
    half = N // 2
    lib = tp.trace_library(mode)
    surf_int, surf_real, coef, n_rows = tp.blank_tables(B, S)
    out = tp.run_trace(
        lib,
        mode,
        launch=tp.random_launch(N),
        surf_int=surf_int,
        surf_real=surf_real,
        coef=coef,
        n_rows=n_rows,
        write_final=False,
        ray_base=half,
        design_base=1,
        threads=(half, 1),
    )
    iters = out["iters"]
    written = iters != trace_layout.ITERS_UNWRITTEN
    expected = np.zeros((B, S, N), dtype=bool)
    expected[1, :, half:] = True
    assert np.array_equal(written, expected)


# ---------------------------------------------------------------------------
# Amalgamation: include order (plan WP1 test_include_order, fix L1.9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_include_order(mode: str) -> None:
    """``conic.metal`` precedes ``trace.metal``, and ``trace_layout.h`` both.

    ``trace.metal`` uses ``conic_ops<R>`` / ``conic_candidates_body<R>`` and the
    ``OT_*`` defines, so a reordered amalgamation would be a compile error at
    best and a silently different kernel at worst.
    """
    source = tp.trace_source(mode)
    i_conic = source.index("OPTILAND_CONIC_METAL")
    i_layout = source.index("OPTILAND_TRACE_LAYOUT_H")
    i_trace = source.index("OPTILAND_TRACE_METAL")
    assert i_conic < i_trace, "conic.metal must precede trace.metal"
    assert i_layout < i_trace, "trace_layout.h must precede trace.metal"
    positions = [source.index(f"// ---- {f} ----") for f in tp.TRACE_FILES]
    assert positions == sorted(positions), tp.TRACE_FILES
    header_end = source.index(f"// ---- {tp.TRACE_FILES[0]} ----")
    for h in tp.headers(mode):
        assert source.index(f"// ---- {h} ----") < header_end, h


@pytest.mark.parametrize("mode", MODES)
def test_entry_points_compile_and_dispatch(mode: str) -> None:
    """The library builds and every entry point survives its first dispatch.

    Metal builds pipeline states lazily (day-1 Q12), so "it compiled" is not
    evidence that a kernel runs; :func:`tp.trace_library` dispatches each entry
    point once at build time and this test pins that it stays that way.
    """
    lib = tp.trace_library(mode)
    assert hasattr(lib, tp.ENTRY[mode])


@pytest.mark.parametrize("mode", MODES)
def test_probe_build_compiles(mode: str) -> None:
    """``-DOPTILAND_TRACE_PROBES`` is a second, separately cached build."""
    src = tp.trace_source(mode, probes=True)
    assert src.startswith("#define OPTILAND_TRACE_PROBES 1")
    lib = tp.trace_library(mode, probes=True)
    assert hasattr(lib, f"probe_copy_{mode}")
    n = 4096
    values = np.linspace(-3.5, 3.5, n)
    src_bufs = tp.to_device(values, mode)
    dst_bufs = tp.empty_device(n, mode)
    getattr(lib, f"probe_copy_{mode}")(*src_bufs, *dst_bufs, threads=[n, 1, 1])
    torch.mps.synchronize()
    got = tp.raw_components(dst_bufs, (n,))
    want = tp.raw_components(src_bufs, (n,))
    for g, w in zip(got, want, strict=True):
        assert np.array_equal(g, w, equal_nan=True)


# ---------------------------------------------------------------------------
# Literal discipline (design 4.2 rules 2 and 5, day-1 probe P7)
# ---------------------------------------------------------------------------


def test_lit_arguments_are_float32_exact() -> None:
    """Every ``lit(v)`` in ``trace.metal`` has a float32-exact ``v``.

    ``lit()`` builds an R from a single float32 word, so a value that is not
    float32-exact (``1e-14``: 1.76e-8 relative error; ``1e-10``: 1.34e-8 --
    probe P7) would silently become a different constant from the one the
    Python source uses.  Such values arrive encoded in ``surf_real`` instead.
    """
    source = tp.TRACE_METAL.read_text()
    bad = tp.non_float32_exact_lits(source)
    assert bad == [], f"lit() arguments that are not float32-exact: {bad}"
    # Exactly TWO call sites convert a runtime float: `pow_scalar`'s fallback
    # `O::pow(x, O::lit(e))` and `index_const`'s `lit(v)`, whose `v` is a
    # coefficient-loop index (`2*(i+1)`, `i+1`).  Both are SIMD-uniform and
    # both are float32-exact for every order a record row can carry; a third
    # dynamic conversion is a reviewable change, not a silent one.
    assert tp.dynamic_lit_arguments(source) == ["e", "v"]
    assert source.count("const float v = float(n);") == 1
    assert source.count("return trace_ops<R>::lit(v);") == 1


def test_lit_scanner_flags_a_known_bad_literal() -> None:
    """The scanner is not vacuous: it flags the two values P7 measured.

    A scan that can only find what it was told to look for proves nothing, so
    predict the exact result on a synthetic source: ``1e-14`` and ``1e-10`` are
    rejected, ``1e3``, ``2**-48``, ``0.5`` and ``-1`` are accepted.
    """
    good = "lit(1e3f) lit(0.5f) lit(-1.0f) lit(3.5527136788005009e-15f) lit(2.0f)"
    assert tp.non_float32_exact_lits(good) == []
    bad = tp.non_float32_exact_lits("lit(1e-14f) lit(1e-10f) lit(1.0f)")
    assert [a for a, _ in bad] == ["1e-14f", "1e-10f"]
    assert tp.lit_arguments("O::lit( 2.0f )") == ["2.0f"]


def test_no_bare_float_literals_in_arithmetic() -> None:
    """No float literal is an operand of ``+ - * /`` outside ``lit()``.

    Design 4.2 rule 2: every constant taking part in the mirrored arithmetic
    goes through ``lit()`` (float32-exact) or an encoded table slot.  A bare
    ``x * 2.0f`` would mix a float32 into an emulated-float64 expression.
    """
    source = tp.TRACE_METAL.read_text()
    bad = tp.bare_float_literal_lines(source)
    assert bad == [], f"bare float literals in arithmetic: {bad}"


def test_bare_float_scanner_flags_real_violations() -> None:
    """The scanner is not vacuous: predict the exact verdict on six snippets."""
    flagged = [
        "R y = x * 2.0f;",
        "R y = 0.5f * x;",
        "a = a - 1e3f;",
        "R z = (x + y) / 4.0f;",
    ]
    for snippet in flagged:
        assert tp.bare_float_literal_lines(snippet) == [(1, snippet)], snippet
    clean = [
        "if (e == -1.0f) return O::recip(x);",
        "a = a * O::lit(1e3f);",
        "out.set(3UL * n + k, inten);",
        "// a comment mentioning x * 2.0f",
        "static inline df64 lit(float v) { return df::make(v, 0.0f); }",
    ]
    for snippet in clean:
        assert tp.bare_float_literal_lines(snippet) == [], snippet


def test_no_fma_no_mul_add_in_trace_metal() -> None:
    """``df::mul_add`` / ``sf::fma`` are never called (design 4.2 rule 5).

    The mirrored Python expressions are fma-free; a fused multiply-add rounds
    differently (once instead of twice, or vice versa) and breaks tier-A
    component equality.
    """
    source = tp.TRACE_METAL.read_text()
    assert tp.fma_call_lines(source) == []
    assert tp.fma_call_lines("x = df::mul_add(a, b, c);") == [
        (1, "x = df::mul_add(a, b, c);")
    ]


def test_trace_metal_header_pragmas_and_guard() -> None:
    """The safe-math pragmas and the include guard are present, in order."""
    source = tp.TRACE_METAL.read_text()
    i_safe = source.index("#pragma METAL fp math_mode(safe)")
    i_contract = source.index("#pragma METAL fp contract(off)")
    i_guard = source.index("#ifndef OPTILAND_TRACE_METAL")
    assert i_safe < i_contract < i_guard
    assert source.rstrip().endswith("#endif  // OPTILAND_TRACE_METAL")
    assert source.count("#ifdef OPTILAND_SF64_CORE_H") == source.count(
        "#endif  // OPTILAND_SF64_CORE_H"
    )


# ---------------------------------------------------------------------------
# Differential probe tests against the LIVE Python objects (design 8.2)
#
# Both paths are fed the SAME float64 arrays, so both see the same representable
# inputs after encoding, and every comparison is on raw components: plan 7.1
# tier A.  N is always above the dual-residency threshold (256) so the reference
# runs the GPU kernels, not the float64 host path.
# ---------------------------------------------------------------------------

N_RAYS = 4096
N_POSE_RAYS = 50_000


@contextlib.contextmanager
def metal_backend(mode: str):
    """Put the process on torch/mps/float64 in ``mode``, autograd OFF, for one test.

    ``be.grad_mode.disable()`` only clears the ``requires_grad`` flag new arrays
    are created with (``torch_backend/config.py:31-33``); it does NOT touch
    ``torch.is_grad_enabled()``, which is True by default.  That distinction is
    load-bearing here: with autograd enabled, ``NewtonRaphsonGeometry.distance``
    takes the DiffOptics branch and returns ``t - F(t)/dF/dt`` instead of the
    primal ``result.t``, which differs in the low word.  The fused path is
    refused outright when ``torch.is_grad_enabled()`` (plan 1.2,
    ``requires_grad``), so the primal path is the ONLY one the kernel mirrors,
    and ``torch.no_grad()`` is what makes the reference that path.
    """
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    be.set_metal_mode(mode)
    try:
        with torch.no_grad():
            assert not torch.is_grad_enabled()
            yield
    finally:
        be.set_metal_mode("df64")
        be.grad_mode.disable()
        be.set_backend("numpy")


def _scalar(value: object) -> float:
    """The float64 value of a 0-d/1-element backend array."""
    return float(np.asarray(be.to_numpy(value)).reshape(-1)[0])


def raw_of(t: object) -> list[np.ndarray]:
    """Raw components of a ``MetalFloat64``: what the GPU buffer holds.

    A host-resident tensor carries exact float64; its raw components are that
    value's encoding, which is precisely what the kernel's buffer would hold.
    """
    if getattr(t, "is_host_resident", False):
        host = np.asarray(t.to_numpy(), dtype=np.float64)
        if t.mode == "df64":
            hi, lo = encode.encode_df64(host)
            return [np.asarray(hi, dtype=np.float32), np.asarray(lo, dtype=np.float32)]
        return [np.asarray(encode.encode_sf64(host), dtype=np.int64)]
    return [c.detach().contiguous().cpu().numpy() for c in t.components]


def assert_raw_equal(got: list[np.ndarray], ref: list[np.ndarray], what: str) -> None:
    """Raw-component equality, with a decoded diff printed for readability."""
    assert len(got) == len(ref), what
    for c, (g, r) in enumerate(zip(got, ref, strict=True)):
        if not np.array_equal(g, r, equal_nan=True):
            bad = int((~(np.isnan(g) & np.isnan(r)) & (g != r)).sum())
            raise AssertionError(
                f"{what}: component {c} differs on {bad}/{g.size} entries; "
                f"first indices {np.flatnonzero(g != r)[:8].tolist()}"
            )


def _metal(a: np.ndarray, mode: str) -> object:
    """A GPU-resident ``MetalFloat64`` holding ``a`` (never the host path)."""
    return MetalFloat64.from_numpy(np.asarray(a, dtype=np.float64), mode, host=False)


# ---------------------------------------------------------------------------
# Which kernel variant the per-op path launches (design 4.2 rule 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_scalar_operand_side_matches_the_per_op_kernels(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python's reflected operators choose the kernel variant; predict each one.

    ``trace.metal`` mirrors the launched kernel, not the written expression:
    ``__rmul__`` / ``__radd__`` SWAP a Python literal to the right, ``__rsub__``
    keeps it on the left, ``__rtruediv__`` becomes ``recip`` then ``mul``, and a
    BACKEND scalar (a material index, a pose cosine) keeps whichever side it is
    written on.  Every row below is an exact prediction; a change here means a
    mirrored MSL expression has to move with it.
    """
    from optiland.backend.torch_backend.metal import ops_elementwise as oe

    calls: list[tuple[str, str | None, float | None]] = []
    original = oe._launch

    def spy(op, launch_mode, *comps, scalar=None, scalar_side=None, **kwargs):
        calls.append((op, scalar_side, None if scalar is None else float(scalar)))
        return original(
            op, launch_mode, *comps, scalar=scalar, scalar_side=scalar_side, **kwargs
        )

    monkeypatch.setattr(oe, "_launch", spy)

    with metal_backend(mode):
        x = _metal(np.linspace(0.1, 0.9, 1024), mode)
        s = be.array(0.5)
        assert s.is_host_resident, "a backend scalar must stay host-resident"
        cases: list[tuple[str, object, list[tuple[str, str | None, float | None]]]] = [
            ("2 * x", lambda: 2 * x, [("mul", "right", 2.0)]),
            ("x * 2", lambda: x * 2, [("mul", "right", 2.0)]),
            ("2 + x", lambda: 2 + x, [("add", "right", 2.0)]),
            ("1 - x", lambda: 1 - x, [("sub", "left", 1.0)]),
            ("x - 1", lambda: x - 1, [("sub", "right", 1.0)]),
            ("2 / x", lambda: 2 / x, [("recip", None, None), ("mul", "right", 2.0)]),
            ("x / 2", lambda: x / 2, [("div", "right", 2.0)]),
            ("s * x", lambda: s * x, [("mul", "left", 0.5)]),
            ("x * s", lambda: x * s, [("mul", "right", 0.5)]),
            ("s / x", lambda: s / x, [("div", "left", 0.5)]),
            ("s - x", lambda: s - x, [("sub", "left", 0.5)]),
            # The Yoda form is the point: rectangular.py:55 writes
            # `self.x_min <= x`, and this row proves it reaches ge(x, x_min).
            ("0.3 <= x", lambda: 0.3 <= x, [("ge", "right", 0.3)]),  # noqa: SIM300
            ("x >= 0.3", lambda: x >= 0.3, [("ge", "right", 0.3)]),
        ]
        for label, run, expected in cases:
            calls.clear()
            run()
            assert calls == expected, f"{label}: {calls} != {expected}"

        # The side is load-bearing: df64's mul adds a.hi*b.lo before a.lo*b.hi,
        # so swapping the operands changes the low word.  (sf64 is correctly
        # rounded and therefore commutative -- which is why a side error is
        # invisible in sf64 and must be caught here.)
        monkeypatch.setattr(oe, "_launch", original)
        u = be.array(1.0 / 1.5168)
        left = raw_of(u * x)
        right = raw_of(x * u)
        differing = sum(
            int((left_c != right_c).sum())
            for left_c, right_c in zip(left, right, strict=True)
        )
    if mode == "df64":
        assert differing > 0, "df64 mul must not be commutative"
    else:
        assert differing == 0, "sf64 mul is correctly rounded, hence commutative"


# ---------------------------------------------------------------------------
# localize / globalize (design 4.5, note 03)
# ---------------------------------------------------------------------------

#: The six poses of plan WP1: zero, one angle each, a pure decentre, all three.
POSES: dict[str, dict[str, float]] = {
    "zero": {},
    "rx": {"rx": 0.17},
    "ry": {"ry": -0.29},
    "rz": {"rz": 0.41},
    "dxdy": {"x": 1.5, "y": -2.25},
    "rxryrz": {
        "x": 0.5,
        "y": -0.75,
        "z": 2.0,
        "rx": 0.11,
        "ry": -0.22,
        "rz": 0.33,
    },
}


def _pose_row_and_flags(cs: object) -> tuple[np.ndarray, int]:
    """The ``surf_real`` pose slots and ``SI_FLAGS`` bits of design 3.4/3.3.

    Every trig value is ``be.cos`` / ``be.sin`` of the LIVE angle tensor, with
    separate slots for ``-a`` (localize) and ``+a`` (globalize): no
    ``cos(-a) == cos(a)`` libm assumption.  The flag bits are the ``if
    self.rx:`` truth tests of ``coordinate_system.py:138-142``.
    """
    row = tp.param_row(
        TX=_scalar(cs.x),
        TY=_scalar(cs.y),
        TZ=_scalar(cs.z),
        NTX=_scalar(-cs.x),
        NTY=_scalar(-cs.y),
        NTZ=_scalar(-cs.z),
        CNRZ=_scalar(be.cos(-cs.rz)),
        SNRZ=_scalar(be.sin(-cs.rz)),
        CNRY=_scalar(be.cos(-cs.ry)),
        SNRY=_scalar(be.sin(-cs.ry)),
        CNRX=_scalar(be.cos(-cs.rx)),
        SNRX=_scalar(be.sin(-cs.rx)),
        CRX=_scalar(be.cos(cs.rx)),
        SRX=_scalar(be.sin(cs.rx)),
        CRY=_scalar(be.cos(cs.ry)),
        SRY=_scalar(be.sin(cs.ry)),
        CRZ=_scalar(be.cos(cs.rz)),
        SRZ=_scalar(be.sin(cs.rz)),
    )
    flags = 0
    if cs.rx:
        flags |= trace_layout.FL_HAS_RX
    if cs.ry:
        flags |= trace_layout.FL_HAS_RY
    if cs.rz:
        flags |= trace_layout.FL_HAS_RZ
    return row, flags


def _pose_rays(n: int, seed: int = 23) -> np.ndarray:
    """``float64[6, n]``: random rays plus axial, grazing and NaN lanes."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-9.0, 9.0, n)
    y = rng.uniform(-9.0, 9.0, n)
    z = rng.uniform(-40.0, 12.0, n)
    ell = rng.uniform(-0.35, 0.35, n)
    em = rng.uniform(-0.35, 0.35, n)
    nn = np.sqrt(1.0 - ell * ell - em * em)
    k = n // 20
    x[:k] = 0.0
    y[:k] = 0.0
    ell[:k] = 0.0
    em[:k] = 0.0
    nn[:k] = 1.0  # axial
    ell[k : 2 * k] = 0.999
    em[k : 2 * k] = 0.0
    nn[k : 2 * k] = np.sqrt(1.0 - 0.999**2)  # grazing
    x[2 * k : 2 * k + 8] = np.nan
    nn[2 * k + 8 : 2 * k + 16] = np.nan
    return np.stack([x, y, z, ell, em, nn])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pose", sorted(POSES))
def test_probe_localize_globalize_bit_exact(mode: str, pose: str) -> None:
    """``probe_localize`` / ``probe_globalize`` == ``CoordinateSystem`` exactly.

    The kernel applies the same translation and the same one, two or three
    sequential rotations, in the same order, on the same encoded trig scalars,
    so the assertion is raw-component equality for EVERY pose -- an eps-scaled
    bound would hide a transposed rotation (note 03).
    """
    lib = tp.trace_library(mode, probes=True)
    planes = _pose_rays(N_POSE_RAYS)
    with metal_backend(mode):
        cs = CoordinateSystem(**POSES[pose])
        row, flags = _pose_row_and_flags(cs)
        expected_flags = 0
        for name, bit in (
            ("rx", trace_layout.FL_HAS_RX),
            ("ry", trace_layout.FL_HAS_RY),
            ("rz", trace_layout.FL_HAS_RZ),
        ):
            if POSES[pose].get(name):
                expected_flags |= bit
        assert flags == expected_flags, pose

        for which in ("localize", "globalize"):
            rays = RealRays(
                *[_metal(planes[k], mode) for k in range(6)],
                _metal(np.ones(N_POSE_RAYS), mode),
                _metal(np.full(N_POSE_RAYS, 0.55), mode),
            )
            getattr(cs, which)(rays)
            ref = [rays.x, rays.y, rays.z, rays.L, rays.M, rays.N]
            got = tp.run_probe_pose(
                lib, mode, which=which, planes=planes, pose=row, flags=flags
            )
            for k, name in enumerate("xyzLMN"):
                assert_raw_equal(
                    [g[k] for g in got], raw_of(ref[k]), f"{which} {name} [{pose}]"
                )


@pytest.mark.parametrize("mode", MODES)
def test_probe_localize_globalize_round_trip_is_not_identity(mode: str) -> None:
    """The tilted pose really moves the rays (the comparison is not vacuous).

    Without this, ``test_probe_localize_globalize_bit_exact`` would pass on a
    kernel whose ``localize`` did nothing, because the Python reference would
    then have to do nothing too -- which it does not.
    """
    lib = tp.trace_library(mode, probes=True)
    planes = _pose_rays(1024)
    with metal_backend(mode):
        cs = CoordinateSystem(**POSES["rxryrz"])
        row, flags = _pose_row_and_flags(cs)
        assert flags == (
            trace_layout.FL_HAS_RX | trace_layout.FL_HAS_RY | trace_layout.FL_HAS_RZ
        )
        got = tp.run_probe_pose(
            lib, mode, which="localize", planes=planes, pose=row, flags=flags
        )
        src = tp.to_device(planes, mode)
        src_raw = tp.raw_components(src, (6, 1024))
    for k in range(6):
        assert not np.array_equal(got[0][k], src_raw[0][k], equal_nan=True), (
            f"plane {k} unchanged by localize"
        )


# ---------------------------------------------------------------------------
# Aperture `contains` (design 4.9)
# ---------------------------------------------------------------------------


def _aperture_cases() -> dict[str, tuple[object, int, np.ndarray]]:
    """The whitelisted apertures with the ``surf_real`` params of design 3.3."""
    radial = RadialAperture(12.5, 3.0)
    radial_inf = RadialAperture(float("inf"), 4.0)
    radial_zero = RadialAperture(0.0)
    offset = OffsetRadialAperture(8.0, 1.5, 2.0, -3.0)
    rect = RectangularAperture(-4.0, 6.5, -2.5, 3.0)
    ellipse = EllipticalAperture(7.0, 3.5, 1.0, -0.5)
    return {
        "radial": (
            radial,
            trace_layout.AP_RADIAL,
            tp.param_row(AP0=radial.r_max**2, AP1=radial.r_min**2),
        ),
        "radial_inf": (
            radial_inf,
            trace_layout.AP_RADIAL,
            tp.param_row(AP0=radial_inf.r_max**2, AP1=radial_inf.r_min**2),
        ),
        "radial_zero": (
            radial_zero,
            trace_layout.AP_RADIAL,
            tp.param_row(AP0=radial_zero.r_max**2, AP1=radial_zero.r_min**2),
        ),
        "offset_radial": (
            offset,
            trace_layout.AP_OFFSET_RADIAL,
            tp.param_row(
                AP0=offset.r_max**2,
                AP1=offset.r_min**2,
                AP2=offset.offset_x,
                AP3=offset.offset_y,
            ),
        ),
        "rect": (
            rect,
            trace_layout.AP_RECT,
            tp.param_row(
                AP0=rect.x_min, AP1=rect.x_max, AP2=rect.y_min, AP3=rect.y_max
            ),
        ),
        "ellipse": (
            ellipse,
            trace_layout.AP_ELLIPSE,
            tp.param_row(
                AP0=ellipse.a**2,
                AP1=ellipse.b**2,
                AP2=ellipse.offset_x,
                AP3=ellipse.offset_y,
            ),
        ),
    }


def _aperture_points(n: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Random points plus NaN, +-inf and exact-rim lanes."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-15.0, 15.0, n)
    y = rng.uniform(-15.0, 15.0, n)
    edges = [
        (12.5, 0.0),  # radial r_max rim (12.5**2 = 156.25, exactly encodable)
        (-12.5, 0.0),
        (0.0, 3.0),  # radial r_min rim
        (-4.0, 0.0),  # rect x_min
        (6.5, 3.0),  # rect corner
        (0.0, -2.5),  # rect y_min
        (8.0, -0.5),  # ellipse rim on +a (a = 7, offset 1, -0.5)
        (1.0, 3.0),  # ellipse rim on +b
        (10.0, -3.0),  # offset-radial r_max rim
        (np.nan, 0.0),
        (0.0, np.nan),
        (np.nan, np.nan),
        (np.inf, 0.0),
        (-np.inf, np.inf),
        (0.0, 0.0),
    ]
    for j, (ex, ey) in enumerate(edges):
        x[j] = ex
        y[j] = ey
    return x, y


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", sorted(_aperture_cases()))
def test_probe_contains_matches_aperture(mode: str, kind: str) -> None:
    """``ap_contains`` == ``aperture.contains`` exactly, NaN and rim included.

    Both operators are inclusive and NaN-false in both representations, so a
    missed (NaN) ray clips to zero intensity exactly as Python's
    ``contains(NaN, NaN) is False`` does.  The probe compares on identical
    encoded inputs, so there is no rim band here (design 8.1 site 2 applies to
    the whole-trace comparison only).
    """
    aperture, code, params = _aperture_cases()[kind]
    lib = tp.trace_library(mode, probes=True)
    x, y = _aperture_points(N_RAYS)
    got = tp.run_probe_contains(lib, mode, x=x, y=y, params=params, code=code)
    with metal_backend(mode):
        ref_mask = aperture.contains(_metal(x, mode), _metal(y, mode))
        ref = np.asarray(
            ref_mask.detach().cpu().numpy() if hasattr(ref_mask, "detach") else ref_mask
        ).astype(bool)
    assert np.array_equal(got.astype(bool), ref), (
        f"{kind}: {int((got.astype(bool) != ref).sum())} of {ref.size} differ"
    )
    # Non-vacuity, predicted per aperture: NaN is outside everywhere; an
    # infinite radius is outside every finite aperture and INSIDE the
    # `r_max = inf` annulus on both paths (`inf <= inf` is true), which is
    # exactly the Hubble obscuration case of plan 7.2.
    assert not ref[9:12].any(), f"{kind}: NaN must be outside"
    if kind == "radial_inf":
        assert bool(ref[12]) and bool(ref[13]), "r_max = inf must admit inf radii"
    else:
        assert not ref[12:14].any(), f"{kind}: infinite radii must be outside"
    if kind != "radial_zero":
        assert ref.any() and not ref.all(), kind
    else:
        assert int(ref.sum()) == 1, "r_max = 0 admits only the origin"


@pytest.mark.parametrize("mode", MODES)
def test_probe_contains_rim_is_inclusive(mode: str) -> None:
    """A point exactly on ``r_max`` is INSIDE on both paths (``<=``, not ``<``)."""
    lib = tp.trace_library(mode, probes=True)
    aperture = RadialAperture(12.5, 3.0)
    params = tp.param_row(AP0=aperture.r_max**2, AP1=aperture.r_min**2)
    x = np.array([12.5, -12.5, 0.0, 0.0, 12.5 + 1e-9, 2.99], dtype=np.float64)
    y = np.array([0.0, 0.0, 3.0, -3.0, 0.0, 0.0], dtype=np.float64)
    got = tp.run_probe_contains(
        lib, mode, x=x, y=y, params=params, code=trace_layout.AP_RADIAL
    ).astype(bool)
    assert got.tolist() == [True, True, True, True, False, False]


# ---------------------------------------------------------------------------
# Interaction (design 4.8)
# ---------------------------------------------------------------------------


def _interact_inputs(n: int, seed: int = 31) -> np.ndarray:
    """``float64[6, n]``: L0, M0, N0, nx, ny, nz with the adversarial lanes.

    Lane 0 is exact grazing (``dot`` is exactly 0, so ``sign(0) == 0``); lanes
    1-2 are NaN; the block from ``n // 2`` is steep incidence, which with
    ``u > 1`` is total internal reflection.
    """
    rng = np.random.default_rng(seed)
    ell = rng.uniform(-0.6, 0.6, n)
    em = rng.uniform(-0.6, 0.6, n)
    nn = np.sqrt(1.0 - ell * ell - em * em)
    nx = rng.uniform(-0.5, 0.5, n)
    ny = rng.uniform(-0.5, 0.5, n)
    nz = -np.sqrt(1.0 - nx * nx - ny * ny)  # StandardGeometry normals have nz < 0
    # exact grazing: dot = 1*0 + 0*0 + 0*(-1) = 0
    ell[0], em[0], nn[0] = 1.0, 0.0, 0.0
    nx[0], ny[0], nz[0] = 0.0, 0.0, -1.0
    # plane normal (nz = +1), still grazing-free
    nx[1], ny[1], nz[1] = 0.0, 0.0, 1.0
    # NaN lanes
    ell[2] = np.nan
    nx[3] = np.nan
    # steep incidence -> TIR when u > 1
    half = n // 2
    ell[half:] = 0.98
    em[half:] = 0.0
    nn[half:] = np.sqrt(1.0 - 0.98**2)
    nx[half:] = 0.0
    ny[half:] = 0.0
    nz[half:] = -1.0
    return np.stack([ell, em, nn, nx, ny, nz])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["refract", "reflect", "tir"])
def test_probe_interact_matches_refract_reflect(mode: str, kind: str) -> None:
    """``interact`` == ``RealRays.refract`` / ``.reflect`` on raw components.

    Covers the ``_align_surface_normal`` sign flip, exact grazing (``sign(0)``
    is 0, so refraction returns the unnormalised ``u * L0`` instead of NaN --
    reproduced, not fixed) and TIR (NaN direction, intensity untouched, and the
    ``TIR`` status bit set exactly where ``root`` is NaN with a finite ``dot``).
    """
    lib = tp.trace_library(mode, probes=True)
    planes = _interact_inputs(N_RAYS)
    index = {"refract": (1.0, 1.5168), "tir": (1.5168, 1.0)}.get(kind, (1.0, 1.0))
    reflective = kind == "reflect"

    with metal_backend(mode):
        # The interaction model passes `material.n(rays.w)`, i.e. BACKEND
        # scalars, so `u` and `u**2` are host float64 values and `u * L0`
        # keeps the scalar on the LEFT.  Python floats here would silently
        # test the other kernel variant (`__rmul__` swaps them to the right).
        materials = [IdealMaterial(n=v) for v in index]
        w_plane = _metal(np.full(N_RAYS, 0.55), mode)
        n1 = materials[0].n(w_plane)
        n2 = materials[1].n(w_plane)
        u_t = n1 / n2
        params = tp.param_row(
            U=_scalar(u_t),
            U2=_scalar(u_t**2),
            NPRE=_scalar(n1),
            NPOST=_scalar(n2),
        )

    got, status = tp.run_probe_interact(
        lib, mode, planes=planes, params=params, reflective=reflective
    )
    with metal_backend(mode):
        materials = [IdealMaterial(n=v) for v in index]
        w_plane = _metal(np.full(N_RAYS, 0.55), mode)
        n1 = materials[0].n(w_plane)
        n2 = materials[1].n(w_plane)
        rays = RealRays(
            _metal(np.zeros(N_RAYS), mode),
            _metal(np.zeros(N_RAYS), mode),
            _metal(np.zeros(N_RAYS), mode),
            _metal(planes[0], mode),
            _metal(planes[1], mode),
            _metal(planes[2], mode),
            _metal(np.full(N_RAYS, 0.75), mode),
            w_plane,
        )
        intensity_before = raw_of(rays.i)
        nx = _metal(planes[3], mode)
        ny = _metal(planes[4], mode)
        nz = _metal(planes[5], mode)
        if reflective:
            rays.reflect(nx, ny, nz)
        else:
            rays.refract(nx, ny, nz, n1, n2)  # exactly interact_real_rays
        ref = [raw_of(rays.L), raw_of(rays.M), raw_of(rays.N)]
        intensity_after = raw_of(rays.i)
        out_l = rays.L.to_numpy()

    for k, name in enumerate("LMN"):
        assert_raw_equal([g[k] for g in got], ref[k], f"{kind} {name}")
    assert_raw_equal(intensity_after, intensity_before, f"{kind} intensity")

    finite_in = np.all(np.isfinite(planes), axis=0)
    expect_tir = np.isnan(out_l) & finite_in
    assert np.array_equal((status & trace_layout.ST_TIR) != 0, expect_tir), (
        f"{kind}: TIR bit misplaced"
    )
    assert int((status & ~np.uint8(trace_layout.ST_TIR)).sum()) == 0
    if kind == "tir":
        assert int(expect_tir.sum()) > 0, "the TIR fixture never reached TIR"
    else:
        assert int(expect_tir.sum()) == 0, f"{kind} must not flag TIR"
    # Exact grazing (lane 0): sign(0) is 0, so the aligned normal and dot are
    # zero and refraction returns the unnormalised u * L0 -- a finite value,
    # not NaN.  Under TIR the root is NaN for every lane, lane 0 included.
    if kind != "tir":
        assert np.isfinite(out_l[0]), "sign(0) grazing artefact not reproduced"


# ---------------------------------------------------------------------------
# pow_scalar (design 4.2 rule 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("e", [0.0, 1.0, 2.0, 3.0, 0.5, -1.0, -2.0, -0.5, 4.0, 5.0])
def test_probe_pow_scalar_matches_pow_tensor_scalar(mode: str, e: float) -> None:
    """``pow_scalar(x, e)`` == ``_pow_tensor_scalar(x, e)`` on raw components.

    The eight fast paths (``ops_elementwise.py:748-763``) and the ``O::pow``
    fallback must agree for negatives, signed zeros and infinities, where
    ``sqrt``/``rsqrt`` deliberately differ from C ``pow``.
    """
    lib = tp.trace_library(mode, probes=True)
    rng = np.random.default_rng(97)
    values = rng.uniform(-6.0, 6.0, N_RAYS)
    specials = [0.0, -0.0, 1.0, -1.0, 2.0, np.inf, -np.inf, np.nan, 1e-8, 1e8]
    values[: len(specials)] = specials
    got = tp.run_probe_pow_scalar(lib, mode, values=values, e=e)
    with metal_backend(mode):
        ref = raw_of(_metal(values, mode) ** e)
    assert_raw_equal(got, ref, f"pow_scalar(x, {e})")


# ---------------------------------------------------------------------------
# Propagation, absorption and OPD (design 4.10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_absorption_order_matches_homogeneous_propagate(mode: str) -> None:
    """The five-op absorption chain runs in ``homogeneous.py:46-53``'s order.

    ``alpha = (4 * pi) * k`` is a host scalar, so the GPU sees, in this order:
    ``/ w``, negate, ``* t``, ``* 1e3``, ``exp``, ``i *``.  Any other
    association changes the low word, which the sensitivity check below
    demonstrates, so the raw-component equality is not vacuous.
    """
    lib = tp.trace_library(mode, probes=True)
    rng = np.random.default_rng(53)
    n = N_RAYS
    x = rng.uniform(-9.0, 9.0, n)
    y = rng.uniform(-9.0, 9.0, n)
    z = rng.uniform(-30.0, -1.0, n)
    ell = rng.uniform(-0.3, 0.3, n)
    em = rng.uniform(-0.3, 0.3, n)
    nn = np.sqrt(1.0 - ell * ell - em * em)
    inten = rng.uniform(0.2, 1.0, n)
    inten[:4] = [1.0, 0.0, 0.5, 0.25]
    opd = rng.uniform(-3.0, 3.0, n)
    w = np.full(n, 0.55)
    t = rng.uniform(-40.0, 40.0, n)  # signed: virtual propagation subtracts OPL
    planes = np.stack([x, y, z, ell, em, nn, inten, opd, w, t])

    with metal_backend(mode):
        material = IdealMaterial(n=1.5168, k=2.5e-7)
        rays = RealRays(
            *[_metal(planes[k], mode) for k in range(6)],
            _metal(inten, mode),
            _metal(w, mode),
        )
        rays.opd = _metal(opd, mode)
        t_r = _metal(t, mode)
        k_value = material.k(rays.w)
        alpha = _scalar(4 * be.pi * k_value)
        n_pre = _scalar(material.n(rays.w))
        assert alpha > 0.0, "the absorbing branch must be live"

        HomogeneousPropagation(material).propagate(rays, t_r)
        rays.opd = rays.opd + t_r * material.n(rays.w)
        ref = [
            raw_of(rays.x),
            raw_of(rays.y),
            raw_of(rays.z),
            raw_of(rays.i),
            raw_of(rays.opd),
        ]
        # sensitivity certificate: the same five values in a different
        # association must NOT reproduce the reference bit for bit.
        a = alpha / _metal(w, mode)
        alt = _metal(inten, mode) * be.exp(-a * (t_r * 1e3))
        alt_raw = raw_of(alt)

    params = tp.param_row(ALPHA=alpha, NPRE=n_pre)
    got = tp.run_probe_propagate(
        lib, mode, planes=planes, params=params, absorbing=True
    )
    for k, name in enumerate(["x", "y", "z", "intensity", "opd"]):
        assert_raw_equal([g[k] for g in got], ref[k], f"propagate {name}")

    differing = sum(
        int((a_c != r_c).sum()) for a_c, r_c in zip(alt_raw, ref[3], strict=True)
    )
    assert differing > 0, (
        "re-associating the absorption chain changed nothing -- the "
        "raw-component comparison cannot detect an order error"
    )


@pytest.mark.parametrize("mode", MODES)
def test_propagate_without_absorption_leaves_intensity_untouched(mode: str) -> None:
    """``ABSORBING`` off skips the chain (the ``k > 0`` gate, homogeneous.py:47)."""
    lib = tp.trace_library(mode, probes=True)
    rng = np.random.default_rng(67)
    n = 1024
    planes = np.zeros((10, n), dtype=np.float64)
    planes[6] = rng.uniform(0.1, 1.0, n)  # intensity
    planes[8] = 0.55  # w
    planes[9] = rng.uniform(-5.0, 5.0, n)  # t
    params = tp.param_row(ALPHA=1.0, NPRE=1.5)
    got = tp.run_probe_propagate(
        lib, mode, planes=planes, params=params, absorbing=False
    )
    src = tp.raw_components(tp.to_device(planes[6], mode), (n,))
    for comp_got, comp_src in zip([g[3] for g in got], src, strict=True):
        assert np.array_equal(comp_got, comp_src, equal_nan=True)


# ---------------------------------------------------------------------------
# Sag, normal and distance (design 4.6-4.7, WP1 part 2)
#
# Every geometry is a LIVE object built under the metal backend, so its
# ``radius`` / ``k`` are the same 0-d ``be.array``s the per-op path multiplies
# by -- which is what decides whether a slot lands on the left or the right of
# the launched kernel (design 4.2 rule 3).
# ---------------------------------------------------------------------------


def _host_float(value: object) -> float:
    """The float64 value of a backend scalar, a tensor entry or a Python float."""
    if hasattr(value, "to_numpy") or hasattr(value, "detach"):
        return float(np.asarray(be.to_numpy(value)).reshape(-1)[0])
    return float(value)


def _geometry_row(geometry: object) -> np.ndarray:
    """The ``surf_real`` slots a geometry owns (design 3.4, slots 18-21 and 27).

    Built here rather than through ``trace_adapters`` so the WP1 probes do not
    depend on WP2's record compiler; ``test_geometry_row_matches_the_adapter``
    pins that the two agree.
    """
    radius = geometry.radius
    conic = getattr(geometry, "k", None)
    return tp.param_row(
        R=_host_float(radius),
        K=0.0 if conic is None else _host_float(conic),
        K1=1.0 if conic is None else _host_float(1 + conic),
        R2=_host_float(radius**2),
        TOL=float(getattr(geometry, "tol", 0.0)),
    )


def _coefficients(geometry: object) -> np.ndarray:
    return np.array(
        [_host_float(c) for c in getattr(geometry, "coefficients", [])],
        dtype=np.float64,
    )


#: Every geometry code, each with the builder that reaches it.  ``StandardGeometry``
#: with an infinite radius is the only way to reach ``GEOM_STD_INF`` (the surface
#: factory collapses ``radius = inf`` to a ``Plane``, WP5 finding 1).
def _geometry_cases() -> dict[str, tuple[object, int]]:
    from optiland.geometries import EvenAsphere, OddAsphere, Plane, StandardGeometry

    even_c = [-1.5e-4, 2.5e-7, -3.5e-10, 4.5e-13, -5.5e-16]
    odd_c = [2.0e-4, -3.0e-6, 1.5e-8, -4.0e-11]
    return {
        "plane": (lambda: Plane(CoordinateSystem()), trace_layout.GEOM_PLANE),
        "std_inf": (
            lambda: StandardGeometry(CoordinateSystem(), be.inf),
            trace_layout.GEOM_STD_INF,
        ),
        "sphere": (
            lambda: StandardGeometry(CoordinateSystem(), 25.0, 0.0),
            trace_layout.GEOM_CONIC,
        ),
        "sphere_neg": (
            lambda: StandardGeometry(CoordinateSystem(), -25.0, 0.0),
            trace_layout.GEOM_CONIC,
        ),
        "conic_m07": (
            lambda: StandardGeometry(CoordinateSystem(), 25.0, -0.7),
            trace_layout.GEOM_CONIC,
        ),
        "parabola": (
            lambda: StandardGeometry(CoordinateSystem(), 25.0, -1.0),
            trace_layout.GEOM_CONIC,
        ),
        "conic_p2": (
            lambda: StandardGeometry(CoordinateSystem(), 25.0, 2.0),
            trace_layout.GEOM_CONIC,
        ),
        # `23.713` and `1 + (-0.3)` both have a NON-ZERO df64 low word, which
        # is what makes `R * s` and `s * R` different kernels: df64's mul adds
        # `a.hi*b.lo` before `a.lo*b.hi`, so when one operand's `lo` is exactly
        # zero (25.0, 30.0, 200.0, inf -- every other radius here) the two
        # orders collapse to the same float32 sum and an operand-side error is
        # INVISIBLE.  Without this row the sag comparison cannot fail on one.
        "conic_inexact": (
            lambda: StandardGeometry(CoordinateSystem(), 23.713, -0.3),
            trace_layout.GEOM_CONIC,
        ),
        "even_inexact": (
            lambda: EvenAsphere(
                CoordinateSystem(), 23.713, -0.3, coefficients=list(even_c)
            ),
            trace_layout.GEOM_EVEN,
        ),
        "even5": (
            lambda: EvenAsphere(
                CoordinateSystem(), 25.0, -0.5, coefficients=list(even_c)
            ),
            trace_layout.GEOM_EVEN,
        ),
        "even_inf": (
            lambda: EvenAsphere(
                CoordinateSystem(), be.inf, 0.0, coefficients=list(even_c)
            ),
            trace_layout.GEOM_EVEN,
        ),
        "odd": (
            lambda: OddAsphere(CoordinateSystem(), 30.0, 0.0, coefficients=list(odd_c)),
            trace_layout.GEOM_ODD,
        ),
    }


def _radius_infinite(geometry: object) -> bool:
    from optiland.geometries.standard import _is_radius_infinite

    return bool(_is_radius_infinite(geometry.radius))


def _surface_points(n: int, seed: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """Points on the used region, plus the vertex, the rim and non-finite lanes.

    ``(0, 0)`` is the odd-asphere scrub case (``r**-1`` is ``+inf`` there) and
    ``(30, 0)`` drives the base radicand negative for ``R = 25, k >= 0``, so
    both the NaN and the finite lanes are exercised in one bundle.
    """
    rng = np.random.default_rng(seed)
    radius = rng.uniform(0.0, 12.0, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    specials = [
        (0.0, 0.0),  # vertex: odd-asphere r**(i-1) scrub, design 4.7
        (-0.0, 0.0),
        (12.0, 0.0),
        (0.0, -12.0),
        (30.0, 0.0),  # outside the base conic for R = 25, k >= 0 -> NaN sag
        (0.0, 30.0),
        (1e-6, -1e-6),
        (np.nan, 0.0),
        (0.0, np.nan),
        (np.inf, 0.0),
        (-np.inf, np.inf),
        (8.0, 8.0),
    ]
    for j, (px, py) in enumerate(specials):
        x[j] = px
        y[j] = py
    return x, y


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(_geometry_cases()))
def test_probe_sag_normal_match(mode: str, name: str) -> None:
    """``sag_of`` / ``normal_of`` == ``geometry.sag`` / ``surface_normal``.

    Raw components, both modes, every geometry code, including ``r = 0`` on the
    odd asphere, ``R = inf`` on an asphere and the ``nz = +1`` (Plane) versus
    ``nz < 0`` (StandardGeometry-derived) sign split of design 4.7.
    """
    build, geom = _geometry_cases()[name]
    lib = tp.trace_library(mode, probes=True)
    x, y = _surface_points(N_RAYS)
    with metal_backend(mode):
        geometry = build()
        row = _geometry_row(geometry)
        coefficients = _coefficients(geometry)
        xm, ym = _metal(x, mode), _metal(y, mode)
        ref_sag = raw_of(be.atleast_1d(geometry.sag(xm, ym)))
        probe_rays = RealRays(
            xm,
            ym,
            _metal(np.zeros_like(x), mode),
            _metal(np.zeros_like(x), mode),
            _metal(np.zeros_like(x), mode),
            _metal(np.ones_like(x), mode),
            _metal(np.ones_like(x), mode),
            _metal(np.full_like(x, 0.55), mode),
        )
        nx, ny, nz = geometry.surface_normal(probe_rays)
        ref_normal = [raw_of(nx), raw_of(ny), raw_of(nz)]
        nz_host = np.asarray(be.to_numpy(nz))
    got_sag = tp.run_probe_sag(
        lib, mode, x=x, y=y, params=row, geom=geom, coefficients=coefficients
    )
    assert_raw_equal(got_sag, ref_sag, f"sag[{name}]")
    got_normal = tp.run_probe_normal(
        lib, mode, x=x, y=y, params=row, geom=geom, coefficients=coefficients
    )
    for k, comp in enumerate(("nx", "ny", "nz")):
        assert_raw_equal([g[k] for g in got_normal], ref_normal[k], f"{comp}[{name}]")
    # The raw sign of nz separates the two normal families (design 4.7): the
    # per-ray alignment in `interact` hides it, so assert it here.
    finite = np.isfinite(nz_host)
    assert finite.any(), name
    if name == "plane":
        assert np.all(nz_host[finite] == 1.0), "Plane normals are (0, 0, +1)"
    else:
        assert np.all(nz_host[finite] < 0.0), "conic-derived normals have nz < 0"


def test_geometry_row_matches_the_adapter() -> None:
    """The hand-built probe row equals WP2's ``GeometryAdapter.fill`` output.

    Slots 18-21 and 27 are the only ones these probes read, and both sides must
    produce them with the SAME host float64 op (design 3.4) or the probe would
    be testing a different surface from the one the driver launches.
    """
    from optiland.backend.torch_backend.metal import trace_layout as L
    from optiland.backend.torch_backend.metal.trace_adapters import GEOMETRY_ADAPTERS

    slots = [L.SR_R, L.SR_K, L.SR_K1, L.SR_R2, L.SR_TOL]
    with metal_backend("df64"):
        for name, (build, geom) in sorted(_geometry_cases().items()):
            geometry = build()
            adapter = GEOMETRY_ADAPTERS[type(geometry)]
            row_int = np.zeros(L.SI_STRIDE, dtype=np.int32)
            row_real = np.zeros(L.SR_STRIDE, dtype=np.float64)
            row_coef = np.zeros(max(1, len(getattr(geometry, "coefficients", []))))
            adapter.fill(geometry, row_int, row_real, row_coef, {})
            mine = _geometry_row(geometry)
            assert [mine[s] for s in slots] == [row_real[s] for s in slots], name
            assert int(row_int[L.SI_GEOM]) == geom, name
            assert bool(row_int[L.SI_FLAGS] & L.FL_RADIUS_INF) == _radius_infinite(
                geometry
            ), name


# ---------------------------------------------------------------------------
# Distance


def _distance_rays(n: int, seed: int = 11) -> np.ndarray:
    """``float64[6, n]``: the conic-kernel bundle plus backward and NaN lanes.

    Lanes, in tenths of the bundle: axial, origin exactly on the vertex plane
    (the rounded-self-hit case), grazing (``L = 0.999``), BACKWARD (``z > 0``
    with ``N > 0``, so ``t < 0`` on a plane and on a conic), ``N = 0`` exactly
    (NaN on ``GEOM_PLANE``, floored on ``GEOM_STD_INF``) and a NaN lane.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-8, 8, n)
    y = rng.uniform(-8, 8, n)
    z = rng.uniform(-40, -1, n)
    ell = rng.uniform(-0.3, 0.3, n)
    em = rng.uniform(-0.3, 0.3, n)
    en = np.sqrt(1 - ell * ell - em * em)
    k = n // 10
    x[:k] = 0.0
    y[:k] = 0.0
    ell[:k] = 0.0
    em[:k] = 0.0
    en[:k] = 1.0  # axial
    z[k : 2 * k] = 0.0  # origin on the vertex plane
    ell[2 * k : 3 * k] = 0.999
    em[2 * k : 3 * k] = 0.0
    en[2 * k : 3 * k] = np.sqrt(1 - 0.999**2)  # grazing
    z[3 * k : 4 * k] = rng.uniform(1.0, 30.0, k)  # backward: t < 0
    ell[4 * k : 4 * k + 8] = 1.0
    em[4 * k : 4 * k + 8] = 0.0
    en[4 * k : 4 * k + 8] = 0.0  # |N| == 0 exactly
    z[4 * k : 4 * k + 8] = 0.0
    x[4 * k + 8 : 4 * k + 12] = np.nan
    en[4 * k + 12 : 4 * k + 16] = np.nan
    z[4 * k + 12 : 4 * k + 16] = 0.0
    # The two |N| ~ 0 lanes sit ON the vertex plane on purpose: off it, the
    # GEOM_STD_INF floor turns them into |t| ~ 1e15, which crosses the
    # round-off floor of `_effective_tolerance` and makes the batch and the
    # per-thread tolerances legitimately disagree (design 4.6).  That is the
    # late-fallback case, exercised on its own in
    # `test_probe_newton_flags_the_tolerance_crossover`, not a mirror bug.
    return np.stack([x, y, z, ell, em, en])


def _aperture_row(aperture: object) -> tuple[int, np.ndarray]:
    """``(SI_APCODE, surf_real row)`` for a whitelisted aperture (design 3.3)."""
    if aperture is None:
        return trace_layout.AP_NONE, tp.param_row()
    if isinstance(aperture, OffsetRadialAperture):
        return trace_layout.AP_OFFSET_RADIAL, tp.param_row(
            AP0=aperture.r_max**2,
            AP1=aperture.r_min**2,
            AP2=aperture.offset_x,
            AP3=aperture.offset_y,
        )
    if isinstance(aperture, RadialAperture):
        return trace_layout.AP_RADIAL, tp.param_row(
            AP0=aperture.r_max**2, AP1=aperture.r_min**2
        )
    raise AssertionError(f"unhandled aperture {type(aperture).__name__}")


def _apertures() -> dict[str, object]:
    return {
        "none": None,
        "annulus": RadialAperture(12.5, 3.0),
        "offaxis": OffsetRadialAperture(25.0, 0.0, 0.0, 130.0),
    }


def _probe_row(geometry: object, aperture: object) -> np.ndarray:
    row = _geometry_row(geometry)
    _, ap_row = _aperture_row(aperture)
    for name in ("AP0", "AP1", "AP2", "AP3"):
        slot = getattr(trace_layout, f"SR_{name}")
        row[slot] = ap_row[slot]
    return row


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ap_name", sorted(_apertures()))
@pytest.mark.parametrize("name", sorted(_geometry_cases()))
def test_probe_distance_matches_geometry_distance(
    mode: str, ap_name: str, name: str
) -> None:
    """``surface_distance`` == ``geometry.distance`` on raw components.

    All five geometry codes, the three aperture states, and the ray lanes of
    :func:`_distance_rays` (axial, self-hit origins, grazing, BACKWARD, ``N =
    0``, NaN).  ``Plane.distance`` takes no ``aperture`` keyword, so a plane row
    never carries ``FL_AP_IN_ROOT`` -- exactly what ``_aperture_aware_distance``
    does (standard_surface.py:61-71).
    """
    build, geom = _geometry_cases()[name]
    aperture = _apertures()[ap_name]
    lib = tp.trace_library(mode, probes=True)
    planes = _distance_rays(N_RAYS)
    with metal_backend(mode):
        geometry = build()
        takes_aperture = geom != trace_layout.GEOM_PLANE
        row = _probe_row(geometry, aperture if takes_aperture else None)
        ap_code, _ = _aperture_row(aperture if takes_aperture else None)
        coefficients = _coefficients(geometry)
        flags = 0
        if _radius_infinite(geometry):
            flags |= trace_layout.FL_RADIUS_INF
        if takes_aperture and aperture is not None:
            flags |= trace_layout.FL_AP_IN_ROOT
        rays = RealRays(
            *[_metal(planes[q], mode) for q in range(6)],
            _metal(np.ones(N_RAYS), mode),
            _metal(np.full(N_RAYS, 0.55), mode),
        )
        if takes_aperture:
            ref_t = geometry.distance(rays, aperture=aperture)
        else:
            ref_t = geometry.distance(rays)
        ref = raw_of(ref_t)
        ref_host = np.asarray(be.to_numpy(ref_t))
    got, status, iters = tp.run_probe_distance(
        lib,
        mode,
        planes=planes,
        params=row,
        geom=geom,
        flags=flags,
        apcode=ap_code,
        coefficients=coefficients,
        max_iter=int(getattr(geometry, "max_iter", 0)),
    )
    assert int((iters == trace_layout.ITERS_UNWRITTEN).sum()) == 0, "unvisited thread"
    assert_raw_equal(got, ref, f"distance[{name}][{ap_name}]")
    # Predicted status: MISS is exactly the NaN set, and no thread may claim the
    # batch/per-thread tolerance split on a bundle this short (|t| < 3.5e3 mm).
    miss = (status & trace_layout.ST_MISS) != 0
    assert np.array_equal(miss, np.isnan(ref_host)), f"{name}/{ap_name}: MISS != NaN"
    assert int((status & trace_layout.ST_TOL_CROSSOVER).sum()) == 0, "unexpected floor"
    # Independent NumPy prediction for the two divisor-only codes: 12 rays
    # (the eight |N| = 0 and the four N = NaN) sit on the vertex plane, so a
    # Plane divides 0 by 0 and misses while the floored STD_INF row returns 0.
    with np.errstate(all="ignore"):
        if geom == trace_layout.GEOM_PLANE:
            predicted = np.isnan(-planes[2] / planes[5])
            assert int(predicted.sum()) == 12, int(predicted.sum())
            assert np.array_equal(miss, predicted)
        elif geom == trace_layout.GEOM_STD_INF:
            n_safe = np.where(np.abs(planes[5]) > tp.N_FLOOR, planes[5], tp.N_FLOOR)
            assert np.array_equal(miss, np.isnan(-planes[2] / n_safe))
            assert int(miss.sum()) == 0, "the |N| floor forbids a miss"
            floored = (status & trace_layout.ST_NZ_FLOORED) != 0
            assert np.array_equal(floored, n_safe != planes[5])
            assert int(floored.sum()) == 12, int(floored.sum())


@pytest.mark.parametrize("mode", MODES)
def test_plane_and_std_inf_differ_at_zero_slope(mode: str) -> None:
    """``GEOM_PLANE`` and ``GEOM_STD_INF`` are NOT interchangeable (WP5 finding 1).

    ``Plane.distance`` is a bare ``-z / N``; the infinite-radius branch of
    ``_conic_intersection_distance`` floors the divisor at ``1e-14``.  A ray with
    ``N == 0`` therefore gets NaN on one and ``-z * 1e14`` on the other, which is
    why the two codes never share a branch in the kernel.
    """
    from optiland.geometries import Plane, StandardGeometry

    lib = tp.trace_library(mode, probes=True)
    n = 512
    half = n // 2
    planes = np.zeros((6, n), dtype=np.float64)
    planes[2, :half] = -5.0  # z != 0: 5 / 0 is +inf on a Plane
    planes[2, half:] = 0.0  # z == 0: 0 / 0 is NaN on a Plane
    planes[5] = 0.0  # N == 0 exactly, everywhere
    with metal_backend(mode):
        plane_row = _geometry_row(Plane(CoordinateSystem()))
        inf_row = _geometry_row(StandardGeometry(CoordinateSystem(), be.inf))
    t_plane, st_plane, _ = tp.run_probe_distance(
        lib, mode, planes=planes, params=plane_row, geom=trace_layout.GEOM_PLANE
    )
    t_inf, st_inf, _ = tp.run_probe_distance(
        lib,
        mode,
        planes=planes,
        params=inf_row,
        geom=trace_layout.GEOM_STD_INF,
        flags=trace_layout.FL_RADIUS_INF,
    )
    # Predicted exactly: the Plane misses only where it divides 0 by 0 and
    # never floors; the STD_INF row floors every ray and never misses.
    assert int((st_plane & trace_layout.ST_MISS).astype(bool).sum()) == half
    assert int((st_plane & trace_layout.ST_NZ_FLOORED).astype(bool).sum()) == 0
    assert int((st_inf & trace_layout.ST_MISS).astype(bool).sum()) == 0
    assert int((st_inf & trace_layout.ST_NZ_FLOORED).astype(bool).sum()) == n
    decoded_plane = _decode(t_plane, mode)
    assert np.all(np.isposinf(decoded_plane[:half])), decoded_plane[:3]
    assert np.all(np.isnan(decoded_plane[half:])), decoded_plane[half:][:3]
    decoded_inf = _decode(t_inf, mode)
    assert np.allclose(decoded_inf[:half], 5.0e14, rtol=1e-12), decoded_inf[:3]
    assert np.all(decoded_inf[half:] == 0.0), decoded_inf[half:][:3]
    # The probe binds the same |N| floor the driver does; a drift here would
    # make every GEOM_STD_INF probe test compare against the wrong constant.
    from optiland.backend.torch_backend.metal.trace import N_FLOOR

    assert tp.N_FLOOR == N_FLOOR == 1e-14


def _decode(raw: list[np.ndarray], mode: str) -> np.ndarray:
    """Decode raw probe components back to float64 (for magnitude checks only)."""
    if mode == "df64":
        return encode.decode_df64(raw[0], raw[1])
    return encode.decode_sf64(raw[0])


@pytest.mark.parametrize("mode", MODES)
def test_aperture_root_preference_is_observable(mode: str) -> None:
    """The aperture REORDERS the two roots -- and it must, or it is untested.

    On an off-axis parabola both roots are genuine forward hits; only the
    aperture picks the used one (WP5 finding 8).  Without ``FL_AP_IN_ROOT`` the
    kernel must return the OTHER root for at least one ray, or
    :func:`test_probe_distance_matches_geometry_distance` would pass with the
    flag ignored.
    """
    from optiland.geometries import StandardGeometry

    lib = tp.trace_library(mode, probes=True)
    # WP5's `off_axis_parabola_far_root` bundle: a 4 mm fan at y = -60,
    # z = -20, climbing at M = 0.95, where both roots are genuine forward hits.
    n = 512
    em = 0.95
    planes = np.zeros((6, n), dtype=np.float64)
    planes[1] = -60.0 + np.linspace(-2.0, 2.0, n)
    planes[2] = -20.0
    planes[4] = em
    planes[5] = np.sqrt(1.0 - em * em)
    aperture = OffsetRadialAperture(25.0, 0.0, 0.0, 130.0)
    ap_code, _ = _aperture_row(aperture)
    with metal_backend(mode):
        geometry = StandardGeometry(CoordinateSystem(), 200.0, -1.0)
        row = _probe_row(geometry, aperture)
        rays = RealRays(
            *[_metal(planes[q], mode) for q in range(6)],
            _metal(np.ones(n), mode),
            _metal(np.full(n, 0.55), mode),
        )
        ref = raw_of(geometry.distance(rays, aperture=aperture))
    with_ap, _, _ = tp.run_probe_distance(
        lib,
        mode,
        planes=planes,
        params=row,
        geom=trace_layout.GEOM_CONIC,
        flags=trace_layout.FL_AP_IN_ROOT,
        apcode=ap_code,
    )
    without_ap, _, _ = tp.run_probe_distance(
        lib, mode, planes=planes, params=row, geom=trace_layout.GEOM_CONIC
    )
    assert_raw_equal(with_ap, ref, "off-axis parabola with aperture")
    differing = sum(
        int((a != b).sum()) for a, b in zip(with_ap, without_ap, strict=True)
    )
    assert differing > 0, "FL_AP_IN_ROOT changed nothing -- the flag is untested"


# ---------------------------------------------------------------------------
# Newton


def _newton_replay(geometry: object, rays: object, aperture: object = None):
    """``_solve_distance_primal``'s loop, plus each ray's own stop index.

    The Python dataclass carries ONE ``iterations`` integer for the whole batch
    (newton_raphson.py:166); the kernel's ``iters`` plane is per thread (WP5
    finding 4).  This replay runs the same helpers on the same objects, so its
    final ``t`` must equal the real solver's bit for bit -- which is asserted
    before its stop indices are believed.
    """
    from optiland.geometries import newton_raphson as nr
    from optiland.geometries.standard import StandardGeometry

    t = StandardGeometry.distance(geometry, rays, aperture=aperture)
    tol = nr._effective_tolerance(geometry.tol, t)
    f_t = geometry._surface_residual(t, rays)
    converged = be.abs(f_t) < tol
    n = int(np.asarray(be.to_numpy(f_t)).size)
    stop = np.full(n, -1, dtype=np.int64)
    for i in range(geometry.max_iter):
        stopped = np.asarray(be.to_numpy(converged)).astype(bool) | ~np.isfinite(
            np.asarray(be.to_numpy(t))
        )
        stop[(stop < 0) & stopped] = i
        if bool(be.to_numpy(be.all(converged))):
            break
        df_dt, scale, _ = geometry._surface_residual_dt(t, rays)
        safe_df_dt, _ = nr._regularize_signed(df_dt, scale)
        step = be.where(converged, be.zeros_like(f_t), f_t / safe_df_dt)
        t = t - step
        f_t = geometry._surface_residual(t, rays)
        converged = be.abs(f_t) < tol
    stop[stop < 0] = geometry.max_iter
    return t, converged, stop, tol


def _newton_rays(n: int, seed: int = 29) -> np.ndarray:
    """A tilted bundle: at normal incidence F(t) is linear and converges in one
    step, so ``max_iter`` could never bite (WP5 finding 5)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-11.0, 11.0, n)
    y = rng.uniform(-11.0, 11.0, n)
    z = np.full(n, -30.0)
    ell = np.full(n, 0.3)
    em = rng.uniform(-0.05, 0.05, n)
    en = np.sqrt(1.0 - ell * ell - em * em)
    x[:4] = 0.0
    y[:4] = 0.0
    em[:4] = 0.0
    ell[:4] = 0.0
    en[:4] = 1.0  # axial: converges at the seed
    x[4:8] = np.nan  # a NaN seed stays NaN and stops the thread at iteration 0
    return np.stack([x, y, z, ell, em, en])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("max_iter", [1, 100])
@pytest.mark.parametrize("kind", ["even", "odd"])
def test_probe_newton_matches_solve_distance_primal(
    mode: str, max_iter: int, kind: str
) -> None:
    """``newton_distance`` == ``_solve_distance_primal`` (t, iters, converged).

    ``t`` is compared on raw components against the REAL solver;
    ``NEWTON_NOT_CONVERGED`` is compared against the solver's own ``converged``
    mask; ``iters`` is compared against a replay whose final ``t`` is first shown
    to equal the solver's bit for bit.
    """
    from optiland.geometries import EvenAsphere, OddAsphere

    lib = tp.trace_library(mode, probes=True)
    planes = _newton_rays(N_RAYS)
    coefficients = (
        [-1.5e-4, 2.5e-7, -3.5e-10] if kind == "even" else [2.0e-4, -3.0e-6, 1.5e-8]
    )
    geom = trace_layout.GEOM_EVEN if kind == "even" else trace_layout.GEOM_ODD
    with metal_backend(mode):
        cls = EvenAsphere if kind == "even" else OddAsphere
        geometry = cls(
            CoordinateSystem(),
            25.0,
            -0.5,
            tol=1e-10,
            max_iter=max_iter,
            coefficients=list(coefficients),
        )
        row = _geometry_row(geometry)
        rays = RealRays(
            *[_metal(planes[q], mode) for q in range(6)],
            _metal(np.ones(N_RAYS), mode),
            _metal(np.full(N_RAYS, 0.55), mode),
        )
        result = geometry._solve_distance_primal(rays)
        ref = raw_of(result.t)
        ref_converged = np.asarray(be.to_numpy(result.converged)).astype(bool)
        ref_finite = np.isfinite(np.asarray(be.to_numpy(result.t)))
        batch_iterations = int(result.iterations)
        replay_rays = RealRays(
            *[_metal(planes[q], mode) for q in range(6)],
            _metal(np.ones(N_RAYS), mode),
            _metal(np.full(N_RAYS, 0.55), mode),
        )
        replay_t, _, replay_stop, tol_eff = _newton_replay(geometry, replay_rays)
        replay_raw = raw_of(replay_t)
    for a, b in zip(replay_raw, ref, strict=True):
        assert np.array_equal(a, b, equal_nan=True), "the replay is not faithful"
    assert tol_eff == 1e-10, "this bundle must not cross the round-off floor"

    got, status, iters = tp.run_probe_distance(
        lib,
        mode,
        planes=planes,
        params=row,
        geom=geom,
        coefficients=np.array(coefficients),
        max_iter=max_iter,
        newton=True,
    )
    assert_raw_equal(got, ref, f"newton t[{kind}][max_iter={max_iter}]")
    not_converged = (status & trace_layout.ST_NEWTON_NOT_CONVERGED) != 0
    assert np.array_equal(not_converged, ~ref_converged & ref_finite)
    assert np.array_equal(iters.astype(np.int64), replay_stop)
    assert int((status & trace_layout.ST_TOL_CROSSOVER).sum()) == 0
    # The bundle carries a NaN lane, so `be.all(converged)` is never true and
    # the batch loop always runs its full cap -- while every thread stops at
    # its own convergence.  That gap IS the design 4.12 / WP5-finding-4
    # divergence, and this is the prediction that pins it.
    assert batch_iterations == max_iter, "the NaN lane keeps all(converged) False"
    assert int(replay_stop.max()) <= max_iter
    if max_iter > 1:
        assert int(replay_stop.max()) < max_iter, "per-thread counts are shorter"
    assert int((iters == 0).sum()) >= 8, "the axial and NaN lanes stop at once"
    if max_iter == 1:
        assert int(not_converged.sum()) > 0, "max_iter = 1 must leave rays short"
    else:
        assert int(not_converged.sum()) == 0, "max_iter = 100 must converge"


@pytest.mark.parametrize("mode", MODES)
def test_probe_newton_flags_the_tolerance_crossover(mode: str) -> None:
    """``TOL_CROSSOVER`` fires exactly where ``8*eps*max(1,|t|) > tol``.

    This is the one semantic divergence of design 4.6: Python's floor uses the
    batch ``max|t|`` and the kernel's uses the thread's own.  The bit is what the
    driver falls back on, so it must be reachable -- at ``tol = 1e-10`` that is
    ``|t| > 3.5e3 mm`` in df64 and ``|t| > 1.1e5 mm`` in sf64.
    """
    from optiland.geometries import EvenAsphere

    lib = tp.trace_library(mode, probes=True)
    threshold = 1e-10 / (8.0 * tp.MACHINE_EPS[mode])
    n = 256
    planes = np.zeros((6, n), dtype=np.float64)
    planes[0] = 1.0
    planes[2] = -np.linspace(0.5 * threshold, 4.0 * threshold, n)
    planes[5] = 1.0
    with metal_backend(mode):
        geometry = EvenAsphere(
            CoordinateSystem(), be.inf, 0.0, tol=1e-10, coefficients=[1e-9]
        )
        row = _geometry_row(geometry)
    _, status, _ = tp.run_probe_distance(
        lib,
        mode,
        planes=planes,
        params=row,
        geom=trace_layout.GEOM_EVEN,
        flags=trace_layout.FL_RADIUS_INF,
        coefficients=np.array([1e-9]),
        max_iter=100,
        newton=True,
    )
    crossed = (status & trace_layout.ST_TOL_CROSSOVER) != 0
    predicted = 8.0 * tp.MACHINE_EPS[mode] * np.maximum(1.0, -planes[2]) > 1e-10
    assert np.array_equal(crossed, predicted)
    assert crossed.any() and not crossed.all(), "both sides of the floor must appear"


# ---------------------------------------------------------------------------
# The whole body against the per-op path (plan 4/WP1 "trace_body in the exact
# Python order"; the tier-A rule of plan 7.1 applied per fixture)
# ---------------------------------------------------------------------------

#: Branch fixtures and the status bit each one MUST raise, so a body that
#: silently stopped reaching a branch fails here instead of comparing two
#: identical no-ops.  ``None`` means "in this mode only" is not required.
_EXPECTED_BITS: dict[str, tuple[tuple[int, str], ...]] = {
    "hubble": ((trace_layout.ST_CLIPPED, "both"),),
    "tir_singlet": (
        (trace_layout.ST_TIR, "both"),
        (trace_layout.ST_MISS, "both"),
    ),
    "miss_bundle": ((trace_layout.ST_MISS, "both"),),
    "rim_bundle": ((trace_layout.ST_CLIPPED, "both"),),
    "rect_aperture": ((trace_layout.ST_CLIPPED, "both"),),
    "ellipse_aperture": ((trace_layout.ST_CLIPPED, "both"),),
    "offset_radial_aperture": ((trace_layout.ST_CLIPPED, "both"),),
    "zero_rmax_aperture": ((trace_layout.ST_CLIPPED, "both"),),
    "exact_grazing_bundle": ((trace_layout.ST_NZ_FLOORED, "both"),),
    "nonconverging_asphere": ((trace_layout.ST_NEWTON_NOT_CONVERGED, "both"),),
    "long_path_asphere": ((trace_layout.ST_TOL_CROSSOVER, "df64"),),
}

#: Every WP5 fixture but the mixed-wavelength one, which the gate refuses
#: before a launch (``mixed_wavelength``) and which the kernel would flag with
#: ``ST_NONUNIFORM_W`` -- covered by its own test below.
_BODY_FIXTURES: tuple[str, ...] = tuple(
    n for n in trace_fixtures.FIXTURES if n != "mixed_wavelength_bundle"
)

#: Rays per differential trace: above the dual-residency threshold (256) and
#: above the tier-A floor of plan 7.1 (N > 1024), so the reference really runs
#: the GPU per-op path and the material cache takes its uniform-key branch.
N_BODY_RAYS = 2048


def _copy_rays(rays: object) -> object:
    """A second bundle holding the SAME encoded words as ``rays``."""
    out = RealRays(
        *(
            be.copy(getattr(rays, attr))
            for attr in ("x", "y", "z", "L", "M", "N", "i", "w")
        )
    )
    out.opd = be.copy(rays.opd)
    return out


def _build_fixture(name: str, n: int) -> tuple[object, object]:
    """``(optic, rays)`` for a WP5 fixture, on the active backend."""
    result = trace_fixtures.FIXTURES[name]()
    optic, factory = result if isinstance(result, tuple) else (result, None)
    rays = (
        factory(optic, n)
        if factory is not None
        else trace_fixtures.pupil_bundle(optic, n)
    )
    return optic, rays


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", _BODY_FIXTURES)
def test_trace_body_matches_the_per_op_path(name: str, mode: str) -> None:
    """``trace_body`` reproduces the per-op path's words, record for record.

    This is the whole point of the kernel: plan 7.1 tier A is raw-component
    equality with the Python path on the same GPU, and the loop's job is to
    call the (already probe-tested) surface steps in the ORDER
    ``_TracingCoordinator.trace`` / ``Surface._trace_real`` call them
    (standard_surface.py:82-101, 299-308).  Both sides read the same encoded
    launch words -- the bundle is packed straight out of its component tensors,
    never through ``to_numpy``/``be.array``, which would re-encode.

    The comparison covers every recorded row (global frame, post-interaction
    direction, post-clip intensity) and the eleven final planes, including the
    ``L0/M0/N0`` the last surface left in ITS local frame.
    """
    with metal_backend(mode):
        optic, rays = _build_fixture(name, N_BODY_RAYS)
        n = int(np.asarray(be.to_numpy(rays.x)).reshape(-1).size)
        assert n > 1024, f"{name}: {n} rays is below the tier-A floor"
        reference = _copy_rays(rays)
        group = optic.surfaces
        group.trace(reference, record=True)

        w0 = _scalar(rays.w)
        records = compile_records(group, w0, mode, record=True)
        out = tp.run_trace(
            tp.trace_library(mode),
            mode,
            launch_bufs=tp.pack_rays(rays, mode, n),
            surf_int=records.surf_int,
            surf_real=records.surf_real,
            coef=records.coef,
            n_rows=records.n_rows,
            write_final=True,
        )

        snap, final = out["snap"], out["final"]
        status, iters = out["status"], out["iters"]
        assert int((iters == trace_layout.ITERS_UNWRITTEN).sum()) == 0
        assert int(status[:, 0, :].sum()) == 0
        assert int(iters[:, 0, :].sum()) == 0

        for s, surface in enumerate(group.surfaces):
            row = int(records.snap_rows[0, s])
            assert row >= 0, f"{name}: surface {s} is not recorded"
            for attr, plane in _SNAP_PLANES:
                assert_raw_equal(
                    [c[plane, 0, row] for c in snap],
                    raw_of(getattr(surface, attr)),
                    f"{name}[{mode}] surface {s} "
                    f"({type(surface.geometry).__name__}) {attr}",
                )
        for attr, plane in _FINAL_PLANES:
            assert_raw_equal(
                [c[plane, 0] for c in final],
                raw_of(getattr(reference, attr)),
                f"{name}[{mode}] final {attr}",
            )

        # `iters` is per-thread and only a Newton row can write it (design
        # 4.12): an exact rule, not a spot check.
        newton = np.isin(
            records.surf_int[0, :, trace_layout.SI_GEOM],
            [trace_layout.GEOM_EVEN, trace_layout.GEOM_ODD],
        )
        assert int(iters[:, ~newton, :].sum()) == 0, (
            f"{name}[{mode}]: a non-Newton row wrote an iteration count"
        )
        if newton.any():
            assert int(iters.sum()) > 0, (
                f"{name}[{mode}]: the Newton rows never iterated"
            )

        for bit, where in _EXPECTED_BITS.get(name, ()):
            if where in ("both", mode):
                assert int(np.count_nonzero(status & bit)) > 0, (
                    f"{name}[{mode}] never raised status bit {bit}: the branch "
                    "the fixture exists for was not taken"
                )


@pytest.mark.parametrize("mode", MODES)
def test_nonuniform_wavelength_raises_its_status_bit(mode: str) -> None:
    """The belt-and-braces ``ST_NONUNIFORM_W`` check of design 4.5.

    The gate refuses a mixed-wavelength bundle before any launch
    (``mixed_wavelength``), so this bit can only appear when the kernel is
    driven around the gate -- which is exactly what this test does.  It is
    OR-ed into every surface row and never into the object row.
    """
    B, S, N = 1, 3, 512
    lib = tp.trace_library(mode)
    surf_int, surf_real, coef, n_rows = tp.mirror_tables(B, S)
    launch = tp.mirror_launch(N)
    launch[trace_layout.Q_W, 0, N // 2 :] = 0.65
    out = tp.run_trace(
        lib,
        mode,
        launch=launch,
        surf_int=surf_int,
        surf_real=surf_real,
        coef=coef,
        n_rows=n_rows,
        write_final=False,
    )
    status = out["status"]
    flagged = (status & trace_layout.ST_NONUNIFORM_W) != 0
    expected = np.zeros((B, S, N), dtype=bool)
    expected[:, 1:, N // 2 :] = True
    assert np.array_equal(flagged, expected)

    uniform = tp.run_trace(
        lib,
        mode,
        launch=tp.mirror_launch(N),
        surf_int=surf_int,
        surf_real=surf_real,
        coef=coef,
        n_rows=n_rows,
        write_final=False,
    )
    assert int(uniform["status"].sum()) == 0


# ---------------------------------------------------------------------------
# Round-0 divergence hooks (plan section 6): each `#ifdef` must be able to
# break the comparison, or the conformance test it certifies proves nothing.
# ---------------------------------------------------------------------------

#: fixture whose trace certifies each site, and the modes it must break.
_BREAK_CERTIFICATES: dict[str, str] = {
    "U_INKERNEL": "cooke",
    "HORNER": "even_asphere_5coeff",
    "COMPOSED_ROTATION": "tilted_triplet_rxryrz",
    "SQR": "cooke",
}


def _body_raw(name: str, mode: str, breaks: tuple[str, ...]) -> list[np.ndarray]:
    """The final planes of one fixture's fused trace, raw components."""
    optic, rays = _build_fixture(name, N_BODY_RAYS)
    n = int(np.asarray(be.to_numpy(rays.x)).reshape(-1).size)
    records = compile_records(optic.surfaces, _scalar(rays.w), mode, record=True)
    out = tp.run_trace(
        tp.trace_library(mode, breaks=breaks),
        mode,
        launch_bufs=tp.pack_rays(rays, mode, n),
        surf_int=records.surf_int,
        surf_real=records.surf_real,
        coef=records.coef,
        n_rows=records.n_rows,
        write_final=True,
    )
    return [c.copy() for c in out["final"]]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("site", sorted(tp.BREAK_SITES))
def test_break_hook_changes_the_trace(site: str, mode: str) -> None:
    """``-DOPTILAND_TRACE_BREAK_<site>`` moves at least one output word.

    Plan section 6 round 0 asks each injection site to be certified rather
    than assumed; a site that cannot change a result is decoration and is
    removed.  The modes each site must break are declared in
    ``tp.BREAK_SITES`` and are asserted BOTH ways: ``U_INKERNEL`` and ``SQR``
    are df64-only (sf64 division is correctly rounded and ``sf::sqr`` IS
    ``mul(a, a)``), so in sf64 they must leave the trace untouched.
    """
    name = _BREAK_CERTIFICATES[site]
    with metal_backend(mode):
        base = _body_raw(name, mode, ())
        broken = _body_raw(name, mode, (site,))
    differs = any(
        not np.array_equal(a, b, equal_nan=True)
        for a, b in zip(base, broken, strict=True)
    )
    if mode in tp.BREAK_SITES[site]:
        assert differs, (
            f"BREAK_{site} did not change {name} in {mode}: the site cannot "
            "certify a conformance test"
        )
    else:
        assert not differs, (
            f"BREAK_{site} changed {name} in {mode}, which tp.BREAK_SITES says "
            "it cannot; the table or the hook is wrong"
        )


def test_break_sqr_site_is_justified() -> None:
    """``df::sqr`` really is a different algorithm from ``mul(x, x)``.

    Plan section 6 keeps the ``BREAK_SQR`` site only if the header implements
    ``sqr`` differently, so the decision is made by inspecting the header
    rather than by assertion: ``sqr_core`` folds the two cross terms into one
    ``fma(2*a.hi, a.lo, ...)`` while ``mul_core`` rounds them separately, and
    ``sf64_core.h`` defines ``sqr(a)`` as literally ``mul(a, a)``.
    """
    df = (tp.KERNEL_DIR / "df64_core.h").read_text(encoding="utf-8")
    sf = (tp.KERNEL_DIR / "sf64_core.h").read_text(encoding="utf-8")
    df_sqr = df[df.index("inline df64 sqr_core(df64 a)") :].split("}", 1)[0]
    df_mul = df[df.index("inline df64 mul_core(df64 a, df64 b)") :].split("}", 1)[0]
    assert "fma(2.0f * a.hi, a.lo" in df_sqr
    assert df_sqr.count("fma(") != df_mul.count("fma(")
    assert "inline sf64 sqr(sf64 a) { return mul(a, a); }" in sf
    print(
        "BREAK_SQR kept: df64 sqr_core has "
        f"{df_sqr.count('fma(')} fma against mul_core's {df_mul.count('fma(')}; "
        "sf64 sqr is mul(a, a), so only df64 can diverge"
    )
