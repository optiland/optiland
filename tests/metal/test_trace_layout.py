"""WP0: the frozen layout contract and its rendered C header.

Every assertion here predicts an exact result (plan 0.2.6): the header text is
predicted by ``render()``, and the constant sets are predicted slot by slot.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from optiland.backend.torch_backend.metal import trace_adapters as A  # noqa: E402
from optiland.backend.torch_backend.metal import trace_layout as L  # noqa: E402


def test_header_is_rendered():
    """kernels/trace_layout.h is exactly what trace_layout.render() produces."""
    path = Path(L.HEADER_PATH)
    assert path.exists(), f"{path} is missing; run trace_layout.py --write"
    assert path.read_text(encoding="utf-8") == L.render(), (
        "kernels/trace_layout.h is stale; regenerate with "
        "`python -m optiland.backend.torch_backend.metal.trace_layout --write`"
    )


def _named(prefix, exclude=()):
    return {
        name: getattr(L, name)
        for name in dir(L)
        if name.startswith(prefix)
        and name not in exclude
        and isinstance(getattr(L, name), int)
    }


def test_constants_are_disjoint_and_dense():
    # --- plane indices: dense 0..n-1, no collisions -----------------------
    q = _named("Q_", exclude=("Q_PLANES",))
    assert sorted(q.values()) == list(range(9)), q
    assert L.Q_PLANES == 9

    f = _named("F_", exclude=("F_PLANES",))
    assert sorted(f.values()) == list(range(11)), f
    assert L.F_PLANES == 11

    s = _named("S_", exclude=("S_PLANES",))
    assert sorted(s.values()) == list(range(8)), s
    assert L.S_PLANES == 8

    # --- surf_int slots ---------------------------------------------------
    si = _named("SI_", exclude=("SI_STRIDE",))
    assert sorted(si.values()) == list(range(7)), si
    assert all(v < L.SI_STRIDE for v in si.values())
    assert L.SI_STRIDE == 8

    # --- surf_real slots: dense 0..31, all below the stride ---------------
    sr = _named("SR_", exclude=("SR_STRIDE",))
    assert len(sr) == 32, sorted(sr)
    assert sorted(sr.values()) == list(range(32)), sr
    assert all(v < L.SR_STRIDE for v in sr.values())
    assert L.SR_STRIDE == 32

    # --- dims slots -------------------------------------------------------
    d = _named("D_", exclude=("D_SIZE",))
    assert sorted(d.values()) == list(range(10)), d
    assert all(v < L.D_SIZE for v in d.values())
    assert L.D_SIZE == 16

    # --- consts slots -----------------------------------------------------
    assert (L.C_EPS, L.C_NFLOOR) == (0, 1)
    assert L.C_SIZE == 4

    # --- geometry and aperture codes: dense, disjoint ---------------------
    geom = _named("GEOM_")
    assert sorted(geom.values()) == list(range(6)), geom
    ap = _named("AP_")
    assert sorted(ap.values()) == list(range(5)), ap


def test_flag_and_status_bits_are_distinct_single_bits():
    """``SI_FLAGS`` carries 10 bits, the status byte exactly 8.

    Bits 0-7 of ``FL_`` are plan 3.2's frozen set.  Bits 8-9
    (``FL_K1_ON_RIGHT``, ``FL_R_ON_RIGHT``) were added by the round-2 fix for
    R2-V1-03: they select the operand side of the two conic products, and they
    live in ``SI_FLAGS`` because it is an int32 that every ``sag_of`` /
    ``normal_of`` call site already has in scope.  ``ST_`` must stay at 8 --
    the status plane is a ``uchar``.
    """
    counts = {"FL_": 10, "ST_": 8}
    for prefix, count in counts.items():
        bits = _named(prefix)
        assert len(bits) == count, sorted(bits)
        for name, value in bits.items():
            assert value > 0 and value & (value - 1) == 0, f"{name} is not a single bit"
        assert sorted(bits.values()) == [1 << i for i in range(count)], bits
    assert max(_named("ST_").values()) <= 0x80, "the status plane is a uchar"
    assert (L.FL_K1_ON_RIGHT, L.FL_R_ON_RIGHT) == (1 << 8, 1 << 9)


def test_iters_sentinel_is_out_of_range():
    """ITERS_UNWRITTEN can never collide with a real iteration count."""
    assert L.ITERS_UNWRITTEN == 0xFF
    assert L.ITERS_UNWRITTEN > 254


def test_buffer_order_is_the_frozen_positional_order():
    assert L.BUFFER_ORDER == (
        "launch",
        "surf_int",
        "surf_real",
        "coef",
        "dims",
        "consts",
        "snap",
        "final",
        "status",
        "iters",
    )
    assert len(set(L.BUFFER_ORDER)) == len(L.BUFFER_ORDER)
    assert set(L.R_BUFFERS) <= set(L.BUFFER_ORDER)
    # df64 binds hi+lo for R buffers, sf64 one long buffer each.
    assert L.R_BUFFERS == ("launch", "surf_real", "coef", "consts", "snap", "final")
    n_plain = len(L.BUFFER_ORDER) - len(L.R_BUFFERS)
    assert n_plain == 4
    assert n_plain + 2 * len(L.R_BUFFERS) == 16
    assert n_plain + len(L.R_BUFFERS) == 10


def test_structural_reasons_has_exactly_six_members():
    assert len(A.STRUCTURAL_REASONS) == 6
    assert (
        frozenset(
            {
                A.FusedTraceSkip.GROUP_TYPE,
                A.FusedTraceSkip.RAYS_TYPE,
                A.FusedTraceSkip.RAYS_SHAPE,
                A.FusedTraceSkip.HOST_RESIDENT,
                A.FusedTraceSkip.REQUIRES_GRAD,
                A.FusedTraceSkip.SKIP,
            }
        )
        == A.STRUCTURAL_REASONS
    )
    assert A.STRUCTURAL_REASONS.isdisjoint(A.FEATURE_REASONS)
    assert frozenset(A.FusedTraceSkip) == A.STRUCTURAL_REASONS | A.FEATURE_REASONS


def test_refusal_reason_values_are_unique_and_snake_case():
    values = [r.value for r in A.FusedTraceSkip]
    assert len(set(values)) == len(values)
    for value in values:
        assert value == value.lower()
        assert value.replace("_", "").isalnum()


def test_adapter_registries_are_filled_with_the_frozen_codes():
    """WP0 shipped the registries empty; WP2 filled them (plan 3.1).

    The original WP0 assertion was ``== {}``, which described the C1 skeleton
    only and has been false since WP2 registered the adapters.  What this file
    owns is the layout side of the registries, so the replacement predicts
    exactly that: the sizes of plan 1.1's supported set, and every registered
    code equal to a frozen ``trace_layout`` code, used exactly once.  The
    registry *contents* (exact-type keying, per-class codes, fixtures) are
    predicted by ``test_trace_adapters.py``.
    """
    assert len(A.GEOMETRY_ADAPTERS) == 4
    assert len(A.APERTURE_ADAPTERS) == 4
    assert len(A.INTERACTION_ADAPTERS) == 1

    geom_codes = sorted(a.code for a in A.GEOMETRY_ADAPTERS.values())
    assert geom_codes == sorted([L.GEOM_PLANE, L.GEOM_CONIC, L.GEOM_EVEN, L.GEOM_ODD])
    aperture_codes = sorted(a.code for a in A.APERTURE_ADAPTERS.values())
    assert aperture_codes == sorted(
        [L.AP_RADIAL, L.AP_OFFSET_RADIAL, L.AP_RECT, L.AP_ELLIPSE]
    )
    # AP_NONE and GEOM_OBJECT/GEOM_STD_INF are written by the record compiler,
    # never by an adapter, so no registered adapter may claim them.
    assert L.AP_NONE not in aperture_codes
    assert L.GEOM_OBJECT not in geom_codes
    assert L.GEOM_STD_INF not in geom_codes

    # One interaction model in v1 (plan 1.1): RefractiveReflectiveModel, code 0.
    ((interaction_cls, interaction_adapter),) = A.INTERACTION_ADAPTERS.items()
    assert interaction_cls.__name__ == "RefractiveReflectiveModel"
    assert interaction_adapter.code == 0


def test_register_rejects_duplicates_and_non_adapters():
    class _Dummy:
        pass

    adapter = A.GeometryAdapter(
        code=L.GEOM_PLANE,
        cls=_Dummy,
        check=lambda obj: None,
        fill=lambda obj, ri, rr, rc, ctx: None,
        grad_params=lambda obj: [],
    )
    try:
        A.register(adapter)
        with pytest.raises(ValueError):
            A.register(adapter)
    finally:
        A.GEOMETRY_ADAPTERS.pop(_Dummy, None)
    with pytest.raises(TypeError):
        A.register(object())
