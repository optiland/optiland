"""Frozen buffer/table layout shared by the fused trace kernel and its host driver.

This module is the single source of truth for every constant the MSL kernel
(``kernels/trace.metal``) and the Python driver (``metal/trace.py``) must agree
on: plane indices, table slots, flag bits, status bits, dimension slots and the
positional buffer order.  ``render()`` turns the constants into the C header
``kernels/trace_layout.h``; the header is checked in and
``tests/metal/test_trace_layout.py::test_header_is_rendered`` fails whenever the
two drift apart.

Nothing here imports torch or numpy: the module is importable from the NumPy
path and from tooling.

Layout frozen at I0 (implementation plan section 3.2).  Changing a constant
requires updating plan section 3.2, this module, the rendered header and
``NOTES/fused-trace-research/status.md`` in one commit.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "AP_ELLIPSE",
    "AP_NONE",
    "AP_OFFSET_RADIAL",
    "AP_RADIAL",
    "AP_RECT",
    "BUFFER_ORDER",
    "C_EPS",
    "C_NFLOOR",
    "C_SIZE",
    "D_B",
    "D_C",
    "D_DESIGN_BASE",
    "D_LAUNCH_STRIDE",
    "D_N",
    "D_NROWS",
    "D_P",
    "D_RAY_BASE",
    "D_S",
    "D_SIZE",
    "D_WRITE_FINAL",
    "FL_ABSORBING",
    "FL_AP_IN_ROOT",
    "FL_HAS_APERTURE",
    "FL_HAS_RX",
    "FL_HAS_RY",
    "FL_HAS_RZ",
    "FL_RADIUS_INF",
    "FL_REFLECTIVE",
    "F_I",
    "F_L",
    "F_L0",
    "F_M",
    "F_M0",
    "F_N",
    "F_N0",
    "F_OPD",
    "F_PLANES",
    "F_X",
    "F_Y",
    "F_Z",
    "GEOM_CONIC",
    "GEOM_EVEN",
    "GEOM_ODD",
    "GEOM_OBJECT",
    "GEOM_PLANE",
    "GEOM_STD_INF",
    "HEADER_PATH",
    "ITERS_UNWRITTEN",
    "Q_I",
    "Q_L",
    "Q_M",
    "Q_N",
    "Q_OPD",
    "Q_PLANES",
    "Q_W",
    "Q_X",
    "R_BUFFERS",
    "Q_Y",
    "Q_Z",
    "SI_APCODE",
    "SI_FLAGS",
    "SI_GEOM",
    "SI_MAXITER",
    "SI_NCOEFF",
    "SI_SNAPROW",
    "SI_STEPCOST",
    "SI_STRIDE",
    "SR_ALPHA",
    "SR_AP0",
    "SR_AP1",
    "SR_AP2",
    "SR_AP3",
    "SR_CNRX",
    "SR_CNRY",
    "SR_CNRZ",
    "SR_CRX",
    "SR_CRY",
    "SR_CRZ",
    "SR_K",
    "SR_K1",
    "SR_NPOST",
    "SR_NPRE",
    "SR_NTX",
    "SR_NTY",
    "SR_NTZ",
    "SR_R",
    "SR_R2",
    "SR_SNRX",
    "SR_SNRY",
    "SR_SNRZ",
    "SR_SRX",
    "SR_SRY",
    "SR_SRZ",
    "SR_STRIDE",
    "SR_TOL",
    "SR_TX",
    "SR_TY",
    "SR_TZ",
    "SR_U",
    "SR_U2",
    "ST_CLIPPED",
    "ST_DF_FLOORED",
    "ST_MISS",
    "ST_NEWTON_NOT_CONVERGED",
    "ST_NONUNIFORM_W",
    "ST_NZ_FLOORED",
    "ST_TIR",
    "ST_TOL_CROSSOVER",
    "S_I",
    "S_L",
    "S_M",
    "S_N",
    "S_OPD",
    "S_PLANES",
    "S_X",
    "S_Y",
    "S_Z",
    "render",
    "write_header",
]

# --------------------------------------------------------------------------
# Launch planes: the ray state handed to the kernel, ``launch R[9][Lb][N]``.
# --------------------------------------------------------------------------
Q_X, Q_Y, Q_Z, Q_L, Q_M, Q_N, Q_I, Q_OPD, Q_W = range(9)
Q_PLANES = 9

# --------------------------------------------------------------------------
# Final planes: the post-trace ray state, ``final R[11][B][N]``.
# L0/M0/N0 are the pre-interaction (local) direction cosines of the last
# surface, which ``RealRays`` keeps after a trace.
# --------------------------------------------------------------------------
F_X, F_Y, F_Z, F_L, F_M, F_N, F_I, F_OPD, F_L0, F_M0, F_N0 = range(11)
F_PLANES = 11

# --------------------------------------------------------------------------
# Snapshot planes: per-surface recorded state, ``snap R[8][B][n_rows][N]``.
# --------------------------------------------------------------------------
S_X, S_Y, S_Z, S_L, S_M, S_N, S_I, S_OPD = range(8)
S_PLANES = 8

# --------------------------------------------------------------------------
# Per-surface integer table, ``surf_int int32[B][S][SI_STRIDE]``.
# --------------------------------------------------------------------------
SI_GEOM, SI_FLAGS, SI_NCOEFF, SI_MAXITER, SI_APCODE, SI_SNAPROW, SI_STEPCOST = range(7)
SI_STRIDE = 8

# --------------------------------------------------------------------------
# Geometry codes (SI_GEOM).
# --------------------------------------------------------------------------
GEOM_OBJECT, GEOM_PLANE, GEOM_STD_INF, GEOM_CONIC, GEOM_EVEN, GEOM_ODD = range(6)

# --------------------------------------------------------------------------
# Per-surface flag bits (SI_FLAGS).
# --------------------------------------------------------------------------
(
    FL_HAS_RX,
    FL_HAS_RY,
    FL_HAS_RZ,
    FL_REFLECTIVE,
    FL_HAS_APERTURE,
    FL_AP_IN_ROOT,
    FL_ABSORBING,
    FL_RADIUS_INF,
) = (1 << i for i in range(8))

# --------------------------------------------------------------------------
# Aperture codes (SI_APCODE).
# --------------------------------------------------------------------------
AP_NONE, AP_RADIAL, AP_OFFSET_RADIAL, AP_RECT, AP_ELLIPSE = range(5)

# --------------------------------------------------------------------------
# Per-surface real table, ``surf_real R[B][S][SR_STRIDE]``.
#   0-5    translation and its negation (localize / globalize)
#   6-11   cos/sin of -rz, -ry, -rx  (localize)
#   12-17  cos/sin of +rx, +ry, +rz  (globalize)
#   18-27  geometry, material and Newton scalars
#   28-31  aperture parameters
# --------------------------------------------------------------------------
SR_TX, SR_TY, SR_TZ, SR_NTX, SR_NTY, SR_NTZ = range(6)
SR_CNRZ, SR_SNRZ, SR_CNRY, SR_SNRY, SR_CNRX, SR_SNRX = range(6, 12)
SR_CRX, SR_SRX, SR_CRY, SR_SRY, SR_CRZ, SR_SRZ = range(12, 18)
SR_R, SR_K, SR_K1, SR_R2, SR_NPRE, SR_NPOST, SR_U, SR_U2, SR_ALPHA, SR_TOL = range(
    18, 28
)
SR_AP0, SR_AP1, SR_AP2, SR_AP3 = range(28, 32)
SR_STRIDE = 32

# --------------------------------------------------------------------------
# Dimensions, ``dims int32[D_SIZE]``.
# --------------------------------------------------------------------------
(
    D_S,
    D_N,
    D_B,
    D_LAUNCH_STRIDE,
    D_RAY_BASE,
    D_NROWS,
    D_C,
    D_P,
    D_DESIGN_BASE,
    D_WRITE_FINAL,
) = range(10)
D_SIZE = 16

# --------------------------------------------------------------------------
# Kernel-wide constants, ``consts R[C_SIZE]``.
# --------------------------------------------------------------------------
C_EPS, C_NFLOOR = 0, 1
C_SIZE = 4

# --------------------------------------------------------------------------
# Status bits, ``status uint8[B][S][N]``.
# --------------------------------------------------------------------------
(
    ST_MISS,
    ST_TIR,
    ST_CLIPPED,
    ST_NEWTON_NOT_CONVERGED,
    ST_TOL_CROSSOVER,
    ST_NZ_FLOORED,
    ST_DF_FLOORED,
    ST_NONUNIFORM_W,
) = (1 << i for i in range(8))

# Write-completion sentinel pre-filled into ``iters`` before every launch.
# The gate maps ``max_iter > 254`` to the ``newton_params`` refusal so a real
# iteration count can never collide with it.
ITERS_UNWRITTEN = 0xFF

# --------------------------------------------------------------------------
# Positional binding order.  df64 binds ``hi, lo`` per R buffer (16 bindings),
# sf64 binds one ``long`` buffer per R buffer (10 bindings).
# --------------------------------------------------------------------------
BUFFER_ORDER = (
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

# Which entries of BUFFER_ORDER are R-typed (two float32 buffers in df64, one
# int64 buffer in sf64); the rest bind as a single plain buffer in both modes.
R_BUFFERS = ("launch", "surf_real", "coef", "consts", "snap", "final")

HEADER_PATH = Path(__file__).resolve().parent / "kernels" / "trace_layout.h"

_HEADER_GUARD = "OPTILAND_TRACE_LAYOUT_H"

_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "launch planes (launch R[9][Lb][N])",
        ("Q_X", "Q_Y", "Q_Z", "Q_L", "Q_M", "Q_N", "Q_I", "Q_OPD", "Q_W", "Q_PLANES"),
    ),
    (
        "final planes (final R[11][B][N])",
        (
            "F_X",
            "F_Y",
            "F_Z",
            "F_L",
            "F_M",
            "F_N",
            "F_I",
            "F_OPD",
            "F_L0",
            "F_M0",
            "F_N0",
            "F_PLANES",
        ),
    ),
    (
        "snapshot planes (snap R[8][B][n_rows][N])",
        ("S_X", "S_Y", "S_Z", "S_L", "S_M", "S_N", "S_I", "S_OPD", "S_PLANES"),
    ),
    (
        "per-surface integer table (surf_int int32[B][S][SI_STRIDE])",
        (
            "SI_GEOM",
            "SI_FLAGS",
            "SI_NCOEFF",
            "SI_MAXITER",
            "SI_APCODE",
            "SI_SNAPROW",
            "SI_STEPCOST",
            "SI_STRIDE",
        ),
    ),
    (
        "geometry codes (SI_GEOM)",
        (
            "GEOM_OBJECT",
            "GEOM_PLANE",
            "GEOM_STD_INF",
            "GEOM_CONIC",
            "GEOM_EVEN",
            "GEOM_ODD",
        ),
    ),
    (
        "per-surface flag bits (SI_FLAGS)",
        (
            "FL_HAS_RX",
            "FL_HAS_RY",
            "FL_HAS_RZ",
            "FL_REFLECTIVE",
            "FL_HAS_APERTURE",
            "FL_AP_IN_ROOT",
            "FL_ABSORBING",
            "FL_RADIUS_INF",
        ),
    ),
    (
        "aperture codes (SI_APCODE)",
        ("AP_NONE", "AP_RADIAL", "AP_OFFSET_RADIAL", "AP_RECT", "AP_ELLIPSE"),
    ),
    (
        "per-surface real table (surf_real R[B][S][SR_STRIDE])",
        (
            "SR_TX",
            "SR_TY",
            "SR_TZ",
            "SR_NTX",
            "SR_NTY",
            "SR_NTZ",
            "SR_CNRZ",
            "SR_SNRZ",
            "SR_CNRY",
            "SR_SNRY",
            "SR_CNRX",
            "SR_SNRX",
            "SR_CRX",
            "SR_SRX",
            "SR_CRY",
            "SR_SRY",
            "SR_CRZ",
            "SR_SRZ",
            "SR_R",
            "SR_K",
            "SR_K1",
            "SR_R2",
            "SR_NPRE",
            "SR_NPOST",
            "SR_U",
            "SR_U2",
            "SR_ALPHA",
            "SR_TOL",
            "SR_AP0",
            "SR_AP1",
            "SR_AP2",
            "SR_AP3",
            "SR_STRIDE",
        ),
    ),
    (
        "dimensions (dims int32[D_SIZE])",
        (
            "D_S",
            "D_N",
            "D_B",
            "D_LAUNCH_STRIDE",
            "D_RAY_BASE",
            "D_NROWS",
            "D_C",
            "D_P",
            "D_DESIGN_BASE",
            "D_WRITE_FINAL",
            "D_SIZE",
        ),
    ),
    ("kernel constants (consts R[C_SIZE])", ("C_EPS", "C_NFLOOR", "C_SIZE")),
    (
        "status bits (status uint8[B][S][N])",
        (
            "ST_MISS",
            "ST_TIR",
            "ST_CLIPPED",
            "ST_NEWTON_NOT_CONVERGED",
            "ST_TOL_CROSSOVER",
            "ST_NZ_FLOORED",
            "ST_DF_FLOORED",
            "ST_NONUNIFORM_W",
        ),
    ),
    ("write-completion sentinel", ("ITERS_UNWRITTEN",)),
)


def render() -> str:
    """Return the text of ``kernels/trace_layout.h`` for the current constants."""
    globals_ = globals()
    out: list[str] = [
        "// Generated by metal/trace_layout.py -- do not edit.",
        "// Regenerate with:",
        "//   python -m optiland.backend.torch_backend.metal.trace_layout --write",
        "",
        f"#ifndef {_HEADER_GUARD}",
        f"#define {_HEADER_GUARD}",
        "",
    ]
    for title, names in _SECTIONS:
        out.append(f"// {title}")
        width = max(len(n) for n in names)
        for name in names:
            value = globals_[name]
            out.append(f"#define OT_{name.ljust(width)} {value}")
        out.append("")
    out.append("// positional binding order (argument order of every entry point)")
    out.append("// R-typed buffers bind as (hi, lo) in df64 and as one long in sf64;")
    out.append("// 16 bindings in df64, 10 in sf64.")
    for i, name in enumerate(BUFFER_ORDER):
        kind = "R" if name in R_BUFFERS else "plain"
        out.append(f"// {i}: {name} ({kind})")
    out.append("")
    out.append(f"#endif  // {_HEADER_GUARD}")
    out.append("")
    return "\n".join(out)


def write_header(path: str | Path | None = None) -> Path:
    """Write ``render()`` to ``path`` (default ``kernels/trace_layout.h``)."""
    target = Path(path) if path is not None else HEADER_PATH
    target.write_text(render(), encoding="utf-8")
    return target


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument(
        "--write", action="store_true", help="regenerate kernels/trace_layout.h"
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when the header is stale"
    )
    args = parser.parse_args(argv)
    if args.write:
        print(f"wrote {write_header()}")
        return 0
    if args.check:
        current = (
            HEADER_PATH.read_text(encoding="utf-8") if HEADER_PATH.exists() else ""
        )
        if current != render():
            print("trace_layout.h is stale; run --write")
            return 1
        print("trace_layout.h is current")
        return 0
    print(render(), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())
