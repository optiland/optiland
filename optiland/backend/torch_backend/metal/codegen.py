"""Generate the flat 1-D elementwise Metal kernels for df64 and sf64.

The op set lives in :data:`OPS`, a declarative table mapping an op name to an
:class:`OpSpec` (arity, df64 expression, sf64 expression, result kind, input
kinds and the header that provides the underlying ``df::``/``sf::`` function).
Kernels are rendered from that table as source text, so callers can pick any
subset of ops (for example only those whose headers exist yet) and compile
one amalgamated library per representation.

Generated kernel names and signatures (buffers bind by position, in order):

``<op>_df64``
    ``device const float* a_hi, a_lo[, b_hi, b_lo[, c_hi, c_lo]]`` then
    ``device float* o_hi, o_lo`` (or ``device bool* o`` for predicates).
``<op>_df64_s`` / ``<op>_df64_rs``
    Binary ops with the second / first operand passed as ``constant float2&``.
``<op>_sf64``, ``<op>_sf64_s``, ``<op>_sf64_rs``
    Same with ``device const long*`` inputs, ``device long*`` outputs and
    ``constant long&`` scalars (IEEE binary64 bit patterns).

``where`` takes a leading ``device const bool*`` condition. Every kernel runs
exactly ``threads`` threads (``thread_position_in_grid`` indexes the flat
buffers directly), so the launcher must dispatch ``threads=[numel, 1, 1]``.

Run ``python -m optiland.backend.torch_backend.metal.codegen`` to write the full
generated text to ``kernels/elementwise_generated.metal`` for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

REPRS: tuple[str, ...] = ("df64", "sf64")
SCALAR_SIDES: tuple[str, ...] = ("right", "left")
_SCALAR_SUFFIX = {"right": "_s", "left": "_rs"}

# Header that provides each family of functions, per representation.
DF_CORE = "df64_core.h"
DF_EXP = "df64_math_exp.h"
DF_TRIG = "df64_math_trig.h"
DF_SPECIAL = "df64_math_special.h"
SF_CORE = "sf64_core.h"
SF_MATH = "sf64_math.h"

GENERATED_FILE = Path(__file__).with_name("kernels") / "elementwise_generated.metal"


class OpSpec(NamedTuple):
    """Declarative description of one elementwise op.

    Attributes:
        arity: Number of inputs (1, 2 or 3).
        df_expr: df64 expression template using ``{a}``, ``{b}``, ``{c}``.
        sf_expr: sf64 expression template using the same placeholders.
        result: ``"value"`` (a df64/sf64 result) or ``"bool"`` (a predicate).
        inputs: Kind of each input, ``"value"`` or ``"bool"``.
        df_header: Kernel header that must be present for the df64 kernel.
        sf_header: Kernel header that must be present for the sf64 kernel.
        sf_inexact: True when the sf64 kernel is the v1 bridge through df64
            (``sf64_math.h``): the result is rounded to 48 bits (plus the df64
            function's error) instead of being correctly rounded binary64.
            :func:`op_machine_eps` reports ``2^-48`` for these ops in sf64 mode.
    """

    arity: int
    df_expr: str
    sf_expr: str
    result: str = "value"
    inputs: tuple[str, ...] = ()
    df_header: str = DF_CORE
    sf_header: str = SF_CORE
    sf_inexact: bool = False


def _op(
    arity: int,
    df_expr: str,
    sf_expr: str,
    *,
    result: str = "value",
    inputs: tuple[str, ...] | None = None,
    df_header: str = DF_CORE,
    sf_header: str = SF_CORE,
) -> OpSpec:
    return OpSpec(
        arity,
        df_expr,
        sf_expr,
        result,
        inputs if inputs is not None else ("value",) * arity,
        df_header,
        sf_header,
        sf_header == SF_MATH,
    )


def _u(fn: str, df_header: str = DF_CORE, sf_header: str = SF_CORE) -> OpSpec:
    """Unary op whose df:: and sf:: function share the op name."""
    return _op(
        1,
        f"df::{fn}({{a}})",
        f"sf::{fn}({{a}})",
        df_header=df_header,
        sf_header=sf_header,
    )


def _b(fn: str, df_header: str = DF_CORE, sf_header: str = SF_CORE) -> OpSpec:
    """Binary op whose df:: and sf:: function share the op name."""
    return _op(
        2,
        f"df::{fn}({{a}}, {{b}})",
        f"sf::{fn}({{a}}, {{b}})",
        df_header=df_header,
        sf_header=sf_header,
    )


def _pred1(fn: str) -> OpSpec:
    return _op(1, f"df::{fn}({{a}})", f"sf::{fn}({{a}})", result="bool")


def _pred2(fn: str) -> OpSpec:
    return _op(2, f"df::{fn}({{a}}, {{b}})", f"sf::{fn}({{a}}, {{b}})", result="bool")


#: The elementwise op table. Insertion order is the generation order.
OPS: dict[str, OpSpec] = {
    # ---- unary, core ------------------------------------------------------
    "neg": _u("neg"),
    "abs": _u("abs"),
    "sign": _u("sign"),
    "sqrt": _u("sqrt"),
    "rsqrt": _u("rsqrt"),
    "recip": _u("recip"),
    "floor": _u("floor"),
    "ceil": _u("ceil"),
    "trunc": _u("trunc"),
    "rint": _u("rint"),
    "round_away": _u("round_away"),
    # ---- unary, exp/log family --------------------------------------------
    "exp": _u("exp", DF_EXP, SF_MATH),
    "exp2": _u("exp2", DF_EXP, SF_MATH),
    "expm1": _u("expm1", DF_EXP, SF_MATH),
    "log": _u("log", DF_EXP, SF_MATH),
    "log1p": _u("log1p", DF_EXP, SF_MATH),
    "log2": _u("log2", DF_EXP, SF_MATH),
    "log10": _u("log10", DF_EXP, SF_MATH),
    "cbrt": _u("cbrt", DF_EXP, SF_MATH),
    # ---- unary, trigonometric ---------------------------------------------
    "sin": _u("sin", DF_TRIG, SF_MATH),
    "cos": _u("cos", DF_TRIG, SF_MATH),
    "tan": _u("tan", DF_TRIG, SF_MATH),
    "asin": _u("asin", DF_TRIG, SF_MATH),
    "acos": _u("acos", DF_TRIG, SF_MATH),
    "atan": _u("atan", DF_TRIG, SF_MATH),
    # ---- unary, hyperbolic and special ------------------------------------
    "sinh": _u("sinh", DF_SPECIAL, SF_MATH),
    "cosh": _u("cosh", DF_SPECIAL, SF_MATH),
    "tanh": _u("tanh", DF_SPECIAL, SF_MATH),
    "asinh": _u("asinh", DF_SPECIAL, SF_MATH),
    "acosh": _u("acosh", DF_SPECIAL, SF_MATH),
    "atanh": _u("atanh", DF_SPECIAL, SF_MATH),
    "erf": _u("erf", DF_SPECIAL, SF_MATH),
    "erfc": _u("erfc", DF_SPECIAL, SF_MATH),
    "erfinv": _u("erfinv", DF_SPECIAL, SF_MATH),
    "lgamma": _u("lgamma", DF_SPECIAL, SF_MATH),
    # ---- unary predicates -------------------------------------------------
    "isnan": _pred1("is_nan"),
    "isinf": _pred1("is_inf"),
    "isfinite": _pred1("is_finite"),
    # ---- binary, core -----------------------------------------------------
    "add": _b("add"),
    "sub": _b("sub"),
    "mul": _b("mul"),
    "div": _b("div"),
    "fmod": _b("fmod"),
    "remainder_py": _b("remainder_py"),
    "copysign": _b("copysign"),
    "minimum": _b("minimum"),
    "maximum": _b("maximum"),
    "fmin": _b("fmin"),
    "fmax": _b("fmax"),
    # ---- binary, transcendental -------------------------------------------
    "pow": _b("pow", DF_EXP, SF_MATH),
    "atan2": _b("atan2", DF_TRIG, SF_MATH),
    "hypot": _b("hypot", DF_SPECIAL, SF_MATH),
    # ---- binary predicates ------------------------------------------------
    "eq": _pred2("eq"),
    "ne": _pred2("ne"),
    "lt": _pred2("lt"),
    "le": _pred2("le"),
    "gt": _pred2("gt"),
    "ge": _pred2("ge"),
    # ---- ternary ----------------------------------------------------------
    # fma(a, b, c) = a*b + c with two roundings (df::mul_add), not a single-
    # rounding fused op; sf::fma is the exact softfloat fma.
    "fma": _op(3, "df::mul_add({a}, {b}, {c})", "sf::fma({a}, {b}, {c})"),
    # lerp(start, end, weight) = start + weight * (end - start)  (torch.lerp)
    "lerp": _op(
        3,
        "df::add({a}, df::mul({c}, df::sub({b}, {a})))",
        "sf::add({a}, sf::mul({c}, sf::sub({b}, {a})))",
    ),
    # addcmul(input, t1, t2) = input + t1 * t2 ; addcdiv = input + t1 / t2
    "addcmul": _op(
        3, "df::add({a}, df::mul({b}, {c}))", "sf::add({a}, sf::mul({b}, {c}))"
    ),
    "addcdiv": _op(
        3, "df::add({a}, df::div({b}, {c}))", "sf::add({a}, sf::div({b}, {c}))"
    ),
    # clamp(x, lo, hi) = min(max(x, lo), hi); NaN in x propagates (torch.clamp).
    "clamp": _op(
        3,
        "df::minimum(df::maximum({a}, {b}), {c})",
        "sf::minimum(sf::maximum({a}, {b}), {c})",
    ),
    # where(cond, a, b)
    "where": _op(
        3, "({a} ? {b} : {c})", "({a} ? {b} : {c})", inputs=("bool", "value", "value")
    ),
}

UNARY: tuple[str, ...] = tuple(n for n, s in OPS.items() if s.arity == 1)
BINARY: tuple[str, ...] = tuple(n for n, s in OPS.items() if s.arity == 2)
TERNARY: tuple[str, ...] = tuple(n for n, s in OPS.items() if s.arity == 3)
#: Ops whose sf64 kernel is the inexact (48-bit) bridge through df64: the 24
#: unary transcendentals plus pow, atan2 and hypot.
SF64_INEXACT: tuple[str, ...] = tuple(n for n, s in OPS.items() if s.sf_inexact)

#: Relative precision of each representation's own arithmetic.
MACHINE_EPS: dict[str, float] = {"df64": 2.0**-48, "sf64": 2.0**-53}


def op_machine_eps(op: str, repr: str) -> float:
    """Relative precision carried by ``op``'s result in representation ``repr``.

    ``2^-53`` for sf64 ops that are correctly rounded binary64 (the core
    arithmetic, rounding, fmod/remainder, min/max, predicates, ...), ``2^-48``
    for every df64 op and for the sf64 ops flagged :attr:`OpSpec.sf_inexact`
    (the v1 bridge rounds the argument and the result to 48 bits; the df64
    function's own error, a few u^2, comes on top, see NOTES/04 section 2.1).

    Args:
        op: Op name from :data:`OPS`.
        repr: ``"df64"`` or ``"sf64"``.

    Returns:
        float: The relative precision.
    """
    _check_repr(repr)
    spec = OPS[op]
    if repr == "sf64" and not spec.sf_inexact:
        return MACHINE_EPS["sf64"]
    return MACHINE_EPS["df64"]


_ARG_NAMES = ("a", "b", "c")

_SF64_PRELUDE = """\
// Bit-pattern <-> sf64 bridges used by every sf64 kernel. sf64 is a plain struct
// with a public `ulong bits` member (see sf64_core.h).
namespace ew {
inline sf64 sf_load(long v) { sf64 r; r.bits = as_type<ulong>(v); return r; }
inline long sf_store(sf64 v) { return as_type<long>(v.bits); }
}  // namespace ew
"""


def header_for(op: str, repr: str) -> str:
    """Return the kernel header that must be present for ``op`` in ``repr``."""
    spec = OPS[op]
    return spec.df_header if repr == "df64" else spec.sf_header


def headers_for(op: str, repr: str) -> tuple[str, ...]:
    """Return every header that must be present for ``op`` in ``repr``.

    The sf64 transcendentals are bridged through the df64 implementations
    (``sf64_math.h`` emits each family only when the matching df64 math header
    preceded it), so an sf64 bridge op needs both ``sf64_math.h`` and the df64
    header that provides the underlying ``df::`` function.
    """
    spec = OPS[op]
    if repr == "df64":
        return (spec.df_header,)
    if spec.sf_header == SF_MATH:
        return (spec.sf_header, spec.df_header)
    return (spec.sf_header,)


def ops_for_headers(
    repr: str, headers: Iterable[str], ops: Iterable[str] | None = None
) -> list[str]:
    """Select the ops whose providing headers are all among ``headers``.

    Args:
        repr: ``"df64"`` or ``"sf64"``.
        headers: Header file names that are available (basenames as in ``kernels/``).
        ops: Optional subset to select from (defaults to every op).

    Returns:
        list[str]: Op names in table order.
    """
    _check_repr(repr)
    have = set(headers)
    names = list(OPS) if ops is None else list(ops)
    for n in names:
        if n not in OPS:
            raise KeyError(f"unknown elementwise op {n!r}")
    return [
        n for n in OPS if n in names and all(h in have for h in headers_for(n, repr))
    ]


def kernel_name(op: str, repr: str, scalar_side: str | None = None) -> str:
    """Return the kernel name for ``op`` in ``repr`` (optionally a scalar variant).

    Args:
        op: Op name from :data:`OPS`.
        repr: ``"df64"`` or ``"sf64"``.
        scalar_side: ``None`` for the tensor-tensor kernel, ``"right"`` when the
            second operand is a scalar (``_s``), ``"left"`` when the first is
            (``_rs``). Only binary ops have scalar variants.

    Returns:
        str: The Metal kernel function name.
    """
    _check_repr(repr)
    spec = OPS[op]
    if scalar_side is None:
        return f"{op}_{repr}"
    if spec.arity != 2:
        raise ValueError(f"scalar variants exist only for binary ops, not {op!r}")
    if scalar_side not in _SCALAR_SUFFIX:
        raise ValueError(
            f"scalar_side must be one of {SCALAR_SIDES}, got {scalar_side!r}"
        )
    return f"{op}_{repr}{_SCALAR_SUFFIX[scalar_side]}"


def kernel_variants(op: str) -> list[str | None]:
    """Return the scalar-side variants generated for ``op`` (``None`` = tensor form)."""
    return [None, *SCALAR_SIDES] if OPS[op].arity == 2 else [None]


def kernel_names(ops: Iterable[str] | None = None, repr: str = "df64") -> list[str]:
    """List every kernel name generated for ``ops`` in ``repr``."""
    names = list(OPS) if ops is None else list(ops)
    return [kernel_name(op, repr, side) for op in names for side in kernel_variants(op)]


def _check_repr(repr: str) -> None:
    if repr not in REPRS:
        raise ValueError(f"repr must be one of {REPRS}, got {repr!r}")


def _params_and_loads(
    spec: OpSpec, repr: str, scalar_side: str | None
) -> tuple[list[str], list[str], dict[str, str]]:
    """Build parameter declarations, load statements and expression bindings."""
    params: list[str] = []
    loads: list[str] = []
    binds: dict[str, str] = {}
    scalar_index = {None: -1, "right": 1, "left": 0}[scalar_side]
    for idx in range(spec.arity):
        name = _ARG_NAMES[idx]
        kind = spec.inputs[idx]
        buf = len(params)
        if kind == "bool":
            params.append(f"device const bool* {name}_mask [[buffer({buf})]]")
            loads.append(f"const bool {name} = {name}_mask[i];")
        elif idx == scalar_index:
            if repr == "df64":
                params.append(f"constant float2& {name}_s [[buffer({buf})]]")
                loads.append(f"const df64 {name} = df::make({name}_s.x, {name}_s.y);")
            else:
                params.append(f"constant long& {name}_s [[buffer({buf})]]")
                loads.append(f"const sf64 {name} = ew::sf_load({name}_s);")
        elif repr == "df64":
            params.append(f"device const float* {name}_hi [[buffer({buf})]]")
            params.append(f"device const float* {name}_lo [[buffer({buf + 1})]]")
            loads.append(f"const df64 {name} = df::make({name}_hi[i], {name}_lo[i]);")
        else:
            params.append(f"device const long* {name}_bits [[buffer({buf})]]")
            loads.append(f"const sf64 {name} = ew::sf_load({name}_bits[i]);")
        binds[name] = name
    return params, loads, binds


def generate_kernel(op: str, repr: str, scalar_side: str | None = None) -> str:
    """Render one kernel function as Metal source.

    Args:
        op: Op name from :data:`OPS`.
        repr: ``"df64"`` or ``"sf64"``.
        scalar_side: ``None``, ``"right"`` or ``"left"`` (see :func:`kernel_name`).

    Returns:
        str: Metal source for the kernel.
    """
    spec = OPS[op]
    name = kernel_name(op, repr, scalar_side)
    params, loads, binds = _params_and_loads(spec, repr, scalar_side)
    expr = (spec.df_expr if repr == "df64" else spec.sf_expr).format(**binds)
    buf = len(params)
    if spec.result == "bool":
        params.append(f"device bool* o [[buffer({buf})]]")
        stores = [f"o[i] = {expr};"]
    elif repr == "df64":
        params.append(f"device float* o_hi [[buffer({buf})]]")
        params.append(f"device float* o_lo [[buffer({buf + 1})]]")
        stores = [f"const df64 r = {expr};", "o_hi[i] = r.hi;", "o_lo[i] = r.lo;"]
    else:
        params.append(f"device long* o [[buffer({buf})]]")
        stores = [f"o[i] = ew::sf_store({expr});"]
    params.append("uint i [[thread_position_in_grid]]")
    sig = f"kernel void {name}(\n    " + ",\n    ".join(params) + ")"
    body = "\n".join("    " + line for line in [*loads, *stores])
    return f"{sig} {{\n{body}\n}}\n"


def build(
    ops: Iterable[str] | None = None,
    reprs: Sequence[str] = REPRS,
) -> str:
    """Render the kernels for ``ops`` in each of ``reprs`` as one source string.

    The text contains only kernel definitions (plus the sf64 load/store bridge);
    it must be appended after the headers that define ``df::``/``sf::``.

    Args:
        ops: Op names to generate (default: every op in :data:`OPS`).
        reprs: Representations to generate, subset of ``("df64", "sf64")``.

    Returns:
        str: Metal source text.
    """
    names = list(OPS) if ops is None else list(ops)
    for n in names:
        if n not in OPS:
            raise KeyError(f"unknown elementwise op {n!r}")
    out = [
        "// ---- elementwise kernels (GENERATED by codegen.py; do not edit) ----",
        "#pragma METAL fp math_mode(safe)",
        "#pragma METAL fp contract(off)",
        "",
    ]
    for repr in reprs:
        _check_repr(repr)
        out.append(f"// ======== {repr} ({len(names)} ops) ========")
        if repr == "sf64":
            out.append(_SF64_PRELUDE)
        for op in names:
            for side in kernel_variants(op):
                out.append(generate_kernel(op, repr, side))
    return "\n".join(out)


def write_generated(
    path: Path | str | None = None, ops: Iterable[str] | None = None
) -> Path:
    """Write the full generated text to disk for inspection.

    Args:
        path: Destination (default :data:`GENERATED_FILE`).
        ops: Optional op subset.

    Returns:
        Path: The written file.
    """
    dest = Path(path) if path is not None else GENERATED_FILE
    dest.write_text(build(ops))
    return dest


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: write the generated kernels to a file."""
    parser = argparse.ArgumentParser(
        description="Generate elementwise df64/sf64 Metal kernels."
    )
    parser.add_argument(
        "--out", type=Path, default=GENERATED_FILE, help="output .metal path"
    )
    parser.add_argument(
        "--ops", nargs="*", default=None, help="subset of op names (default: all)"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the op table and exit"
    )
    args = parser.parse_args(argv)
    if args.list:
        for name, spec in OPS.items():
            print(
                f"{name:14s} arity={spec.arity} result={spec.result:5s} "
                f"df={spec.df_header} sf64={'inexact' if spec.sf_inexact else 'exact'}"
            )
        return 0
    dest = write_generated(args.out, args.ops)
    n = len(kernel_names(args.ops, "df64")) + len(kernel_names(args.ops, "sf64"))
    print(f"wrote {n} kernels to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
