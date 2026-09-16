"""Assemble, compile, verify and launch the elementwise Metal kernel libraries.

One :class:`MetalLibrary` per representation (``df64`` or ``sf64``) holds the
amalgamated source (headers + reduction/matmul kernels + generated elementwise
kernels), the compiled ``torch.mps`` library object, and a ``launch`` method
that binds plain contiguous component tensors to the elementwise kernels. The
reduction and matmul kernels (``sum_df64``, ``matmul_df64``, ...) are attributes
of ``lib`` as well; their launch conventions are documented in
``kernels/reduce.metal`` and ``kernels/matmul.metal``.

Amalgamation order (the contract documented in the headers themselves):

* ``df64``: ``df64_core.h`` -> ``df64_constants.h`` -> ``df64_math_exp.h`` ->
  ``df64_math_trig.h`` -> ``df64_math_special.h``; the special header needs the
  exp header (``df::exp``, ``log_ext``, ``prod3``), trig is independent.
* ``sf64``: ``vendor/softfloat64.metal`` -> the whole df64 stack ->
  ``sf64_core.h`` (compiles its df64 conversions only when ``df64_core.h``
  preceded it) -> ``sf64_math.h`` (emits each transcendental family only when
  the matching ``OPTILAND_DF64_MATH_*_H`` guard macro was seen).
* then ``reduce.metal`` and ``matmul.metal`` (their sf64 twins are emitted only
  when ``OPTILAND_SF64_CORE_H`` is defined), then the generated kernels.

Launch contract (see NOTES/02-design.md section 2.4a): kernels bind buffers by
position and ignore strides, so every input must be a contiguous MPS tensor of
the right dtype and the same *shape* (equal ``numel`` with different shapes is
rejected too), and must not be a lazily negated view (``Tensor.is_neg()``: the
kernel would read the un-negated storage). Broadcasting is the caller's job;
:func:`broadcast_contiguous` does it for a set of component tensors. An ``out=``
buffer may alias an input exactly (same storage address, dtype and shape:
in-place elementwise ops are safe because every thread reads its element before
writing it) but must not partially overlap an input or another output buffer.
Launches above :data:`DEFAULT_CHUNK` = 2^30 elements are split into several
dispatches. Scalars (``scalar=``) must be real Python/NumPy numbers (not bool,
str, tensors or wrapped values).
"""

from __future__ import annotations

import numbers
import operator
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from . import codegen, encode
from . import compile as _compile

if TYPE_CHECKING:
    from collections.abc import Sequence

_KERNEL_DIR = _compile._KERNEL_DIR

#: Header stack per representation, in include order. Missing files are skipped.
HEADERS: dict[str, tuple[str, ...]] = {
    "df64": (
        "df64_core.h",
        "df64_constants.h",
        "df64_math_exp.h",
        "df64_math_trig.h",
        "df64_math_special.h",
    ),
    # Order is the contract documented in sf64_core.h / sf64_math.h: the vendored
    # softfloat first, then the whole df64 stack (sf64_core.h compiles its df64
    # conversions only when df64_core.h preceded it), then sf64_core.h, then the
    # sf64 transcendental bridge.
    "sf64": (
        "vendor/softfloat64.metal",
        "df64_core.h",
        "df64_constants.h",
        "df64_math_exp.h",
        "df64_math_trig.h",
        "df64_math_special.h",
        "sf64_core.h",
        "sf64_math.h",
    ),
}

#: Reduction / matmul kernel files appended after the headers of every stack.
KERNEL_FILES: tuple[str, ...] = ("reduce.metal", "matmul.metal")

#: Headers that may only be included when all of the listed headers are present.
#: ``sf64_math.h`` degrades per family (each df:: bridge is guarded by the
#: matching ``OPTILAND_DF64_MATH_*_H`` macro), so it only needs the two cores;
#: ``df64_math_special.h`` calls into the exp header.
HEADER_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "df64_math_special.h": ("df64_math_exp.h",),
    "sf64_math.h": ("df64_core.h", "sf64_core.h"),
}

#: Headers that must exist for a representation to be usable at all.
REQUIRED_HEADERS: dict[str, tuple[str, ...]] = {
    "df64": ("df64_core.h", "df64_constants.h"),
    "sf64": (
        "vendor/softfloat64.metal",
        "df64_core.h",
        "df64_constants.h",
        "sf64_core.h",
    ),
}

#: Default dispatch chunk: 2^30 threads per launch (larger launches are split).
DEFAULT_CHUNK = 1 << 30


def op_machine_eps(op: str, repr: str = "df64") -> float:
    """Relative precision of ``op``'s result in ``repr``.

    ``2^-53`` for the correctly rounded sf64 core ops, ``2^-48`` for every df64
    op and for the sf64 ops that go through the inexact df64 bridge
    (:data:`codegen.SF64_INEXACT`); see :func:`codegen.op_machine_eps`.
    """
    return codegen.op_machine_eps(op, repr)


_SELFTEST_LEN = 8


def check_selftest(values: Sequence[float]) -> list[str]:
    """Compare the ``df64_selftest`` probe vector against the expected values.

    Args:
        values: The eight floats written by the ``df64_selftest`` kernel.

    Returns:
        list[str]: Human-readable descriptions of every mismatching probe
        (empty when the library is sound).
    """
    v = [float(x) for x in values]
    problems: list[str] = []
    if len(v) != _SELFTEST_LEN:
        return [f"self-test vector has {len(v)} entries, expected {_SELFTEST_LEN}"]

    def bits(x: float) -> int:
        return int(np.float32(x).view(np.uint32))

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    expect(
        v[7] == 1.0,
        f"completion sentinel out[7] = {v[7]!r}, expected 1.0 (kernel did not run)",
    )
    expect(
        v[0] == 2.0**-30,
        f"TwoSum error term out[0] = {v[0]!r}, expected 2^-30 (reassociation?)",
    )
    expect(
        v[1] == 2.0**-46,
        f"TwoProd error term out[1] = {v[1]!r}, expected 2^-46 "
        "(fma not fused / contraction?)",
    )
    expect(
        bits(v[2]) == 0x3F2AAAAB,
        f"2/3 out[2] bits = {bits(v[2]):#010x}, expected 0x3f2aaaab (RNE)",
    )
    expect(
        bits(v[3]) == 0x3F800002,
        f"1+3*2^-24 out[3] bits = {bits(v[3]):#010x}, expected 0x3f800002 (RNE tie)",
    )
    expect(v[4] == 0.0, f"(1e-9 + 1) - 1 out[4] = {v[4]!r}, expected 0 (reassociation)")
    third = v[5] + v[6]
    expect(abs(third - 1.0 / 3.0) < 1e-14, f"df::div(1, 3) = {third!r}, |err| >= 1e-14")
    return problems


def broadcast_contiguous(*tensors: Any) -> tuple[Any, ...]:
    """Broadcast tensors against each other and make every result contiguous.

    Args:
        *tensors: Component tensors (float32 hi/lo, int64 bits or bool masks).

    Returns:
        tuple[torch.Tensor, ...]: Broadcast, contiguous tensors in the same order.
    """
    import torch

    if not tensors:
        return ()
    return tuple(t.contiguous() for t in torch.broadcast_tensors(*tensors))


class MetalLibrary:
    """Compiled elementwise kernel library for one representation.

    Attributes:
        repr: ``"df64"`` or ``"sf64"``.
        headers: Header files that were found and included, in order.
        missing_headers: Header files from the stack that do not exist yet.
        kernel_files: Reduction / matmul kernel files included after the headers.
        ops: Op names whose kernels were generated and compiled.
        kernel_names: Every compiled elementwise kernel function name.
        source: The complete Metal source that was compiled.
        lib: The ``torch.mps`` library object.
        chunk_size: Maximum threads per dispatch (larger launches are split).
    """

    def __init__(
        self,
        repr: str = "df64",
        ops: Sequence[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK,
        selftest: bool = True,
    ) -> None:
        """Assemble and compile the library.

        Args:
            repr: ``"df64"`` or ``"sf64"``.
            ops: Optional subset of :data:`codegen.OPS` to compile. Ops whose
                providing header is missing are dropped either way.
            chunk_size: Maximum threads per dispatch (``<= 2^30``).
            selftest: Run ``df64_selftest`` after compiling.

        Raises:
            FileNotFoundError: If a required header is missing.
            RuntimeError: If the self-test probes mismatch (fast math, contraction).
        """
        if repr not in HEADERS:
            raise ValueError(f"repr must be one of {tuple(HEADERS)}, got {repr!r}")
        # An integral type is required (0.9 used to pass the range check and
        # become 0, 1.5 silently 1); bool is an int subclass but not a size.
        if isinstance(chunk_size, bool):
            raise TypeError("chunk_size must be an integer, got bool")
        try:
            chunk = operator.index(chunk_size)
        except TypeError as exc:
            raise TypeError(
                f"chunk_size must be an integer, got {type(chunk_size).__name__}"
            ) from exc
        if not 0 < chunk <= DEFAULT_CHUNK:
            raise ValueError(f"chunk_size must be in (0, 2^30], got {chunk}")
        self.repr = repr
        self.chunk_size = chunk
        present = [h for h in HEADERS[repr] if (_KERNEL_DIR / h).is_file()]
        present = [
            h
            for h in present
            if all(p in present for p in HEADER_PREREQUISITES.get(h, ()))
        ]
        self.headers: tuple[str, ...] = tuple(present)
        self.missing_headers: tuple[str, ...] = tuple(
            h for h in HEADERS[repr] if h not in present
        )
        missing_required = [h for h in REQUIRED_HEADERS[repr] if h not in present]
        if missing_required:
            raise FileNotFoundError(
                f"{repr}: required kernel header(s) missing from {_KERNEL_DIR}: "
                f"{missing_required}"
            )
        self.kernel_files: tuple[str, ...] = tuple(
            k for k in KERNEL_FILES if (_KERNEL_DIR / k).is_file()
        )
        basenames = {os.path.basename(h) for h in present}
        self.ops: tuple[str, ...] = tuple(codegen.ops_for_headers(repr, basenames, ops))
        self.kernel_names: tuple[str, ...] = tuple(codegen.kernel_names(self.ops, repr))
        self.source = (
            _compile.kernel_source(*present, *self.kernel_files)
            + "\n"
            + codegen.build(self.ops, reprs=(repr,))
        )
        self.lib = _compile.compile_library(self.source)
        self.selftest_values: list[float] | None = None
        if selftest:
            self.run_selftest()

    # ------------------------------------------------------------------
    def run_selftest(self) -> list[float]:
        """Run ``df64_selftest`` and raise ``RuntimeError`` on any mismatch."""
        import torch

        out = torch.zeros(_SELFTEST_LEN, dtype=torch.float32, device="mps")
        self.lib.df64_selftest(out, threads=[1, 1, 1])
        values = out.cpu().numpy().astype(float).tolist()
        self.selftest_values = values
        problems = check_selftest(values)
        if problems:
            raise RuntimeError(
                f"Metal {self.repr} library self-test FAILED; the kernels were "
                "compiled without safe math / with contraction and must not be "
                "used:\n  - " + "\n  - ".join(problems) + f"\n  probe vector: {values}"
            )
        return values

    def has(self, op: str) -> bool:
        """Return whether ``op`` was compiled into this library."""
        return op in self.ops

    def is_inexact(self, op: str) -> bool:
        """Whether ``op`` is evaluated through the inexact 48-bit df64 bridge.

        Always False for the df64 library (every df64 op carries 2^-48); True
        for the sf64 transcendentals of ``sf64_math.h`` (:data:`codegen.SF64_INEXACT`).
        """
        return self.repr == "sf64" and codegen.OPS[op].sf_inexact

    def machine_eps(self, op: str) -> float:
        """Relative precision of ``op``'s result here (:func:`op_machine_eps`)."""
        return codegen.op_machine_eps(op, self.repr)

    def __repr__(self) -> str:
        return (
            f"MetalLibrary(repr={self.repr!r}, ops={len(self.ops)}, "
            f"kernels={len(self.kernel_names)}, headers={list(self.headers)})"
        )

    # ------------------------------------------------------------------
    def _check_tensor(self, t: Any, dtype: Any, what: str) -> None:
        import torch

        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{what}: expected a torch.Tensor, got {type(t).__name__}")
        if t.device.type != "mps":
            raise ValueError(f"{what}: expected an mps tensor, got device {t.device}")
        if t.dtype != dtype:
            raise TypeError(f"{what}: expected dtype {dtype}, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(
                f"{what}: tensor must be contiguous (kernels ignore strides); "
                "use broadcast_contiguous()"
            )
        if t.is_neg() or t.is_conj():
            raise ValueError(
                f"{what}: lazily negated / conjugated views are not supported "
                "(the kernel would read the raw storage); call resolve_neg() / "
                "resolve_conj() first"
            )

    def _components(self, value: Any, kind: str, what: str) -> list[Any]:
        """Validate one operand and return its flat list of buffer tensors."""
        import torch

        if kind == "bool":
            self._check_tensor(value, torch.bool, what)
            return [value]
        if self.repr == "df64":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(
                    f"{what}: df64 operands are (hi, lo) tuples of float32 tensors"
                )
            hi, lo = value
            self._check_tensor(hi, torch.float32, f"{what}.hi")
            self._check_tensor(lo, torch.float32, f"{what}.lo")
            if hi.shape != lo.shape:
                raise ValueError(
                    f"{what}: hi shape {tuple(hi.shape)} != lo shape {tuple(lo.shape)}"
                )
            return [hi, lo]
        self._check_tensor(value, torch.int64, what)
        return [value]

    def _encode_scalar(self, scalar: Any) -> Any:
        # Real numbers only: bool, str/bytes (float() would parse them), tensors
        # and wrapped values (float() would silently take their item) are misuse.
        if isinstance(scalar, bool) or not isinstance(
            scalar, (numbers.Real, np.floating, np.integer)
        ):
            raise TypeError(
                "scalar must be a real Python/NumPy number, got "
                f"{type(scalar).__name__}"
            )
        if self.repr == "df64":
            return encode.df64_scalar(float(scalar))
        return encode.sf64_scalar(float(scalar))

    def _alloc_out(self, result: str, shape: Any) -> tuple[Any, list[Any]]:
        import torch

        if result == "bool":
            o = torch.empty(shape, dtype=torch.bool, device="mps")
            return o, [o]
        if self.repr == "df64":
            hi = torch.empty(shape, dtype=torch.float32, device="mps")
            lo = torch.empty(shape, dtype=torch.float32, device="mps")
            return (hi, lo), [hi, lo]
        o = torch.empty(shape, dtype=torch.int64, device="mps")
        return o, [o]

    def launch(
        self,
        op: str,
        *inputs: Any,
        scalar: Any = None,
        scalar_side: str | None = None,
        out: Any = None,
    ) -> Any:
        """Run one elementwise kernel over contiguous component tensors.

        Args:
            op: Op name from :data:`codegen.OPS`.
            *inputs: Tensor operands in op order. For ``df64`` each value operand
                is a ``(hi, lo)`` tuple of float32 MPS tensors; for ``sf64`` an
                int64 MPS tensor. ``where`` takes a bool tensor first. When
                ``scalar`` is given, the scalar operand is omitted from ``inputs``.
            scalar: Python float for the scalar variant of a binary op.
            scalar_side: ``"right"`` (default when ``scalar`` is given) makes the
                scalar the second operand; ``"left"`` makes it the first.
            out: Optional preallocated output (same form as an operand).

        Returns:
            The result components: ``(hi, lo)`` for df64 values, an int64 tensor
            for sf64 values, or a bool tensor for predicates. The shape is that
            of the first tensor operand.

        Raises:
            KeyError: If ``op`` is not compiled into this library.
            TypeError, ValueError: On operand validation failures.
        """
        if op not in codegen.OPS:
            raise KeyError(f"unknown elementwise op {op!r}")
        if op not in self.ops:
            raise KeyError(
                f"op {op!r} is not available in the {self.repr} library "
                f"(header {codegen.header_for(op, self.repr)!r} missing?)"
            )
        spec = codegen.OPS[op]
        if scalar is not None and scalar_side is None:
            scalar_side = "right"
        if scalar is None and scalar_side is not None:
            raise ValueError("scalar_side given without scalar")
        name = codegen.kernel_name(op, self.repr, scalar_side)
        scalar_index = {None: -1, "right": 1, "left": 0}[scalar_side]
        expected = spec.arity - (1 if scalar is not None else 0)
        if len(inputs) != expected:
            raise TypeError(
                f"{op}: expected {expected} tensor operand(s), got {len(inputs)}"
            )

        buffers: list[Any] = []
        it = iter(inputs)
        shape = None
        numel = None
        for idx in range(spec.arity):
            if idx == scalar_index:
                buffers.append(self._encode_scalar(scalar))
                continue
            comps = self._components(next(it), spec.inputs[idx], f"{op} operand {idx}")
            if shape is None:
                shape, numel = tuple(comps[0].shape), comps[0].numel()
            for c in comps:
                if tuple(c.shape) != shape:
                    raise ValueError(
                        f"{op}: operand {idx} has shape {tuple(c.shape)}, expected "
                        f"{shape} (broadcast on the caller side)"
                    )
            buffers.extend(comps)
        assert shape is not None and numel is not None

        if out is None:
            result, out_bufs = self._alloc_out(spec.result, shape)
        else:
            out_bufs = self._components(out, spec.result, f"{op} out")
            for o in out_bufs:
                if tuple(o.shape) != shape:
                    raise ValueError(
                        f"{op}: out has shape {tuple(o.shape)}, expected {shape}"
                    )
            self._check_out_overlap(op, buffers, out_bufs)
            result = out

        if numel > 0:
            kernel = getattr(self.lib, name)
            self._dispatch(kernel, buffers, out_bufs, numel)
        return result

    @staticmethod
    def _check_out_overlap(op: str, buffers: list[Any], out_bufs: list[Any]) -> None:
        """Reject out= buffers that alias each other or partially overlap an input.

        Exact aliasing of an input (same storage address, dtype and numel) is the
        in-place case and is allowed; every other overlap makes a thread read
        an element another thread has already overwritten.
        """
        import torch

        # Byte range inside the tensor's storage. On MPS ``data_ptr()`` is the
        # Metal buffer object's address plus the byte offset, and distinct
        # buffers are allocated only a few hundred bytes apart, so ranges are
        # only comparable between tensors that share a storage.
        def span(t: Any) -> tuple[int, int, int]:
            start = t.storage_offset() * t.element_size()
            return t.untyped_storage().data_ptr(), start, start + t.nbytes

        def overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
            return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]

        outs = [span(o) for o in out_bufs]
        for i, a in enumerate(outs):
            for b in outs[:i]:
                if overlap(a, b):
                    raise ValueError(f"{op}: out buffers overlap each other")
        for o, a in zip(out_bufs, outs, strict=True):
            for t in buffers:
                if not isinstance(t, torch.Tensor):
                    continue
                b = span(t)
                if overlap(a, b) and not (b == a and t.dtype == o.dtype):
                    raise ValueError(
                        f"{op}: out buffer partially overlaps an input "
                        "(only exact in-place aliasing is supported)"
                    )

    def _dispatch(
        self, kernel: Any, buffers: list[Any], out_bufs: list[Any], numel: int
    ) -> None:
        import torch

        args = buffers + out_bufs
        flat = [a.view(-1) if isinstance(a, torch.Tensor) else a for a in args]
        chunk = self.chunk_size
        for start in range(0, numel, chunk):
            end = min(start + chunk, numel)
            if start == 0 and end == numel:
                view_args = flat
            else:
                view_args = [
                    a[start:end] if isinstance(a, torch.Tensor) else a for a in flat
                ]
            kernel(*view_args, threads=[end - start, 1, 1])


_LIBRARIES: dict[str, MetalLibrary] = {}


def get_library(repr: str = "df64", **kwargs: Any) -> MetalLibrary:
    """Return the process-wide :class:`MetalLibrary` for ``repr`` (built on first use).

    Args:
        repr: ``"df64"`` or ``"sf64"``.
        **kwargs: Passed to :class:`MetalLibrary` on first construction only.

    Returns:
        MetalLibrary: The singleton for this representation.
    """
    lib = _LIBRARIES.get(repr)
    if lib is None:
        lib = MetalLibrary(repr, **kwargs)
        _LIBRARIES[repr] = lib
    return lib


def reset_libraries() -> None:
    """Forget the singletons (compiled Metal libraries stay cached by hash)."""
    _LIBRARIES.clear()


__all__ = [
    "DEFAULT_CHUNK",
    "HEADERS",
    "HEADER_PREREQUISITES",
    "KERNEL_FILES",
    "REQUIRED_HEADERS",
    "MetalLibrary",
    "broadcast_contiguous",
    "check_selftest",
    "get_library",
    "op_machine_eps",
    "reset_libraries",
]
