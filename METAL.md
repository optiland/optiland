# Optiland on Apple GPUs: emulated float64 with Metal

This fork adds a `mps` device to Optiland's torch backend with **float64 precision emulated on
the GPU**. Apple GPUs have no native binary64 (Metal has no `double`), so float64 tensors on
`mps` are represented by a tensor subclass, `MetalFloat64`, whose arithmetic runs in custom Metal
kernels compiled through `torch.mps.compile_shader`.

```python
import optiland.backend as be

be.set_backend("torch")
be.set_device("mps")          # Apple GPU
be.set_precision("float64")   # emulated: double-single by default

from optiland.samples.objectives import CookeTriplet
lens = CookeTriplet()
rays = lens.trace(Hx=0, Hy=1, wavelength=0.55, num_rays=64)   # runs on the GPU
```

`be.set_precision("float32")` on `mps` uses native float32 tensors (fast, single precision).

## Representations

| mode | storage | precision | range | speed |
|---|---|---|---|---|
| `df64` (default) | two float32 (hi, lo) | ~48 significant bits, ~1e-14 relative per op | float32 range; values below ~2e-31 lose the low word (GPU denormal flush), below ~1.2e-38 become 0 | fastest |
| `sf64` | int64 IEEE binary64 bit pattern (software float, metal-softfloat) | 53 bits, correctly rounded per op; bit-exact with NumPy for + − × ÷ √ | full binary64 | ~2–3× slower; transcendental functions go through df64 (48-bit) in this version |

```python
from optiland.backend.torch_backend import metal
metal.set_mode("sf64")   # before creating optics; "df64" to switch back
```

Ray-trace results with `df64` agree with NumPy float64 to ~1e-12 mm in position for the sample
systems (see `NOTES/06-oracle-report.md` in the project root); `sf64` agrees to float64 rounding.
The whole upstream test suite passes under the emulated device (`OPTILAND_TEST_MPS=1`: 4,000
passed, 2 pre-existing/ordering failures).

## Performance, honestly

The emulation is validated for correctness first. Speed is bounded by per-operation dispatch,
not by the kernels: a sequential Optiland trace is thousands of small tensor ops, and each costs
~40 µs of Python/torch dispatch on top of the launch. Measured on an M1 Max (Cooke triplet,
`NOTES/07-final-report.md`): the GPU path is slower than NumPy on one CPU core up to ~1e5 rays
per trace and only reaches parity around 1e6 rays. Use it when you need float64 results on the
GPU (e.g. to compose with other GPU work) or as the base for fused per-surface kernels, which is
where the real speed-up lies (see `metal/conic.py` for the pattern). For throughput today, use
`optiland.parallel` with CPU workers.

Measured saturation on the M1 Max (`scripts/metal_saturation.py`, 1e5-ray Cooke traces,
`NOTES/08-parallel-saturation.md` in the project root):

| pool | throughput | saturates at |
|---|---|---|
| NumPy workers | 20 traces/s per worker, 99 traces/s with 8 | 8 workers (the performance cores); 10 adds 2%, 12+ loses |
| torch CPU float64 workers (1 thread each) | 20 traces/s per worker, 71 traces/s with 8 | 8 workers |
| GPU (`mps`, df64) processes | 2.4 traces/s per process, 8.2 traces/s with 6 | ~6 processes; each also occupies a CPU core for dispatch |

A single kernel launch costs ~56 µs regardless of size and is GPU-bound only above ~1e6
elements, which is why per-op dispatch cannot keep the 32 GPU cores busy. The
design-independent ways to change that are (1) fusing the recorded aten op stream into one
generated kernel per elementwise chain, (2) a trace-interpreter kernel that takes the surface
list as a buffer, and (3) batching many designs into one launch; all three reuse this
library's kernels, encoders and launcher.

## Fused trace-interpreter kernel

`kernels/trace.metal` walks the whole surface list per ray thread in **one** launch, instead of
the ~40 small tensor ops per surface the per-op path dispatches. It is additive: a 17-line hook
in `SurfaceGroup.trace` chooses between the kernel and the unchanged upstream loop, and every
bundle the kernel does not support falls back to that loop transparently.

### Switches

| variable | values | effect |
|---|---|---|
| `OPTILAND_METAL_FUSED_TRACE` | `1` (default), `0`, `require` | `0` turns the hook off entirely (the package stays inert); `require` raises `MetalFallbackError` when a *candidate* bundle is refused for a *feature* reason, on the late fallback, on mirror drift and when the kernel is unavailable |
| `OPTILAND_METAL_TRACE_DIAG` | `0` (default), `1` | keep the per-surface status and iteration planes of the last trace and count `fused_trace:diag:<bit>` |
| `OPTILAND_METAL_FUSED_TRACE_MAX_STEPS` | int, default `2**26` | weighted surface-steps per command buffer (the chunking budget) |
| `OPTILAND_METAL_FUSED_TRACE_MIN_RAYS` | int, default `0` | refuse bundles below this size with `min_rays` |
| `OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION` | float, default `0.25` | share of `torch.mps.recommended_max_memory()` one trace may allocate |
| `OPTILAND_METAL_FUSED_TRACE_DRIFT` | `refuse` (default), `warn` | developer escape hatch while re-mirroring a changed Python function; never set it in a measurement run |
| `OPTILAND_TEST_MPS_GRAD` | `1` (default), `0` | test-suite only: run the `torch-mps` parametrization with autograd off, which is what makes a bundle eligible |
| `OPTILAND_TEST_MPS_STATS_FILE` | path | test-suite only: append one JSON line per test with the candidate census and the `metal_stats()` delta |

**`require` is structural-safe.** A *candidate* is a bundle that passes every structural check:
`type(group) is SurfaceGroup`, `type(rays) is RealRays`, `rays.x` a GPU-resident `MetalFloat64`
with `numel > 256`, `skip == 0`, autograd off, uniform shapes. Chief rays, 6-ring bundles,
`ParaxialRays`, `PolarizedRays` and grad-on traces are *not* candidates, so they pass through
`require` untouched and only ever count their reason.

**Rollback.** Set `OPTILAND_METAL_FUSED_TRACE=0` (read per trace, so it works at runtime), or
revert the hook commit; nothing else in the package is on the per-op path.

### What the kernel takes (v1)

Geometries `Plane`, `StandardGeometry` (finite and infinite radius), `EvenAsphere`,
`OddAsphere`; apertures none, `RadialAperture`, `OffsetRadialAperture`, `RectangularAperture`,
`EllipticalAperture`; `RefractiveReflectiveModel` (refraction and reflection, no coating, no
BSDF); `HomogeneousPropagation` with absorption; one flat `CoordinateSystem` per surface with
any tilt/decenter; `ObjectSurface` at index 0; `RealRays` only; one wavelength per bundle; both
`df64` and `sf64`. That covers all 29 shipped sample systems end to end.

Everything else is refused with a counted reason and runs on the per-op path: Zernike /
Chebyshev / Forbes / toroidal / biconic / grid-sag / NURBS / polynomial geometries, gratings,
thin-lens and diffractive interaction models, coatings, BSDFs, polarization, polygon and file
apertures, nested `reference_cs` chains, GRIN propagation, non-finite indices, mixed-wavelength
bundles, autograd, `skip != 0`, `SequencedSurfaceGroup` (its loop is not hooked at all) and
bundles that would exceed the memory budget.

### Counters and the census

`be.metal_stats()` gains, beside the existing `gpu:*` launch counts:

* `gpu:fused_trace` — one per dispatched slab;
* `fused_trace:candidates`, `:traces`, `:designs`, `:surface_steps`, `:chunks`,
  `:late_fallback`, `:readback`, `:unvisited`, `:tier1_canary_mismatch`;
* `fused_trace_skip:<reason>` — one per refusal, over the closed `FusedTraceSkip` key set.

Two identities are checked per test and per system by a census that does **not** import the
gate -- `tests/conftest.py::_is_fuse_candidate`, which the suite and the oracle both wrap
`SurfaceGroup.trace` with -- so a silently-disabled gate fails a test instead of hiding:

```
census_candidates == fused_trace:candidates
fused_trace:candidates == fused_trace:traces
                        + sum(fused_trace_skip:<feature reason>)
                        + fused_trace:late_fallback
```

### Autograd

The fused path is for forward traces only. With `be.grad_mode` enabled, or with any
participating tensor requiring grad, the gate refuses with `requires_grad` and the per-op Metal
path runs, so gradients keep flowing exactly as before. Suite sweeps that want the kernel
exercised set `OPTILAND_TEST_MPS_GRAD=0`.

### Mirror fingerprints and drift

Every Python function the MSL reproduces ("mirror, never improve": the kernel repeats the
Python expression's association and operand order, so agreement is raw-component equality)
is fingerprinted in `metal/trace_mirror.py` by an AST hash that ignores docstrings and
formatting. `fused_trace` calls `trace_mirror.check_all()` once per process; any mismatch emits
one `FusedTraceDriftWarning` naming the qualnames and then refuses every candidate with
`mirror_drift` until the row is re-verified:

```
python -m optiland.backend.torch_backend.metal.trace_mirror --check
python -m optiland.backend.torch_backend.metal.trace_mirror --update --verified \
    "<qualname>=<why the MSL still mirrors it>"
```

A second class of rows, `CONTRACT`, is not hashed; those are host-consumed helpers whose
*values* named adapter tests compare.

### Batch API

The same kernel carries a design axis: `optiland.raytrace.batch_trace.trace_batch` traces
B designs x N rays in one launch and returns a result whose `install(optic, b)` writes one
design's recorded rows back onto an optic; `optiland.tolerancing.batched` builds Monte-Carlo
and sensitivity runs on it for the operands in `BATCHABLE_OPERANDS`, falling back to the
existing per-design loop for anything else.

## How it works

* `optiland/backend/torch_backend/metal/kernels/` — Metal Shading Language sources: double-single
  arithmetic (`df64_core.h`, error-free transformations with FMA, JMP2017 algorithms), transcendental
  functions (`df64_math_*.h`, QD-style argument reduction + series), software binary64
  (`sf64_core.h` over the vendored `metal-softfloat`), generated elementwise kernels, compensated
  reductions, batched matmul, and a fused conic-intersection kernel for the ray-trace hot path.
* `tensor.py` — the `MetalFloat64` subclass: reports `dtype=torch.float64`, `device=mps:0`; aten ops
  are dispatched to the kernels (`ops_*.py`); autograd works unchanged (derivative formulas are aten
  ops that dispatch the same way).
* **Dual residency** — tensors with at most `OPTILAND_METAL_HOST_THRESHOLD` elements (default 256)
  live on the CPU in real float64 and their ops run there exactly, with no GPU launch; large ray
  arrays live on the GPU. This keeps Optiland's scalar bookkeeping (coordinate transforms, paraxial
  quantities, `.item()` reads) off the launch queue.
* `optiland/parallel.py` — `evaluate_parallel(fn, jobs, workers=[WorkerConfig(...), ...])` runs
  independent jobs in spawned worker processes, each with its own backend/device/precision (CPU
  float64 workers plus one or two GPU workers is the recommended layout).

## Requirements and safety

* macOS 15+ (tested on macOS 26, Apple M1 Max), PyTorch ≥ 2.14 with MPS.
* `PYTORCH_MPS_FAST_MATH` must be unset or `0` before the first Metal compile in the process; the
  library refuses to run otherwise (fast math would break the error-free transformations), and it
  runs an arithmetic self-test at load.
* Diagnostics: `metal.stats()` counts GPU launches (`gpu:<op>`), host-path ops (`host:<op>`) and
  CPU fallbacks (`cpu_fallback:<op>`); `OPTILAND_METAL_STRICT=1` turns fallbacks into errors.
* CPU fallbacks (decoded to CPU float64, re-encoded): dense factorizations (`lstsq`, `solve`,
  `eigh`, `svd`, ...), FFTs and complex arithmetic (polarization, PSF/MTF), a few SciPy-based paths.

## Tests and tools

* `tests/metal/` — kernel accuracy vs mpmath, softfloat bit-exactness, dispatch-layer contracts,
  fused conic kernel, host path, end-to-end oracle.
* `tests/metal/test_trace_*.py` — the fused trace-interpreter kernel: layout contract, mirror
  fingerprints, per-function units, adapters, writeback, eligibility gate and hook, conformance
  against the per-op path, batch API and finite differences.
* `OPTILAND_TEST_MPS=1 pytest tests` adds a `torch-mps` parametrization to every backend test;
  `scripts/metal_suite.py` runs it one file per process and summarizes failures.
* `scripts/metal_oracle_e2e.py` compares traces and analyses against NumPy; `scripts/metal_benchmark.py`
  times workloads across numpy / torch-cpu / mps-f32 / mps-df64 / mps-sf64 with correctness checks;
  `scripts/metal_saturation.py` measures throughput versus the number of CPU or GPU worker processes.
