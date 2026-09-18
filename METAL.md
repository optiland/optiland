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

**With the fused trace-interpreter kernel (2026-09-18, M1 Max, Cooke triplet, medians, after
the five levers):** the surface loop (`SurfaceGroup.trace`) runs in 17 ms for 1e5 rays and 45 ms
for 1e6 (NumPy on one core: 22 and 236 ms; the per-op GPU path: 104 and 135 ms); the reverse
telephoto and the aspheric singlet gain 6.1x and 5.6x over NumPy at 1e6 rays. End to end,
`optic.trace` is 72 ms at 1e5 rays and 177 ms at 1e6 (NumPy 44 and 457 ms): ray generation and
the analyses around the kernel still run on the per-op path and cost a fixed ~45 ms per trace,
so NumPy wins below ~2e5 rays per trace and the GPU wins above. Batched designs through
`optiland.raytrace.batch_trace.trace_batch` (post-stop variables, shared launch): 1000 designs
x 1000 rays in 0.36 s, 100 x 100,000 in 0.37 s, 10,000 x 100 in 2.7 s, against 0.75 / 0.85 /
3.64 s steady state for eight NumPy worker processes (plus ~5 s pool start-up) and 2.9 / 4.3 /
22.8 s for one core. Every number carries a correctness record against NumPy. The kernel's
occupancy is set by `[[max_total_threads_per_threadgroup]]` on its entry points and a
spherical-only twin serves systems without Newton-solved surfaces; the dispatch layer runs
host-resident scalar ops through a fast path (11 us per op). Full tables, targets and the
measured remaining levers: `NOTES/09-fused-trace-report.md` in the project root.

The paragraph below describes the per-op path, which every unsupported feature still uses.


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
with `numel > 256`, `skip == 0`, no participating tensor carrying `requires_grad`, uniform
shapes. Chief rays, 6-ring bundles, `ParaxialRays`, `PolarizedRays` and traces whose tensors
carry `requires_grad` are *not* candidates, so they pass through `require` untouched and only
ever count their reason. Autograd merely *enabled*, with no tensor flagged, leaves a bundle a
candidate -- see the documented limit under "Autograd".

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
apertures, nested `reference_cs` chains, `CoordinateSystem` subclasses, GRIN propagation,
non-finite indices, mixed-wavelength
bundles, tensors carrying `requires_grad`, `skip != 0`, `SequencedSurfaceGroup` (its loop is
not hooked at all) and bundles that would exceed the memory budget.

Two refusals are about the *form* a supported configuration arrives in rather than the feature
itself, because the kernel mirrors one specific Python expression and cannot mirror the other:

* a bundle whose `is_normalized` flag is clear (`propagation_model`). `HomogeneousPropagation.
  propagate` then re-normalises the directions after every surface, which the kernel does not
  do; the flag is a public `RealRays` attribute and nothing in Optiland clears it.
* in `df64` only, an asphere coefficient stored as a backend tensor rather than a Python or
  NumPy scalar (`geometry_type`). `EvenAsphere.sag` evaluates `Ci * r2 ** (i + 1)` as an
  array-array op for a tensor and as a host-scalar op for a scalar, and the two round
  differently at ~1 ulp; the record carries the host-scalar form. `Variable.update` stores a
  tensor when the caller passes one (`be.array(v)`); the batch API's own value arrays are NumPy
  scalars and are unaffected.

Every mirrored class is keyed by **exact type** -- `SurfaceGroup`, `RealRays`, `Surface` /
`ImageSurface`, the geometry / aperture / interaction registries, `HomogeneousPropagation` and
`CoordinateSystem` -- because a subclass overrides a mirrored method without touching the base
source the fingerprint is taken from, so the drift check cannot see it (R3-V2-03). A subclass is
refused with the family's own counted reason (`group_type`, `rays_type`, `surface_type`,
`geometry_type`, `aperture_type`, `interaction_type`, `propagation_model`, `reference_cs`) and
the per-op path runs. A *material* subclass is the one exception, and it is deliberate: `n` and
`k` are host reads that the record and the per-op path make through the same call, so the two
answers move together.

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

The fused path is for forward traces only. The gate refuses with `requires_grad` when
`torch.is_grad_enabled()` is True **and** a participating tensor actually carries
`requires_grad` -- the state `be.grad_mode.enable()` produces -- and the per-op Metal path runs,
so gradients keep flowing exactly as before. Suite sweeps that want the kernel exercised set
`OPTILAND_TEST_MPS_GRAD=0`.

**Documented limit (R2-V1-06): autograd merely *enabled* is not refused, and a Newton
geometry is not bit-exact there.** `be.grad_mode.disable()` clears the flag new arrays are
created with; it does **not** clear `torch.is_grad_enabled()`, which is the default True and
is the flag `NewtonRaphsonGeometry.distance` branches on: with it set, the per-op path applies
the DiffOptics one-step implicit correction `t - F(t)/stopgrad(dF/dt)` after the primal solve,
and the kernel mirrors the primal solve. No tensor carries `requires_grad` in that state, so
the gate does not refuse and `SurfaceGroup.trace` -- hence `optic.trace`, every analysis and
the GUI -- fuses and disagrees with the per-op path it mirrors, on the rays whose extra
refinement step moves a word. Measured on 4096-ray bundles (differing quantities / worst ray
count): `aspheric_singlet` 11/594 df64 and 0 sf64, `even_asphere_5coeff` 20/592 and 13/1,
`odd_asphere_singlet` 26/672 and 17/22, `nonconverging_asphere` 43/1112 and 21/1112; a group
with no Newton row (e.g. a Cooke triplet) is unaffected in both modes.

Wrap the trace in `torch.no_grad()` for raw-component agreement -- that is what `trace_batch`
does for both its legs, and what every conformance fixture does. The limit is pinned by
`tests/metal/test_trace_adversarial_round2.py::test_r2v106_locked`, so a later change of policy
(refusing the configuration at the gate) fails a test rather than drifting.

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

The table's *coverage* is checked too, not only its rows:
`tests/metal/test_trace_mirror_sources.py::test_every_executed_physics_function_is_fingerprinted_or_exempt`
profiles a per-op trace of ten fixtures, collects every `optiland.*` function that really runs
inside `SurfaceGroup.trace`, and requires each one to be either fingerprinted or named in an
explicit exemption list with the reason it carries no physics. Five functions were reaching the
kernel's scope without a row when that census was first run (R3-V2-02), and an edit to any of
them left the kernel serving the physics it was verified against while the Python path had
moved. An upstream merge that adds a function to the trace path now fails that test until
somebody decides which of the two it is.

That census drops `optiland.backend.*` as arithmetic both paths share, which is true of the
elementwise kernels and false of `metal/conic.py`: that module is the per-op GPU path's own
conic solver -- the reference the conformance tests measure the kernel against -- and the fused
kernel runs none of it. A root-order edit there moved the per-op answer while the kernel kept
serving the physics it was verified against, with no warning and no counter (R3-V2-04). Three
rows (`conic_candidates`, `_ConicMetal.forward`, `_scalar_float`) and a carve-out with a
GPU census leg behind it close that;
`test_every_executed_conic_function_is_fingerprinted_or_exempt` is what keeps it closed, and
`test_no_executed_mirrored_row_is_hidden_from_the_census` states the residue exactly: the only
fingerprint rows a census still drops are the four `ops_elementwise` `CONTRACT` rows, which are
guarded by value tests by design. The same finding removed a duplicated constant: the conic
solver's epsilon was written as the literals `2.0**-48` / `2.0**-53` although it is
`MACHINE_EPS[mode]`, the number the kernel reads as `consts[C_EPS]`; there is one copy now.

### Notices, warning filters and threads

The driver's notices -- the one-shot `FusedTraceUnavailableWarning`, `FusedTraceDriftWarning`,
and the `OPTILAND_METAL_TRACE_DIAG=1` note that the kernel's Newton loop did not converge --
all announce a *fallback the caller never asked for*: the per-op path runs and the caller still
gets an answer. None of them can become an exception in your process. Under
`warnings.simplefilter("error")` (`python -W error`, `pytest -W error`, a
`filterwarnings = error` ini section) the promotion is caught and the same text is logged on
the `optiland.backend.torch_backend.metal.trace` logger instead, which `logging.lastResort`
still prints to stderr (R3-V1-06). Without such a filter they are ordinary warnings and
`pytest.warns` sees them as before. The loud channel is unchanged and is not a warning:
`OPTILAND_METAL_FUSED_TRACE=require` raises `MetalFallbackError`, and every refusal is counted
under `fused_trace_skip:*` whether or not anybody is listening.

**Documented limit (R3-V1-05): do one trace on the main thread before tracing from threads.**
Several threads whose *first* trace in the process is concurrent can abort or hang the
interpreter -- on the fused path **and on the per-op path** (measured at 4 threads: fused df64
7/8 and sf64 3 aborts in 6 runs; per-op df64 7/8, its failure a hang past 300 s). A sampled
hang wedges inside torch's own `metal gpu stream` dispatch queue, on kernels both paths use, so
this is not a fused-trace defect and there is no trigger in this package to fix. After one
main-thread trace, concurrent tracing is fine: 4 and 6 threads x 20 traces are clean and
raw-component equal. Pinned by `test_r3v105_locked_*`.

**Documented limit (R3-V1-07): a recorded row is a view that pins its whole `snap`.** The
writeback is zero-copy, so `surface.y` from a fused trace is a view of the one `snap` buffer and
keeps `8 * n_rows` times its own bytes alive -- 64x for an 8-surface system. A loop that keeps
one row per trace grew by +624 MB per iteration at N = 1,250,000 where the per-op path grew by
+9.7 MB; loops that do not retain rows are flat, so it is retention, not a leak. Copy the row
(`be.copy(row)`) or read records through `SurfaceGroup.x/y/...`, which stacks into fresh
storage -- every shipped analysis already does the latter. Pinned by `test_r3v107_locked`.

**Documented limit (R3-V1-09): a large trace makes the process hold about a gigabyte until you
call `torch.mps.empty_cache()`.** `snap` is one contiguous buffer per raw component, and
PyTorch's MPS caching allocator reserves a 1 024 MiB heap for a single allocation above ~8-12 MB
(measured with plain torch, no Optiland). An 8-surface system at N = 60,000 therefore leaves the
process holding 1,072.6 MiB for a 33.6 MiB live set, where the per-op path -- whose largest
single block is one ray plane -- holds 48.6 MiB and does not cross until ~1e6 rays. It is
allocator granularity, not retention: `empty_cache()` returns it to 0.6 MiB, the number is
reached at the first trace and does not grow, and the memory budget of
`OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION` is computed on *live* bytes, which are within 1.3 %
of the design's model. Pinned by `test_r3v109_locked*`; the related note about what plan 9.2's
T8 benchmark row measures is R3-V1-08 in `NOTES/fused-trace-research/documented-limits.md`.

### Batch API

The same kernel carries a design axis: `optiland.raytrace.batch_trace.trace_batch` traces
B designs x N rays in one launch and returns a result whose `install(optic, b)` writes one
design's recorded rows back onto an optic; `optiland.tolerancing.batched` builds Monte-Carlo
and sensitivity runs on it for the operands in `BATCHABLE_OPERANDS`, falling back to the
existing per-design loop for anything else.

`trace_batch` is a **primal** batch tracer: both its legs -- the fused launch and the
contract loop it is verified against -- run inside `torch.no_grad()`.
`NewtonRaphsonGeometry.distance` returns the DiffOptics one-step implicit correction
`t - F(t)/(dF/dt)` instead of the primal `result.t` whenever `torch.is_grad_enabled()` is True,
and the kernel mirrors the primal solve, so a batch traced with autograd merely enabled would
not agree with its own reference on any Newton system. Note that `be.grad_mode.disable()` does
*not* clear `torch.is_grad_enabled()`: it decides which leaves carry `requires_grad`, which is
what the gate reads. Derivatives come from `batch_trace.fd_jacobian`, not from autograd through
the returned planes.

**What a batch leaves on your optic.** `trace_batch` mutates the system design by design and
puts it back in a `finally` -- the *objects* it found, not a replay of their values. Restoring
the values is not enough and used to lose two things: `ThicknessVariable.get_value()` reads
`cs.z[i+1] - cs.z[i]` rather than `surface.thickness`, so the round trip handed back a
1-ulp-moved `MetalFloat64` where the caller had a Python `float`, and `Variable("index")`
mutates through `IdealMaterial(n, k=0)`, so restoring its value restored an *ideal* glass and
threw the caller's dispersion and absorption away. Both are fixed: after a call, every
surface's `thickness`, `material_post`, `semi_aperture`, geometry attributes and coordinate
system are the objects they were. The *records* on `optic.surfaces` are not part of that
promise -- they are trace output, and `install(optic, b)` is what writes them.

That fixes the restore, not the meaning of an `index` design. While the batch runs, each
`index` row is applied by `OpticUpdater.set_index` as `IdealMaterial(n=value, k=0)`, so **no**
row of an `index` variable traces a dispersive system -- not even the row carrying the nominal
index, and 23 of the 29 shipped samples are absorbing. That is the shared `Variable.update`
path every optimizer and every tolerancing run uses, so the sequential loop does exactly the
same thing; use a `material` variable if a design has to keep a real glass.

**How equal a batched tolerancing frame is.** `monte_carlo_batched` and `sensitivity_batched`
equal `MonteCarlo.run` / `SensitivityAnalysis.run` cell for cell only when every operand's
bundle is larger than 1024 rays (`optiland.tolerancing.batched.TIER_A_MIN_RAYS`). At or below
that, `materials/base.py`'s `_MAX_VALUE_KEY_ARRAY_SIZE` makes the per-op path evaluate a
dispersive index on the whole wavelength array instead of on the one uniform representative the
kernel always uses, and the frames agree only to `64 * eps_mode * scale` (in df64; sf64 is
exact). The frame says which applies: `df.attrs["tier"]` is `"A"` or `"B"`, beside
`df.attrs["num_rays"]`. Measured on a Cooke triplet's `rms_spot_size` at 12 hexapolar rings
(469 rays), 16 samples: 14 of 16 rows differ, worst 1.1e-14 against a bound of 2.3e-13.

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
