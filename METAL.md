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
* `OPTILAND_TEST_MPS=1 pytest tests` adds a `torch-mps` parametrization to every backend test;
  `scripts/metal_suite.py` runs it one file per process and summarizes failures.
* `scripts/metal_oracle_e2e.py` compares traces and analyses against NumPy; `scripts/metal_benchmark.py`
  times workloads across numpy / torch-cpu / mps-f32 / mps-df64 / mps-sf64 with correctness checks.
