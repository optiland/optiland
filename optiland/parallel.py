"""Multi-process evaluation of independent optical jobs (CPU workers + GPU).

Optiland's backend configuration is process-global, so parallel design
evaluation uses *processes*, each with its own backend/device/precision. On
Apple silicon the workers must be started with the ``spawn`` context: forking
after Metal has been initialized hangs (Apple documents fork-without-exec as
unsafe on all its platforms). Every worker that selects ``device='mps'`` owns
its own Metal command queue through torch; the GPU time-slices between them.
Measured on an M1 Max with the fused trace kernel (NOTES/08-parallel-saturation.md,
section 6): one GPU worker saturates the device for batched sweeps and for
traces of a million rays, a CPU pool of one worker per performance core is the
throughput for traces below ~2e5 rays, and putting both kinds of job into one
first-come pool loses on both sides. :func:`run_scheduled` therefore routes
each :class:`Job` by its shape (rays, traces, designs) with a cost model, keeps
a CPU pool and a GPU worker, and returns results in job order.

Example::

    from optiland.parallel import evaluate_parallel, WorkerConfig

    def job(optic_dict, radius):          # importable module-level function
        import optiland.backend as be
        from optiland.optic import Optic
        optic = Optic.from_dict(optic_dict)
        optic.surface_group.radii[1] = radius   # illustrative
        ...
        return float(rms)

    results = evaluate_parallel(job, [(lens.to_dict(), r) for r in radii],
                                workers=[WorkerConfig("mps", "float64")] * 2
                                        + [WorkerConfig("cpu", "float64")] * 6)

Kramer Harrison's Optiland; parallel helper added by the Optiland-Metal fork, 2026.
"""

from __future__ import annotations

import contextlib
import heapq
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "MEASURED_HOST",
    "CostModel",
    "Job",
    "Plan",
    "WorkerConfig",
    "configure_worker",
    "default_model",
    "evaluate_parallel",
    "gpu_available",
    "host_signature",
    "performance_cores",
    "plan_jobs",
    "run_scheduled",
    "trace_rays",
]


@dataclass(frozen=True)
class WorkerConfig:
    """Backend configuration for one worker process.

    Attributes:
        device: ``'cpu'``, ``'cuda'`` or ``'mps'``.
        precision: ``'float32'`` or ``'float64'``.
        backend: ``'torch'`` or ``'numpy'`` (numpy ignores device/precision).
        metal_mode: ``'df64'`` (double-single, fast) or ``'sf64'`` (software
            binary64, exact) for float64 on ``mps``.
        threads: CPU math threads for this worker (torch/numpy/BLAS); ``1`` keeps
            eight CPU workers from oversubscribing the eight performance cores.
        fused: value for ``OPTILAND_METAL_FUSED_TRACE`` in the worker (``'1'``,
            ``'0'`` or ``'require'``), or ``None`` to leave the environment alone.
    """

    device: str = "cpu"
    precision: str = "float64"
    backend: str = "torch"
    metal_mode: str = "df64"
    threads: int = 1
    fused: str | None = None


def configure_worker(config: WorkerConfig) -> None:
    """Apply ``config`` in the current process (called by the pool initializer)."""
    os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")
    if config.fused is not None:
        os.environ["OPTILAND_METAL_FUSED_TRACE"] = config.fused
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, str(config.threads))
    import optiland.backend as be

    be.set_backend(config.backend)
    if config.backend == "torch":
        import torch

        torch.set_num_threads(config.threads)
        be.set_device(config.device)
        be.set_precision(config.precision)
        if config.device == "mps" and config.precision == "float64":
            from optiland.backend.torch_backend import metal

            metal.set_mode(config.metal_mode)


_WORKER_CONFIG: WorkerConfig | None = None
_WORKER_INDEX: int = -1


def _initializer(configs: Sequence[WorkerConfig], counter: Any) -> None:
    global _WORKER_CONFIG, _WORKER_INDEX
    with counter.get_lock():
        _WORKER_INDEX = counter.value
        counter.value += 1
    _WORKER_CONFIG = configs[_WORKER_INDEX % len(configs)]
    configure_worker(_WORKER_CONFIG)


def _run(payload: tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]) -> Any:
    fn, args, kwargs = payload
    return fn(*args, **kwargs)


def evaluate_parallel(
    fn: Callable[..., Any],
    jobs: Iterable[Any],
    *,
    workers: int | Sequence[WorkerConfig] = 8,
    chunksize: int = 1,
    context: str = "spawn",
) -> list[Any]:
    """Run ``fn(*job)`` for every job in worker processes; results in job order.

    Args:
        fn: Module-level (picklable by reference) function. It receives one job's
            items as positional arguments (a job that is not a tuple is passed as a
            single argument). It runs with the worker's backend already configured.
        jobs: Iterable of jobs (tuples of picklable arguments, e.g. an
            ``Optic.to_dict()`` and parameters).
        workers: Number of CPU float64 torch workers, or an explicit sequence of
            ``WorkerConfig`` (one process per entry; workers cycle through the
            sequence).
        chunksize: Jobs per task handed to a worker.
        context: Multiprocessing start method; keep ``'spawn'`` on macOS.

    Returns:
        list: ``fn``'s return values in job order.
    """
    if isinstance(workers, int):
        configs: Sequence[WorkerConfig] = [WorkerConfig()] * workers
    else:
        configs = list(workers)
    if not configs:
        raise ValueError("at least one worker is required")
    ctx = mp.get_context(context)
    counter = ctx.Value("i", 0)
    payloads = [(fn, job if isinstance(job, tuple) else (job,), {}) for job in jobs]
    with ctx.Pool(
        len(configs), initializer=_initializer, initargs=(configs, counter)
    ) as pool:
        return pool.map(_run, payloads, chunksize=chunksize)


# ---------------------------------------------------------------------------
# Shape-aware scheduling: a CPU pool plus a GPU worker, jobs routed by cost
# ---------------------------------------------------------------------------

CPU = "cpu"
GPU = "gpu"


def performance_cores() -> int:
    """Performance cores (macOS ``hw.perflevel0``), else ``os.cpu_count()``."""
    if sys.platform == "darwin":
        # perflevel0 is the performance cluster on Apple silicon; the key names
        # differ between macOS releases, so several are tried in order.
        for key in ("hw.perflevel0.physicalcpu", "hw.perflevel0.logicalcpu"):
            try:
                out = subprocess.run(
                    ["sysctl", "-n", key],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                value = int(out.stdout.strip().split()[0]) if out.stdout.strip() else 0
            except (OSError, ValueError, subprocess.SubprocessError):
                value = 0
            if value > 0:
                return value
    return max(1, os.cpu_count() or 1)


MEASURED_HOST = "Apple M1 Max"


def host_signature() -> str:
    """The CPU brand string (``machdep.cpu.brand_string`` on macOS) or the platform."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            value = out.stdout.strip()
            if value:
                return value
        except (OSError, subprocess.SubprocessError):
            pass
    import platform

    return f"{platform.machine()} {platform.processor()}".strip()


def gpu_available() -> bool:
    """Whether a Metal device is available to a worker (torch with MPS)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional
        return False
    return bool(torch.backends.mps.is_available())


def trace_rays(num_rays: int, distribution: str = "hexapolar") -> int:
    """Rays that ``optic.trace(num_rays=..., distribution=...)`` launches per field.

    Optiland's distributions interpret ``num_rays`` differently (rings for the
    hexapolar pattern, points per axis for the uniform grid, ...), so the
    count is read off the distribution itself; it is cheap and exact.
    """
    from optiland.distribution import create_distribution

    dist = create_distribution(distribution)
    dist.generate_points(num_rays)
    return int(len(dist.x))


@dataclass(frozen=True)
class Job:
    """One unit of work for :func:`run_scheduled`.

    ``fn(*args, **kwargs)`` runs in a worker whose backend the scheduler picked.
    The shape fields are the cost hints (they never change what ``fn`` does):

    Attributes:
        kind: ``'trace'`` (one or more full traces of ``rays`` rays each),
            ``'batch'`` (a ``trace_batch`` call: ``designs`` designs of ``rays``
            rays), ``'cpu'`` or ``'gpu'`` (forced resource; the shape fields
            still order the queue).
        rays: rays per trace (``'trace'``) or per design (``'batch'``).
        traces: traces per job (``'trace'``: e.g. fields x wavelengths).
        designs: designs per job (``'batch'``).
        surfaces: surfaces in the system including object and image (the work
            per ray is ``surfaces - 1`` surface steps).
    """

    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    kind: str = "trace"
    rays: int = 0
    traces: int = 1
    designs: int = 1
    surfaces: int = 8

    def __post_init__(self) -> None:
        if self.kind not in ("trace", "batch", CPU, GPU):
            raise ValueError(f"unknown job kind {self.kind!r}")

    @property
    def steps_per_trace(self) -> int:
        return max(0, self.surfaces - 1) * max(0, self.rays)

    @property
    def steps(self) -> int:
        """Surface steps the job performs in total (its work)."""
        return self.steps_per_trace * max(1, self.traces) * max(1, self.designs)


@dataclass(frozen=True)
class CostModel:
    """Predicted seconds per job on a CPU worker (in a full pool) and on the GPU worker.

    The shipped coefficients are the measurements of NOTES/08-parallel-saturation.md
    (M1 Max, Cooke triplet: 7 surface steps per ray) after the fused-kernel levers;
    :meth:`calibrate` refits them on the current machine. The model only has to
    rank resources per job and balance queues, so 20-30% error is harmless.

    Attributes:
        cpu_fixed: seconds per trace of Optiland bookkeeping on a CPU worker.
        cpu_per_step: seconds per surface-step on a CPU worker in a full pool at
            the knee; above ``cpu_knee`` steps per trace the pool is memory-bound
            and the cost grows by ``cpu_bandwidth_growth`` per decade.
        cpu_batch_per_design: extra seconds per design of a CPU batch (the
            sequential contract loop: variable update, records, generation).
        gpu_fixed: seconds per trace on the GPU worker (ray generation and the
            driver, both still per-op) and ``gpu_per_step`` its kernel rate.
        gpu_batch_fixed / gpu_batch_per_design / gpu_batch_per_step: the same
            for a ``trace_batch`` call, where one launch carries every design.
        co_run_cpu / co_run_gpu: slowdown of each side while the other is busy
            (they share cores and memory bandwidth; the GPU worker's host thread
            suffers most). Measured on the mixed batch pool of
            NOTES/08-parallel-saturation.md section 6: the CPU workers ran 1.3x
            slower and the GPU worker 1.7x slower than each alone.
    """

    cpu_fixed: float = 0.003
    cpu_per_step: float = 1.1e-7
    cpu_knee: float = 7e5
    cpu_bandwidth_growth: float = 0.4
    cpu_batch_per_design: float = 0.0025
    gpu_fixed: float = 0.055
    gpu_per_step: float = 1.7e-8
    gpu_batch_fixed: float = 0.09
    gpu_batch_per_design: float = 2.7e-4
    gpu_batch_per_step: float = 4.0e-9
    co_run_cpu: float = 1.3
    co_run_gpu: float = 1.7
    host: str = MEASURED_HOST

    def _cpu_step_rate(self, steps_per_trace: float) -> float:
        growth = 0.0
        if steps_per_trace > self.cpu_knee > 0:
            growth = self.cpu_bandwidth_growth * math.log10(
                steps_per_trace / self.cpu_knee
            )
        return self.cpu_per_step * (1.0 + growth)

    def seconds(self, job: Job, resource: str) -> float:
        """Predicted seconds of ``job`` on ``resource`` (``'cpu'`` or ``'gpu'``)."""
        traces = max(1, job.traces)
        designs = max(1, job.designs)
        spt = float(job.steps_per_trace)
        if resource == CPU:
            if job.kind == "batch":
                per_design = self.cpu_batch_per_design + self.cpu_fixed
                return designs * (per_design + self._cpu_step_rate(spt) * spt)
            return traces * (self.cpu_fixed + self._cpu_step_rate(spt) * spt)
        if resource == GPU:
            if job.kind == "batch":
                return (
                    self.gpu_batch_fixed
                    + designs * self.gpu_batch_per_design
                    + self.gpu_batch_per_step * spt * designs
                )
            return traces * (self.gpu_fixed + self.gpu_per_step * spt)
        raise ValueError(f"unknown resource {resource!r}")

    @classmethod
    def calibrate(
        cls,
        *,
        cpu_workers: int | None = None,
        gpu: bool | None = None,
        cpu_config: WorkerConfig | None = None,
        gpu_config: WorkerConfig | None = None,
        context: str = "spawn",
    ) -> CostModel:
        """Refit the coefficients on this machine (about a minute; pools included).

        Probes trace the Cooke triplet: a full CPU pool at 1e3 and 1e5 rays per
        trace (fixed and per-step cost in the pool; every worker probes at the
        same time, so the pool's memory contention is in the numbers), the GPU
        worker at 1e3, 1e5
        and 1e6 rays (fixed and per-step) and one 100 x 1e5 batch. The
        bandwidth growth and the co-run factors keep their shipped values.
        """
        base = cls()
        n_cpu = cpu_workers or performance_cores()
        cpu_cfg = cpu_config or WorkerConfig(backend="numpy")
        use_gpu = gpu_available() if gpu is None else gpu
        # CPU pool: n_cpu jobs of each size so every worker holds one (pool regime)
        small = evaluate_parallel(
            _probe_trace,
            [(1_000, 20)] * n_cpu,
            workers=[cpu_cfg] * n_cpu,
            context=context,
        )
        large = evaluate_parallel(
            _probe_trace,
            [(100_000, 8)] * n_cpu,
            workers=[cpu_cfg] * n_cpu,
            context=context,
        )
        t_small = _median(r["seconds"] for r in small)
        t_large = _median(r["seconds"] for r in large)
        steps_small, steps_large = 7 * small[0]["rays"], 7 * large[0]["rays"]
        cpu_per_step = max(1e-9, (t_large - t_small) / (steps_large - steps_small))
        cpu_fixed = max(1e-4, t_small - cpu_per_step * steps_small)
        values = {
            "cpu_fixed": cpu_fixed,
            "cpu_per_step": cpu_per_step,
            "cpu_knee": float(steps_large),
        }
        if use_gpu:
            gpu_cfg = gpu_config or WorkerConfig(
                backend="torch",
                device="mps",
                precision="float64",
                metal_mode="df64",
                fused="1",
            )
            probes = evaluate_parallel(
                _probe_trace,
                [(1_000, 20), (100_000, 8), (1_000_000, 3)],
                workers=[gpu_cfg],
                context=context,
            )
            t0, t1, t2 = (r["seconds"] for r in probes)
            s0, s1, s2 = (7 * r["rays"] for r in probes)
            gpu_per_step = max(1e-10, (t2 - t1) / (s2 - s1))
            gpu_fixed = max(1e-4, t0 - gpu_per_step * s0)
            batch = evaluate_parallel(
                _probe_batch, [(100, 100_000, 1)], workers=[gpu_cfg], context=context
            )[0]
            b_steps = 7 * batch["rays"] * batch["designs"]
            # keep the shipped split between fixed and per-design; scale per-step
            residual = (
                batch["seconds"]
                - base.gpu_batch_fixed
                - batch["designs"] * base.gpu_batch_per_design
            )
            gpu_batch_per_step = max(1e-11, residual / b_steps)
            values.update(
                gpu_fixed=gpu_fixed,
                gpu_per_step=gpu_per_step,
                gpu_batch_per_step=gpu_batch_per_step,
            )
        values["host"] = host_signature()
        return cls(**{**asdict(base), **values})

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the coefficients as JSON (see :func:`default_model`)."""
        p = os.fspath(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> CostModel:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _cache_path(host: str) -> str:
    base = os.environ.get("OPTILAND_SCHEDULER_CACHE")
    if not base:
        home = os.path.expanduser("~")
        if sys.platform == "darwin":
            base = os.path.join(home, "Library", "Caches", "optiland")
        else:
            xdg = os.environ.get("XDG_CACHE_HOME", os.path.join(home, ".cache"))
            base = os.path.join(xdg, "optiland")
    slug = "".join(ch if ch.isalnum() else "-" for ch in host).strip("-").lower()
    return os.path.join(base, f"scheduler-{slug or 'host'}.json")


def default_model(calibrate: str | bool = "auto", **calibrate_kwargs: Any) -> CostModel:
    """The cost model for this machine: shipped, cached, or calibrated.

    ``calibrate='auto'`` (also the ``OPTILAND_SCHEDULER_CALIBRATE`` variable:
    ``auto`` / ``1`` / ``0``) uses the shipped coefficients on the machine they
    were measured on (:data:`MEASURED_HOST`), a cached calibration for any
    other host, and calibrates once (about a minute) when there is no cache,
    saving it under the user's cache directory (``OPTILAND_SCHEDULER_CACHE``
    overrides the directory). ``True`` always recalibrates; ``False`` never
    does (shipped coefficients, whatever the host). A newer chip with more
    performance cores changes both sides of the balance, which is why the
    model is refitted rather than scaled.
    """
    env = os.environ.get("OPTILAND_SCHEDULER_CALIBRATE")
    if env is not None and calibrate == "auto":
        table = {"1": True, "true": True, "0": False, "false": False}
        calibrate = table.get(env.lower(), "auto")
    host = host_signature()
    if calibrate is False:
        return CostModel() if host == MEASURED_HOST else CostModel(host=host)
    path = _cache_path(host)
    if calibrate == "auto":
        if host == MEASURED_HOST:
            return CostModel()
        if os.path.exists(path):
            try:
                return CostModel.load(path)
            except (OSError, ValueError, TypeError):
                pass
    model = CostModel.calibrate(**calibrate_kwargs)
    with contextlib.suppress(OSError):
        model.save(path)
    return model


def _median(values: Iterable[float]) -> float:
    xs = sorted(values)
    return xs[len(xs) // 2] if xs else 0.0


def _probe_trace(rays: int, repeat: int) -> dict[str, Any]:
    """Calibration probe: ``repeat`` Cooke-triplet traces of about ``rays`` rays."""
    import optiland.backend as be
    from optiland.samples.objectives import CookeTriplet

    rings = max(1, round((math.sqrt(1 + 4 * (rays - 1) / 3) - 1) / 2))
    lens = CookeTriplet()
    _sync = _synchronizer()
    lens.trace(
        Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=rings, distribution="hexapolar"
    )
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = lens.trace(
            Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=rings, distribution="hexapolar"
        )
        _sync()
    seconds = (time.perf_counter() - t0) / repeat
    return {"rays": int(be.size(out.x)), "seconds": seconds}


def _probe_batch(designs: int, rays: int, repeat: int) -> dict[str, Any]:
    """Calibration probe: one ``trace_batch`` of ``designs`` Cooke triplets."""
    import numpy as np

    from optiland.optimization.variable import Variable
    from optiland.raytrace.batch_trace import trace_batch
    from optiland.samples.objectives import CookeTriplet

    rings = max(1, round((math.sqrt(1 + 4 * (rays - 1) / 3) - 1) / 2))
    optic = CookeTriplet()
    var = Variable(optic, "radius", surface_number=5)
    physical = float(var.variable.inverse_scale(var.value))
    rng = np.random.default_rng(0)
    values = np.array(
        [
            [float(var.variable.scale(physical * (1 + 0.02 * (2 * u - 1))))]
            for u in rng.random(designs)
        ]
    )
    _sync = _synchronizer()
    trace_batch(
        optic,
        [var],
        values,
        Hx=0.0,
        Hy=1.0,
        wavelength=0.55,
        num_rays=rings,
        record="image",
    )
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        trace_batch(
            optic,
            [var],
            values,
            Hx=0.0,
            Hy=1.0,
            wavelength=0.55,
            num_rays=rings,
            record="image",
        )
        _sync()
    return {
        "designs": designs,
        "rays": 1 + 3 * rings * (rings + 1),
        "seconds": (time.perf_counter() - t0) / repeat,
    }


def _synchronizer() -> Callable[[], None]:
    import optiland.backend as be

    if be.get_backend() == "torch" and be.get_device() == "mps":
        import torch

        return torch.mps.synchronize
    return lambda: None


@dataclass
class Plan:
    """The schedule :func:`plan_jobs` produced.

    Attributes:
        assignment: resource per job (``'cpu'`` or ``'gpu'``), in job order.
        order: job indices per resource in submission order (longest first).
        seconds: predicted seconds per job on its assigned resource.
        makespan: predicted seconds per resource.
        cpu_workers / gpu_workers: pool sizes the plan assumed.
    """

    assignment: list[str]
    order: dict[str, list[int]]
    seconds: list[float]
    makespan: dict[str, float]
    cpu_workers: int
    gpu_workers: int

    @property
    def uses_gpu(self) -> bool:
        return bool(self.order.get(GPU))


def plan_jobs(
    jobs: Sequence[Job],
    *,
    cpu_workers: int,
    gpu_workers: int,
    model: CostModel | None = None,
) -> Plan:
    """Assign every job to the CPU pool or the GPU worker(s) by predicted finish time.

    Longest-processing-time list scheduling: jobs are taken in decreasing
    predicted cost and each goes to the resource on which it would finish
    earliest given what is already queued there. Small traces therefore stay on
    the CPU pool, batched sweeps and million-ray traces go to the GPU, and the
    GPU only takes a small job when every CPU slot is busy far enough ahead.
    Forced kinds (``'cpu'``, ``'gpu'``) are honoured when the resource exists;
    a ``'gpu'`` job with no GPU worker runs on the CPU with a warning.
    """
    model = model or CostModel()
    jobs = list(jobs)
    n = len(jobs)
    cpu_workers = max(1, int(cpu_workers))
    gpu_workers = max(0, int(gpu_workers))
    t_cpu = [model.seconds(j, CPU) for j in jobs]
    t_gpu = [model.seconds(j, GPU) if gpu_workers else math.inf for j in jobs]
    forced_gpu = [j.kind == GPU for j in jobs]
    forced_cpu = [j.kind == CPU for j in jobs]
    if gpu_workers == 0 and any(forced_gpu):
        warnings.warn(
            "jobs of kind 'gpu' run on the CPU pool: no GPU worker",
            RuntimeWarning,
            stacklevel=2,
        )
    # Co-run penalties apply when both resources will be busy; decide after a dry pass.
    mixed = (
        gpu_workers > 0
        and any(not c for c in forced_cpu)
        and any(not g for g in forced_gpu)
    )
    pen_cpu = model.co_run_cpu if mixed else 1.0
    pen_gpu = model.co_run_gpu if mixed else 1.0
    key = [
        max(t_cpu[i], t_gpu[i] if math.isfinite(t_gpu[i]) else 0.0) for i in range(n)
    ]
    ordered = sorted(range(n), key=lambda i: -key[i])
    cpu_heap = [0.0] * cpu_workers
    gpu_heap = [0.0] * gpu_workers
    heapq.heapify(cpu_heap)
    heapq.heapify(gpu_heap)
    assignment = [CPU] * n
    seconds = [0.0] * n
    for i in ordered:
        c = t_cpu[i] * pen_cpu
        g = t_gpu[i] * pen_gpu if gpu_workers else math.inf
        finish_cpu = cpu_heap[0] + c
        finish_gpu = (gpu_heap[0] + g) if gpu_workers else math.inf
        if forced_cpu[i] or gpu_workers == 0:
            choose_gpu = False
        elif forced_gpu[i]:
            choose_gpu = True
        else:
            choose_gpu = finish_gpu < finish_cpu
        if choose_gpu:
            assignment[i] = GPU
            seconds[i] = g
            heapq.heapreplace(gpu_heap, gpu_heap[0] + g)
        else:
            assignment[i] = CPU
            seconds[i] = c
            heapq.heapreplace(cpu_heap, cpu_heap[0] + c)
    order = {
        CPU: sorted(
            (i for i in range(n) if assignment[i] == CPU), key=lambda i: -seconds[i]
        ),
        GPU: sorted(
            (i for i in range(n) if assignment[i] == GPU), key=lambda i: -seconds[i]
        ),
    }
    makespan = {
        CPU: max(cpu_heap) if cpu_heap else 0.0,
        GPU: max(gpu_heap) if gpu_heap else 0.0,
    }
    return Plan(assignment, order, seconds, makespan, cpu_workers, gpu_workers)


def run_scheduled(
    jobs: Sequence[Job],
    *,
    cpu_workers: int | None = None,
    gpu_workers: int | None = None,
    model: CostModel | None = None,
    cpu_config: WorkerConfig | None = None,
    gpu_config: WorkerConfig | None = None,
    context: str = "spawn",
    return_plan: bool = False,
) -> list[Any] | tuple[list[Any], Plan]:
    """Run ``jobs`` on a CPU pool plus a GPU worker, each job where it finishes soonest.

    Defaults: one NumPy float64 worker per performance core minus one core per
    GPU worker (the GPU worker's host thread needs a performance core of its
    own); one GPU worker (torch, ``mps``, emulated float64, fused trace on)
    when a Metal device is available, else none. The plan comes from
    :func:`plan_jobs` with ``model``, by default :func:`default_model` (the
    shipped coefficients on the machine they were measured on, a cached or
    fresh calibration elsewhere). The GPU pool is only started when the plan
    sends it work.

    Returns:
        ``fn`` results in job order, or ``(results, plan)`` with ``return_plan``.
    """
    jobs = list(jobs)
    if not jobs:
        return ([], plan_jobs([], cpu_workers=1, gpu_workers=0)) if return_plan else []
    if gpu_workers is None:
        gpu_workers = 1 if gpu_available() else 0
    if cpu_workers is None:
        cpu_workers = max(1, performance_cores() - gpu_workers)
    if model is None:
        model = default_model()
    plan = plan_jobs(
        jobs, cpu_workers=cpu_workers, gpu_workers=gpu_workers, model=model
    )
    cpu_cfg = cpu_config or WorkerConfig(backend="numpy")
    gpu_cfg = gpu_config or WorkerConfig(
        backend="torch", device="mps", precision="float64", metal_mode="df64", fused="1"
    )
    ctx = mp.get_context(context)
    pools: dict[str, Any] = {}
    results: list[Any] = [None] * len(jobs)
    try:
        if plan.order[GPU]:
            counter = ctx.Value("i", 0)
            pools[GPU] = ctx.Pool(
                plan.gpu_workers,
                initializer=_initializer,
                initargs=([gpu_cfg] * plan.gpu_workers, counter),
            )
        if plan.order[CPU]:
            counter = ctx.Value("i", 0)
            pools[CPU] = ctx.Pool(
                plan.cpu_workers,
                initializer=_initializer,
                initargs=([cpu_cfg] * plan.cpu_workers, counter),
            )
        pending: list[tuple[int, Any]] = []
        for resource in (GPU, CPU):
            pool = pools.get(resource)
            if pool is None:
                continue
            for i in plan.order[resource]:
                job = jobs[i]
                pending.append(
                    (i, pool.apply_async(_run, ((job.fn, job.args, dict(job.kwargs)),)))
                )
        for i, handle in pending:
            results[i] = handle.get()
    finally:
        for pool in pools.values():
            pool.close()
            pool.join()
    return (results, plan) if return_plan else results
