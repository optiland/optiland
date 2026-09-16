"""Multi-process evaluation of independent optical jobs (CPU workers + GPU).

Optiland's backend configuration is process-global, so parallel design
evaluation uses *processes*, each with its own backend/device/precision. On
Apple silicon the workers must be started with the ``spawn`` context: forking
after Metal has been initialized hangs (Apple documents fork-without-exec as
unsafe on all its platforms). Every worker that selects ``device='mps'`` owns
its own Metal command queue through torch; the GPU time-slices between them.
Measurements on an M1 Max show the GPU saturates with about two submitting
processes, so the recommended layout is a few GPU workers plus CPU workers, or
all CPU workers for tiny per-job workloads.

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

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["WorkerConfig", "evaluate_parallel", "configure_worker"]


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
    """

    device: str = "cpu"
    precision: str = "float64"
    backend: str = "torch"
    metal_mode: str = "df64"
    threads: int = 1


def configure_worker(config: WorkerConfig) -> None:
    """Apply ``config`` in the current process (called by the pool initializer)."""
    os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
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
    """Run ``fn(*job)`` for every job in worker processes and return the results in order.

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
    with ctx.Pool(len(configs), initializer=_initializer, initargs=(configs, counter)) as pool:
        return pool.map(_run, payloads, chunksize=chunksize)
