"""Non-gating benchmark for weighted wavefront evaluation.

Run from the repository root::

    python -m benchmarks.wavefront_evaluation

The output is descriptive timing data, not a pass/fail performance test.
"""

# ruff: noqa: I002

import argparse
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable

import numpy as np
import torch

import optiland.backend as be
from optiland._types import BEArrayT
from optiland.wavefront import WavefrontData, WavefrontRemoval, evaluate_wavefront


def _median_ms(operation: Callable[[], None], warmup: int, repeats: int) -> float:
    """Return the median wall-clock duration of an operation in milliseconds."""
    for _ in range(warmup):
        operation()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        timings.append((time.perf_counter() - start) * 1_000.0)
    return statistics.median(timings)


def _median_backward_ms(
    prepare: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> float:
    """Return median Torch backward time with graph construction excluded."""
    for _ in range(warmup):
        prepare().backward()
    timings = []
    for _ in range(repeats):
        objective = prepare()
        start = time.perf_counter()
        objective.backward()
        timings.append((time.perf_counter() - start) * 1_000.0)
    return statistics.median(timings)


def _end_to_end_operation(
    prepare: Callable[[], torch.Tensor],
) -> Callable[[], None]:
    """Create an operation that builds an autograd graph and runs backward."""

    def operation() -> None:
        prepare().backward()

    return operation


def _sample_values(
    sample_count: int, scenario: str, dtype_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct affine samples or an adversarial cancelling piston reduction."""
    dtype = getattr(np, dtype_name)
    indices = np.arange(sample_count, dtype=dtype)
    if scenario == "fallback":
        opd = np.ones(sample_count, dtype=dtype)
        large = 2.0 ** (24 if dtype_name == "float32" else 53)
        block_count = (sample_count - 1) // 4
        block = np.array([large, large + 2.0, -large, -large], dtype=dtype)
        opd[: 4 * block_count] = np.tile(block, block_count)
        return opd, indices, indices, np.ones(sample_count, dtype=dtype)
    if scenario == "asymmetric":
        x = np.linspace(-1.0, 1.0, sample_count, dtype=dtype)
        y = np.sin(indices * 0.37)
        weights = np.linspace(0.5, 1.5, sample_count, dtype=dtype)
        opd = 0.2 + 0.3 * x - 0.4 * y + 0.01 * np.cos(indices)
        return opd, x, y, weights

    theta = indices * (2.0 * math.pi / sample_count)
    x = np.cos(theta)
    y = np.sin(theta)
    weights = np.ones(sample_count, dtype=dtype)
    opd = 0.2 + 0.3 * x - 0.4 * y + 0.01 * np.cos(3.0 * theta)
    return opd, x, y, weights


def _build_evaluator(
    opd: BEArrayT,
    x: BEArrayT,
    y: BEArrayT,
    weights: BEArrayT,
    scenario: str,
    api: str,
) -> Callable[[BEArrayT], BEArrayT]:
    """Prepare fixed metadata outside timing, including the cached container."""
    remove: WavefrontRemoval = "piston" if scenario == "fallback" else "piston_tilt"
    effective_weights = None if scenario == "symmetric" else weights
    data = WavefrontData(
        pupil_x=x,
        pupil_y=y,
        pupil_z=x * 0.0,
        opd=opd,
        intensity=x * 0.0 + 1.0,
        radius=1.0,
    )

    def evaluate(values: BEArrayT) -> BEArrayT:
        if api == "cached":
            data.opd = values
            return data.evaluate(weights=effective_weights, remove=remove).rms
        return evaluate_wavefront(
            values, x=x, y=y, weights=effective_weights, remove=remove
        ).rms

    return evaluate


def _numpy_case(
    sample_count: int, scenario: str, dtype_name: str, api: str
) -> Callable[[], None]:
    """Construct a representative NumPy affine evaluation operation."""
    opd, x, y, weights = _sample_values(sample_count, scenario, dtype_name)
    evaluator = _build_evaluator(opd, x, y, weights, scenario, api)

    def evaluate() -> None:
        evaluator(opd)

    return evaluate


def _torch_cases(
    sample_count: int, scenario: str, dtype_name: str, api: str
) -> tuple[Callable[[], None], Callable[[], None], Callable[[], torch.Tensor]]:
    """Construct Torch no-grad, autograd-forward, and backward operations."""
    values = _sample_values(sample_count, scenario, dtype_name)
    base_opd, x, y, weights = (
        torch.tensor(value, dtype=getattr(torch, dtype_name)) for value in values
    )
    evaluator = _build_evaluator(base_opd, x, y, weights, scenario, api)

    def no_grad_forward() -> None:
        with torch.no_grad():
            evaluator(base_opd)

    def autograd_forward() -> None:
        opd = base_opd.detach().requires_grad_()
        evaluator(opd)

    def prepare_backward() -> torch.Tensor:
        opd = base_opd.detach().requires_grad_()
        return evaluator(opd)

    return no_grad_forward, autograd_forward, prepare_backward


def _print_environment(warmup: int, repeats: int) -> None:
    """Print the runtime metadata needed to interpret benchmark timings."""
    cpu = (
        os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or platform.machine()
        or "unknown"
    )
    thread_environment = ", ".join(
        f"{name}={os.environ.get(name, '<unset>')}"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    print(f"Python: {platform.python_version()} ({sys.implementation.name})")
    print(f"NumPy: {np.__version__}")
    print(f"Torch: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {cpu}; logical CPUs: {os.cpu_count()}")
    print(
        "Threads: "
        f"torch intra-op={torch.get_num_threads()}, "
        f"torch inter-op={torch.get_num_interop_threads()}, {thread_environment}"
    )
    print(f"Timing: warmup={warmup}, repeats={repeats}")


def main() -> None:
    """Run representative NumPy/Torch CPU timings and print a compact table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--samples", nargs="+", type=int, default=[32, 1_024, 4_096])
    parser.add_argument("--library-config", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if any(count < 8 for count in args.samples):
        parser.error("--samples must contain counts of at least 8")

    _print_environment(args.warmup, args.repeats)
    print(f"Dtype: {args.dtype}")
    if args.library_config:
        np.show_config()
        print(torch.__config__.show())
    print("Non-gating benchmark; times are median wall-clock milliseconds.")
    print("Cached container setup and trace generation are excluded from timing.")
    print(
        "Symmetric: implicit equal weights. Fallback: multiple large cancelling blocks."
    )
    header = (
        f"{'api':<12}{'scenario':<11}{'samples':>9}{'numpy_fwd':>12}{'torch_ng':>11}"
        f"{'torch_ag':>11}{'backward':>11}{'end_to_end':>13}"
    )
    print(header)
    print("-" * len(header))
    cases = (
        (api, scenario, count)
        for api in ("standalone", "cached")
        for scenario in ("asymmetric", "symmetric", "fallback")
        for count in args.samples
    )
    for api, scenario, sample_count in cases:
        be.set_backend("numpy")
        numpy_ms = _median_ms(
            _numpy_case(sample_count, scenario, args.dtype, api),
            args.warmup,
            args.repeats,
        )

        be.set_backend("torch")
        no_grad, autograd, prepare_backward = _torch_cases(
            sample_count, scenario, args.dtype, api
        )
        no_grad_ms = _median_ms(no_grad, args.warmup, args.repeats)
        autograd_ms = _median_ms(autograd, args.warmup, args.repeats)
        backward_ms = _median_backward_ms(prepare_backward, args.warmup, args.repeats)

        end_to_end_ms = _median_ms(
            _end_to_end_operation(prepare_backward),
            args.warmup,
            args.repeats,
        )
        print(
            f"{api:<12}{scenario:<11}{sample_count:>9}{numpy_ms:>12.3f}"
            f"{no_grad_ms:>11.3f}{autograd_ms:>11.3f}"
            f"{backward_ms:>11.3f}{end_to_end_ms:>13.3f}"
        )


if __name__ == "__main__":
    main()
