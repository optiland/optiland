"""Performance benchmark harness for the NSQ Monte Carlo tracer.

Measures rays/sec against scene surface count, ray count, and max trace
depth, for both the NumPy and PyTorch backends, so later acceleration work
(BVH, batched traversal, ...) has a measured baseline to improve on. This is
a measurement harness, not a correctness test suite -- see
``tests/nonsequential/validation/`` for the closed-form benchmarks and
invariants that gate correctness.

Run directly::

    .venv/Scripts/python.exe -m benchmarks.nonsequential.run

Kramer Harrison, 2026
"""

from __future__ import annotations
