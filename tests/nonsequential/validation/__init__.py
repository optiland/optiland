"""NSQ validation suite (§6, D8, PR16).

Closed-form analytic benchmarks (§6.1) and cross-cutting invariants (§6.2)
for the non-sequential engine, run in CI alongside the rest of
``tests/nonsequential/``. See
``docs/gallery/nonsequential/validation_report.rst`` for the narrative
summary of what each module checks and why, and for the two benchmarks
(prism at minimum deviation, integrating sphere) deliberately deferred out
of this pass -- named rather than silently missing.

Kramer Harrison, 2026
"""
