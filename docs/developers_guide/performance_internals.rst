.. _performance_internals:

Performance Internals
=====================

Optiland has one reference implementation of each calculation: the generic
code, which runs on all backends. Some calculations also have a fast path.
In a stated domain, a fast path gives results that are bit-identical to the
results of the generic code. Outside that domain, the generic code runs. No
fast path changes the public API.

This page gives the rules for fast paths and the domain of each one. Read it
before you change a module in the overview table, or code that a fast path
calls.

The equivalence rule
--------------------

Each number in a fast-path result has the same bits as the number that the
generic code gives. This includes the sign of a zero, and the position and
the sign bit of each NaN, for example of a ray that misses a surface. A
result that is only close is not equal.

The tests check this rule with ``assert_bits_equal`` from ``tests/utils.py``.
It compares the dtype, the shape, the positions of NaN, the bytes of all
other entries and the sign bits of NaN. Compiled code can give a NaN another
sign bit. A test of such code passes ``ignore_nan_sign=True``, and the
section of that fast path says why. Each test computes a result twice in one
process, once with the fast path and once without it, and compares the two.
If you change an expression of the generic code, these tests fail until you
make the same change in the fast path. Do not change a test to accept a
difference.

Two facts about floating-point arithmetic limit what a fast path can do:

- Addition and multiplication are not associative. ``(a * b) * c`` and
  ``a * (b * c)`` can differ in the last bit, so a fast path must do the
  operations of the generic code in the same order.
- An element-wise NumPy function gives the same value for an element at any
  position in an array, and for any length of the array. This is true for
  contiguous float64 arrays, one NumPy build and one CPU. NumPy can use
  another loop for strided or overlapping memory, and a SIMD loop can differ
  in the last bit from the function of the C library.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 20 26 28 26

   * - Fast path
     - Module
     - Domain
     - How to check or switch off
   * - Evaluation epoch of a direct trace
     - ``optiland/surfaces/surface_group.py`` and ``optiland/_epoch.py``
     - ``SurfaceGroup.trace`` of more than 128 rays when no epoch is open,
       on all backends
     - Spy on ``SurfaceGroup._trace_surfaces`` and read
       ``optiland._epoch.current()``. Call ``_trace_surfaces`` to trace
       without an epoch.
   * - Field runs
     - ``optiland/fields/field_types/_runs.py``
     - ``AngleField.get_ray_origins`` on NumPy with float64, for 4096 rays
       or more in runs of 64 rays or more on average
     - Patch ``optiland.fields.field_types.angle.field_runs`` to return
       None, as ``tests/test_ray_generation_paths.py`` does.

Evaluation epoch of a direct trace
----------------------------------

An evaluation epoch (``optiland._epoch``) memoizes values that depend only
on a frozen optic state, for example the refractive index of a material for
one wavelength array. The optimizer evaluations and ``RealRayTracer`` open
an epoch. Many analyses, the paraxial tracer and scripts that trace explicit
rays call ``SurfaceGroup.trace`` directly.

``SurfaceGroup.trace`` reuses an open epoch. If no epoch is open, it opens
one for a bundle of more than 128 rays (``_MAX_RAYS_WITHOUT_EPOCH`` in
``optiland/surfaces/surface_group.py``). For a bundle of up to 128 rays, the
regular material cache costs less than the bookkeeping of an epoch. The
limit comes from a measurement with the system of the gate scenario
``surfaces_trace_small``, so measure it again when the trace changes.

The reference is ``SurfaceGroup._trace_surfaces`` without an epoch. The
epoch changes how a material evaluates a constant wavelength array: in the
epoch, the material evaluates the array at one element and broadcasts the
result. Without an epoch, a constant array of up to 1024 values is evaluated
at each element. The bits are the same because the material formulas are
element-wise. A custom material must also be element-wise, because constant
wavelength arrays are evaluated at one element.

A trace changes only the ray records of the surfaces, and no memoized value
depends on them, so the epoch contract holds during a trace. The
``OPTILAND_EPOCH_CHECK=1`` debug guard compares the optics that are passed
to ``evaluation_epoch`` at the start and at the end of the epoch. A direct
trace passes no optics, so the guard cannot see changes to the optic during
a direct trace. To check a trace with the guard, trace inside
``with _epoch.evaluation_epoch((optic,)):``. The trace then reuses that
epoch.

Field runs
----------

``RealRayTracer.trace`` repeats each field point for each pupil point, so a
large bundle has few different field points, in long runs.
``AngleField.get_ray_origins`` is the only user of these runs. For each run
of equal field points, it computes the validity checks of the field angles,
their tangents and the launch distances once. ``FieldRuns.expand`` then
repeats the results to one value per ray. The pupil offsets are computed
per ray. For a finite object, the values are repeated to one value per ray
before the sag of the object surface and the coordinate transform, because
these take any geometry.

The domain is the NumPy backend with float64, 1-D float64 arrays of one
length, at least 4096 rays (``_MIN_RAYS``) and an average run of at least
64 rays (``_MIN_RUN``). The runs are found by the bits of the field
coordinates, not by their values: ``-0.0 == 0.0``, but the tangent of
``-0.0`` is ``-0.0``. The reference is the per-ray expressions of
``AngleField.get_ray_origins``. NumPy computes each element of a contiguous
float64 array with the same loop, whatever its length, so each ray gets the
same bits. The runs keep the order of the rays, so a validity check reports
the same first bad field as the check on one value per ray.

Torch never uses runs. A gradient through a repeated value would add its
parts in another order.

How to verify a fast path
-------------------------

1. Find the tests of the fast path in the overview table and run them. Run
   them on both backends where both apply.
2. For a new case, write a test that computes the same result with the fast
   path and without it, in one process, and compares each array with
   ``assert_bits_equal``. Include rays that miss a surface or are clipped,
   so that the result has NaN and zero intensity.
3. Make a fault on purpose in the fast path, for example another order of
   two operations, and check that the test fails. Then undo the fault.
4. Run the performance gate (see below) to check that the fast path is
   faster.

Measuring performance
---------------------

``benchmarks/perf_regression.py`` is the performance gate. It times two git
refs on the same machine and compares the time ratios of its scenarios. Its
module docstring describes how to run it and when it fails.
