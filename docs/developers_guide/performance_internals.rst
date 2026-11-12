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
   * - Traces in parts
     - ``optiland/surfaces/_chunked_trace.py`` and
       ``optiland/_trace_probe.py``
     - NumPy traces of 131072 rays or more, in parts of 65536 rays, for the
       types of ``_allowed_types``, outside the domain of the compiled trace
       plan
     - Spy on ``optiland.surfaces._chunked_trace.trace_in_parts`` (see
       below). It runs on Windows only (``_ENABLED``). Set ``_MIN_RAYS``
       above the bundle size to switch it off.
   * - Compiled trace plan
     - ``optiland/surfaces/_trace_plan.py`` and
       ``optiland/backend/numpy_backend/_trace_kernel.py``
     - ``SurfaceGroup.trace`` of real rays with the NumPy backend, in the
       domain listed below
     - Spy on ``optiland.surfaces._trace_plan.trace_with_plan`` (see
       below). Set its private switch ``_ENABLED`` to False to switch it
       off.

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

Traces in parts
---------------

Each step of the generic trace makes new arrays of the bundle size. On
64-bit Windows, the heap gets blocks of about 1 MB or more from the
operating system (VirtualAlloc) and gives them back when they are freed.
Each new array of that size then pays a page fault for each page. A bundle
of 131072 rays (1 MiB per float64 array) is the smallest power of two above
that limit, and a part of 65536 rays (512 KiB per array) stays below it. So
``trace_in_parts`` traces a NumPy bundle of 131072 rays or more through all
surfaces in parts of 65536 rays. A trace of one million rays is then about
three times faster on Windows.

On Linux, the parts cost time instead. glibc gives the memory of each part
back to the system after the part, so each new part pays the page faults.
With the scenarios of the performance gate, a trace in parts took 1.4 to 1.6
times the time of the whole bundle, and with that memory kept, it was not
faster than the whole bundle. So ``trace_in_parts`` runs only on Windows
(``_ENABLED``), and returns False on Linux and macOS (not measured). The
tests set ``_ENABLED`` to True, so that they test the parts on each
platform.

``SurfaceGroup._trace_surfaces`` calls ``trace_in_parts`` when the compiled
trace plan (see below) returns False. The reference is the loop over the
surfaces in ``SurfaceGroup._trace_surfaces``. It runs on the other backends
and each time ``trace_in_parts`` returns False. The domain is in
``_allowed_types``:

- the NumPy backend, and rays of exactly the type ``RealRays`` whose arrays
  are 1-D float64 arrays of one length;
- the surfaces ``Surface``, ``ObjectSurface`` and ``ImageSurface``;
- the geometries ``StandardGeometry``, ``Plane``, ``EvenAsphere``,
  ``OddAsphere``, ``PolynomialGeometry``, ``BiconicGeometry`` and
  ``ToroidalGeometry``;
- the interaction models ``RefractiveReflectiveModel`` and
  ``ThinLensInteractionModel``, with no BSDF;
- no coating, or a ``SimpleCoating``, ``FresnelCoating``,
  ``ThinFilmCoating``, ``PolarizerCoating``, ``RetarderCoating`` or
  ``CartesianJonesCoating``;
- no aperture, or a ``RadialAperture``, ``OffsetRadialAperture``,
  ``EllipticalAperture`` or ``RectangularAperture``;
- ``HomogeneousPropagation`` in each medium.

The type checks are exact (``type(x) is T``), because a subclass can add a
step. Each type in this list traces each ray independently, or reports each
decision that uses all rays through the probe (``optiland._trace_probe``).
A step can take one decision for all rays of a call only if both outcomes
give each ray the same bits, as the early exit and the plane seed of the
Newton-Raphson solver do. Two decisions can change the bits. The tolerance
floor of the Newton-Raphson solver depends on the largest ``|t|`` of the
start values, so a part that uses it asks for the whole bundle
(``require_whole_bundle``). Absorption applies only if a ray has k > 0, so
each part reports this decision (``note_decision``), and the whole bundle is
traced if the parts report different decisions.

The ray arrays must be float64. For contiguous float64 arrays, one NumPy
build and one CPU, an element-wise NumPy function gives the same value for
an element at any position in an array, so a part gets the values that its
rays get in the whole bundle. The part size is also chosen for 8-byte
values.

The types of the materials are not checked, only their propagation models.
A material must compute the value of a ray from that ray alone, as the
built-in ones do. A part can have one wavelength while the whole bundle has
several. The part is then evaluated at one element and broadcast, and the
whole bundle element by element. Each built-in coating also computes the
value of a ray from that ray alone.

No step may write into a ray array in place. Each part holds read-only views
of the input arrays, so such a write raises. Each fallback runs the generic
loop on the whole bundle from the unchanged input: after an exception in a
part, a request for the whole bundle, different decisions, or a part whose
arrays differ from those of the first part. Nothing is written to the rays
or to the surfaces until all parts are traced.

The tests in ``tests/test_chunked_trace.py`` set ``_CHUNK_SIZE`` and
``_MIN_RAYS`` to small values, so that small bundles are traced in many
parts. They have one system for each allowed type.

The compiled trace plan
-----------------------

The generic trace runs each step of each surface as NumPy operations on the
whole bundle. For a small bundle, the Python work of the steps costs more
than the arithmetic. For a large bundle, each step reads and writes arrays
of the bundle size. ``SurfaceGroup._trace_surfaces`` therefore first calls
``trace_with_plan`` (``optiland/surfaces/_trace_plan.py``). In its domain,
one compiled kernel (``optiland/backend/numpy_backend/_trace_kernel.py``)
traces the rays through all surfaces, so the Python cost is paid once per
trace.

At each trace, ``trace_with_plan`` reads the live objects and builds a plan:
two small tables with one row per surface, an integer table (the kind of
each step) and a float table (the values of each step). The build reads a
few attributes per surface, so it costs little compared with the NumPy
operations of the generic trace. No plan is kept between traces, so a
change to a surface, a material or a coordinate system applies at the next
trace. If one object is outside the domain, the build stops, nothing
changes, and the trace in parts or the generic loop runs.

A trace with a plan has three steps:

1. The kernel ``trace_plan`` traces the rays in blocks of 256 rays. For
   each block, each surface runs short loops over the rays of the block,
   one function per step of the generic trace. These loops vectorize, and
   the buffers of a block stay in the L1 or L2 cache while the block goes
   through all surfaces.
2. Only if a medium absorbs: the kernel writes the exponent
   ``-alpha * t * 1e3`` of each ray in each absorbing medium, and a NumPy
   pass then computes the intensity from the input intensity with
   ``numpy.exp`` and the expression of ``RealRays.attenuate``, in the order
   of the generic trace, in parts of 65536 rays. A ray without power keeps
   ``i = 0``, also when a later miss gives a factor that is not finite.
3. The new arrays go to the rays, and the records go to the surfaces.

The first trace compiles the kernel, which takes some seconds. Numba keeps
the compiled code in its cache on disk, so later processes load it in a
fraction of a second.

Domain
~~~~~~

Each limit has its reason:

- The NumPy backend: the kernel works on NumPy arrays, and Torch needs the
  generic trace for its gradients. The precision float64
  (``be.set_precision``): the generic trace converts the translations with
  ``be.array`` to the backend precision. Numba with its JIT compiler: with
  ``NUMBA_DISABLE_JIT=1``, the kernel would run as Python code.
- Rays of exactly the type ``RealRays``: other ray types, for example
  ``PolarizedRays``, have other steps. The arrays ``x``, ``y``, ``z``,
  ``L``, ``M``, ``N``, ``i``, ``opd`` and ``w`` are 1-D float64
  ``np.ndarray`` (exact type) of one length, with at least one ray: the
  kernel reads float64 arrays, and the bit-identity argument is made for
  float64. ``is_normalized`` is True: otherwise
  ``HomogeneousPropagation`` normalizes the directions after each
  propagation.
- At least one surface: an empty list gives no tables.
- Surfaces of exactly the types ``Surface``, ``ImageSurface`` and
  ``ObjectSurface``: no subclass, because it can override a step.
- Geometries of exactly the types ``StandardGeometry`` and ``Plane``: each
  other geometry has its own distance and normal. The radius and the conic
  constant are 0-d float64 arrays: NumPy squares a 0-d array as ``R * R``,
  and the builder does the same. A NumPy scalar would be squared with
  ``pow``.
- A ``CoordinateSystem`` (exact type) without a reference coordinate system,
  with 0-d float64 coordinates: a reference frame adds transforms.
- No rotation, no mirror and no aperture: the kernel has no step for them.
- A ``RefractiveReflectiveModel`` (exact type) whose ``parent_surface`` is
  the surface: a shared model takes the geometry and the media of another
  surface. No coating and no BSDF: per-ray NumPy code changes the
  intensity, the polarization or the direction.
- ``HomogeneousPropagation`` (exact type) in the media before and after
  each surface: no GRIN, because the index changes along the path. The
  index at the surface is then ``material.n(w)``, the ``boundary_index`` of
  ``BasePropagationModel``.
- One refractive index and one absorption coefficient
  ``alpha = 4 * pi * k / lambda`` per medium: the kernel takes one value per
  surface. ``n(w)`` and ``k(w)`` are float64 values: a Python float, a NumPy
  float64, a 0-d array, or a 1-D array of length 1 or of the ray count with
  one value. A bundle of several wavelengths in a dispersive glass is
  outside this domain.

The types of the materials are not checked. As for the traces in parts, a
material must compute the value of a ray from that ray alone, and give the
same value at each call during a trace: the plan evaluates each material
once per trace. If a material raises an error, the plan returns False, and
the generic trace raises the error.

Tables and records
~~~~~~~~~~~~~~~~~~

The docstring of ``_trace_kernel.py`` has the one table of the columns:
their meaning, unit, the steps that read them and their default. The
builder starts each row from the defaults, and ``NO_ROW = -1`` is the
default of each column that names a row of another table. New columns go at
the end, so a column keeps its number.

``trace_with_plan`` owns the records of the surfaces that it traces. Before
the kernel runs, it replaces the records of the previous trace with one
read-only empty array, so a large trace does not hold two sets of records.
Then it sets the records as the generic trace does: with ``record=True``,
``x``, ``y``, ``z``, ``L``, ``M``, ``N``, ``intensity`` and ``opd`` get the
ray state after the surface, and ``u`` and ``aoi`` an empty array. Each
record is a row of one array per name, so a record keeps that array alive.
With ``record=False``, it calls ``Surface.reset``. When it returns False,
the records are unchanged, and ``SurfaceGroup._trace_surfaces`` resets
them. The surfaces before ``skip`` are reset before the plan.

The scalar rule
~~~~~~~~~~~~~~~

The builder computes each value of a surface in Python with the expression
of the generic code, so the kernel gets the same value. Examples are
``1 + k``, ``R**2``, the ratio ``u = n1 / n2`` of the refractive indices
and ``u**2``, and the absorption coefficient. Python floats give the same
bits as NumPy float64 for ``+ - * /``: each is one correctly rounded IEEE
754 operation. ``x**2`` is different. NumPy computes ``x**2`` as ``x * x``
only for an array (also a 0-d array). For a NumPy or Python scalar, NumPy
and Python call the C function ``pow``, which does not promise the rounding
of ``x * x``. For example, with the C library of Windows, ``pow(u, 2)`` and
``u * u`` differ in the last bit for about one value in 2000 (``u`` from 0.5
to 2). In ``RealRays.refract``, ``u`` is an array if one index is an array,
and a scalar if both indices are scalars, for example the ``numpy.float64``
that ``IdealMaterial`` returns for one ray. The builder repeats the case
(``_index_ratio``).

Rules for compiled code
~~~~~~~~~~~~~~~~~~~~~~~

- Repeat the NumPy expression in the same order (see the equivalence rule).
- Do not use ``fastmath``. It lets LLVM change the order of the operations
  and fuse a multiplication and an addition.
- Use ``error_model="numpy"``, so that a division by zero gives inf or NaN,
  as in NumPy.
- Do not compute ``exp``, ``sin``, ``cos`` or ``tan`` in the kernel when
  the generic code computes them with NumPy: two implementations can
  differ in the last bit. Compute them with NumPy before or after the
  kernel, as the intensity pass does with ``numpy.exp``. ``sqrt`` is
  correctly rounded in both, so the kernel computes it.
- Write each step as a function that Numba inlines into ``trace_plan``
  (``inline="always"``). A function that calls an inline overload, such as
  the conic arithmetic of ``optiland/backend/numpy_backend/conic.py``,
  stays a separate compiled function: after Numba inlines an overload, it
  removes code that it finds dead, and in ``trace_plan`` (Numba 0.65) it
  removed stores into the block buffers.

A NaN of the plan is at the same position as in the generic trace, but its
sign bit can differ. One cause: when both operands of an addition or a
multiplication are NaN, x86 returns the first one, and LLVM can swap the
operands of these operations. A NaN marks a ray that missed a surface, and
only its position matters. The tests of the plan pass
``ignore_nan_sign=True`` for this reason.

Add a step to the plan
~~~~~~~~~~~~~~~~~~~~~~

1. In ``_trace_plan.py``, accept the new type in ``_types()`` with an exact
   type check, and fill its columns in a ``_*_columns`` method of
   ``_Builder``. Return False for each value that the kernel cannot take.
2. In ``_trace_kernel.py``, add the step kind or the columns at the end of
   the tables, with their rows in the layout table of the docstring, and
   add the step as a function with the expressions of the generic code.
3. Extend the domain list on this page. The module docstring of
   ``_trace_plan.py`` refers to it.
4. Add a system with the new step to ``SYSTEMS`` in
   ``tests/test_trace_plan.py``. The tests compare the plan with the
   generic trace, bit for bit, for many ray counts, with and without
   records. Add a test for each new fallback.
5. Run the performance gate (see below). A change to ``trace_plan`` can
   change how LLVM vectorizes the loops, also for systems that do not use
   the new step.

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

Check which path a trace takes
------------------------------

``SurfaceGroup._trace_surfaces`` first calls ``trace_with_plan`` (the
compiled trace plan) and, if it returns False, ``trace_in_parts``. It looks
up both functions in their modules at each trace, so a spy on a module
attribute shows which path traced the rays. This spy on
``optiland.surfaces._chunked_trace.trace_in_parts`` shows whether a trace
ran in parts of 65536 rays. The Fresnel coatings put the system outside the
domain of the compiled trace plan:

.. code:: python

   import numpy as np

   import optiland.surfaces._chunked_trace as chunked
   from optiland.rays import RealRays
   from optiland.samples.objectives import CookeTriplet

   results = []
   original = chunked.trace_in_parts


   def spy(surfaces, rays, record):
       results.append(original(surfaces, rays, record))
       return results[-1]


   chunked.trace_in_parts = spy
   try:
       n = 200_000
       rays = RealRays(
           np.zeros(n), np.linspace(-2.0, 2.0, n), np.zeros(n),
           np.zeros(n), np.zeros(n), np.ones(n), np.ones(n), np.full(n, 0.55),
       )
       optic = CookeTriplet()
       optic.surfaces.set_fresnel_coatings()
       optic.surfaces.trace(rays)
   finally:
       chunked.trace_in_parts = original
   print(results)  # [True] on Windows: the trace ran in parts of 65536 rays

``False`` means that the generic loop traced the whole bundle. A spy on
``optiland.surfaces._trace_plan.trace_with_plan`` works in the same way: it
gives ``[True]`` for the Cooke triplet without coatings, and ``[False]``
with them. Compare a system with the domain of the plan to find the limit
that it breaks. With ``NUMBA_DISABLE_JIT=1``, the plan always gives
``False``. In a test, replace the module attribute with ``monkeypatch``
instead, for example
``monkeypatch.setattr(trace_plan, "trace_with_plan", spy)``, and switch the
plan off with ``monkeypatch.setattr(trace_plan, "_ENABLED", False)``.

Measuring performance
---------------------

``benchmarks/perf_regression.py`` is the performance gate. It times two git
refs on the same machine and compares the time ratios of its scenarios. Its
module docstring describes how to run it and when it fails.
