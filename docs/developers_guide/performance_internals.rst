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
   * - Host-value scope of a CUDA trace
     - ``optiland/_host_values.py`` and
       ``optiland/surfaces/surface_group.py``
     - The generic trace in ``SurfaceGroup.trace`` of rays on a CUDA
       device
     - Spy on ``optiland._host_values._copy`` to count the copies. Set the
       private switch ``_ENABLED`` of ``optiland._host_values`` to False
       to switch it off.

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

A trace with a plan has four steps:

1. A compiled kernel traces the rays in blocks of 256 rays (see "Two
   kernels" below). For each block, each surface runs short loops over the
   rays of the block, one function per step of the generic trace. These
   loops vectorize, and the buffers of a block stay in the L1 or L2 cache
   while the block goes through all surfaces.
2. Only if a surface is an even asphere: ``_floor_applies`` checks the
   tolerance of its Newton-Raphson iteration (see "Even aspheres" below).
   If the generic code would raise the tolerance, ``trace_with_plan``
   returns False.
3. Only if a medium absorbs: the kernel writes the exponent
   ``-alpha * t * 1e3`` of each ray in each absorbing medium, with the
   absorption coefficient of the medium or of the ray, and the clip mask of
   each aperture. A NumPy pass then computes the intensity from the input
   intensity with ``numpy.exp`` and the expressions of
   ``RealRays.attenuate`` and ``RealRays.clip``, in the order of the generic
   trace, in parts of 65536 rays. A ray without power keeps ``i = 0``, also
   when a later miss gives a factor that is not finite. Without absorption,
   the kernel applies the clips to the intensity itself.
4. The new arrays go to the rays, and the records go to the surfaces.

The first trace compiles the kernel, which takes some seconds. Numba keeps
the compiled code in its cache on disk, so later processes load it in a
fraction of a second.

Domain
~~~~~~

Each limit has its reason:

- The NumPy backend: the kernel works on NumPy arrays, and Torch needs the
  generic trace for its gradients. The precision float64
  (``be.set_precision``): the generic trace converts the translations and
  the rotation angles with ``be.array`` to the backend precision. Numba with
  its JIT compiler: with ``NUMBA_DISABLE_JIT=1``, the kernel would run as
  Python code.
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
- Geometries of exactly the types ``StandardGeometry``, ``EvenAsphere`` and
  ``Plane``: each other geometry has its own distance and normal. The
  radius and the conic constant are 0-d float64 arrays: NumPy squares a 0-d
  array as ``R * R``, and the builder does the same. A NumPy scalar would
  be squared with ``pow``.
- For an ``EvenAsphere``, the coefficients are a 1-D float64 array or a
  list of float64 numbers (Python int or float, NumPy float64, or 0-d
  float64 arrays), ``tol`` is such a number, and ``max_iter`` is a Python
  int in the range of int64: the kernel takes float64 values and an int64
  iteration limit. ``OddAsphere`` is a subclass of ``EvenAsphere`` with
  another sag, so it uses the generic trace, as the other Newton-Raphson
  geometries do.
- A ``CoordinateSystem`` (exact type) without a reference coordinate
  system: a reference frame adds transforms. Its coordinates and angles are
  0-d float64 arrays: the kernel takes one value per surface.
- No aperture, or exactly a ``RadialAperture`` or an
  ``OffsetRadialAperture``: each other aperture has its own ``contains``.
  Its radii and offsets are numbers (Python int or float, NumPy float64, or
  0-d float64 arrays): the builder computes ``r_max**2`` and ``r_min**2``
  with the objects of the aperture, as ``contains`` does.
- A ``RefractiveReflectiveModel`` (exact type, a refraction or a mirror)
  whose ``parent_surface`` is the surface: a shared model takes the geometry
  and the media of another surface. No coating and no BSDF: per-ray NumPy code changes the
  intensity, the polarization or the direction.
- ``HomogeneousPropagation`` (exact type) in the media before and after
  each surface: no GRIN, because the index changes along the path. The
  index at the surface is then ``material.n(w)``, the ``boundary_index`` of
  ``BasePropagationModel``.
- ``n(w)`` and ``k(w)`` are float64 values: a Python float, a NumPy
  float64, a 0-d array, or a 1-D array of length 1 or of the ray count. The
  kernel takes one value per surface or one value per ray. An array with
  different values, for example the index of a glass in a bundle of several
  wavelengths, goes into a row of the table of per-ray values, and so does
  an absorption coefficient ``alpha = 4 * pi * k / lambda`` that differs
  from ray to ray. Each array gets one row.

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
With ``record=False``, it calls ``Surface.reset``. When it returns False
before the kernel runs, the records are unchanged. Only the tolerance floor
of an even asphere gives False after the kernel, and the records of the
previous trace are then released. After each False,
``SurfaceGroup._trace_surfaces`` resets the records. The surfaces before
``skip`` are reset before the plan.

Two kernels
~~~~~~~~~~~

``_trace_kernel.py`` compiles one loop nest, ``_trace_blocks``, into two
kernels. ``trace_plan`` traces all plans. ``trace_basic_plan`` traces a
basic plan: a plan without rows of per-ray values and without even
aspheres, for example of a trace with one wavelength through spheres and
conics. Two flags of the loop nest, ``per_ray`` and ``newton``, are
constants in each kernel. Numba removes the branches that the constants do
not take before it compiles a kernel, so ``trace_basic_plan`` has no code
for per-ray values and for the Newton-Raphson iteration. It does not take
the tables of per-ray values, coefficients and seeds, and does not allocate
the buffers of the iteration.

The second kernel exists because this code made the other steps slower.
With the code for per-ray values in its loop nest, the kernel traced the
Cooke triplet with one wavelength in 5 to 7 percent more time (30000 and
200000 rays), although it ran none of that code. The code of the
Newton-Raphson iteration added about 1 percent at 200000 rays, and 1 us
to each kernel call of 64 rays, 0.2 us of it for its three tables and its
two buffers. The branches of that code run once per surface and block of
256 rays, and LLVM vectorizes the same loops with and without that code.
The cause is other machine code of the larger loop nest.

Both kernels run the same steps on a basic plan, so they give the same
bits. Keep it so: a step that a basic plan uses must not depend on a flag.
``tests/test_trace_plan.py`` checks which kernel each plan gets, and traces
a basic plan of each system with both kernels and compares the bits.

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

Even aspheres
~~~~~~~~~~~~~

An ``EvenAsphere`` has no closed-form intersection with a ray.
``NewtonRaphsonGeometry._solve_distance_primal`` starts at a seed and runs a
Newton-Raphson iteration on the ray parameter t. The kernel repeats this
solve (``_newton`` in ``_trace_kernel.py``):

- The seed is the distance to the base conic, as
  ``StandardGeometry.distance`` computes it: with ``geometry.radius``, not
  ``paraxial_radius``, and with the root preference of the aperture.
- A ray that misses the base conic starts on the vertex plane z = 0. The
  ray is kept only if the iteration converges to a root in front of it,
  t > 0. Otherwise it misses the surface.
- The rays of a block iterate together, with the rules of the generic
  solve: the residual F(t) = sag(x + t L, y + t M) - (z + t N), the slopes
  of the sag from the normalized surface normal, the floors of n_z and of
  dF/dt, the strict test ``abs(F) < tol``, the early exit when no ray is
  active, ``max_iter``, and NaN for a ray with a NaN residual. A ray that
  converged, or that has a NaN residual, takes no more steps, so each ray
  takes the steps that it takes in the generic solve.

The sag and its slopes use Horner's method, as ``EvenAsphere`` does. The
builder computes the products i * C_i in Python, with the objects of the
geometry, as ``EvenAsphere._surface_normal`` does.

One value depends on all rays. ``_effective_tolerance`` raises ``tol`` to
the round-off floor ``8 * eps * max(1, max|t|)``, with the largest
``abs(t)`` over the seeds of all rays of the call. If one seed is NaN,
NumPy's max is NaN, and Python's ``max(1.0, nan)`` is 1.0, so the floor is
``8 * eps``. The kernel iterates with ``tol`` and writes a summary of the
seeds of each asphere: a flag for a NaN seed, and the largest ``abs(t)`` of
the other seeds. After the kernel, ``_floor_applies`` computes the floor.
It is above ``tol`` only if a seed is above ``tol / (8 * eps)``: about
5.6e4 mm for the default ``tol = 1e-10``, and 563 mm for ``tol = 1e-12``.
Then ``trace_with_plan`` returns False and the generic trace runs, so such
a trace pays for both traces.

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
  kernel, as the intensity pass does with ``numpy.exp`` and the builder
  with the cosines and sines of the rotations. ``sqrt`` is correctly
  rounded in both, so the kernel computes it.
- Do not use a power function in an expression of the generic code that
  the kernel repeats, for example ``r2**i`` with i > 2. NumPy calls the
  ``pow`` of its C library for each element, and compiled code computes a
  power in another way, so the last bit can differ. ``EvenAsphere`` uses
  Horner's method, which needs only multiplications and additions, and the
  kernel repeats it.
- Write each step as a function that Numba inlines into the loop nest
  ``_trace_blocks`` (``inline="always"``). A function that calls an inline
  overload, such as the conic arithmetic of
  ``optiland/backend/numpy_backend/conic.py``, stays a separate compiled
  function: after Numba inlines an overload, it removes code that it finds
  dead, and in the kernel (Numba 0.65) it removed stores into the block
  buffers.

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
5. Run the performance gate (see below). A change to the loop nest
   ``_trace_blocks`` can change how LLVM vectorizes the loops, also for
   systems that do not use the new step. If a basic plan gets slower, put the new step behind
   a flag of the loop nest, so that ``trace_basic_plan`` has no code for it
   (see "Two kernels").

Host-value scope (CUDA)
-----------------------

The trace checks some tensor values in Python, with ``bool()`` or
``math.isinf()``. On a CUDA device, each such check copies the value to the
host, and the copy waits until the device has done all queued work (a CUDA
synchronization). ``SurfaceGroup._trace_surfaces`` therefore opens a
host-value scope (``optiland/_host_values.py``) around the loop of the
generic trace when the rays are on a CUDA device. The scope copies the
values of these checks to the host in one device-to-host copy, and the
checks read the copies.

``_host_value_scope`` in ``optiland/surfaces/surface_group.py`` collects
the values:

- the rotation angles of each coordinate system and of its reference
  coordinate systems, for ``_needs_rotation`` in
  ``optiland/coordinate_system.py``;
- the radius of each geometry, for ``_is_radius_infinite`` in
  ``optiland/geometries/standard.py``;
- k of each medium with exactly ``HomogeneousPropagation``, for the check
  ``any(k > 0)`` in ``HomogeneousPropagation.propagate``.

The scope copies ``any(k > 0)`` only inside an evaluation epoch and when the
wavelengths do not require gradients, that is when
``BaseMaterial._epoch_key`` gives a key. Only then does the material memo of
the epoch give the trace the same k array again, so k is not evaluated
twice. Otherwise each absorption check still waits. The Newton-Raphson
warning of the Torch path (``_warn_rejected_rays``) also copies its four
counts in one copy, without the scope.

The invariants:

- A check that reads a copy takes the branch that it takes on the tensor,
  so the results and the gradients do not change. float64 holds each
  float32 and float64 value exactly, so a copy has the truth value, the
  sign and the infinity of its tensor.
- ``_needs_rotation`` checks ``requires_grad`` first, so an angle that
  requires a gradient is applied, also at zero, without a host read.
- The copies are taken when the scope opens and are not updated. Code in
  the scope must not change these tensors in place. The Optiland trace does
  not, and a custom geometry, material or surface must not either.
- A value without a copy keeps its own check: ``host`` and ``absorbs``
  return None, and the caller checks the value as before. NumPy arrays, CPU
  tensors, complex tensors, tensors with more than one element (for
  ``host``) and tensors on another device than the first CUDA tensor have
  no copy.
- The scope keeps each copied tensor alive, so no other object can get its
  id while the scope is open. The scope lives in a context variable, so
  other threads do not see it.

To batch a new check:

1. Find the check in the trace: ``bool(x)``, ``float(x)``, ``x.item()``,
   or a backend call such as ``be.any(x)``, on a tensor ``x``.
2. Add ``x`` to the values in ``_host_value_scope``: to ``scalars`` for a
   one-element value, or to a new kind of copy in ``_copy`` for another
   reduction, as ``absorbs`` does for ``any(k > 0)``.
3. In the check, read the copy first (``host(x)``). If it is None, check
   the tensor as before.
4. Add the check to ``test_a_cuda_trace_does_not_wait_in_the_batched_checks``
   in ``tests/test_host_values.py``, and a system that takes both branches
   of the check to the bit tests there.

The tests run without a GPU. They add the CPU to ``_DEVICE_TYPES``, so the
scope copies CPU tensors with the same code. Each test computes a trace or a
gradient with the scope and without it (``_ENABLED = False``) and compares
the bits. The tests do not use the ``set_test_backend`` fixture, because it
switches grad mode on: the wavelengths then require gradients, and the
scope does not copy ``any(k > 0)``. Only
``test_a_cuda_trace_does_not_wait_in_the_batched_checks`` needs a CUDA
device. It finds the source of each CUDA synchronization with
``torch.cuda.set_sync_debug_mode("warn")`` and checks that no
synchronization comes from a batched check.

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
that it breaks. For a system with an even asphere, a spy on
``optiland.surfaces._trace_plan._floor_applies`` shows whether the kernel
ran and the tolerance floor gave False (``[True]``), or the build stopped
(no call). With ``NUMBA_DISABLE_JIT=1``, the plan always gives
``False``. In a test, replace the module attribute with ``monkeypatch``
instead, for example
``monkeypatch.setattr(trace_plan, "trace_with_plan", spy)``, and switch the
plan off with ``monkeypatch.setattr(trace_plan, "_ENABLED", False)``.

Measuring performance
---------------------

``benchmarks/perf_regression.py`` is the performance gate. It times two git
refs on the same machine and compares the time ratios of its scenarios. Its
module docstring describes how to run it and when it fails.
