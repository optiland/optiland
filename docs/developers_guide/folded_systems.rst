Folded Systems and the Scalar Paraxial Domain
=============================================

Optiland's first-order (paraxial) machinery supports beam paths folded by
plane mirrors. This page states exactly what that support covers, the
conventions it uses, and where it deliberately stops.

Supported domain
----------------

    Fold-aware scalar paraxial support applies to **piecewise-centered
    systems whose propagation segments are connected by plane fold
    mirrors**. Powered surfaces must be normal to the local beam segment.
    General tilted or decentered powered systems, oblique powered mirrors,
    anamorphic pupils, and transverse first-order coupling require a vector
    paraxial model and are **not approximated silently**.

Concretely, the following work with full first-order analysis and ray
aiming (paraxial, iterative, and robust):

- straight systems entered along global ``+z``, and on-axis retro systems;
- systems rigidly translated away from global ``x = y = 0``;
- systems entered along an arbitrary finite direction (``+x``, ``-z``, ...);
- one or more plane fold mirrors at arbitrary 3-D orientations, including
  non-45-degree folds and out-of-plane periscopes;
- finite and infinite conjugates with **angle** fields;
- stops before a fold, after a fold, or between folds;
- powered surfaces on any reflection-parity leg, provided each powered
  surface is normal to its local propagation segment -- in **either**
  local-axis authoring (local ``+z`` along or against the beam).

Geometries outside this domain raise
:class:`~optiland.paraxial_path.UnsupportedParaxialGeometryError` from
first-order calls instead of returning plausible numbers. Real ray tracing
(``optic.surfaces.trace``) remains available for any physically valid
geometry. Inside the ray aimers, where paraxial values only seed a
real-ray Newton solve, the same finding is reported as a
:class:`~optiland.paraxial_path.ParaxialDomainWarning` so that e.g. a
slightly tilted stop mirror can still be aimed (the aimed rays are
verified against real traces). Both are importable from the package root
(``optiland.UnsupportedParaxialGeometryError``,
``optiland.ParaxialDomainWarning``).

Straight ``+z`` systems with tilted powered surfaces or interior decenters
remain fully supported with their historical first-order numbers (the
tilt/decenter is ignored by the scalar model, as it always was), but that
approximation is now surfaced as a :class:`ParaxialDomainWarning` instead
of staying silent. Real ray tracing accounts for the perturbation exactly.

The signed unfolded axial coordinate
------------------------------------

``SurfaceGroup.positions`` is a **1-D scalar coordinate along the unfolded
optical axis**, with each reflection reversing the direction of travel
(spacings after an odd number of mirrors are negative -- the classic
sequential mirror convention). It is *not* a Cartesian coordinate:

- ``positions`` -- unfolded signed scalar axial coordinate (paraxial use);
- ``global_z_positions`` -- only the global z component of the vertices;
- ``vertices_gcs`` -- full three-dimensional vertex positions.

While every leg runs along ``+z`` (entered along ``+z``), ``positions``
equals global z bit-for-bit. The per-operation metadata behind all of this
lives in one object, :class:`optiland.paraxial_path.ParaxialPath`, built by
``SurfaceGroup.build_paraxial_path()``; it carries vertices, local axes,
beam directions, reflection parity, orientation signs, the entry frame, and
the unsupported-geometry diagnostics. Geometry is mutable, so the path is
rebuilt per high-level operation rather than cached.

Orientation-aware effective power
---------------------------------

The same physical powered surface on a folded arm can be authored two ways:
local ``+z`` along the beam, or against it (with the opposite radius sign).
Real rays treat both identically; the scalar paraxial model maps authored
values to effective ones using the orientation sign

.. math::

   s_k = p_k \,\operatorname{sgn}(\hat z_k \cdot \mathbf d_k),
   \qquad R_{\mathrm{eff},k} = s_k R_k, \qquad f_{\mathrm{eff},k} = s_k f_k,

where :math:`p_k` is the reflection parity before surface *k*,
:math:`\hat z_k` its local ``+z`` axis, and :math:`\mathbf d_k` the incoming
beam direction. The **canonical default authoring has sign** :math:`+1`;
equivalent centered surfaces authored with the local axis reversed are
normalized to the same effective scalar power -- on straight and folded
paths alike. Authored radii and focal lengths are never mutated;
serialization is unaffected.

The sign applies exactly to **centered/collinear** powered surfaces
(:math:`\bigl|1 - |\hat z_k \cdot \mathbf d_k|\bigr| \le` the dtype-aware
angular tolerance). A genuinely oblique powered surface is never re-signed
by a heuristic: on folded/off-axis paths it is rejected as out of domain,
and on straight legacy paths it keeps its historical raw value together
with the scalar-approximation advisory.

One validated scalar sequence for trace and matrix APIs
-------------------------------------------------------

``Paraxial.ray_transfer_matrix()`` and ``Paraxial.f2_range()`` are
assembled from the **same validated scalar sequence** as the explicit
paraxial tracer (``trace_generic``): one internal preparation step
(``ParaxialRayTracer.prepare_scalar_sequence``) builds or reuses the
:class:`~optiland.paraxial_path.ParaxialPath`, validates the scalar domain,
surfaces straight-system advisories, applies the collinear orientation
signs to radii and explicit focal lengths (preserving radius infinities
exactly), and applies the reverse transform exactly once. The matrix API
and the explicit trace therefore can never disagree on the same
prescription because they selected different power conventions. Both
accept an optional prebuilt ``path=`` so a high-level operation pays the
path construction once.

Entry frame and pupil coordinates
---------------------------------

Object-space constructs live on the **entry line**, anchored at the first
physical surface's vertex (a decentered first vertex shifts the pupil line
with it). The entrance pupil's apparent point is
:math:`\mathbf r_{\mathrm{EP}} = \mathbf r_1 + EPL\,\mathbf d_0`; pupil
offsets are laid out on a transverse basis :math:`(\mathbf u_0, \mathbf
v_0)`:

- :math:`\mathbf v_0` is global ``+y`` projected off the entry axis
  (the meridional direction), :math:`\mathbf u_0` completes a right-handed
  frame;
- within the **dtype-aware** angular tolerance of a ``+/-y`` entry
  (``1e-10`` in float64, widened in float32 to stay above round-off) the
  projection degenerates -- the unavoidable pole of any deterministic
  gauge -- and global ``+x`` is projected instead;
- for a ``+z`` entry the basis reduces exactly to global ``+x/+y``;
- the gauge is a **global reference** and never depends on the object
  plane's orientation or roll -- a tilted object plane does not steer the
  axis or roll the field/pupil axes.

Through three-dimensional folds, pupil labels ``(Px, Py)`` and field labels
map onto rotated/flipped stop-local and image-local axes. Magnitudes are
preserved exactly; users comparing against a straight design should compare
magnitudes or transform frames explicitly.

Real-space pupil points are exposed directly:

- ``Paraxial.entrance_pupil_point_gcs()`` --
  :math:`\mathbf r_1 + EPL\,\mathbf d_0`;
- ``Paraxial.exit_pupil_point_gcs()`` --
  :math:`\mathbf r_I + p_I\,XPL\,\mathbf d_I` (image vertex, parity and
  beam direction at the image plane);
- ``Paraxial.entrance_pupil_axial_position()`` -- the axial scalar
  (``entrance_pupil_z()`` is a deprecated legacy alias of it and, despite
  the name, is **not** a Cartesian z).

The chief-ray wavefront reference sphere and the Huygens PSF normalization
use these 3-D points (never ``global_z + axial_scalar``).

Iterative and robust aiming
---------------------------

The Newton/Broyden aiming core solves two **true transverse degrees of
freedom** ``(xi, eta)`` in the entry frame
(:class:`optiland.rays.ray_aiming.parameterization.LaunchParameterization`):

- infinite conjugate: the launch point moves in the transverse plane at
  fixed field direction;
- finite conjugate: the object point stays fixed and the direction rotates
  in a per-ray orthonormal tangent basis around the seed direction, so
  every trial and line-search candidate is a unit vector.

Stop residuals are measured in the stop surface's local transverse
coordinates, and each solve reports seed residual, final residual,
iteration count, convergence status, whether the Jacobian conditioning
fallback was used, and how many iteration-time Jacobian refreshes ran
(:class:`~optiland.rays.ray_aiming.parameterization.SolveReport`); a
returned finite ray is not evidence of convergence. Pupil-map warm-start
caches store local transverse offsets, so rigid pose changes cannot reuse
an incompatible global-coordinate map.

The Newton core's 2x2 solves share one scale-invariant conditioning
authority (:func:`optiland.utils.solve_2x2`, also used by the
real-image-height field solve): each ray's Jacobian is normalized by its
largest entry, judged by a reciprocal Frobenius-condition estimate against
a machine-epsilon threshold, and solved with its determinant **sign
preserved** -- the determinant is never clamped to an arbitrary positive
value, so a Newton step can never be silently reversed. Initial Jacobians
use **central** finite differences with dtype- and scale-aware steps
(:math:`h = \varepsilon_{\mathrm{mach}}^{1/3} S`, with :math:`S` a
physical stop scale for infinite conjugates and a dimensionless unit for
finite ones). Before each Newton solve the reciprocal condition is
checked; ill-conditioned rays get a fresh central-difference refresh,
then the sign-preserving paraxial diagonal (reported as
``fallback_used``), and finally a held zero step that surfaces as
non-convergence -- a step is never fabricated. The Broyden update is
skipped for zero or round-off-level accepted steps.

Robust aiming reports
---------------------

``RobustRayAimer.last_report`` exposes a
:class:`~optiland.rays.ray_aiming.robust.RobustSolveReport` for the most
recent call: per-field
:class:`~optiland.rays.ray_aiming.robust.RobustFieldReport` entries (the
final polish :class:`SolveReport`, the chief seed strategy --
``initial_guess`` / ``cached_map`` / ``warm_map`` / ``direct_paraxial`` /
``marching`` / ``scan`` -- cached-map reuse, the edge-probe fallback
count, and a per-field fallback flag) plus aggregate ray counts, worst
seed/final residuals, and the largest polish iteration count. Robust
aiming may return NaN for individual vignetted/unreachable rays;
``converged`` is defined as ``num_converged == num_rays`` with exact
counts retained. On a total field failure, ``last_report`` is published
*before* the ``ValueError`` propagates. A cached map reused as a normal
warm start is reported via ``used_cached_map``/``chief_seed_strategy``,
not as a fallback. Reading the report::

    aimer = RobustRayAimer(optic)
    rays = aimer.aim_rays((0.0, 0.7), 0.55, (Px, Py))
    report = aimer.last_report
    if not report.converged:
        print(f"{report.num_converged}/{report.num_rays} rays converged")
    for field in report.field_reports:
        print(field.chief_seed_strategy, field.final_polish.final_residual)

The last-resort chief **scan** (used when the warm/direct solves and field
marching all fail, e.g. one-dimensional fields beyond 90 degrees) sweeps
candidates along the transverse line **through the fresh paraxial seed**:
the zero-offset candidate is exactly the seed, every displacement is
transverse to the entry direction, the sweep direction follows the field
coordinates, and the candidate set is invariant under rigid translation of
the system. Among converged candidates the one nearest the seed is
selected. Scan use is reported as ``chief_seed_strategy = "scan"``.

``total_track`` semantics
-------------------------

``total_track`` is the span of the **unfolded signed axial surface
coordinates** -- the track length of the scalar paraxial system, which the
``total_track`` optimization operand constrains. For a folded system this
differs from the bounding extent in global z; use
``SurfaceGroup.global_z_span`` for the old global-z meaning.

Explicitly unsupported (rejected, not approximated)
---------------------------------------------------

- powered mirrors at oblique incidence (astigmatic; ``OBLIQUE_POWERED_MIRROR``);
- tilted powered refractive surfaces on folded paths
  (``TILTED_REFRACTIVE_SURFACE``);
- plane refractive interfaces that steer the nominal axis (Snell steering);
- transversely decentered vertex chains on folded paths
  (``NONCOLLINEAR_VERTEX_CHAIN``);
- non-object surfaces at infinity on a folded arm (``NONOBJECT_INFINITY``);
- object-space telecentric aiming for non-``+z`` entry;
- ``paraxial_image_height`` and ``real_image_height`` field types on
  folded/off-axis paths: their defining coordinate lives on the image-side
  leg, which a fold moves off the global z axis;
- ``object_height`` field types on systems entered off global ``+z``: the
  heights are global (x, y) coordinates on the object surface and match the
  entry frame only for ``+z`` entry. A ``+z``-entered system folded
  downstream of the object keeps ``object_height`` support (subject to the
  usual scalar-domain diagnostics for any first-order quantity involved);
- ambiguous two-dimensional angle fields at or beyond 90 degrees total
  (``AMBIGUOUS_WIDE_ANGLE_FIELD``): the component-angle representation does
  not uniquely define a 3-D direction there. One-dimensional wide-angle
  fields keep working;
- component field angles at an odd multiple of 90 degrees
  (``SINGULAR_ANGLE_TANGENT``): floating-point ``tan`` returns a huge
  finite number at the pole instead of failing, so every tangent
  evaluation on component angles (ray origins, paraxial object positions,
  chief-ray scaling, wavefront tilt correction) rejects angles within a
  precision-derived width (:math:`90\sqrt{\varepsilon}` degrees: ~1.3e-6
  in float64, ~0.031 in float32) of the pole. Nonsingular wide angles
  such as 89, 91, 95 or 105 degrees remain supported.

z-bound geometry mutations (rejected before any state changes)
---------------------------------------------------------------

Every operation that interprets an unfolded axial scalar as a writable
global ``cs.z`` rejects folded/off-axis prescriptions **preflight-atomically**
-- the guard fires before the first read-modify or write, so a rejected
system is left exactly as it was:

- ``set_thickness`` and thickness updates (including through pickups and
  solves);
- ``ThicknessSolve`` / marginal- and chief-ray height thickness solves;
- ``image_solve`` and ``QuickFocusSolve.apply()``;
- ``QuickFocusSolve.optimal_focus_distance()`` directly -- the value it
  returns is a Cartesian global-z focus coordinate, which is not
  frame-correct for a folded output arm (guarded before any ray is
  traced);
- ``ThroughFocusAnalysis`` (moves the image plane along z);
- ``OpticUpdater.scale_system()`` (guarded at entry, before reading
  thicknesses or scaling the first geometry);
- ``SurfaceGroup.flip()`` and ``OpticUpdater.flip()`` (guarded before list
  reversal, material swaps, coordinate or thickness changes);
- surface-group composition (``SurfaceGroup.__add__``): both operands must
  be global-z compatible before anything is copied or shifted; laterally
  translated straight systems remain supported;
- relative-coordinate reconstruction
  (``SurfaceGroup._update_coordinate_systems``): insertions and removals
  preflight the would-be chain before mutating the surface list, and the
  method itself is gated;
- multi-configuration thickness updates preflight **every selected
  configuration** before mutating any, so a folded configuration later in
  the group cannot leave earlier configurations changed.

A future implementation may make these path-aware (translating downstream
geometry along the actual beam legs); until then they reject rather than
corrupt.

Huygens PSF normalization
-------------------------

The scalar and vectorial Huygens PSF normalize against the full 3-D image
surface vertex through one shared, gradient-safe construction
(:func:`optiland.psf.huygens_fresnel.image_vertex_grid`): the vertex is
broadcast onto the image grid instead of passed through ``be.full``,
whose torch implementation extracts a Python scalar and would silently
sever trainable image-vertex coordinates from the autograd graph.

Remaining limitations
---------------------

- Genuinely tilted or decentered powered surfaces on *straight* (global-z)
  systems keep the historical behavior: first-order results ignore the
  tilt/decenter and a :class:`ParaxialDomainWarning` advisory is emitted.
  Gating those would break long-standing legacy systems; the hard
  validation currently applies to folded/off-axis paths only. (Centered
  surfaces authored with a reversed local axis are *not* part of this
  limitation -- they are normalized consistently.)
- The wavefront/OPD and PSF stacks are validated in the supported scalar
  folded domain; general folded wavefront analysis beyond it is untested.
- The wavefront reference-sphere radius is float-typed by long-standing
  design, so the *total* derivative of a Huygens normalization with
  respect to geometry includes that term only through the forward value;
  the image-vertex path itself is fully differentiable.
- A general 4x4 first-order coordinate-break model (tangential/sagittal
  treatment of oblique powered mirrors, anamorphic pupils, transverse
  coupling) is future work and out of scope here.
