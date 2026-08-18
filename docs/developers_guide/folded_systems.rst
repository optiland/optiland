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
verified against real traces).

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
beam direction. Classic z-authored systems always have :math:`s_k = +1`, so
their behavior is unchanged. Authored radii and focal lengths are never
mutated; serialization is unaffected.

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
- within ``1e-10`` of a ``+/-y`` entry the projection degenerates (the
  unavoidable pole of any deterministic gauge) and global ``+x`` is
  projected instead;
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
  (``entrance_pupil_z()`` is a legacy alias of it and, despite the name, is
  **not** a Cartesian z).

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
iteration count, and convergence status
(:class:`~optiland.rays.ray_aiming.parameterization.SolveReport`); a
returned finite ray is not evidence of convergence. Pupil-map warm-start
caches store local transverse offsets, so rigid pose changes cannot reuse
an incompatible global-coordinate map.

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
- ``object_height``, ``paraxial_image_height`` and ``real_image_height``
  field types on folded/off-axis paths (their coordinate semantics are
  still z-bound);
- thickness/image/quick-focus solves and thickness updates on folded paths
  (they write axial offsets into global ``cs.z``; the guard fires *before*
  any mutation);
- ambiguous two-dimensional angle fields at or beyond 90 degrees total
  (``AMBIGUOUS_WIDE_ANGLE_FIELD``): the component-angle representation does
  not uniquely define a 3-D direction there. One-dimensional wide-angle
  fields keep working.

Remaining limitations
---------------------

- Tilted or decentered powered surfaces on *straight* (global-z) systems
  keep the historical behavior: first-order results silently ignore the
  tilt/decenter. Gating those would break long-standing legacy systems;
  the validation currently applies to folded/off-axis paths only.
- The wavefront/OPD and PSF stacks are validated in the supported scalar
  folded domain; general folded wavefront analysis beyond it is untested.
- A general 4x4 first-order coordinate-break model (tangential/sagittal
  treatment of oblique powered mirrors, anamorphic pupils, transverse
  coupling) is future work and out of scope here.
