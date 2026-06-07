Operands and Variables Reference
=================================

This page provides hand-authored reference tables for all operand types and
variable types in the optimization framework, complementing the API reference.

For the full API, see :doc:`/api/api_optimization`.

Operand Types
-------------

Operands define what to measure; they are added to an
:class:`~optiland.optimization.problem.OptimizationProblem` via
:meth:`~optiland.optimization.problem.OptimizationProblem.add_operand`.

The ``input_data`` dict always includes ``"optic"`` (the :class:`~optiland.optic.Optic`
instance). Additional keys are listed per operand below.

Weights and merit function contribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``weight`` argument to :meth:`~optiland.optimization.problem.OptimizationProblem.add_operand`
scales the operand's contribution to the merit function relative to all other
operands.  For operands evaluated across the pupil or across a spectrum (e.g.,
``rms_spot_size``), Optiland also automatically accounts for the intrinsic
``weight`` assigned to the :class:`~optiland.fields.Field` and
:class:`~optiland.wavelengths.Wavelength` objects defined in the ``Optic``.

Paraxial Operands
~~~~~~~~~~~~~~~~~

These operands compute paraxial (first-order) system properties and require
only ``{"optic": lens}`` in ``input_data``.

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Operand key
     - What it measures
     - Additional ``input_data``
   * - ``f1``
     - First (front) focal length
     - *(none)*
   * - ``f2``
     - Second (rear) focal length / EFL
     - *(none)*
   * - ``F1``
     - Front focal distance (vertex to front focal point)
     - *(none)*
   * - ``F2``
     - Rear focal distance (vertex to rear focal point)
     - *(none)*
   * - ``P1``
     - Front principal plane position
     - *(none)*
   * - ``P2``
     - Rear principal plane position
     - *(none)*
   * - ``N1``
     - Front nodal point position
     - *(none)*
   * - ``N2``
     - Rear nodal point position
     - *(none)*
   * - ``EPD``
     - Entrance pupil diameter
     - *(none)*
   * - ``EPL``
     - Entrance pupil position (from first surface)
     - *(none)*
   * - ``XPD``
     - Exit pupil diameter
     - *(none)*
   * - ``XPL``
     - Exit pupil position (from last surface)
     - *(none)*
   * - ``magnification``
     - Lateral magnification
     - *(none)*
   * - ``total_track``
     - Total track length (first surface to image plane)
     - *(none)*

Aberration Operands
~~~~~~~~~~~~~~~~~~~

Seidel third-order aberration contributions.  Per-surface operands require
``surface_number``; the ``_sum`` variants sum over all surfaces.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Operand key
     - What it measures
     - Additional ``input_data``
   * - ``seidel``
     - Full Seidel aberration vector
     - ``{"seidel_number": int}``
   * - ``TSC``
     - Transverse spherical aberration (per surface)
     - ``{"surface_number": int}``
   * - ``SC``
     - Longitudinal spherical aberration (per surface)
     - ``{"surface_number": int}``
   * - ``CC``
     - Sagittal coma (per surface)
     - ``{"surface_number": int}``
   * - ``TCC``
     - Tangential coma (per surface)
     - ``{"surface_number": int}``
   * - ``TAC``
     - Transverse astigmatism (per surface)
     - ``{"surface_number": int}``
   * - ``AC``
     - Longitudinal astigmatism (per surface)
     - ``{"surface_number": int}``
   * - ``TPC``
     - Transverse Petzval sum (per surface)
     - ``{"surface_number": int}``
   * - ``PC``
     - Longitudinal Petzval sum (per surface)
     - ``{"surface_number": int}``
   * - ``DC``
     - Distortion (per surface)
     - ``{"surface_number": int}``
   * - ``TAchC``
     - Transverse axial chromatic aberration (per surface)
     - ``{"surface_number": int}``
   * - ``LchC``
     - Longitudinal chromatic aberration (per surface)
     - ``{"surface_number": int}``
   * - ``TchC``
     - Transverse chromatic aberration (per surface)
     - ``{"surface_number": int}``
   * - ``TSC_sum``
     - Sum of TSC over all surfaces
     - *(none)*
   * - ``SC_sum``
     - Sum of SC over all surfaces
     - *(none)*
   * - ``CC_sum``
     - Sum of CC over all surfaces
     - *(none)*
   * - ``TCC_sum``
     - Sum of TCC over all surfaces
     - *(none)*
   * - ``TAC_sum``
     - Sum of TAC over all surfaces
     - *(none)*
   * - ``AC_sum``
     - Sum of AC over all surfaces
     - *(none)*
   * - ``TPC_sum``
     - Sum of TPC over all surfaces
     - *(none)*
   * - ``PC_sum``
     - Sum of PC over all surfaces
     - *(none)*
   * - ``DC_sum``
     - Sum of DC over all surfaces
     - *(none)*
   * - ``TAchC_sum``
     - Sum of TAchC over all surfaces
     - *(none)*
   * - ``LchC_sum``
     - Sum of LchC over all surfaces
     - *(none)*
   * - ``TchC_sum``
     - Sum of TchC over all surfaces
     - *(none)*

Real Ray Operands
~~~~~~~~~~~~~~~~~

These operands trace one or more real rays and extract properties.

.. list-table::
   :header-rows: 1
   :widths: 22 43 35

   * - Operand key
     - What it measures
     - Additional ``input_data``
   * - ``real_x_intercept``
     - X coordinate where a single ray intersects a surface (global frame)
     - ``{"surface_number": int, "Hx": float, "Hy": float, "Px": float, "Py": float, "wavelength": float}``
   * - ``real_y_intercept``
     - Y coordinate of single-ray surface intercept (global frame)
     - *(same as above)*
   * - ``real_z_intercept``
     - Z coordinate of single-ray surface intercept (global frame)
     - *(same as above)*
   * - ``real_x_intercept_lcs``
     - X intercept in local coordinate system of the surface
     - *(same as above)*
   * - ``real_y_intercept_lcs``
     - Y intercept in local coordinate system of the surface
     - *(same as above)*
   * - ``real_z_intercept_lcs``
     - Z intercept in local coordinate system of the surface
     - *(same as above)*
   * - ``real_L``
     - Direction cosine L of a traced ray at a surface
     - *(same as above)*
   * - ``real_M``
     - Direction cosine M of a traced ray at a surface
     - *(same as above)*
   * - ``real_N``
     - Direction cosine N of a traced ray at a surface
     - *(same as above)*
   * - ``clearance``
     - Signed clearance margin between ray footprint and aperture edge
     - ``{"surface_number": int, "Hx": float, "Hy": float, "Px": float, "Py": float, "wavelength": float}``
   * - ``AOI``
     - Angle of incidence at a surface (radians)
     - *(same as clearance)*
   * - ``rms_spot_size``
     - RMS spot radius at a surface across a pupil distribution
     - ``{"surface_number": int, "Hx": float, "Hy": float, "wavelength": float, "num_rays": int}``
   * - ``OPD_difference``
     - RMS wavefront error (OPD) for a field / wavelength
     - ``{"Hx": float, "Hy": float, "wavelength": float, "num_rays": int}``

Lens / Mechanical Operands
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 43 35

   * - Operand key
     - What it measures
     - Additional ``input_data``
   * - ``edge_thickness``
     - Edge (peripheral) thickness of a lens element
     - ``{"surface_number": int, "wavelength": float}``

Registering Custom Operands
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a new function to ``optiland/optimization/operand/operand.py`` and register
it in the ``METRIC_DICT`` at the top of that file, or use the
:class:`~optiland.optimization.operand.operand.OperandRegistry` API at runtime:

.. code-block:: python

   from optiland.optimization.operand.operand import OperandRegistry

   def my_operand(optic, surface_number):
       return optic.surfaces.radii[surface_number] ** 2

   OperandRegistry.register("my_operand", my_operand)

   problem.add_operand(
       operand_type="my_operand",
       target=0.0,
       weight=1.0,
       input_data={"optic": lens, "surface_number": 3},
   )


Variable Types
--------------

Variables define what to optimize; they are added via
:meth:`~optiland.optimization.problem.OptimizationProblem.add_variable`.

All variable types accept ``surface_number`` (int), optional ``min_val`` /
``max_val`` bounds, and an optional ``scaler`` (default: identity).

.. list-table::
   :header-rows: 1
   :widths: 25 35 22 18

   * - Variable type string
     - What it controls
     - Additional required kwargs
     - Notes
   * - ``"radius"``
     - Surface radius of curvature
     - *(none)*
     - Singularity near flat surfaces; prefer ``"reciprocal_radius"``
   * - ``"reciprocal_radius"``
     - Curvature (1/R) of a surface
     - *(none)*
     - More numerically stable for near-flat surfaces
   * - ``"thickness"``
     - Air gap or element thickness after surface
     - *(none)*
     - Controls axial separation to the next surface
   * - ``"conic"``
     - Conic constant K of a surface
     - *(none)*
     - Only meaningful on conic or aspheric geometries
   * - ``"index"``
     - Refractive index of the material after a surface
     - ``wavelength`` (float, µm)
     - Bypasses dispersion model; use with care
   * - ``"material"``
     - Glass catalog selection for a surface
     - ``glass_selection`` (list of glass name strings)
     - Used with :class:`~optiland.optimization.GlassExpert`; categorical
   * - ``"decenter"``
     - X or Y decentration of a surface
     - ``axis`` (``"x"`` or ``"y"``)
     - Requires a coordinate break surface
   * - ``"tilt"``
     - Tilt angle of a surface about X or Y axis
     - ``axis`` (``"x"`` or ``"y"``)
     - Angle in degrees; requires coordinate break
   * - ``"asphere_coeff"``
     - Even-asphere polynomial coefficient A_i
     - ``coeff_number`` (int, 0-based)
     - Requires aspheric surface geometry
   * - ``"polynomial_coeff"``
     - XY polynomial surface coefficient
     - ``coeff_number`` (int, 0-based)
     - For polynomial freeform surfaces
   * - ``"chebyshev_coeff"``
     - Chebyshev polynomial surface coefficient
     - ``coeff_number`` (int, 0-based)
     - For Chebyshev freeform surfaces
   * - ``"zernike_coeff"``
     - Zernike surface coefficient
     - ``coeff_number`` (int, 0-based)
     - For Zernike freeform surfaces
   * - ``"forbes_coeff"``
     - Forbes Q2D aspheric coefficient
     - ``coeff_number`` (int, 0-based)
     - For Forbes aspheric surfaces
   * - ``"forbes_normal_slope_coeff"``
     - Forbes QNormal slope coefficient
     - ``coeff_number`` (int, 0-based)
     - For Forbes normal-slope surfaces
   * - ``"norm_radius"``
     - Normalization radius of a surface geometry
     - *(none)*
     - Relevant for freeform surfaces with normalization
   * - ``"nurbs_points"``
     - NURBS control point positions
     - *(none)*
     - For NURBS surface geometries
   * - ``"nurbs_weights"``
     - NURBS control point weights
     - *(none)*
     - For NURBS surface geometries
   * - ``"grid_sag"``
     - Grid sag surface height values
     - *(none)*
     - For grid sag surface geometries

Adding Bounds
~~~~~~~~~~~~~

Pass ``min_val`` and/or ``max_val`` to clamp the variable search range:

.. code-block:: python

   problem.add_variable(lens, "radius", surface_number=2, min_val=10.0, max_val=200.0)

   # With a curvature scaler for near-flat surfaces
   from optiland.optimization.scaling.reciprocal import ReciprocalScaler
   problem.add_variable(lens, "reciprocal_radius", surface_number=4,
                        scaler=ReciprocalScaler())

Creating Custom Variable Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass :class:`~optiland.optimization.variable.base.VariableBehavior` and
implement ``get_value()``, ``update_value()``, and ``__str__()``:

.. code-block:: python

   from optiland.optimization.variable.base import VariableBehavior

   class MyVariable(VariableBehavior):
       """Controls the semi-diameter of a surface."""

       def get_value(self):
           return self._surfaces[self.surface_number].semi_diameter

       def update_value(self, new_value):
           self._surfaces[self.surface_number].semi_diameter = new_value

       def __str__(self):
           return f"SemiDiameter, Surface {self.surface_number}"

Register the new type in the ``Variable`` class dispatch table in
``optiland/optimization/variable/variable.py`` so it can be referenced by
string in :meth:`~optiland.optimization.problem.OptimizationProblem.add_variable`.

Operands vs. Constraints
------------------------

Optiland provides two mechanisms for specifying targets:

* ``add_operand`` -- **soft target**. The operand enters the merit function as a
  weighted squared residual. The optimizer balances all operands simultaneously;
  no individual target is guaranteed to be met exactly. Use this for preferences,
  tolerances, and weighted trade-offs.
* ``add_constraint`` -- **hard constraint**. The constraint is enforced at every
  step via the KKT active-set method. The optimizer cannot trade it off against
  the merit. Use this for absolute requirements: exact focal length, minimum
  edge thickness, maximum chief-ray angle, etc.

Both take the same spec shape:

.. code-block:: python

   # Soft: EFL enters the merit function with weight 1.0
   problem.add_operand(
       operand_type="f2",
       target=50.0,
       weight=1.0,
       input_data={"optic": lens},
   )

   # Hard: EFL must equal 50 mm (KKT-enforced)
   problem.add_constraint(
       operand_type="f2",
       target=50.0,           # equality constraint
       input_data={"optic": lens},
   )

   # Hard inequality: total track <= 80 mm
   problem.add_constraint(
       operand_type="total_track",
       max_val=80.0,
       input_data={"optic": lens},
   )

Hard constraints require ``method="dls"`` or ``method="lm"``; ``method="auto"``
selects ``"dls"`` automatically when constraints are present. See
:ref:`hard_constraints` in the framework guide for the full workflow and diagnostics.

See Also
--------

* :doc:`framework` -- extension recipes and architecture details, including :ref:`hard_constraints`
* :doc:`/api/api_optimization` -- full API reference
