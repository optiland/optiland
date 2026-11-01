.. _optimization_merit_constraints:

.. meta::
   :description: Build a classical spot-size or wavefront merit function with one call, and keep a design buildable with hard constraints and physical thickness limits.

Default Merit and Constraints
=============================

Two additions make a typical lens-design setup short: a **default merit
function** that replaces a hand-assembled list of RMS operands, and **hard
constraints** that keep the design buildable. Both are experimental and both
are opt-in: Optiland never adds them for you. For choosing an optimizer and
running it, see the :doc:`optimization_guide`. A worked example is
:doc:`Tutorial 3f <examples/Tutorial_3f_Constrained_Default_Merit>`.

.. _default_merit:

The default merit
-----------------

``problem.add_default_merit(optic)`` adds a *residual block*: one object that
returns the per-ray errors of a whole set of rays in a single vectorized call,
instead of one ``Operand`` per number.

.. code-block:: python

   block = problem.add_default_merit(
       lens,
       kind="spot",            # or "wavefront"
       rings=3, arms=6,        # pupil quadrature: 18 rays per field and wavelength
       reference="centroid",   # or "chief"
       fields="all",           # or [0, 2]
       wavelengths="all",
       weight=1.0,
   )

Each ray contributes a residual

.. math::

   r = \sqrt{q \, w_\mathrm{field} \, w_\lambda}\; \cdot\; \mathrm{weight}\; \cdot\; e

where :math:`e` is the transverse error (``dx`` and ``dy`` on the image
surface for ``kind="spot"``) or the OPD in waves (``kind="wavefront"``), and
:math:`q` is the ray's quadrature weight. Because the weights sum to one,
``sum(r ** 2)`` is an estimate of the *weighted RMS squared*, summed over
fields and wavelengths. The field and wavelength weights of the optic are used
as they are at each evaluation, so you can change them between runs.

Choices that matter:

* ``reference="centroid"`` measures from the weighted centroid of each spot;
  ``"chief"`` from the chief ray. A spot is always measured from the
  reference of the **primary** wavelength, so lateral color stays in the merit.
  Wavefront errors are measured at their own wavelength, like
  :class:`~optiland.wavefront.opd.OPD`.
* ``rings`` and ``arms`` set the accuracy/cost trade-off (next section).
  Three rings and six arms are usually enough for a lens with moderate
  aberrations during optimization; raise them to confirm the final design.
* The rays are fixed when the block is built. After large changes to the
  aperture or fields, build a new block.

``problem.info()``, ``problem.merit_info()`` and ``problem.block_info()`` show
a block as one summary row (kind, rays, rows, merit and share of the total).
The block's rows follow the operands in ``residual_vector()``, ``fun_array()``
and ``sum_squared()``, so every optimizer sees them, including SciPy and torch.
Mixing is normal: keep an ``f2`` operand for the focal length and let the block
handle image quality.

.. code-block:: python

   problem.add_operand("f2", target=50.0, weight=10.0, input_data={"optic": lens})
   problem.add_default_merit(lens, kind="spot")

``problem.clear_residual_blocks()`` removes all blocks.

Writing your own block
~~~~~~~~~~~~~~~~~~~~~~

Subclass :class:`~optiland.optimization.ResidualBlock` for any per-ray or
per-sample merit term: implement ``size()``, ``residuals(context)`` (already
weighted), ``trace_requests()`` (the rays you need, so they are traced together
with those of other blocks and operands) and ``describe()`` (the summary row),
and register it with ``problem.add_residual_block(block)``.

Gaussian quadrature pupil sampling
----------------------------------

Averaging over the pupil is a numerical integral. A hexapolar grid weights every
ray equally and needs many rays; **Gaussian quadrature** picks few rays and
gives each a weight, so the same integral is accurate with far fewer traces.

Optiland's ``"gaussian_quad"`` distribution (after Forbes, 1988) places
Gauss-Legendre rings in :math:`\rho^2`, equally spaced arms around each ring,
and returns per-sample weights that sum to one:

.. code-block:: python

   from optiland.distribution import create_distribution

   pupil = create_distribution("gaussian_quad")
   pupil.generate_points(num_rings=3, num_spokes=6)
   pupil.x, pupil.y, pupil.weights

With ``num_spokes=None`` it uses ``4 * (num_rings + 1)`` arms, for which the
integral is exact for polynomials up to degree ``num_rings`` in :math:`x` and
:math:`y`. A smooth wavefront error is well described by low-order polynomials
in the pupil, which is why a handful of rays is enough. A merit with sharp
features (vignetting, strong high-order aberration) needs more rings. A good
check: evaluate your final design with ``rings=3`` and ``rings=6``; if the
merit changes by more than a few percent, increase rings during optimization.

``GaussianQuadrature(half_pupil=True)`` keeps only the :math:`x \geq 0` half
pupil and doubles the weights of the off-axis arms. Use it only when the optic
and field are symmetric about the :math:`y`-:math:`z` plane.

Hard constraints
----------------

A penalty operand (``min_val`` / ``max_val`` on an operand) is part of the
merit: it is *traded off* against image quality and can be violated at the
optimum. A **hard constraint** is a condition the optimizer must satisfy; it is
not in the merit at all.

.. code-block:: python

   problem.add_constraint("f2", target=50.0, input_data={"optic": lens})      # equality
   problem.add_constraint("center_thickness", min_val=2.0, max_val=12.0,
                          input_data={"optic": lens, "surface_number": 2})    # inequality

``target`` makes an equality; ``min_val`` and/or ``max_val`` make an
inequality (``target`` and bounds cannot be combined). ``weight`` scales the
constraint rows, and ``tol`` (default ``1e-8``) is the feasibility tolerance
in row units. Inspect them with ``problem.constraint_info()``; remove them with
``problem.clear_constraints()``. Constraints are evaluated in the same pass
as the merit, so they add no extra ray traces.

**Only** ``minimize(problem, "dls")`` (and
``LeastSquares(...).optimize(method_choice="dls")``) enforces constraints. Any
other optimizer raises ``ConfigurationError`` with that hint rather than
silently ignoring them. The result of a ``"dls"`` run reports:

* ``result.max_constraint_violation``: the largest violation (0 when feasible),
* ``result.active_set``: labels of the constraints that are binding at the
  solution,
* ``result.multipliers``: a dict of Lagrange multipliers by label. A large
  multiplier means the constraint is expensive: relaxing it would improve the
  merit,
* ``result.constraint_report``: a list with one dict per constraint row
  (``label``, ``kind``, ``residual``, ``violation``, ``feasible``).

If the starting design violates a constraint, the optimizer first restores
feasibility with damped steps. If the constraints cannot all be satisfied, the
run ends with ``success=False`` and a report instead of a half-feasible design
presented as a success. Variable bounds (``min_val`` / ``max_val`` on
``add_variable``) are enforced as inequality constraints as well.

.. _physical_constraints:

Physical thickness limits
-------------------------

A freely optimized lens will happily shrink an element to zero thickness or
let an edge go negative. ``add_physical_constraints`` adds sensible
manufacturing limits on the gaps between lens surfaces in one call:

.. code-block:: python

   problem.add_physical_constraints(
       lens,
       min_center_glass=1.5,   # mm, center thickness of every glass element
       min_edge_glass=1.0,     # mm, edge thickness of every glass element
       min_center_air=0.5,     # mm, air gaps (center)
       min_edge_air=0.5,       # mm, air gaps (edge)
       max_center=15.0,        # mm, largest center thickness, glass and air
   )

A limit left as ``None`` is not applied. There are also ``max_center_glass``,
``max_center_air``, ``max_edge_glass`` and ``max_edge_air`` (keyword-only) to
treat glass and air differently. The gap from the object and the gap to the
image are never limited. Edge thickness is measured at the semi-apertures of the two
surfaces, which Optiland computes from the paraxial trace when they are not
set; check them with ``lens.info()``.

``mode="constraint"`` (default) emits hard constraints for the ``"dls"``
optimizer. ``mode="penalty"`` emits ordinary operands with ``min_val`` /
``max_val`` instead, which any optimizer accepts but which may be violated
slightly at the optimum:

.. code-block:: python

   problem.add_physical_constraints(lens, min_center_glass=1.5, mode="penalty", weight=10.0)
   LeastSquares(problem).optimize()          # works with the default "lm"

The method returns the new ``Constraint`` (or ``Operand``) objects, so you can
inspect or remove them individually. It never runs automatically: a design
problem without it is optimized exactly as before.

A recipe
--------

.. code-block:: python

   from optiland.optimization import OptimizationProblem, minimize

   problem = OptimizationProblem()
   problem.add_default_merit(lens, kind="spot", rings=3, arms=6)
   problem.add_operand("f2", target=50.0, weight=10.0, input_data={"optic": lens})

   for k in range(1, n_surfaces - 1):
       problem.add_variable(lens, "radius", surface_number=k)
   for k in range(1, n_surfaces - 2):
       problem.add_variable(lens, "thickness", surface_number=k, min_val=0.5)

   problem.add_physical_constraints(
       lens, min_center_glass=1.5, min_edge_glass=1.0, min_center_air=0.5, max_center=15.0
   )
   result = minimize(problem, "dls")
   problem.merit_info()
   problem.constraint_info()
