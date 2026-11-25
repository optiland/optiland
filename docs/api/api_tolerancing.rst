Tolerancing
===========

This section covers the tolerancing capabilities of Optiland. Tolerancing is used to determine
acceptable limits of variation in the manufacturing of optical components and systems.
These variations can arise due to imperfections in the manufacturing process, such as slight 
deviations in the shape or alignment of optical elements. Tolerancing ensures that despite these 
imperfections, the performance of an optical system will still fall within an acceptable range.

In Optiland, tolerancing is performed via sensitivity analysis, inverse sensitivity analysis or
Monte Carlo simulations.

For a guide to the whole workflow, see :doc:`../tolerancing_guide`. For tolerancing of
multilayer thin film coatings specifically, see the :doc:`api_thin_film` module.

Tolerancing in Optiland requires 4 key components:

1. Optic - the optical system to be analyzed.
2. Operands - the metrics which are assessed e.g., wavefront error.
3. Tolerances or perturbations - the variations applied to the optic or a surface of an optic e.g., surface tilt.
   Tolerances (``Tolerancing.add_tolerance``) are signed deviations from nominal within limits, e.g., ±0.05 mm.
   Perturbations (``Tolerancing.add_perturbation``) set absolute values of an optimization variable.
4. Compensators - an *optional* parameter of the optical system that can be adjusted to counteract the effects of a perturbation.

Core and Analyses
-----------------

.. autosummary::
   :toctree: tolerancing/
   :caption: Tolerancing Modules

   tolerancing.base
   tolerancing.compensator
   tolerancing.core
   tolerancing.distributions
   tolerancing.inverse_sensitivity
   tolerancing.iso10110_bridge
   tolerancing.monte_carlo
   tolerancing.perturbation
   tolerancing.presets
   tolerancing.registry
   tolerancing.sensitivity_analysis
   tolerancing.tolerance

Tolerance Types
---------------

The built-in tolerance types, registered in ``tolerancing.registry.tolerance_registry``
under the names used by ``Tolerancing.add_tolerance``.

.. autosummary::
   :toctree: tolerancing/
   :caption: Tolerance Type Modules

   tolerancing.types.element
   tolerancing.types.irregularity
   tolerancing.types.material
   tolerancing.types.surface
