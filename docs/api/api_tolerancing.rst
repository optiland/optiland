Tolerancing
===========

This section covers the tolerancing capabilities of Optiland. Tolerancing is used to determine
acceptable limits of variation in the manufacturing of optical components and systems.
These variations can arise due to imperfections in the manufacturing process, such as slight 
deviations in the shape or alignment of optical elements. Tolerancing ensures that despite these 
imperfections, the performance of an optical system will still fall within an acceptable range.

In Optiland, tolerancing is performed via sensitivity analysis or Monte Carlo simulations.

For tolerancing of multilayer thin film coatings specifically, see the :doc:`api_thin_film` module.

Tolerancing in Optiland requires 4 key components:

1. Optic - the optical system to be analyzed.
2. Operands - the metrics which are assessed e.g., wavefront error.
3. Perturbations - the variations applied to the optic or a surface of an optic e.g., surface tilt.
4. Compensators - an *optional* parameter of the optical system that can be adjusted to counteract the effects of a perturbation.

.. autosummary::
   :toctree: tolerancing/
   :caption: Tolerancing Modules

   tolerancing.compensator
   tolerancing.core
   tolerancing.monte_carlo
   tolerancing.perturbation
   tolerancing.sensitivity_analysis
