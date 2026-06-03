Optimization Guide
==================

Optiland provides a unified optimization framework for optical system design.
This section covers the architecture, usage patterns, and extension points.

**The single rule:** does Optiland run the loop, or do you?

* **Optiland runs it** → :func:`optiland.optimization.minimize`
* **You run it** (custom loss, batched data, optic as a differentiable layer) → :class:`optiland.ml.OpticalSystemModule`

.. toctree::
   :maxdepth: 2

   method_selection
   framework
   operands_variables
   migration

Persona Routing
---------------

+-------------------------+-----------------------------------------------+----------------------------------------+
| Persona                 | Need                                          | Entry point                            |
+=========================+===============================================+========================================+
| Optical engineer        | DLS on doublet/triplet; bounds; reproducible  | ``minimize(problem, "dls")``           |
+-------------------------+-----------------------------------------------+----------------------------------------+
| Computational physicist | Custom stopping, inspect per-iter state       | ``minimize(…, stop=, observers=)``     |
+-------------------------+-----------------------------------------------+----------------------------------------+
| Data scientist          | Optic as differentiable layer; batched GPU    | ``OpticalSystemModule``                |
+-------------------------+-----------------------------------------------+----------------------------------------+
| Educator                | Watch a solver iterate; compare methods       | ``minimize(…, disp=True, observers=…)``|
+-------------------------+-----------------------------------------------+----------------------------------------+

See Also
--------

* :doc:`/api/api_optimization` — full API reference
* :doc:`/learning_guide` — step-by-step tutorials (§3)
