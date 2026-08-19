Tolerancing Framework
=====================

The tolerancing framework in Optiland is designed to evaluate the **sensitivity** and **robustness** of optical systems by analyzing the
effects of small perturbations in system parameters. It provides a modular and extensible approach, largely reusing components from the
optimization framework to maintain consistency and simplicity.

Core Concepts
-------------

The tolerancing framework is built around the following key components:

- **Variables**: System parameters subject to perturbations, such as surface curvatures, thicknesses, or material indices. These are tied to the same variable definitions used in the optimization framework.

- **Operands**: Metrics used to quantify the system’s performance or deviations due to perturbations. These are also reused from the optimization framework, enabling compatibility with a wide range of analysis metrics.

- **Compensators**: Parameters that can be adjusted to minimize the impact of perturbations. Compensators operate by performing an optimization to restore system performance within acceptable limits.

- **Perturbations**: Defined changes to system parameters (e.g., a shift in lens position or a change in curvature) that simulate real-world manufacturing tolerances or environmental variations.

- **Samplers**: Distributions, from which random perturbations are drawn during sensitivity analyses or Monte Carlo simulations. These can be customized to model specific manufacturing processes or environmental conditions.

Core Classes
------------

The tolerancing framework centers around the following classes:

- **Tolerancing**: This is the core class that orchestrates all tolerancing operations. It manages variables, operands, compensators, and perturbations, providing a unified interface for sensitivity and robustness analyses.

- **SensitivityAnalysis**: This class evaluates the effect of individual perturbations on system performance. It computes the changes in operand values for each perturbation independently to identify critical sensitivities in the design.

- **MonteCarlo**: This class performs stochastic simulations by applying random perturbations to system variables. The results provide insights into the statistical robustness of the design under realistic tolerances.

Workflow
--------

1. **Create a Tolerancing Instance**: Instantiate an empty `Tolerancing` object.
2. **Define Perturbations**: Specify perturbations, together with an appropriate sampling distribution, and add them to the `Tolerancing` instance.
3. **Add Operands and (optionally) Compensators**: Identify which metrics will be evaluated and specify any compensators (e.g., adjusting lens positions or tilts to counteract perturbations).
4. **Run Analysis**: Pass the `Tolerancing` class instance to either the `SensitivityAnalysis` or `MonteCarlo` class, depending on the desired type of study, and run the analysis.
5. **Interpret Results**: Use the output to identify sensitive parameters or assess the statistical robustness of the design. Optionally visualize the output using built-in plotting functions, or export the results as a `pandas.DataFrame` for further analysis.

.. tip::
   See the :ref:`learning_guide` for specific demonstrations of both sensitivity and Monte Carlo analyses using the tolerancing framework.

How to Extend This
------------------

**Scenario:** Add a custom perturbation sampling strategy to Optiland.

The ``Perturbation`` class in ``optiland/tolerancing/perturbation.py`` is not subclassed per
perturbation type — instead it wraps an existing optimization ``Variable`` (identified by a
``variable_type`` string, e.g. ``"radius"`` or ``"thickness"``) together with a sampler that
supplies new values. Its ``apply()`` method draws a value from the sampler and calls
``self.variable.update(value)``; ``reset()`` restores the variable's original value.

**Step 1:** To add a new *sampling* strategy, subclass ``BaseSampler`` in
``optiland/tolerancing/perturbation.py`` and implement ``sample()``.
**Step 2:** To perturb a new *kind of parameter*, add or reuse a variable type in the
optimization framework's ``Variable``/``VariableBehavior`` classes (see
:doc:`optimization_framework`) — ``Perturbation`` will pick it up automatically via
``variable_type``.
**Step 3:** Register perturbations on a ``Tolerancing`` instance with
``Tolerancing.add_perturbation(variable_type, sampler, **kwargs)``.
**Step 4:** Add tests in ``tests/test_tolerancing/``.

For step-by-step guidance, see :ref:`extension_recipes`.
