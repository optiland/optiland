optiland.wavefront.wavefront\_data
==================================

.. automodule:: optiland.wavefront.wavefront_data

   
   .. rubric:: Classes

   .. autosummary::
   
      WavefrontData

Generated data have no quadrature metadata by default. Opt in to a source-order
snapshot when constructing the analysis::

   analysis = OPD(
       optic, field, wavelength, distribution=distribution,
       assume_sample_order=True,
   )
   data = analysis.get_data(field, wavelength)

``assume_sample_order=True`` asserts that the chosen trace route preserves
one-to-one association with distribution samples. It copies available distribution
weights before tracing and validates aligned one-dimensional shapes before and
after tracing. Shape checks cannot detect same-length permutations. The option
does not audit, certify, or inspect tracing implementations; callers using custom
tracers must establish the association themselves. Focused association tests cover
the built-in unpolarized sequential routes with paraxial, iterative, and robust
aiming. Without the opt-in, or if the distribution has no quadrature rule,
``quadrature_weights`` is ``None``.

Gaussian-quadrature values describe normalized area in the distribution's
sampling disk. Interpretation as physical pupil or stop area depends on the
producer and aiming map; no mapping Jacobian or exit-pupil conversion is added.
The snapshot is not multiplied by intensity or modified for clipping.
Source weights must use the active backend; masked NumPy arrays are rejected
before copying.

Cached evaluation
-----------------

``WavefrontData.evaluate`` evaluates the stored, already referenced OPD without
tracing again. The removal convention is always explicit::

   equal_sample = data.evaluate(remove="piston")
   quadrature = data.evaluate(remove="piston", use_quadrature=True)

The first call gives every sample equal weight. The second uses the stored
quadrature snapshot and raises if no snapshot is available. Supplying both
``weights`` and ``use_quadrature=True`` is an error.

For ``remove="piston_tilt"``, the affine basis uses the stored ``pupil_x`` and
``pupil_y`` coordinates. Quadrature metadata continues to describe the distribution's
sampling measure; opting in does not change the affine basis to source coordinates.

Explicit ``weights`` are final effective weights. Where ``data.intensity`` has
been audited as a compatible local factor in the same quadrature measure, compose
the intended measure at the call site, exactly once::

   q = data.quadrature_weights
   intensity_weighted = data.evaluate(
       remove="piston",
       weights=q * data.intensity,
   )

This multiplication is not valid merely because an array is named ``intensity``.
If ``transmitted_power`` already contains integrated per-ray power, pass it once,
unchanged, instead of multiplying by quadrature or intensity again::

   power_weighted = data.evaluate(
       remove="piston",
       weights=transmitted_power,
   )

The native convenience method never infers or multiplies intensity, apodization,
or quadrature. As a conservative safety precondition, each positive-weight sample
must have finite, strictly positive stored intensity. Intensity is used only for
this safety check, not numerical weighting, so any aligned real numeric dtype is
accepted and need not match the OPD floating precision. Effective support and
selected intensity values are validated before numerical RMS evaluation or fitting.
In particular, ``data.evaluate(remove="piston")`` selects every sample and rejects
an unusable one; it does not silently select only positive-intensity rays.
Explicit zero weights may exclude unusable samples, but the resulting RMS is
conditional on that supplied support. It is not a trace-completeness, throughput,
or missing-power verdict, and weights are never silently masked or altered.
Exclusion defines a discrete sample statistic; accuracy for a clipped or obscured
continuous pupil still requires an appropriate sampling rule and convergence study.
Use ``evaluate_wavefront`` directly for supplied sample datasets whose evaluation
is intentionally independent of native ray intensity.
