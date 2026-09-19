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
