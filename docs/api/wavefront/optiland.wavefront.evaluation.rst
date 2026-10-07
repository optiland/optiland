optiland.wavefront.evaluation
==============================

.. automodule:: optiland.wavefront.evaluation

   The evaluator consumes already referenced OPD samples in waves and never traces
   an optical system. ``weights`` are final effective evaluation weights: the
   evaluator does not infer or apply quadrature, intensity, or apodization factors.
   ``weights=None`` requests an equal-sample statistic.

   On positive-weight support :math:`S`, the objective is
   :math:`R = \sqrt{\sum_{i\in S}\alpha_i r_i^2}`, with
   :math:`\alpha_i = w_i / \sum_{j\in S}w_j`. ``none`` uses the supplied OPD
   directly, ``piston`` subtracts its weighted mean, and ``piston_tilt`` removes
   the weighted affine least-squares fit in the supplied coordinate chart.
   Fitting and RMS use identical support and weights, with population/integration
   normalization rather than a degrees-of-freedom correction.

   NumPy masked arrays are rejected rather than treating their masks as sample
   support. When plain sequence weights are converted to the OPD dtype, evaluation
   also rejects any nonzero source value that becomes zero. Underflow already
   present in a caller-created backend array cannot be recovered; such an existing
   zero retains its normal zero-weight meaning.

   Complex numeric elements in Python sequences are rejected before conversion
   to a real dtype, even when their imaginary part is zero. Existing complex
   arrays are likewise rejected rather than projecting them onto their real parts.

   The advertised matrix is NumPy CPU and Torch CPU with ``float32`` or
   ``float64``. Active-backend arrays preserve dtype/device and Torch autograd
   attachment; additional Torch devices depend on backend SVD support.

   First-order eager Torch gradients through ``Tensor.backward()`` and
   ``torch.autograd.grad`` are supported for supplied OPD while
   coordinates, weights, and support are held fixed, for full affine rank,
   normal-range nonzero RMS and derivatives, and normal-range nonzero normalized
   weighted components. The latter means ``sqrt_alpha * (r / max(abs(r)))`` and
   its norm, where ``sqrt_alpha`` denotes the normalized square-root weights and
   ``r`` the selected residuals. Exact zero components are allowed. The analytical
   RMS backward uses the weighted direction directly, so a representable small
   derivative is not erased by multiplying and then dividing by a small OPD scale.
   For example, constant float64 OPD ``1e-250`` with weights ``[1e-100, 1]``
   retains the first derivative near ``1e-100``.

   Higher-order derivatives, forward-mode differentiation, ``torch.func`` transforms,
   coordinate/weight derivatives, and underflow of those normalized components
   are outside the contract. Metadata may remain graph-attached for forward
   compatibility, but its derivatives are not guaranteed and some are deliberately
   omitted by the custom backward. Detach metadata to construct a fixed-metadata
   objective. Subnormal forward RMS values
   remain supported. This is not a guarantee for arbitrary end-to-end optical-system
   derivatives when traced coordinates or weights also vary.

   For example, fixed metadata may be detached explicitly before optimizing OPD::

      fixed_weights = effective_weights.detach()
      result = evaluate_wavefront(
          opd,
          x=traced_x.detach(),
          y=traced_y.detach(),
          weights=fixed_weights,
          remove="piston_tilt",
      )
      result.rms.backward()

   Affine coefficients use the centered form
   ``a + b*(x-x_ref) + c*(y-y_ref)``. The returned condition number is computed
   from the weighted design after centering and weighted-standard-deviation scaling.
   The supplied ``x`` and ``y`` values define the caller's two-dimensional
   coordinate chart. Slopes are measured in waves per caller coordinate unit;
   they are not automatically three-dimensional optical directions.

   Absolute piston and affine-intercept coefficients are reported in the input
   dtype and can therefore round when a large common OPD offset is not exactly
   compatible with a small mean correction. Piston and affine residuals retain
   the anchor and correction separately, so that rounding of the reported absolute
   coefficient does not discard otherwise representable centered residuals.
   Reconstructing every input sample from one rounded absolute coefficient is not
   guaranteed at such offsets. Both centering passes use every positive weight;
   there is no epsilon-based weight cutoff or dominant-support substitute
   objective.

   Mixed-sign reductions retain addition errors through a reduction tree and
   sum the remaining expansion, including the subtraction errors in the second
   centering pass. They do not treat a finite ordinary sum as proof of accuracy.
   Products and divisions still round in the input dtype; universal correctly
   rounded means and invariance of every last bit under permutation are not
   guaranteed, particularly after overflow or underflow.

   The default rank cutoff is ``eps(dtype) * max(n_used, 3)``. Consequently,
   splitting a weighted sample changes the default threshold through ``n_used``
   even when it preserves the mathematical integration measure. Use an explicit
   ``rcond`` when comparing rank decisions across different sample counts.

   ``used_mask`` describes only the strictly positive-weight statistical support.
   It is not a ray-trace completeness, clipping, throughput, or missing-power
   verdict. The evaluator cannot infer why a sample has zero weight.

   Finite excluded samples receive diagnostic residual extrapolations when the
   fitted subtraction is representable. If that extrapolation overflows, the fit
   and RMS remain valid and the excluded residual remains visibly nonfinite.
   Invalid excluded OPD or coordinates likewise remain visibly invalid rather than
   becoming apparent zero-OPD measurements. These invalid diagnostics do not enter
   the positive-support autograd graph. Finite representable diagnostics retain
   their dependence on the fitted coefficients, including at zero slopes.
   Selected residuals preserve the required OPD graph. A reconstructed objective
   reproduces the supported derivative only when its RMS implementation is itself
   stable over the same range. For example::

      rebuilt = evaluate_wavefront(
          result.residual_opd[result.used_mask],
          weights=fixed_weights[result.used_mask],
          remove="none",
      )

   An ordinary scale/square/square-root expression can still lose small derivatives.

   A fit raises ``WavefrontEvaluationNumericalError`` if finite caller-unit
   coefficients or residuals on positive-weight support cannot be represented in
   the input dtype. Unrepresentable zero-weight diagnostics do not invalidate the
   positive-support fit.

   ``rcond`` accepts Python and NumPy real scalars, excluding booleans. Strings,
   arrays, and Torch tensors raise ``TypeError`` rather than being coerced with
   ``float()``. Nonfinite values and values outside ``[0, 1)`` raise ``ValueError``.

   .. rubric:: Optical reference and weighting

   OPD construction/reference geometry, statistical detrending, and integration
   measure are separate choices. This evaluator handles the latter two for an
   already referenced dataset. Matching detrending alone does not establish
   equivalence between optical analyses with different reference construction,
   coordinate mapping, sampling, aiming, or clipping.

   Compatible local intensity ``I`` and a compatible nonnegative scalar amplitude
   envelope ``A`` give distinct objectives through ``q * I`` and ``q * A``.
   Complex field amplitudes are not valid statistical weights.

   For Gaussian intensity ``I(rho) = exp(-rho**2 / (2*sigma**2))`` and amplitude
   envelope ``A(rho) = exp(-G*rho**2)``, the algebraic conversion is
   ``G = 1 / (4*sigma**2)``. If the chart radius corresponds to aperture radius
   ``R`` and ``w_b`` is the ``1/e**2`` intensity radius, ``sigma = w_b/(2*R)``.
   Thus ``sigma=1/2`` corresponds to ``G=1`` and ``sigma=1/6`` to ``G=9``.
   The latter is a strong-apodization convergence case, not a low-density default.

   .. rubric:: Functions

   .. autosummary::

      evaluate_wavefront

   .. rubric:: Classes

   .. autosummary::

      WavefrontEvaluationResult
      WavefrontEvaluationNumericalError
