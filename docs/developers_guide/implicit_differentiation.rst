.. _implicit_differentiation:

Implicit Differentiation
========================

Optiland solves two problems iteratively that sit in the middle of the
differentiable ray-tracing path:

1. **Ray-surface intersection** — finding the propagation distance ``t`` at
   which a ray meets a non-analytic surface (:class:`NewtonRaphsonGeometry`).
2. **Real image height fields** — finding the object-space field parameters
   whose traced chief ray lands on a requested image height
   (:class:`RealImageHeightField`).

Both are root-finding problems. Differentiating them naively — by letting
autograd record every Newton iteration — makes the graph grow linearly with
the iteration count. With ``max_iter=100`` and a large ray batch, that
dominates memory and dwarfs the cost of the actual optics (issue #335).

Instead, both use **implicit differentiation**: the iteration runs without a
graph, and a single grad-attached correction supplies the derivative.

The construction
----------------

Take the ray-surface intersection. For a sag surface
:math:`z = s(x, y; \theta)` and a ray
:math:`\mathbf p(t) = \mathbf p_0 + t\,\mathbf d`, define the residual

.. math::

   F(t, \theta) = s(x_0 + tL,\; y_0 + tM;\; \theta) - (z_0 + tN).

The intersection distance is the root :math:`F(t^\star, \theta) = 0`. At a
converged **simple** root (:math:`F_t \neq 0`), the implicit function theorem
gives the derivative directly, without differentiating the iteration:

.. math::

   \frac{\mathrm{d}t^\star}{\mathrm{d}\theta} = -\frac{F_\theta}{F_t}.

The implementation runs the Newton iteration inside ``torch.no_grad()`` to get
a detached root :math:`\bar t`, then evaluates one grad-attached correction:

.. math::

   t_{\mathrm{implicit}} = \bar t -
   \frac{F(\bar t, \theta)}{\operatorname{stopgrad}\big(F_t(\bar t, \theta)\big)}.

Because :math:`F \approx 0` at convergence, the forward value is essentially
unchanged, but the expression now carries the correct first derivative.
Detaching the denominator is deliberate: a first derivative does not require
differentiating the inverse Jacobian.

The field solve uses the same construction with a full :math:`2\times2`
Jacobian, since the x and y image coordinates are coupled:

.. math::

   \frac{\mathrm{d}\mathbf q^\star}{\mathrm{d}\theta}
   = -\mathbf J_q^{-1}\mathbf R_\theta,
   \qquad
   \mathbf J_q = \frac{\partial \mathbf R}{\partial \mathbf q}.

What this guarantees
--------------------

**First derivatives only.** Double backward runs and stays finite, so these
geometries compose into larger graphs, but the second derivative is not
guaranteed to match the unrolled Newton system — the detached denominator is
exactly what makes it differ. Treat first order as the contract.

**Forward values are preserved within solver tolerance**, not bitwise. The
differentiable path applies one final Newton correction, which normally
*improves* the root.

**Graph size is independent of the iteration count.** Raising ``max_iter``
changes how long the primal solve runs, not how large the autograd graph is.

The assumptions
---------------

The derivative is exact only when all of the following hold:

1. the primal solve converged to the intended physical root;
2. the root stays on the same branch under a small parameter perturbation;
3. :math:`F_t` (or :math:`\det \mathbf J_q`) is not zero or numerically
   singular;
4. ``sag()`` and ``_surface_normal()`` describe the same differentiable
   surface;
5. cached tensors derived from trainable parameters are attached to the graph
   when the correction runs.

Where an assumption fails, the code does not quietly return a confident
answer:

- **Non-converged rays** keep their forward value but receive a *detached*
  distance, and a ``RuntimeWarning`` reports the residual and iteration count.
  A failed root never carries a fabricated gradient.
- **Near-singular denominators** are floored with a sign-preserving,
  dtype- and scale-aware threshold,
  :math:`\tau = C\,\epsilon_{\mathrm{dtype}}\max(1, |s_xL| + |s_yM| + |N|)`.
  Preserving the sign matters: replacing a small negative denominator with a
  positive constant reverses the Newton step. This is a *regularization*, not
  the exact physics — at a true tangent intersection the sensitivity really is
  singular.
- **A singular field Jacobian or a non-convergent field solve** raises, rather
  than returning an implicitly differentiated result.

.. note::
   Thresholds are derived from the working dtype. A hardcoded ``1e-14`` is
   below float32 round-off and would never trigger, which is why the code
   builds them from ``machine_eps`` instead.

Caches must not be built under ``no_grad``
------------------------------------------

This is the subtle failure mode, and it is worth understanding before adding
any cache to a geometry.

Because the primal solve deliberately runs inside ``torch.no_grad()``, the
*first* call into a lazily-built cache can happen while grad recording is
disabled. A tensor stacked under ``no_grad`` is permanently detached — calling
``requires_grad_(True)`` on the original coefficients afterwards cannot
reconnect it. The differentiable correction then reuses that detached cache,
the residual still depends on the ray coordinates, everything looks healthy,
and the gradient with respect to the cached parameters silently collapses to
zero.

The Forbes geometries hit exactly this with their prepared-coefficient cache.
``ForbesGeometryBase._ensure_coeffs`` therefore builds the cache under an
explicit ``torch.enable_grad()``, so it stays differentiable regardless of the
caller's ambient grad context::

    def _ensure_coeffs(self) -> None:
        if self._coeffs_dirty:
            if torch is not None and be.get_backend() == "torch":
                with torch.enable_grad():
                    self._prepare_coeffs()
            else:
                self._prepare_coeffs()

Geometries that cannot use that approach may instead override
``NewtonRaphsonGeometry._invalidate_cached_derived_state_for_autograd``, which
is called after the no-grad primal solve and immediately before the
differentiable correction.

The same hazard applies to any helper that reads a grad-carrying tensor out as
a scalar. ``be.full_like(x, value)`` and ``be.array(tensor)`` both detach;
prefer ``be.full_like(x, 0.0) + value`` or plain broadcasting when ``value``
may require grad.

Testing changes to these paths
------------------------------

A gradient that is merely *nonzero* proves very little — a detached
intermediate usually still produces a plausible-looking number. Compare
against finite differences, and check the finite-difference estimate is itself
stable at two step sizes before trusting it:

- ``tests/test_nr_implicit_diff_distance.py`` — AD/FD for the scalar solve.
- ``tests/test_nr_implicit_diff_forbes_cache.py`` — cache lifecycle and
  repeated backward.
- ``tests/test_real_image_height_implicit_diff.py`` — field solve accuracy,
  cross-coupling, AD/FD and graph size.
- ``tests/test_nr_implicit_diff_graph_complexity.py`` — graph size versus
  ``max_iter``.

References
----------

- Congli Wang, Ni Chen, and Wolfgang Heidrich, "dO: A Differentiable Engine
  for Deep Lens Design of Computational Imaging Systems," *IEEE Transactions
  on Computational Imaging*, 2022.
  `doi:10.1109/TCI.2022.3212837 <https://doi.org/10.1109/TCI.2022.3212837>`_
