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
a detached root :math:`\bar t`, classifies every ray at that final root (still
graph-free), then evaluates one grad-attached correction **only for the rays
whose root is regular**:

.. math::

   t_{\mathrm{valid}} = \bar t -
   \frac{F(\bar t, \theta)}{\operatorname{stopgrad}\big(F_t(\bar t, \theta)\big)}.

Because :math:`F \approx 0` at convergence, the forward value is essentially
unchanged, but the expression now carries the correct first derivative.
Detaching the denominator is deliberate: a first derivative does not require
differentiating the inverse Jacobian. For regular rays the detached
denominator is used *unclipped* — regularity guarantees it is safely above
the singularity threshold, so the returned derivative is the exact
first-order physics, never a floored surrogate.

A ray is **regular** when all of the following hold at the final root: the
solve converged; every relevant quantity (root, residual, ray components,
normal, denominator) is finite; :math:`|n_z|` is above a dtype-aware
threshold, so the surface is a numerically valid local height function; and
:math:`|F_t|` is above a dtype- and scale-aware threshold, so the root is
simple. Rejected rays keep their detached primal forward value, are **never**
evaluated through a grad-attached residual branch — PyTorch can propagate
``NaN`` through the backward pass from an invalid branch even when it is
later masked by ``where`` — and therefore contribute exactly zero gradient to
shared trainable parameters. Regular rays in the same batch remain fully
differentiable. Rejections are reported by one grouped ``RuntimeWarning``
per call, split by category (non-converged, non-finite, tangent/grazing,
near-vertical surface).

The field solve uses the same construction with a full :math:`2\times2`
Jacobian, since the x and y image coordinates are coupled:

.. math::

   \frac{\mathrm{d}\mathbf q^\star}{\mathrm{d}\theta}
   = -\mathbf J_q^{-1}\mathbf R_\theta,
   \qquad
   \mathbf J_q = \frac{\partial \mathbf R}{\partial \mathbf q}.

The Jacobian is validated with a **scale-invariant** condition test: each
field's matrix is normalized by its largest-magnitude entry,
:math:`\mathbf A = \mathbf J / s`, and classified as unusable when the
reciprocal Frobenius-condition estimate
:math:`\rho_F = |\det\mathbf A| / \lVert\mathbf A\rVert_F^2` falls to
round-off level (:math:`\rho_F \le C\,\epsilon_{\mathrm{dtype}}`). A global
unit or magnification rescaling of the Jacobian therefore never changes the
singular/non-singular classification — a well-conditioned but
small-magnitude matrix such as :math:`10^{-3}\mathbf I` is accepted in every
dtype. The linear solve itself operates on the normalized system
:math:`\mathbf A\,\Delta\mathbf q = \mathbf R / s`, which avoids
raw-determinant overflow and underflow. In the implicit correction the solve
is *strict*: a singular or non-convergent field solve raises rather than
returning an implicitly differentiated result, and the determinant is never
clipped to fabricate a step.

Chief-ray and coordinate conventions of the field solve
-------------------------------------------------------

The chief ray of :class:`RealImageHeightField` is constructed by aiming at
the **paraxial entrance-pupil center**. This is not exact stop-center aiming:
in systems with strong pupil aberration or a tilted/decentered stop, the
traced ray does not necessarily cross the physical stop at its center. Exact
nested stop-center aiming is a separate problem handled by the ray-aiming
machinery.

The requested image height :math:`(h_x, h_y)` is defined in the **global**
``x``/``y`` coordinates of the chief ray's traced intercept on the image
surface (surfaces globalize ray coordinates after each trace). For a tilted
or decentered image surface the solve drives the global transverse
coordinates to the target, not coordinates local to the image surface.

Surface-derivative consistency
------------------------------

The implicit construction assumes ``sag()`` and ``_surface_normal()``
describe the *same* differentiable surface. For normalized polynomial
surfaces this includes the chain-rule factors: the Chebyshev sag uses
:math:`T_i(x/\mathrm{norm}_x)`, so every slope carries a
:math:`1/\mathrm{norm}` factor. The Chebyshev polynomials and their
derivatives are evaluated with the stable recurrences
(:math:`T_n = 2xT_{n-1} - T_{n-2}`, :math:`T_n' = n\,U_{n-1}`), which give
exact finite endpoint values :math:`T_n'(\pm 1) = (\pm 1)^{n-1} n^2` where
the ``cos``/``arccos`` closed form produces an autograd :math:`0/0`.

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
3. :math:`F_t` is not zero or numerically singular (a converged **regular
   simple root**), and for the field solve :math:`\mathbf J_q` passes the
   scale-invariant condition test;
4. :math:`|n_z|` is above the dtype-aware threshold, so the surface is a
   numerically valid local height function;
5. ``sag()`` and ``_surface_normal()`` describe the same differentiable
   surface;
6. cached tensors derived from trainable parameters are attached to the graph
   when the correction runs.

Where an assumption fails, the code does not quietly return a confident
answer:

- **Rejected rays** (non-converged, non-finite, tangent/grazing, or
  near-vertical surface) keep their *detached* primal forward value, are
  never evaluated through a grad-attached residual branch, and contribute
  zero gradient. One grouped ``RuntimeWarning`` reports the counts,
  iteration count and worst residual by category. Regular rays in the same
  batch remain differentiable.
- **Near-singular denominators** may be floored — sign-preserving, with the
  dtype- and scale-aware threshold
  :math:`\tau = C\,\epsilon_{\mathrm{dtype}}\max(1, |s_xL| + |s_yM| + |N|)`
  — but only inside the *graph-free primal numerical solver*, where the
  floor is a step-size safeguard. Preserving the sign matters: replacing a
  small negative denominator with a positive constant reverses the Newton
  step. A root whose true sensitivity is singular is **excluded** from
  implicit differentiation; a clipped denominator is never presented as the
  exact physical derivative.
- **A singular field Jacobian or a non-convergent field solve** raises,
  rather than returning an implicitly differentiated result. The final
  implicit correction always uses the strict solve; the inverse Jacobian is
  never silently regularized there.

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
- ``tests/test_nr_implicit_diff_singular_roots.py`` — the singular-root
  policy: mixed valid/invalid batches, tangent roots, ``n_z`` validity.
- ``tests/test_nr_implicit_diff_chebyshev.py`` — sag/normal consistency,
  endpoint behavior and implicit AD/FD for normalized Chebyshev surfaces.
- ``tests/test_field_jacobian_condition.py`` — scale invariance of the
  2x2 condition test and the normalized solve.
- ``tests/test_nr_implicit_diff_forbes_cache.py`` — cache lifecycle and
  repeated backward.
- ``tests/test_real_image_height_implicit_diff.py`` — field solve accuracy,
  cross-coupling, AD/FD, VJP block-diagonal verification and graph size.
- ``tests/test_nr_implicit_diff_graph_complexity.py`` — graph size versus
  the executed primal iteration count.

References
----------

- Congli Wang, Ni Chen, and Wolfgang Heidrich, "dO: A Differentiable Engine
  for Deep Lens Design of Computational Imaging Systems," *IEEE Transactions
  on Computational Imaging*, 2022.
  `doi:10.1109/TCI.2022.3212837 <https://doi.org/10.1109/TCI.2022.3212837>`_
