.. _nsq_limitations_and_roadmap:

NSQ Limitations & Roadmap
=========================

This page is the **canonical, single source of truth** for the non-sequential
(NSQ) engine's beta status, its v1 capability envelope, its known limitations,
and the development roadmap. The README, the package ``__init__`` docstring, the
:ref:`gallery overview <gallery_nonsequential>`, and the
:ref:`developer guide <nonsequential_raytracing>` all link here rather than
keeping their own copies.

Beta status
-----------

The NSQ engine is a **beta release**. The public symbols documented in
:ref:`api_nonsequential` are stable and will not be removed without a
deprecation cycle, but internal implementation details (backend classes,
geometry helper methods) may still change as the API stabilizes toward a frozen
1.0. Differentiability is interior-correct for refractive and reflective
surfaces; **visibility gradients are not yet supported** (see Limitations).

v1 capability envelope
----------------------

NSQ ships **two engines for two jobs**:

- **Forward mode (NumPy backend).** 1 × 10\ :sup:`7`\+ rays at ``max_depth`` 16,
  fully batched with live ray compaction and hard pixel binning. This is the
  production path for illumination and stray-light analysis.
- **Gradient mode (PyTorch backend).** ~1 × 10\ :sup:`5` rays at ``max_depth``
  16 on a single GPU. A full autograd graph is built through the Monte Carlo
  loop, so memory scales as ``O(num_rays × max_depth)`` (compaction is disabled
  to keep fixed tensor shapes). This is the path for optimization and ML layers.

Switch engines through ``optiland.backend``::

    import optiland.backend as be
    be.set_backend("torch")   # enables gradient mode

Limitations (v1)
----------------

1. **Visibility gradients are zero — the biggest physics gap.** When a ray
   silhouette moves across a surface boundary (vignetting, occlusion,
   which-surface-hit), that boundary contributes **no** gradient. The
   detached-sample / attached-weight estimator differentiates interior
   interactions (Fresnel, refraction, dispersion) but not the discrete
   visibility decision. Reparameterization (roadmap #1) closes this gap. A
   dedicated test
   (``tests/nonsequential/test_nsq_geometric_gradients.py``) explicitly
   asserts this behaviour.
2. **Mesh geometry is forward-only.** Analytic conics
   (:class:`~optiland.nonsequential.components.geometry.ConicGeometry`,
   :class:`~optiland.nonsequential.components.geometry.SphereGeometry`,
   :class:`~optiland.nonsequential.components.geometry.ParaboloidGeometry`) are
   differentiable; :class:`~optiland.nonsequential.components.geometry.MeshGeometry`
   is not — calling ``backward()`` through a mesh interaction will raise.
3. **Source geometry is not differentiable.** Source sampling
   (aperture position, emission angle, emitter extent) runs in NumPy, so
   ``aperture_radius``, ``half_angle_deg`` and the extended-source dimensions
   cannot carry gradients. Passing a ``requires_grad`` tensor for one of these
   raises :class:`NotImplementedError` rather than detaching silently, so a
   dead design variable is never mistaken for a live one. Source
   ``total_flux`` *is* differentiable. ``SpectralDetector`` extents are
   likewise detached (the detector accumulates into a NumPy histogram);
   :class:`~optiland.nonsequential.detectors.IrradianceDetector` extents are
   differentiable.
4. **No polarization.** Stokes tracking is not present in v1 (roadmap #6).
5. **Gradient-mode memory cap.** The ~1 × 10\ :sup:`5`-ray envelope is a hard
   constraint of the naive autograd strategy. Path Replay Backpropagation
   (roadmap #3) lifts this cap.

Roadmap
-------

Ordered by priority. Each item names the seam or reference paper.

1. **Reparameterization for visibility gradients** — warped-area
   reparameterization (Loubet et al. 2019; Bangaru et al. 2020) so
   silhouette / vignetting / which-surface-hit gradients become correct. Closes
   the single biggest physics limitation of the beta.
2. **Optimization integration** — wire NSQ into Optiland's ``Variable`` /
   operand system and optimizers for end-to-end illumination and stray-light
   design.
3. **Path Replay Backpropagation (PRB)** — constant-memory, unbiased gradients
   via the ``gradient_mode`` seam already present in ``TorchBackend`` (Vicini
   et al. 2021); lifts the ~1 × 10\ :sup:`5`-ray cap.
4. **GUI integration** — NSQ scene building and visualization in ``optiland_gui``.
5. **Volumetric media** — Beer–Lambert bulk absorption and volume scattering.
6. **Polarization** — Stokes tracking through Fresnel coatings, done properly.
7. **Dr.Jit / Mitsuba 3 backend** — high-performance
   :class:`~optiland.nonsequential.backends.base.TracerBackend` plugin via scene
   translation.

Get involved
------------

**Try it and tell us what you build.** If you use NSQ for illumination design,
stray-light analysis, or differentiable optics, open a
`GitHub issue <https://github.com/HarrisonKramer/optiland/issues>`_ and describe
your use case — your feedback directly shapes the roadmap.

**Contribute.** The roadmap items above (especially reparameterization, PRB, and
GUI integration) are open for contributors. Development happens on the
``feat/nonsequential`` branch; PRs target ``master``. The canonical engine
reference is the :doc:`developer guide
</developers_guide/nonsequential_raytracing>`.

References
----------

- Loubet, Holzschuch & Jakob, *Reparameterizing Discontinuous Integrands for
  Differentiable Rendering*, SIGGRAPH Asia 2019.
- Bangaru, Li & Durand, *Unbiased Warped-Area Sampling for Differentiable
  Rendering*, SIGGRAPH Asia 2020.
- Vicini, Speierer & Jakob, *Path Replay Backpropagation of Light Paths*,
  SIGGRAPH 2021.
