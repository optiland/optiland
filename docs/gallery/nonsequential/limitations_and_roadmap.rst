.. _nsq_limitations_and_roadmap:

NSQ Limitations & Roadmap
=========================

This page is the **canonical, single source of truth** for the non-sequential
(NSQ) engine's pre-release status, its capability envelope, its known
limitations, and the development roadmap. The package ``__init__`` docstring,
the :ref:`gallery overview <gallery_nonsequential>`, and the
:ref:`developer guide <nonsequential_raytracing>` all link here rather than
keeping their own copies. See :ref:`nsq_validation_report` for what the engine
is actually checked against in CI (closed-form benchmarks and invariants).

Pre-release status
-------------------

NSQ has **never shipped in a tagged Optiland release** — it exists on
``master`` as an actively-developed feature. The public API may therefore
still change without a deprecation cycle; once NSQ ships in a tagged
release, Optiland's usual API-stability guarantee applies. Differentiability
is interior-correct for refractive and reflective surfaces;
**visibility gradients are not yet supported** (see Limitations).

Capability envelope
--------------------

NSQ ships **two engines for two jobs**:

- **Forward mode (NumPy backend).** 1 × 10\ :sup:`7`\+ rays at ``max_depth`` 16,
  fully batched with live ray compaction and hard pixel binning. This is the
  production path for illumination and stray-light analysis, and the only
  backend that supports bounded ghost-path splitting.
- **Gradient mode (PyTorch backend).** ~1 × 10\ :sup:`5` rays at ``max_depth``
  16 on a single GPU. A full autograd graph is built through the Monte Carlo
  loop, so memory scales as ``O(num_rays × max_depth)`` (compaction is disabled
  to keep fixed tensor shapes). This is the path for optimization and ML layers.

Switch engines through ``optiland.backend``::

    import optiland.backend as be
    be.set_backend("torch")   # enables gradient mode

Physics coverage, as of this page:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Capability
     - Status
   * - Fresnel reflection/refraction
     - Differentiable, both backends.
   * - AR / thin-film coatings
     - Implemented via ``optiland.coatings`` (unpolarized only); NSQ and the
       sequential engine agree on R by construction.
   * - Mirror reflectance
     - Required, explicit (constant, wavelength-dependent, or coating). No
       implicit perfect-mirror default.
   * - Bulk (Beer-Lambert) absorption
     - Implemented, differentiable w.r.t. ``k`` and path length. Volumetric
       *scattering* (not just absorption) is still roadmap item 5.
   * - Diffuse/scatter BSDFs
     - Lambertian (with a differentiable reflect/transmit split via
       ``transmissive_fraction``), Harvey-Shack, tabulated.
   * - Medium sidedness
     - Geometric (``n_geom``-based), correct for index-matched/cemented
       interfaces. ``Volume`` adds construction-time watertightness
       validation for compound components.
   * - Detectors
     - Unified intersection pass, ``absorb`` flag for transmissive
       (non-terminating) detectors, attached ``total_flux``, true
       differentiable Gaussian splat.
   * - Rare-path sampling
     - Importance biasing (both backends), bounded splitting (NumPy only),
       unbiased Russian roulette.
   * - Reproducibility
     - Counter-based PCG32, bit-identical random decisions across
       ``batch_size``/compaction/backend (see the developer guide).
   * - Photometric units
     - Radiometric core (W) with a lumen/lux conversion layer
       (``optiland.nonsequential.units``).
   * - Diagnostics
     - Self-diagnosing ``SimulationResult.report()``: depth truncation,
       roulette loss, unreached geometry, per-detector undersampling.
   * - Volumetric scattering
     - Not implemented (roadmap item 5).
   * - Polarization
     - Not implemented (roadmap item 6).
   * - Visibility gradients
     - Zero (roadmap item 1) — see Limitations below.

Limitations
------------

1. **Visibility gradients are zero — the biggest physics gap.** When a ray
   silhouette moves across a surface boundary (vignetting, occlusion,
   which-surface-hit), that boundary contributes **no** gradient. The
   detached-sample / attached-weight estimator differentiates interior
   interactions (Fresnel, refraction, dispersion, BSDF lobes, absorption)
   but not the discrete visibility decision. Reparameterization (roadmap #1)
   closes this gap. A dedicated test
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
   :class:`~optiland.nonsequential.detectors.IrradianceDetector` extents and
   ``total_flux`` are differentiable.
4. **No polarization.** Stokes tracking is not present yet (roadmap #6), so
   polarization-sensitive coatings (``BaseCoatingPolarized``) raise rather
   than being silently averaged to a scalar.
5. **No volumetric scattering.** Bulk absorption (Beer-Lambert) is
   implemented; a photon being scattered *within* a volume, rather than only
   attenuated, is not (roadmap #5).
6. **Gradient-mode memory cap.** The ~1 × 10\ :sup:`5`-ray envelope is a hard
   constraint of the naive autograd strategy. Path Replay Backpropagation
   (roadmap #3) lifts this cap.
7. **No acceleration structure.** Intersection is ``O(rays × surfaces)`` per
   bounce on both backends — fine for the scene sizes NSQ targets today, but
   a scaling limit for very high surface-count scenes. A
   ``benchmarks/nonsequential/`` harness measures this rather than
   estimating it, so a future BVH/batched-traversal effort has a baseline
   (roadmap #7).

Roadmap
-------

Ordered by priority. Each item names the seam or reference paper.

1. **Reparameterization for visibility gradients** — warped-area
   reparameterization (Loubet et al. 2019; Bangaru et al. 2020) so
   silhouette / vignetting / which-surface-hit gradients become correct. Closes
   the single biggest physics limitation.
2. **Optimization integration** — wire NSQ into Optiland's ``Variable`` /
   operand system and optimizers for end-to-end illumination and stray-light
   design.
3. **Path Replay Backpropagation (PRB)** — constant-memory, unbiased gradients
   via the ``gradient_mode`` seam already present in ``TorchBackend`` (Vicini
   et al. 2021); lifts the ~1 × 10\ :sup:`5`-ray cap.
4. **GUI integration** — NSQ scene building and visualization in ``optiland_gui``.
5. **Volumetric scattering** — beyond the Beer-Lambert absorption already
   implemented, actual in-volume scatter events.
6. **Polarization** — Stokes tracking through Fresnel coatings, done properly.
7. **Acceleration structure** — a BVH or batched-traversal scheme, sized
   against the ``benchmarks/nonsequential/`` measurements rather than guessed.
8. **Dr.Jit / Mitsuba 3 / OptiX backend** — high-performance
   :class:`~optiland.nonsequential.backends.base.TracerBackend` plugin
   consuming the same ``SceneIR`` the reference backends interpret.

Get involved
------------

**Try it and tell us what you build.** If you use NSQ for illumination design,
stray-light analysis, or differentiable optics, open a
`GitHub issue <https://github.com/HarrisonKramer/optiland/issues>`_ and describe
your use case — your feedback directly shapes the roadmap.

**Contribute.** The roadmap items above (especially reparameterization, PRB, and
GUI integration) are open for contributors. Development happens on ``master``;
PRs target ``master`` directly. The canonical engine reference is the
:doc:`developer guide </developers_guide/nonsequential_raytracing>`.

References
----------

- Loubet, Holzschuch & Jakob, *Reparameterizing Discontinuous Integrands for
  Differentiable Rendering*, SIGGRAPH Asia 2019.
- Bangaru, Li & Durand, *Unbiased Warped-Area Sampling for Differentiable
  Rendering*, SIGGRAPH Asia 2020.
- Vicini, Speierer & Jakob, *Path Replay Backpropagation of Light Paths*,
  SIGGRAPH 2021.
