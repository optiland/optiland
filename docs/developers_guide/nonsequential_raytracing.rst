.. _nonsequential_raytracing:

Non-Sequential Ray Tracing
==========================

This guide is written for **contributors who extend the non-sequential (NSQ)
engine** — adding geometries, BSDFs, detectors, or a new execution backend. It
documents the architecture, the differentiability contract, and the extension
seams. For *user*-facing tutorials see the
:ref:`NSQ gallery <gallery_nonsequential>`; for the API reference see
:ref:`api_nonsequential`; for the current capability envelope and known gaps
see :ref:`nsq_limitations_and_roadmap` (canonical).

.. note::

   **Pre-release.** NSQ has never shipped in a tagged Optiland release, so
   the public API may still change without a deprecation cycle. This page is
   the canonical engine reference; see also
   :doc:`/gallery/nonsequential/limitations_and_roadmap`.

.. _nsq_arch:

1. Architecture — description, IR, and execution
-------------------------------------------------

NSQ separates **what a scene is** from **how it is traced** through two
seams, not one:

- :class:`~optiland.nonsequential.scene.NSQScene` is a *declarative*
  container. It holds sources, components (built from
  :class:`~optiland.nonsequential.components.base.BaseComponent` primitives),
  and detectors with their parameters and coordinate systems. It performs no
  ray math.
- :mod:`optiland.nonsequential.ir` lowers that scene into a
  ``SceneIR`` — a plain-data description (``PrimitiveIR``, ``BsdfIR``,
  ``MediumIR``, ``SamplingPolicy``) with no Python-side per-hit callables.
  This is the seam a non-reference backend (e.g. a future OptiX or
  Dr.Jit/Mitsuba backend) consumes; it never has to understand
  :class:`BaseComponent` subclasses or Python dispatch.
- A :class:`~optiland.nonsequential.backends.base.TracerBackend` *executes*
  the lowered IR. ``NumpyBackend`` and ``TorchBackend`` are *reference
  interpreters* of that IR: ``BaseComponent.interact()`` is a private
  implementation detail of these two backends, not the engine contract —
  dispatch is driven by ``PrimitiveIR.bsdf``/``interaction`` data, not by
  Python class identity.

This is why the same scene can be traced multiple ways without changing
scene-building code:

- **today** on NumPy (fast forward path),
- **today** on PyTorch (differentiable),
- **later** on a high-performance backend (Dr.Jit / Mitsuba, OptiX) via the
  same ``SceneIR`` — only a new ``TracerBackend`` is required.

:class:`~optiland.nonsequential.tracer.NSQTracer` is a thin coordinator: it
selects a backend (or accepts one explicitly) and delegates the loop.

.. _nsq_backends:

2. Backends
-----------

The active backend is chosen from ``optiland.backend`` unless one is passed
explicitly to :class:`~optiland.nonsequential.tracer.NSQTracer` or
``NSQScene.trace(backend=...)``.

**NumpyBackend** (forward, production)
   Raw-NumPy outer loop (does *not* go through ``optiland.backend``), with live
   **ray compaction** (dead rays are dropped between bounces) and hard pixel
   binning. Designed for 1e7+ rays at ``max_depth`` 16. This is the engine for
   real illumination and stray-light work, and the only backend that supports
   bounded ghost-path splitting (``SamplingPolicy.split_depth``).

**TorchBackend** (differentiable)
   ``be``-unified array ops that build a full PyTorch autograd graph through the
   Monte Carlo loop. Compaction is **disabled** so tensor shapes stay fixed for
   the graph, which is why memory scales as ``O(num_rays × max_depth)`` and the
   practical envelope is ~1e5 rays at depth 16 on a single GPU. This is the
   engine for optimization and ML layers. It forces ``split_depth=0`` and warns
   if a scene sets otherwise, since fixed tensor shapes are required for the
   autograd graph.

There is **no separate GPU array backend.** GPU acceleration in the
differentiable path comes from PyTorch device placement; the forward NumPy path
is CPU-batched. A ``benchmarks/nonsequential/`` harness measures rays/sec
against surface count, ray count, and trace depth for both backends, as a
baseline for future acceleration work (BVH / batched traversal) — run it with
``python -m benchmarks.nonsequential.run``.

.. _nsq_megakernel:

3. Wavefront megakernel
-----------------------

The loop is **breadth-first per bounce**: at each depth the backend intersects
the entire live wavefront against all components and detectors, applies
interactions, then advances. Termination is a deterministic
``max_depth`` cap, plus an unbiased Russian-roulette kill below
``rr_start_flux`` (replacing the older biased flux-truncation) — depth
truncation is the only *inherent, reported bias*; roulette is unbiased in
expectation. Rays that exceed ``max_depth`` are counted in
``SimulationResult.num_rays_depth_killed``, and both loss mechanisms are
tracked separately in ``SimulationResult.diagnostics``.

.. code-block:: python

   result = scene.trace(num_rays=1_000_000, max_depth=16, seed=42)
   print(result.report())

.. _nsq_sampling:

4. Rare-path sampling policy
-----------------------------

:class:`~optiland.nonsequential.ir.scene_ir.SamplingPolicy` (set via
``scene.sampling_policy``) controls how the tracer spends rays on rare paths
— faint ghosts, high-order reflections — without changing the expected
result:

- **Importance biasing** (``reflect_prob``, all backends): the reflect/refract
  branch is still drawn from a *detached* probability with a compensating
  *attached* weight (the same detached-sample/attached-weight estimator
  Fresnel splitting always used — see :ref:`nsq_diff_contract`), but the
  probability itself can be biased away from the physical Fresnel value.
  ``"fresnel"`` (default) reproduces the original unbiased-at-p=R behaviour;
  ``"auto"`` clamps into ``[0.25, 0.75]``; an explicit float fixes it. A 4%
  uncoated ghost at ``reflect_prob=0.25`` gets ~12x variance reduction at
  zero memory cost.
- **Bounded splitting** (``split_depth``, NumPy forward engine only): both
  the reflect and refract branches are spawned (rather than one sampled) while
  ``bounce < split_depth``, then the tracer reverts to single-branch
  sampling. Live rays are capped at ``split_budget × batch_size``; excess is
  Russian-rouletted rather than silently dropped.
  ``SimulationResult.diagnostics.split_budget_saturated`` reports whether
  that cap was ever hit.
- **Russian roulette** (``rr_start_flux``): below this fraction of a ray's
  initial flux, kill with probability *p* and boost survivors by
  ``1/(1-p)``. Unbiased in expectation; ``SimulationResult.total_flux_lost``
  and ``diagnostics.rr_killed_flux_fraction`` should be near zero for a
  well-configured scene, making a nonzero value a genuine variance signal
  rather than an expected bookkeeping entry.

.. _nsq_rng:

5. Reproducibility — PCG32
----------------------------

Random numbers are drawn from a vectorised, counter-based PCG32
(:mod:`optiland.nonsequential.rng`) keyed by
``(seed, ray_id, bounce, event_slot)``, where ``event_slot`` is a small enum
(:class:`~optiland.nonsequential.rng.EventSlot`: ``FRESNEL_BRANCH``,
``SCATTER_BRANCH``, ``BSDF_U1``, ``BSDF_U2``, ``BSDF_LOBE_BRANCH``, ``RR``,
source-sampling slots, ...). There is no shared mutable stream, so each
ray's random numbers depend only on its own identity, not on how many other
rays or components happen to be in the scene.

This makes the *random-decision stream* bit-identical across ``batch_size``,
compaction on/off, and — for a fixed ``(ray_id, bounce, slot)`` — between the
NumPy and Torch backends, verified by a fixed-vector conformance suite
(``tests/nonsequential/test_nsq_rng_conformance.py``) any third-party backend
can run to prove conformance.

**Honest scope of the guarantee.** Same random decisions, same code path;
*final floating-point results* agree only to documented tolerance across
backends, because arithmetic order, FMA usage, and transcendental
implementations differ between NumPy, Torch CPU, and Torch CUDA. Do not
expect bit-identical detector maps across backends — expect bit-identical
*decisions* and numerically close maps.

.. _nsq_diff_contract:

6. Differentiability — the contract
-------------------------------------

Fresnel splitting is the crux: a ray must stochastically *either* reflect *or*
refract, yet the choice must not block gradients. NSQ uses a
**detached-sample / attached-weight** estimator
(:mod:`optiland.nonsequential.components.refractive`), and the same pattern
covers every other stochastic branch in the engine: the BSDF
reflect/transmit lobe split (``LambertianBSDF.transmissive_fraction``,
D-5), rare-path importance biasing (see the sampling-policy section above), and Russian roulette.

.. admonition:: Theory box — detached-sample / attached-weight
   :class: note

   Let :math:`R` be the (attached, differentiable) Fresnel reflectance. The
   *branch* is sampled from a **detached** copy :math:`\hat R = \operatorname{detach}(R)`:

   .. math::

      b \sim \text{Bernoulli}(\hat R), \qquad
      w =
      \begin{cases}
        R \,/\, \hat R & \text{if } b = \text{reflect} \\[4pt]
        (1-R) \,/\, (1-\hat R) & \text{if } b = \text{refract}
      \end{cases}

   In the forward pass :math:`w = 1` exactly (the value cancels), so the
   estimator is **unbiased**: :math:`\mathbb{E}[w] = \hat R\,(R/\hat R) +
   (1-\hat R)\,((1-R)/(1-\hat R)) = 1`. But :math:`w` carries
   :math:`\partial w/\partial R`, so the throughput weight applied to ray flux
   propagates ``∂flux/∂R`` into material and geometry parameters. (TIR rays use
   :math:`w \equiv 1`; full reflection is deterministic.)

   **What this does *not* capture:** the detached sampling decision means the
   *which-branch / which-surface* (visibility) choice contributes **zero**
   gradient. Silhouette, vignetting, and occlusion boundaries are therefore not
   differentiated. Correcting this needs warped-area reparameterization
   (Loubet et al. 2019; Bangaru et al. 2020); see
   :ref:`nsq_limitations_and_roadmap`. For the constant-memory replacement of
   the naive autograd graph, see Path Replay Backpropagation (Vicini et al.
   2021).

The estimator strategy is exposed through a ``gradient_mode`` seam on
:class:`~optiland.nonsequential.backends.torch_backend.TorchBackend`
(``"autograd"`` today; PRB is the planned second mode). The naive autograd
graph is what imposes the ``O(num_rays × max_depth)`` memory scaling.

.. _nsq_coatings:

7. Coatings, mirror reflectance, and absorption
--------------------------------------------------

**Coatings** on a refractive surface come from ``optiland.coatings`` directly
— attach one via ``SurfaceConfig(coating=...)`` on a ``LensConfig``/
``DoubletConfig`` face, or pass ``coating=`` to ``RefractiveComponent``
directly. This is the same model class the sequential engine uses, so R/T
agree between the two engines by construction (asserted by a dedicated
cross-engine test). An unpolarized coating (``SimpleCoating``) is required;
a polarization-sensitive coating (``BaseCoatingPolarized`` — Jones-matrix
based) raises ``NotImplementedError`` naming the coating and surface rather
than being silently degraded to its scalar average. With no coating
attached, a refractive surface falls back to bare Fresnel, as before.

**Mirrors** have no implicit reflectance. ``MirrorConfig.reflectance`` (or
``SurfaceConfig.reflectance`` for a per-surface override) is *required*: a
constant, a ``callable(wavelength_um) -> reflectance``, or an unpolarized
coating. Constructing a mirror without one raises, so nobody silently gets a
100% reflector.

**Bulk absorption** follows Beer-Lambert:
``flux *= exp(-4*pi*k*L/wavelength_um)`` over the geometric path length ``L``
through a glass whose extinction coefficient ``k`` (from
``NSQMaterial.k()``) is nonzero — automatic once a lens/doublet uses an
absorbing material, no separate configuration. Absorbed flux is tracked
separately as ``SimulationResult.total_flux_bulk_absorbed`` (distinct from
surface-``AbsorbingComponent`` absorption) so the flux ledger still closes.
Both ``k`` and the geometric path length are differentiable.

.. _nsq_volumes:

8. Volumes and watertightness validation
-------------------------------------------

Medium sidedness (which index a ray is *entering* vs *leaving*) is resolved
**geometrically**, not by comparing index values: each geometry's
intersection returns an unflipped geometric normal ``n_geom`` (pointing out
of the solid), and ``entering = (direction · n_geom) < 0`` decides the side.
This replaced an index-proximity heuristic that silently misclassified
index-matched or nearly-matched interfaces (cemented doublets, oil
immersion) — the fix is direction-agnostic *by construction*.

:class:`~optiland.nonsequential.components.volume.Volume` is a
construction-time correctness check built on top of that fix: it validates
that a compound component's boundary surfaces actually close up (every rim
meets a neighbour's rim within ``WATERTIGHT_TOL``) and are consistently
outward-oriented (a ray-parity test from the interior centroid), raising
:class:`~optiland.nonsequential.components.volume.NonWatertightVolumeError`
at construction rather than letting a geometry gap leak flux silently at
trace time. ``Lens``, ``Doublet``, and ``Mirror`` build a ``Volume``
internally; ``NSQScene.add_lens``/``add_doublet``/``add_mirror`` keep their
existing signatures.

Each ray also carries a runtime medium-nesting stack
(``NSQRayBundle.medium_stack``/``medium_depth``), pushed and popped by
``RefractiveComponent.interact`` alongside ``n_current`` on every
transmitted ray. This never feeds back into n1/n2 -- the geometric
resolution above stays the sole source of truth -- it is a cross-check: a
pop attempted on an empty stack (a ray exiting a volume it never entered)
is counted in ``SimulationResult.diagnostics.medium_stack_underflows``
rather than raised, so one bad ray does not abort an otherwise-good trace.
A nonzero count usually means two separately constructed ``NSQMaterial``
instances stand in for what should be one physical medium, or a genuine
geometry defect.

.. _nsq_detectors:

9. Differentiable detectors
-----------------------------

Detectors join ``scene.surfaces`` in the single intersection pass — there is
no separate, duplicated nearest-hit dispatch. Every detector config has an
``absorb`` field (default ``True``); setting ``absorb=False`` makes the
detector **transmissive**: the hit is recorded and the ray continues on its
unchanged direction, enabling mid-system beam sampling (e.g. tilting a
detector into a converging beam without terminating it).

Detectors accumulate flux into a pixel grid via **splatting**, controlled by
the ``splat`` / ``splat_sigma`` config fields:

- ``splat="bilinear"`` (default) — differentiable w.r.t. landing position *and*
  flux; distributes each ray's flux across the four nearest pixels.
- ``splat="hard"`` — nearest-pixel binning; not differentiable in landing
  position. Used by the forward NumPy path.
- ``splat="gaussian"`` — a true, differentiable Gaussian splat (width
  ``splat_sigma``, in pixels), renormalised at the truncation radius so no
  energy is lost.

Accumulation uses ``index_add`` / scatter so it stays in the autograd graph;
the detector's stored ``.data`` tensor is attached, and so is
``IrradianceMap.total_flux`` (a backend array, not a Python float) — a
natural loss expression like ``result.detectors["D1"].total_flux`` carries
gradients. A separate ``total_flux_float`` property is available for
printing. Results (:class:`~optiland.nonsequential.results.IrradianceMap`,
etc.) are read out after the trace.

.. _nsq_diagnostics:

10. Diagnostics
-----------------

``SimulationResult.diagnostics``
(:class:`~optiland.nonsequential.diagnostics.Diagnostics`) is computed during
the trace at negligible extra cost (a few running counters, one pass over
detector results) and turns several previously-silent failure modes into
explicit, inspectable data: depth-truncated flux fraction, Russian-roulette
loss fraction, flux-conservation error, components no ray ever hit, and
per-detector sampling quality (mean hits/pixel, and the ray count needed for
~5% relative error if undersampled). ``result.report()`` renders it as text
with a threshold-based warning list; ``repr(result)`` shows a warning count.

A companion introspection test (the ignored-config audit,
``tests/nonsequential/test_nsq_diagnostics.py``) asserts that every field of
every ``*Config`` dataclass is actually consumed somewhere in the lowering
path — a config field that is accepted and silently ignored is a test
failure, not a discovered-in-production surprise.

.. _nsq_materials:

11. Materials adapter and photometric units
-----------------------------------------------

:class:`~optiland.nonsequential.materials.NSQMaterial` is a **thin adapter** over
``optiland.materials.BaseMaterial`` — *not* a value type. Its ``n(wavelength_um)``
returns the index as a ``be``-backed value with **no grad-severing casts** (no
``float()``, no ``np.asarray()``), so dispersion parameters stay differentiable
under the Torch backend. ``k(wavelength_um)`` (extinction coefficient) feeds
Beer-Lambert absorption (see Coatings, mirror reflectance, and absorption above). Wavelengths are in **micrometres**.

- Vacuum is ``VACUUM`` (``n = 1``, ``k = 0``;
  ``NSQMaterial(optiland_material=None)``).
- Catalog glass via ``NSQMaterial.from_glass("N-BK7")``, which resolves through
  ``optiland.materials.Material(name)``.
- A constant-index material (no dispersion formula) is
  ``NSQMaterial(optiland_material=IdealMaterial(n=1.5))`` — useful when you
  want a deterministic index without touching the glass catalog, e.g. in
  performance benchmarking.

The trace loop itself is **radiometric** (watts) throughout.
:mod:`optiland.nonsequential.units` is a conversion layer on top: sources may
be specified in lumens (``total_flux_lumens``, converted once at
scene-construction time via ``lumens_to_watts``), and
``to_photometric(result.detectors["D1"], quantity="illuminance")`` converts a
traced (radiometric) detector result to lux/lumens by integrating the
trace's spectral content against a CIE ``v_lambda`` curve. Converting a
monochromatic result outside the visible band, or a spectrum with negligible
V(λ) overlap, **raises** rather than silently returning ~0 — the same
"loud failure over silent wrong answer" policy as everywhere else in NSQ.

.. _nsq_extensions:

12. Extension recipes
------------------------

All new components must be written through ``import optiland.backend as be`` so
they work on both backends. Cross-link
:ref:`extension_recipes` for the general pattern.

**Add a geometry.** Subclass the geometry base and implement intersection and
surface-normal in ``be`` ops, returning the ray-facing normal *and* the
unflipped geometric normal ``n_geom`` (see Volumes and watertightness validation above). For differentiability, keep the
intersection distance ``t`` and ``normals`` attached to geometry parameters
(analytic conics do this; see
:mod:`optiland.nonsequential.components.geometry.analytic.conic`). Mesh
geometry is forward-only.

**Add a BSDF.** Subclass :class:`~optiland.nonsequential.bsdf.base.BaseBSDF` and
implement ``sample()`` returning ``(directions, flux_weights, transmitted)``.
Follow the same contract as Fresnel: **detach the sampled direction and the
reflect/transmit hemisphere choice**, keep the **weight attached** so flux
gradients flow (see :mod:`optiland.nonsequential.bsdf.lambertian` — its
``transmissive_fraction`` parameter is the reference implementation of a
detached branch draw with an attached weight). Implement ``reflectance()``
for the total hemispherical reflectance used by flux bookkeeping.

**Add a detector.** Subclass :class:`~optiland.nonsequential.detectors.BaseDetector`,
accumulate via the splat helpers, support the ``absorb`` flag, and expose a
result object with an attached ``total_flux``.

**Add a backend.** Subclass
:class:`~optiland.nonsequential.backends.base.TracerBackend` and interpret
``SceneIR`` (see Architecture above) rather than reaching into ``BaseComponent`` subclasses
directly — a conformance test (drift guard) asserts that every ``BsdfIR`` /
``MediumIR`` variant is handled, so a backend that forgets a variant fails a
test rather than silently mishandling it at runtime. This is the seam for
Dr.Jit / Mitsuba / OptiX.

.. _nsq_serialization:

13. Serialization
--------------------

:mod:`optiland.nonsequential.serialization` writes a **versioned scene JSON**
(``nsq_schema_version``, currently ``1`` — NSQ has never shipped in a tagged
release, so there is exactly one schema and no migration machinery):

- Tensors are stored **by value** (``detach().cpu()``); ``requires_grad`` is not
  persisted — reload, then re-mark leaves for optimization.
- Mesh geometry is referenced **by file path**, not embedded.
- **Results are not serialized** — re-trace to regenerate them.
- A scene whose ``nsq_schema_version`` does not match the current loader is
  refused with a version-mismatch error naming both versions.

.. _nsq_convert:

14. Converting from a sequential ``Optic``
----------------------------------------------

:func:`~optiland.nonsequential.convert.sequential_to_nonsequential` builds an
``NSQScene`` from an existing sequential ``Optic`` (singlets → ``Lens``,
doublets → ``Doublet``, image surface → ``IrradianceDetector``), carrying
over unpolarized coatings and (defaulted where necessary) mirror
reflectance. The returned scene's ``scene.conversion_report`` is a
:class:`~optiland.nonsequential.convert.ConversionReport` — structured data
listing exactly what was carried over, defaulted, estimated, or dropped
(coated/uncoated surfaces, defaulted mirror reflectance, estimated
apertures, dropped polarization), rather than requiring the caller to parse
warning text.

.. _nsq_units:

15. Units convention
------------------------

.. warning::

   The single biggest source of confusion:

   - **Wavelengths** are in **micrometres (µm)** everywhere, including
     ``SpectralDetectorConfig.wl_min``/``wl_max``.
   - **Positions / lengths** are in **millimetres (mm)**.
   - **Rotation angles** (``rx``, ``ry``, ``rz``) are in **radians**.
   - The trace loop is **radiometric** (watts); photometric quantities
     (lumens, lux) go through the conversion layer described in the materials/units section above.

References
----------

- Loubet, Holzschuch & Jakob, *Reparameterizing Discontinuous Integrands for
  Differentiable Rendering*, SIGGRAPH Asia 2019.
- Bangaru, Li & Durand, *Unbiased Warped-Area Sampling for Differentiable
  Rendering*, SIGGRAPH Asia 2020.
- Vicini, Speierer & Jakob, *Path Replay Backpropagation*, SIGGRAPH 2021.
