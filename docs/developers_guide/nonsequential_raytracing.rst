.. _nonsequential_raytracing:

Non-Sequential Ray Tracing
==========================

This guide is written for **contributors who extend the non-sequential (NSQ)
engine** — adding geometries, BSDFs, detectors, or a new execution backend. It
documents the architecture, the differentiability contract, and the extension
seams. For *user*-facing tutorials see the
:ref:`NSQ gallery <gallery_nonsequential>`; for the API reference see
:ref:`api_nonsequential`; for the v1 envelope and known gaps see
:ref:`nsq_limitations_and_roadmap` (canonical).

.. note::

   **Beta.** The NSQ engine is differentiable and torch-native. The public API
   is stabilizing toward a frozen 1.0. This page is the canonical engine
   reference; see also :doc:`/gallery/nonsequential/limitations_and_roadmap`.

.. _nsq_arch:

1. Architecture — description vs. execution
-------------------------------------------

NSQ separates **what a scene is** from **how it is traced**:

- :class:`~optiland.nonsequential.scene.NSQScene` is a *declarative* container.
  It holds sources, components, and detectors with their parameters and
  coordinate systems. It performs no ray math.
- A :class:`~optiland.nonsequential.backends.base.TracerBackend` *executes* the
  scene. It owns the Monte Carlo loop, intersection, interaction, and detector
  accumulation.

This seam is the reason the same scene can be traced three ways without
changing scene-building code:

- **today** on NumPy (fast forward path),
- **now** on PyTorch (differentiable),
- **later** on a high-performance backend (Dr.Jit / Mitsuba) via scene
  translation — only a new ``TracerBackend`` is required.

:class:`~optiland.nonsequential.tracer.NSQTracer` is a thin coordinator: it
selects a backend (or accepts one explicitly) and delegates the loop.

.. _nsq_backends:

2. Backends
-----------

The active backend is chosen from ``optiland.backend`` unless one is passed
explicitly to :class:`~optiland.nonsequential.tracer.NSQTracer`.

**NumpyBackend** (forward, production)
   Raw-NumPy outer loop (does *not* go through ``optiland.backend``), with live
   **ray compaction** (dead rays are dropped between bounces) and hard pixel
   binning. Designed for 1e7+ rays at ``max_depth`` 16. This is the engine for
   real illumination and stray-light work.

**TorchBackend** (differentiable)
   ``be``-unified array ops that build a full PyTorch autograd graph through the
   Monte Carlo loop. Compaction is **disabled** so tensor shapes stay fixed for
   the graph, which is why memory scales as ``O(num_rays × max_depth)`` and the
   practical envelope is ~1e5 rays at depth 16 on a single GPU. This is the
   engine for optimization and ML layers.

There is **no separate GPU array backend.** GPU acceleration in the
differentiable path comes from PyTorch device placement; the forward NumPy path
is CPU-batched.

.. _nsq_megakernel:

3. Wavefront megakernel
-----------------------

The loop is **breadth-first per bounce**: at each depth the backend intersects
the entire live wavefront against all components and detectors, applies
interactions, then advances. Termination is a deterministic
``max_depth`` cap (plus a flux-fraction kill threshold) — termination is fixed
and deterministic, **not** a probabilistic survival decision. Rays that exceed
``max_depth`` are counted in ``SimulationResult.num_rays_depth_killed``.

.. code-block:: python

   result = scene.trace(num_rays=1_000_000, max_depth=16, seed=42)

.. _nsq_diff_contract:

4. Differentiability — the contract
-----------------------------------

Fresnel splitting is the crux: a ray must stochastically *either* reflect *or*
refract, yet the choice must not block gradients. NSQ uses a
**detached-sample / attached-weight** estimator
(:mod:`optiland.nonsequential.components.refractive`).

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
   differentiated in v1. Correcting this needs warped-area reparameterization
   (Loubet et al. 2019; Bangaru et al. 2020); see
   :ref:`nsq_limitations_and_roadmap`. For the constant-memory replacement of
   the naive autograd graph, see Path Replay Backpropagation (Vicini et al.
   2021).

The estimator strategy is exposed through a ``gradient_mode`` seam on
:class:`~optiland.nonsequential.backends.torch_backend.TorchBackend`
(``"autograd"`` today; PRB is the planned second mode). The naive autograd
graph is what imposes the ``O(num_rays × max_depth)`` memory scaling.

.. _nsq_detectors:

5. Differentiable detectors
---------------------------

Detectors accumulate flux into a pixel grid via **splatting**, controlled by the
``splat`` / ``splat_sigma`` config fields:

- ``splat="bilinear"`` (default) — differentiable w.r.t. landing position *and*
  flux; distributes each ray's flux across the four nearest pixels.
- ``splat="hard"`` — nearest-pixel binning; not differentiable in landing
  position. Used by the forward NumPy path.
- ``splat="gaussian"`` — currently falls back to bilinear; full Gaussian splat
  is a TODO.

Accumulation uses ``index_add`` / scatter so it stays in the autograd graph; the
detector's stored ``.data`` tensors are attached. Results
(:class:`~optiland.nonsequential.results.IrradianceMap`, etc.) are read out
after the trace.

.. _nsq_materials:

6. Materials adapter
--------------------

:class:`~optiland.nonsequential.materials.NSQMaterial` is a **thin adapter** over
``optiland.materials.BaseMaterial`` — *not* a value type. Its ``n(wavelength_um)``
returns the index as a ``be``-backed value with **no grad-severing casts** (no
``float()``, no ``np.asarray()``), so dispersion parameters stay differentiable
under the Torch backend. Wavelengths are in **micrometres**.

- Vacuum is ``VACUUM`` (``n = 1``; ``NSQMaterial(optiland_material=None)``).
- Catalog glass via ``NSQMaterial.from_glass("N-BK7")``, which resolves through
  ``optiland.materials.Material(name)``.

.. _nsq_extensions:

7. Extension recipes
--------------------

All new components must be written through ``import optiland.backend as be`` so
they work on both backends. Cross-link
:ref:`extension_recipes` for the general pattern.

**Add a geometry.** Subclass the geometry base and implement intersection and
surface-normal in ``be`` ops. For differentiability, keep the intersection
distance ``t`` and ``normals`` attached to geometry parameters (analytic conics
do this; see :mod:`optiland.nonsequential.components.geometry.analytic.conic`).
Mesh geometry is forward-only.

**Add a BSDF.** Subclass :class:`~optiland.nonsequential.bsdf.base.BaseBSDF` and
implement ``sample()`` returning ``(directions, flux_weights)``. Follow the same
contract as Fresnel: **detach the sampled direction**, keep the **weight
attached** so flux gradients flow (see
:mod:`optiland.nonsequential.bsdf.lambertian`). Implement ``reflectance()`` for
the total hemispherical reflectance used by flux bookkeeping.

**Add a detector.** Subclass :class:`~optiland.nonsequential.detectors.BaseDetector`,
accumulate via the splat helpers, and expose a result object.

**Add a backend.** Subclass
:class:`~optiland.nonsequential.backends.base.TracerBackend` and implement the
megakernel loop. This is the seam for Dr.Jit / Mitsuba.

.. _nsq_serialization:

8. Serialization
----------------

:mod:`optiland.nonsequential.serialization` writes a **versioned scene JSON**:

- Tensors are stored **by value** (``detach().cpu()``); ``requires_grad`` is not
  persisted — reload, then re-mark leaves for optimization.
- Mesh geometry is referenced **by file path**, not embedded.
- **Results are not serialized** — re-trace to regenerate them.

.. _nsq_units:

9. Units convention
-------------------

.. warning::

   The single biggest source of confusion:

   - **Wavelengths** are in **micrometres (µm)**.
   - **Positions / lengths** are in **millimetres (mm)**.
   - **Rotation angles** (``rx``, ``ry``, ``rz``) are in **radians**.

References
----------

- Loubet, Holzschuch & Jakob, *Reparameterizing Discontinuous Integrands for
  Differentiable Rendering*, SIGGRAPH Asia 2019.
- Bangaru, Li & Durand, *Unbiased Warped-Area Sampling for Differentiable
  Rendering*, SIGGRAPH Asia 2020.
- Vicini, Speierer & Jakob, *Path Replay Backpropagation*, SIGGRAPH 2021.
