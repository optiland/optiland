.. _api_nonsequential:

Nonsequential Ray Tracing API
=============================

Public API reference for the non-sequential (NSQ) engine. For the conceptual
overview and tutorials see the :ref:`NSQ gallery <gallery_nonsequential>`; for
the capability envelope and known limitations see
:ref:`nsq_limitations_and_roadmap`; for the architecture and
differentiability contract see the
:ref:`developer guide <nonsequential_raytracing>`.

.. note::

   **Pre-release.** NSQ has never shipped in a tagged Optiland release, so
   the API may still change without a deprecation cycle. Scenes are built
   almost entirely through the ``*Config`` dataclasses — the
   :ref:`Configuration reference <nsq_config_reference>` below gives a
   parameter table for each.

.. contents:: On this page
   :local:
   :depth: 2

Scene & tracer
--------------

.. autoclass:: optiland.nonsequential.scene.NSQScene
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: optiland.nonsequential.tracer.NSQTracer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: optiland.nonsequential.tracer.SimulationResult
   :members:
   :undoc-members:
   :show-inheritance:

Sources
-------

.. automodule:: optiland.nonsequential.sources
   :members:
   :undoc-members:
   :show-inheritance:

Components
----------

.. automodule:: optiland.nonsequential.components
   :members:
   :undoc-members:
   :show-inheritance:

Geometry
--------

.. automodule:: optiland.nonsequential.components.geometry
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: optiland.nonsequential.components.geometry.analytic.AnnularPlaneGeometry
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: optiland.nonsequential.components.geometry.analytic.CylindricalFrustumGeometry
   :members:
   :undoc-members:
   :show-inheritance:

BSDF and scattering
-------------------

.. automodule:: optiland.nonsequential.bsdf
   :members:
   :undoc-members:
   :show-inheritance:

Detectors
---------

.. automodule:: optiland.nonsequential.detectors
   :members:
   :undoc-members:
   :show-inheritance:

Results
-------

.. automodule:: optiland.nonsequential.results
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostics
-----------

.. automodule:: optiland.nonsequential.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Photometric units
------------------

.. automodule:: optiland.nonsequential.units
   :members:
   :undoc-members:
   :show-inheritance:

Materials
---------

.. automodule:: optiland.nonsequential.materials
   :members:
   :undoc-members:
   :show-inheritance:

Backends
--------

.. automodule:: optiland.nonsequential.backends
   :members:
   :undoc-members:
   :show-inheritance:

Scene IR, sampling policy, and RNG
-------------------------------------

The declarative, backend-portable scene description (see the
:ref:`developer guide <nsq_arch>`), the rare-path sampling policy, and the
counter-based PCG32 RNG.

.. automodule:: optiland.nonsequential.ir
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: optiland.nonsequential.rng
   :members:
   :undoc-members:
   :show-inheritance:

Serialization
-------------

.. automodule:: optiland.nonsequential.serialization
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

.. autoclass:: optiland.nonsequential.visualization.NSQViewer2D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: optiland.nonsequential.visualization.NSQViewer3D
   :members:
   :undoc-members:
   :show-inheritance:

Converter
---------

.. automodule:: optiland.nonsequential.convert
   :members:
   :undoc-members:
   :show-inheritance:

.. _nsq_config_reference:

Configuration reference
-----------------------

Scenes are built by passing a ``*Config`` dataclass to the corresponding
``scene.add_*`` builder. The tables below list every field, its type, units, and
default. **Differentiable** fields are typed ``float | Tensor``: under the
PyTorch backend they may be passed as ``torch.Tensor`` leaves to receive
gradients via ``loss.backward()``.

Conventions: wavelengths in **micrometres (µm)**, lengths/positions in
**millimetres (mm)**, angles in **degrees** where a field name ends in
``_deg``, otherwise radians.

Sources
~~~~~~~

**PointSourceConfig** — infinitesimal point emitting into a cone.

.. list-table::
   :widths: 22 22 12 44
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``spectrum``
     - ``Spectrum``
     - *(required)*
     - Wavelength distribution (µm).
   * - ``total_flux``
     - float | Tensor
     - ``1.0``
     - Total emitted flux [W].
   * - ``half_angle_deg``
     - float
     - ``90.0``
     - Emission cone half-angle [deg]; 90 = hemisphere, 180 = isotropic.
   * - ``medium``
     - ``NSQMaterial`` | None
     - ``None``
     - Embedding medium (default vacuum).

**CollimatedSourceConfig** — parallel beam.

.. list-table::
   :widths: 22 22 12 44
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``spectrum``
     - ``Spectrum``
     - *(required)*
     - Wavelength distribution (µm).
   * - ``total_flux``
     - float | Tensor
     - ``1.0``
     - Total emitted flux [W].
   * - ``aperture_radius``
     - float | Tensor
     - ``1.0``
     - Beam semi-diameter [mm].
   * - ``profile``
     - str
     - ``"tophat"``
     - Spatial profile: ``"tophat"`` or ``"gaussian"``.
   * - ``gaussian_sigma``
     - float | None
     - ``None``
     - Gaussian sigma [mm]; defaults to ``aperture_radius / 2``.
   * - ``medium``
     - ``NSQMaterial`` | None
     - ``None``
     - Embedding medium (default vacuum).

**ExtendedSourceConfig** — spatially + angularly extended emitter.

.. list-table::
   :widths: 22 22 12 44
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``spectrum``
     - ``Spectrum``
     - *(required)*
     - Wavelength distribution (µm).
   * - ``total_flux``
     - float | Tensor
     - ``1.0``
     - Total emitted flux [W].
   * - ``width``
     - float | Tensor
     - ``1.0``
     - Source width [mm].
   * - ``height``
     - float | Tensor
     - ``1.0``
     - Source height [mm].
   * - ``aperture_radius``
     - float | None
     - ``None``
     - Circular aperture radius [mm]; overrides width/height when set.
   * - ``half_angle_deg``
     - float
     - ``90.0``
     - Emission cone half-angle [deg]. Rays are uniform within the cone
       below 90; at 90 and above they are cosine-weighted over the full
       hemisphere (Lambertian), and values above 90 behave as 90.
   * - ``medium``
     - ``NSQMaterial`` | None
     - ``None``
     - Embedding medium (default vacuum).

Components
~~~~~~~~~~

**LensConfig** — single refractive element (up to four surfaces).

.. list-table::
   :widths: 24 22 12 42
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``r1``
     - float | Tensor
     - *(required)*
     - Front vertex radius of curvature [mm]; + = centre of curvature on +z.
   * - ``r2``
     - float | Tensor
     - *(required)*
     - Back vertex radius of curvature [mm].
   * - ``thickness``
     - float | Tensor
     - *(required)*
     - Centre thickness [mm].
   * - ``material``
     - str | ``NSQMaterial``
     - *(required)*
     - Glass name (e.g. ``"N-BK7"``) or an ``NSQMaterial`` instance.
   * - ``front_aperture_radius``
     - float | Tensor
     - *(required)*
     - Front-face semi-diameter [mm].
   * - ``back_aperture_radius``
     - float | None
     - ``None``
     - Back-face semi-diameter [mm]; defaults to ``front_aperture_radius``.
   * - ``conic1``
     - float | Tensor
     - ``0.0``
     - Front-face conic constant (0 = sphere).
   * - ``conic2``
     - float | Tensor
     - ``0.0``
     - Back-face conic constant.
   * - ``front`` / ``back`` / ``edge`` / ``rim``
     - ``SurfaceConfig`` | None
     - ``None``
     - Per-surface overrides (see SurfaceConfig).

**DoubletConfig** — cemented achromatic doublet.

.. list-table::
   :widths: 24 22 12 42
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``r1``
     - float | Tensor
     - *(required)*
     - Front radius of curvature [mm].
   * - ``r2``
     - float | Tensor
     - *(required)*
     - Cemented-interface radius of curvature [mm].
   * - ``r3``
     - float | Tensor
     - *(required)*
     - Back radius of curvature [mm].
   * - ``thickness1``
     - float | Tensor
     - *(required)*
     - Crown element thickness [mm].
   * - ``thickness2``
     - float | Tensor
     - *(required)*
     - Flint element thickness [mm].
   * - ``material1`` / ``material2``
     - str | ``NSQMaterial``
     - *(required)*
     - Crown / flint glass name or ``NSQMaterial``.
   * - ``aperture_radius``
     - float | Tensor
     - *(required)*
     - Common semi-diameter for all surfaces [mm].
   * - ``conic1`` / ``conic2`` / ``conic3``
     - float | Tensor
     - ``0.0``
     - Conic constants of front / cemented / back faces.
   * - ``front`` / ``cemented`` / ``back`` / ``edge``
     - ``SurfaceConfig`` | None
     - ``None``
     - Per-surface overrides.

**MirrorConfig** — single reflective surface.

.. list-table::
   :widths: 24 22 12 42
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``radius``
     - float | Tensor
     - *(required)*
     - Vertex radius of curvature [mm]; − = concave (normal toward +z).
   * - ``reflectance``
     - float | Callable | ``BaseCoating``
     - *(required)*
     - Mirror reflectance: a constant in [0, 1], a
       ``callable(wavelength_um) -> reflectance``, or an unpolarized
       ``optiland.coatings`` coating. **No implicit default** — a mirror
       built without one raises rather than reflecting 100%.
   * - ``conic``
     - float | Tensor
     - ``0.0``
     - Conic constant (0 = sphere, −1 = paraboloid).
   * - ``aperture_radius``
     - float | Tensor
     - ``25.0``
     - Semi-diameter [mm].
   * - ``surface``
     - ``SurfaceConfig`` | None
     - ``None``
     - Per-surface override (e.g. attach a scatter BSDF or override
       ``reflectance``).

**SurfaceConfig** — optional per-surface overrides within a compound component.
All fields default to ``None`` (use the compound-level default).

.. list-table::
   :widths: 24 28 12 36
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``bsdf``
     - ``BaseBSDF`` | None
     - ``None``
     - Scatter model. Rays routed to it are scattered *instead of* being
       refracted or specularly reflected.
   * - ``scatter_fraction``
     - float
     - ``1.0``
     - Probability that a ray striking this surface is routed through
       ``bsdf`` rather than following the specular/refractive path. The
       default of 1.0 makes the surface a pure diffuser; use a smaller
       value for a partially scattering surface. Ignored when ``bsdf``
       is ``None``.
   * - ``coating``
     - ``BaseCoating`` | None
     - ``None``
     - An ``optiland.coatings`` coating (e.g. an AR coating) for a refractive
       surface; its reflectance/transmittance replace bare Fresnel, so NSQ
       and the sequential engine agree on R. Must be unpolarized
       (``SimpleCoating``) — a polarization-sensitive coating
       (``BaseCoatingPolarized``) raises rather than being silently
       degraded to its scalar average. Ignored on absorbing surfaces.
   * - ``aperture_radius``
     - float | None
     - ``None``
     - Semi-diameter override [mm].
   * - ``interaction``
     - ``InteractionType`` | None
     - ``None``
     - Force a specific interaction (REFRACTIVE / REFLECTIVE / ABSORBING).
   * - ``reflectance``
     - float | Callable | ``BaseCoating`` | None
     - ``None``
     - Required when ``interaction`` selects ``InteractionType.REFLECTIVE``:
       overrides the compound's ``MirrorConfig.reflectance`` for this
       surface only.

Detectors
~~~~~~~~~

**IrradianceDetectorConfig** — 2-D spatial flux map.

.. list-table::
   :widths: 22 26 12 40
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``width`` / ``height``
     - float
     - *(required)*
     - Detector extent [mm].
   * - ``num_pixels_x`` / ``num_pixels_y``
     - int
     - ``256``
     - Pixel grid resolution.
   * - ``splat``
     - ``"bilinear"`` | ``"gaussian"`` | ``"hard"``
     - ``"bilinear"``
     - Splatting mode. ``bilinear`` is differentiable in landing position;
       ``hard`` is nearest-pixel (not differentiable in landing position);
       ``gaussian`` is a true, differentiable Gaussian splat of width
       ``splat_sigma``.
   * - ``splat_sigma``
     - float
     - ``0.5``
     - Gaussian splat sigma in pixels (used when ``splat="gaussian"``).
   * - ``absorb``
     - bool
     - ``True``
     - Whether a hit terminates the ray. ``False`` makes the detector
       transmissive: the hit is recorded and the ray continues unchanged,
       enabling mid-system beam sampling.

**SpectralDetectorConfig** — per-wavelength irradiance.

.. list-table::
   :widths: 22 26 12 40
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``width`` / ``height``
     - float
     - *(required)*
     - Detector extent [mm].
   * - ``num_pixels_x`` / ``num_pixels_y``
     - int
     - ``256``
     - Pixel grid resolution.
   * - ``wl_min`` / ``wl_max``
     - float
     - ``0.4`` / ``0.7``
     - Spectral binning range [µm] (same convention as ``Spectrum`` —
       visible light is ``0.4``-``0.7``).
   * - ``num_bins``
     - int
     - ``100``
     - Number of wavelength bins.
   * - ``splat``
     - ``"bilinear"`` | ``"gaussian"`` | ``"hard"``
     - ``"bilinear"``
     - Only ``hard`` binning is currently implemented for this detector.
   * - ``splat_sigma``
     - float
     - ``0.5``
     - Reserved for future use.
   * - ``absorb``
     - bool
     - ``True``
     - Whether a hit terminates the ray.

**FarFieldDetectorConfig** — angular intensity pattern.

.. list-table::
   :widths: 22 14 12 52
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``num_theta``
     - int
     - ``90``
     - Number of polar-angle bins.
   * - ``num_phi``
     - int
     - ``360``
     - Number of azimuthal-angle bins.
   * - ``absorb``
     - bool
     - ``True``
     - Whether a hit terminates the ray.

**RayDatabaseConfig** — full phase-space record.

.. list-table::
   :widths: 22 14 12 52
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``width`` / ``height``
     - float
     - *(required)*
     - Detector extent [mm].
   * - ``max_rays``
     - int
     - ``0``
     - Maximum rays to store (0 = unlimited).
   * - ``absorb``
     - bool
     - ``True``
     - Whether a hit terminates the ray.
