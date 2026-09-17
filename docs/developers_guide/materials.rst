.. _material_architecture:

Materials: responsibilities and callers
=======================================

The material interface describes optical properties. A material supplies real
refractive index ``n(wavelength, **kwargs)`` and dimensionless extinction
coefficient ``k(wavelength, **kwargs)``, with wavelength in micrometers. The
propagation model controls travel through the medium; surface interaction models
control refraction, reflection and coatings. Neither belongs in a dispersion
equation. Spatial GRIN propagation remains a separate, unimplemented feature.

Choosing and constructing a material
------------------------------------

.. list-table:: Public material responsibilities
   :header-rows: 1
   :widths: 22 42 36

   * - Type
     - Responsibility and state owner
     - Construction and principal callers
   * - ``BaseMaterial``
     - Property interface, subclass registry, propagation attachment, cache
       context and native deserialization dispatch. No optical equation.
     - Subclassed by material implementations; queried by all runtime consumers.
   * - ``IdealMaterial``
     - Constant index and extinction in live backend arrays. Index 1 is air
       only when extinction is zero.
     - Constructed by users, ``MaterialFactory`` and file readers for constant
       media; used in tracing and index optimization.
   * - ``AbbeMaterial``
     - d-line index/Abbe specification. The selected polynomial or Buchdahl
       model owns the live parameters and prediction equation.
     - Users, glass optimization and readers with index/Abbe data. Polynomial
       evaluation is limited to 0.38–0.75 µm; Buchdahl has no declared limits.
   * - ``AbbeMaterialE``
     - e-line index/Abbe specification using the e-line Buchdahl model.
       Shares parameter delegation with the d-line wrapper.
     - Users and readers with e-line specifications; the same tracing and
       persistence callers as ``AbbeMaterial``.
   * - ``MaterialFile``
     - Existing refractiveindex.info YAML adapter. Owns live coefficients,
       sampled arrays and thermal fields, and retains filename-based JSON.
       Shared helpers evaluate its equations and interpolation.
     - Users with a YAML file and the ``Material`` catalog wrapper. Its normal
       table evaluation defaults to endpoint clamping, including one-sample
       tables; ``bounds="raise"`` opts into strict table support.
   * - ``Material``
     - Catalog resolution, name/reference identity and lookup provenance on top
       of ``MaterialFile``. It does not implement another dispersion engine.
     - ``MaterialFactory``, ``MaterialCatalog.get``, named-glass file readers,
       glass selection and user code.
   * - ``DataMaterial``
     - Owns an immutable, spatially uniform sampled or analytic definition and
       optional independent extinction table. Requires no file, catalog or
       registration. Native JSON
       contains the complete optical definition and copied metadata.
     - ``from_samples``, ``from_coefficients`` or a native definition dictionary;
       OSLO sampled-index input and explicit material overrides. Used by the
       same property, plotting and persistence callers as other materials.

``AbbeModel`` is a prediction interface, not a material. ``AbbePolynomialModel``,
``BuchdahlDModel`` and ``BuchdahlEModel`` own their equations and reference-line
conventions. Fresh predictions derive coefficients from current parameters;
the wrappers expose those exact parameters and provide material persistence.

``MaterialSpec`` and ``MatchPolicy`` describe catalog lookup choices.
``MaterialRegistry`` owns discovery, registered YAML sources and resolution;
``MaterialCatalog`` is its catalog-scoped browsing facade. These are not optical
evaluators. ``MaterialFactory`` converts surface specifications into material
instances: an existing material is accepted directly, and a dictionary there
continues to describe a lookup specification. Restore an inline native object
with ``BaseMaterial.from_dict`` before passing it to a surface.

Owned data and persistence
--------------------------

.. code-block:: python

   from optiland.materials import BaseMaterial, DataMaterial

   measured = DataMaterial.from_samples(
       [0.4, 0.6, 0.8], [1.6, 1.55, 1.5], name="measured glass",
       extinction={
           "kind": "tabulated_k",
           "wavelengths_um": [0.45, 0.75],
           "values": [0.001, 0.002],
       },
       metadata={"source": "laboratory measurement"},
   )
   analytic = DataMaterial.from_coefficients(
       "formula 2", [0, 1.03961212, 0.00600069867,
                     0.231792344, 0.0200179144, 1.01046945, 103.560653],
       wavelength_range=(0.3, 2.5), name="analytic example",
   )
   restored = BaseMaterial.from_dict(measured.to_dict())
   assert restored.definition == measured.definition

The ``definition`` contains a ``dispersion`` tagged as ``tabulated`` or
``formula``, an optional ``extinction`` tagged as ``tabulated_k``, and reserved
``thermal: null``. Formula identifiers are ``formula 1`` through ``formula 9``,
using the existing refractiveindex.info conventions. A formula remains analytic;
it is never replaced with fitted or sampled indices. Negative finite indices
are permitted in the generic material interface. Import/export adapters enforce
their own format's physical restrictions.

Sample arrays are sorted together and require at least two distinct positive
wavelengths. All optical data must be finite; extinction must be nonnegative.
Each table rejects queries beyond its own support by default. Missing extinction means
zero at all finite positive wavelengths. Formula bounds are optional and enforced
when supplied. Scalar evaluation returns a one-element array; array queries keep
their shape and Torch wavelength gradients. Imported coefficients are constants,
not trainable parameters: create a replacement definition to change them.

The frozen definition and its tuple fields cannot be edited. Input containers,
serialized containers and finite JSON metadata are copied; metadata and display
names do not affect optical evaluation. ``BaseMaterial.from_dict`` dispatches
the class and restores the registered propagation model. Existing public
material JSON formats continue to work; there are no aliases for earlier
unreleased material prototypes.

Choosing a table bounds policy
----------------------------------------

``DataMaterial`` accepts a keyword-only ``bounds="raise"`` or ``bounds="clamp"``
on its constructor and both factories. ``MaterialFile`` and its catalog wrapper
``Material`` expose the same option with ``"clamp"`` as their existing default.
The option applies to every tabulated property of the material, with each n/k
table using its own wavelength interval. It does not change the stored samples
or the support reported by ``spectral_range``.

.. code-block:: python

   from optiland.materials import DataMaterial, Material, MaterialFile

   measured = DataMaterial.from_samples(
       [0.4, 0.8], [1.6, 1.4], bounds="clamp",
   )
   assert measured.n(0.9).item() == 1.4
   strict_file = MaterialFile("measurements.yml", bounds="raise")
   strict_catalog = Material("N-BK7", catalog="schott", bounds="raise")

Clamping holds the nearest endpoint value. It is an explicit assumption about
unknown dispersion or absorption outside the supplied interval, not additional
measurement data. The clamped interpolation's wavelength derivative is zero
outside the table; later thermal corrections can still depend on wavelength.
Gradients with respect to live file-table endpoint values remain available.
The default exception for
owned data prevents silently tracing with this assumption. A strict one-sample
file table accepts only its recorded wavelength; clamping makes it constant.

The policy is read-only after construction, is part of material cache state and
is written as ``bounds`` in native material JSON. Construct a replacement to
choose another policy. Old public ``MaterialFile``/``Material`` JSON without this
field still loads with clamping, and new saves record it explicitly. This allows
applications to opt into strict table evaluation now and retain their chosen
behavior if a later breaking release changes the file-material default.

Analytic equations are independent of table interpolation: a declared
``DataMaterial`` formula range is always enforced, while ``MaterialFile`` keeps
its existing analytic extrapolation behavior. On ``from_coefficients``, the
``bounds`` option therefore affects only an optional tabulated extinction curve.
Missing extinction remains the documented zero assumption. Invalid/nonpositive
``DataMaterial`` wavelengths still fail even with clamping enabled.

OSLO can preserve owned index samples but cannot encode this explicit clamping
choice. Its writer rejects clamped ``DataMaterial`` objects before replacing the
destination; use native JSON to retain the policy. Catalog-name exports in
external formats retain the destination program's evaluation conventions.

Shared implementation ownership
-------------------------------

* ``dispersion.py`` owns all nine formula equations and coefficient arity checks.
  Both ``DataMaterial`` and ``MaterialFile`` call it.
* ``buchdahl.py`` owns the Buchdahl coordinate transform and polynomial
  evaluation for any order. The existing three-term Abbe models call it;
  higher-order models can reuse it with their own coefficients. Reference
  wavelengths, alpha, fitted constants, validity limits and coefficient
  generation belong to the individual model, not this algebra module.
* ``spectral.py`` owns finite sample validation and linear interpolation, with
  an explicit bounds policy chosen by the caller. It preserves the constant
  single-sample behavior required by existing YAML data.
* ``definition.py`` owns private immutable records and native definition
  validation/encoding. It contains no file access, backend arrays or cache.
* ``rii.py`` decodes YAML spectral blocks to plain intermediate records.
  ``MaterialFile`` binds those values to its existing live arrays; it does not
  retain a second immutable copy of mutable optical state.
* ``BaseMaterial`` owns property caching. Mutable built-ins report all live
  optical state; ``DataMaterial`` reports its immutable definition and bounds
  policy. New custom
  subclasses evaluate uncached unless they explicitly implement ``_cache_state``.
  Graph-bearing queries bypass result caching.

Runtime and persistence callers
-------------------------------

.. list-table:: Consumer boundaries
   :header-rows: 1
   :widths: 28 72

   * - Consumer
     - Contract and responsibility
   * - Surface groups and paraxial/real ray tracing
     - Query index at the ray wavelength. Material ownership is independent of
       whether rays are traced sequentially or through the nonsequential adapter.
   * - Interaction models and thin-film calculations
     - Query n and k, then own complex-index conventions, Fresnel coefficients,
       layer interference and polarization. Materials do not implement coatings.
   * - ``HomogeneousPropagation``
     - Advances rays and uses extinction for attenuation, and owns
       propagation distance units and absorption conversion.
   * - ``NSQMaterial``
     - Adapts a material to nonsequential tracing and attaches a surface BSDF;
       it delegates optical dispersion instead of duplicating its equations.
   * - Optimization material variables
     - Update live Ideal/Abbe parameters or choose catalog glass. Owned imported
       definitions are constant data and are replaced as complete definitions.
   * - Native JSON
     - Uses ``to_dict`` and ``BaseMaterial.from_dict``. Data definitions are
       self-contained; file/catalog materials retain their public reference format.
   * - OSLO reader/writer
     - Reader constructs sampled ``DataMaterial`` when distinct saved indices
       are present. Writer preserves homogeneous positive n tables with no
       extinction; other owned definitions require native JSON. The label is
       never interpreted as a catalog reference.
   * - Zemax and CODE V writers
     - Reject ``DataMaterial`` before opening the destination. An n/Abbe fit or
       a guessed glass name would lose the imported definition.
       All three external writers preflight both incident and outgoing media
       for custom propagation before air, catalog, model-glass or mirror
       shortcuts. This check does not implement spatial-material export.
   * - Plotting and information display
     - Use ``display_name`` and ``spectral_range('n'/'k')``. ``plot_nk`` intersects
       requested and known limits and requires an explicit range if none are
       known. It plots in µm and converts to host arrays only at the display boundary.

Adding catalog transmission/thermal data or new file encodings requires a
separate feature with its own physical validation and persistence tests. Material
temperature/pressure keywords are not automatically connected to the global
``Optic`` environment; that caller-level change is outside this material rework.

Extending material models
-------------------------

A fitted glass model with live index, Abbe number or partial-dispersion
parameters is a ``BaseMaterial`` implementation (or a prediction strategy
behind a material wrapper). It is not an immutable imported data definition.
Reuse ``buchdahl_coordinate(wavelength, reference_wavelength, alpha)`` and
``evaluate_buchdahl(index, coefficients, omega)`` for Buchdahl models. Pass
coefficients in increasing power order, starting with the linear term.
Different fits can have different orders and reference conventions without
sharing regression assets or becoming the same optical model.

Derive coefficients from the current parameters during evaluation; retaining
a derived tensor graph across optimizer steps or backward calls gives stale
values or invalid gradients. Reuse immutable regression data separately from
that derived state. Use backend operations and preserve parameter and wavelength
gradients. A new subclass is uncached by default; opt in only with a complete
``_cache_state`` including every optical parameter and model choice. Caching
does not refresh coefficients stored by the subclass itself.

Serialize the model's defining parameters in ``to_dict`` alongside
``super().to_dict()``. Its ``from_dict`` reconstructs those parameters;
``BaseMaterial.from_dict`` restores propagation centrally. Do not serialize
derived coefficients as a second authority. Native loading and GUI background
workers must import the material class so its registry entry exists. GUI undo
and document replacement use the same native persistence contract.

Provide ``display_name`` for labels and ``spectral_range('n'/'k')`` for known
validity limits, validating the property name through the base contract.
Information displays use this label for unregistered material families;
explicit formatter registrations and a configured default still take precedence.
Unknown limits remain ``None``; plotting then requires an explicit range.
Consumers must not infer catalog identity from a display label, or assume an
``abbe`` member is numeric: ``BaseMaterial.abbe()`` is a method while the Abbe
wrappers expose a live parameter. Callers needing a spectral-line quantity
should evaluate ``n`` at the stated wavelengths. An optimization variable owns
its choice of model and must preserve all independent model parameters when
updating one of them. Import/export mappings need explicit, tested semantics;
sharing an equation does not make different fitted models interchangeable.

A future spatial material is another ``BaseMaterial`` implementation. It can
compose an existing material as its homogeneous spectral background, without
forcing its field or propagation controls into ``DataMaterial.definition``.
Keep position/environment arguments in ``n/k(..., **kwargs)`` and require the
position needed by the optical law. In an opted-in cache, spatial context must
participate in the key even when all wavelengths are equal.

This extension point does not make tracing, paraxial analysis, wavefront launch
phase, drawings or coatings position-aware. A GRIN implementation must update
those callers or reject unsupported use, and define medium coordinates,
boundary events, index/gradient consistency and a single owner of optical-path
and attenuation integration. Custom propagation must remain explicit so
homogeneous-only exporters can reject it. A spatial law attached to the default
homogeneous propagator is not a supported GRIN configuration.
