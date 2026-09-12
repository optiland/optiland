Materials
=========

This section provides a comprehensive overview of the materials available in Optiland.
A material is a set of properties that define the optical behavior of a medium. Specifically in Optiland, a material defines the refractive
index and extinction coefficient of a medium at any wavelength. A material may be described simply by refractive index and abbe number, or it may be a more complex
model based on Sellmeier, or other, coefficients. Optiland provides a wide range of materials, which are outlined here.

Optiland includes a database of common materials based on refrativeindex.info. Any material in this database
can be accessed by name via the Material class.

Abbe Material Model
-------------------

The :class:`~optiland.materials.abbe.AbbeMaterial` class utilizes a robust, data-driven statistical dispersion model to resolve the ambiguity of the two-parameter ($n_d, V_d$) definition. While standard approximations like the "Normal Line" rule often fail for anomalous glasses, Optiland's model is derived from a principal component analysis (PCA) of over 1,000 commercial optical glasses.

The model construction involves:
1.  **Dimensionality Reduction:** Using PCA to quantify the effective degrees of freedom in standard optical glasses.
2.  **Basis Selection:** Applying Sparse Regression (LassoLarsIC) to the Buchdahl dispersion formula to identify the minimum set of coefficients required for accurate spectral reconstruction.

This approach allows for accurate refractive index prediction across the visible and near-infrared spectrum, even for glasses that deviate from the standard "normal line".

For a detailed walkthrough of the model derivation and validation, please refer to the :doc:`Abbe Material Model Building <../references/AbbeMaterial_Model_Building>` notebook.

Parameter Ownership
~~~~~~~~~~~~~~~~~~~

``AbbeMaterial`` selects the d-line polynomial or Buchdahl model;
``AbbeMaterialE`` selects the e-line Buchdahl model. Importers and glass
optimization workflows construct these wrappers when index/Abbe data is the
available description. Tracing, plotting and native persistence use the same
material interface as other optical materials.

The selected model owns the live ``index`` and ``abbe`` arrays. The wrapper's
properties expose those exact arrays, so assignment or in-place mutation through
either object changes the same optical parameters. Serialization reads those
parameters. A small internal mixin shares this delegation between the two
wrappers; it implements no dispersion equations and is not a material type.

The prediction models own the fitted equations and reference-line conventions.
Each fresh prediction recomputes the small derived coefficient vector from the
current parameters, retaining fresh Torch graphs after backward or a ``no_grad``
query. The polynomial model loads its fixed fit matrix once on construction.
The material cache tracks model inputs and fitted constants, rather than treating
previously computed coefficients as independent optical inputs.

Catalog-Scoped Lookup
---------------------

:class:`~optiland.materials.material.Material` accepts an optional ``catalog=``
keyword (e.g. ``"schott"``, ``"ohara"``) to restrict lookup to a specific
manufacturer.  The ``match_policy`` keyword controls fuzzy-match behavior:

.. code-block:: python

   from optiland.materials import Material, MatchPolicy

   glass = Material("N-BK7", catalog="schott")              # exact-catalog lookup
   glass = Material("N-BK7", match_policy=MatchPolicy.BEST) # silent fuzzy
   glass = Material("N-BK7", catalog="schott",
                    match_policy=MatchPolicy.STRICT)         # raise on non-exact

Discovery and User Catalogs
----------------------------

:class:`~optiland.materials.catalog.MaterialCatalog` is a read-only view into
any registered catalog:

.. code-block:: python

   from optiland.materials import MaterialCatalog

   MaterialCatalog.available()              # list all catalogs
   MaterialCatalog("schott").list()         # all Schott glass names
   MaterialCatalog("schott").search("bk7") # fuzzy search within catalog
   MaterialCatalog("schott").get("N-BK7")  # returns a Material instance

:class:`~optiland.materials.registry.MaterialRegistry` is the global singleton
that manages built-in and user-registered materials:

.. code-block:: python

   from optiland.materials import MaterialRegistry

   reg = MaterialRegistry.instance()
   reg.register("MyGlass", "internal", yaml_payload_dict)  # programmatic
   reg.register_file("path/to/my_glass.yml")               # single YAML file
   reg.load_catalog("~/.optiland/catalogs/my_company/")    # directory

User YAML files must follow the `refractiveindex.info <https://refractiveindex.info>`_
format.  Files placed under ``~/.optiland/catalogs/<catalog_name>/`` are
auto-discovered on the first registry access.

Evaluation and Caching
----------------------

All optical material classes implement the
:class:`~optiland.materials.base.BaseMaterial` property interface. Surface groups,
interaction models, propagation, thin-film calculations and the nonsequential
material adapter query ``n`` and ``k``. Those consumers retain responsibility for
ray transport and attenuation; materials report optical properties.

``BaseMaterial`` shares one evaluation/cache path between ``n`` and ``k``. Cache
validity includes the wavelength values, shape and dtype, backend execution
context, keyword arguments and optical state. Ideal, Abbe and file/catalog materials
track their live parameter arrays; changing a parameter invalidates old results.
Their calculations convert parameters for the active backend without replacing
the original arrays, preserving existing Torch parameter identity and gradients.

``IdealMaterial.n`` and ``IdealMaterial.k`` retain the dtype of their stored index
and extinction coefficient, including for array-valued wavelength queries.
Changing the default backend precision after creating the material does not
recast these parameters or their returned values. Conversion still follows the
active backend and device. This preserves mixed-precision calculations such as
object-space NA with a float64 material and a Python scalar aperture value.

Each concrete custom material opts into caching by implementing ``_cache_state``.
An immutable model may return ``()``. A mutable model can use
``self._state_key((self.parameter, ...))`` for all numerical state used by its
calculations. Return ``None`` for untracked state or trainable parameters.
Subclasses that add behavior must explicitly implement the hook even when their
parent implements it. Without that hook, material evaluation remains supported
and executes afresh. Keep the numerical implementation in ``_calculate_n/k``;
do not duplicate cache handling in the public methods.

Caller-owned wavelength buffers are inspected on every lookup. A NumPy alias can
modify Torch storage without incrementing its version, and inference tensors can
also be changed in place. Content checking is therefore O(N), including on cache
hits. Uniform, non-trainable queries still evaluate one wavelength and return a
broadcast result; trainable queries retain elementwise evaluation so every
wavelength receives its own derivative. Cached constants are bypassed before
lookup when an input or tracked parameter requires a fresh gradient graph.

.. autosummary::
   :toctree: materials/
   :caption: Material Modules

   materials.base
   materials.abbe
   materials.ideal
   materials.material_file
   materials.material
   materials.material_spec
   materials.material_utils
   materials.catalog
   materials.registry
