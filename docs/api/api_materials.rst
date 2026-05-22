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

.. autosummary::
   :toctree: materials/
   :caption: Material Modules

   materials.abbe
   materials.ideal
   materials.material_file
   materials.material
   materials.material_spec
   materials.catalog
   materials.registry
