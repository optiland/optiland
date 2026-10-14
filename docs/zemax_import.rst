Zemax import: glass resolution
==============================

``load_zemax_file("design.zmx")`` resolves each glass the way OpticStudio
does: from the catalogs the file declares on its ``GCAT`` line, by exact name.
It never falls back to a fuzzy search across every catalog in the material
database, so an imported lens gets the glass the file asks for or a model of
it, never a different glass with a similar name.

Resolution order
----------------

For each ``GLAS <name> <solve> <?> <nd> <vd> ...`` line:

1. **Model glasses.** A solve type of ``1`` or the name ``___BLANK`` marks an
   explicit model glass. It becomes an ``AbbeMaterial`` built from nd/vd,
   without searching the catalogs.
2. **Declared catalogs, exact name.** Every ``GCAT`` catalog is searched for
   the exact name (case insensitive), with
   ``Material(name, catalog=..., match_policy="strict")``.

   * Found in one catalog: that glass is used.
   * Found in several: the glass whose nd/vd best match the ``GLAS`` line is
     used, and the first catalog in ``GCAT`` order wins a tie or a line
     without nd/vd. A ``.zmx`` file lists catalogs once for the whole file, so
     nd/vd are what tell, for example, Schott F2 from CDGM F2 on a given
     surface.
3. **MIL-number names.** A six-digit name such as ``517642`` (or
   ``517.642``) that no catalog contains is a model glass with
   nd = 1.517 and vd = 64.2, taken from the name.
4. **Not found.** Any other glass becomes an ``AbbeMaterial`` from the nd/vd
   on the ``GLAS`` line, and an ``OptilandMaterialWarning`` names the surface
   and glass. If the line has no nd/vd, import fails with ``ValueError``.

A glass modelled with vd = 0 is undispersed and becomes an ``IdealMaterial``.

A ``GCAT`` catalog that is not in the material database (see
``MaterialRegistry.instance().list_catalogs()``) is skipped with a warning, so
its glasses are modelled from nd/vd. A file without ``GCAT`` models every
glass.

Glasses from AGF catalogs
-------------------------

OpticStudio evaluates each glass from its AGF catalog entry: the dispersion
formula code and the coefficients on its ``CD`` line. The material database
holds refractiveindex.info data instead, which can differ in fit, wavelength
range or catalog edition, and it has no vendor, custom or in-house catalogs.
To use the AGF files you already have, pass the folders that hold them:

.. code-block:: python

   from optiland.fileio import load_zemax_file

   optic = load_zemax_file("design.zmx", agf_dirs=["C:/Users/me/Documents/Zemax/Glasscat"])

or list them in the ``OPTILAND_AGF_PATH`` environment variable, separated as
in ``PATH``. With AGF folders, step 2 above changes for each ``GCAT``
catalog that has an AGF file (``SCHOTT`` as ``SCHOTT.AGF``, ignoring case):

* The catalog is searched in its AGF file by exact name (case insensitive),
  not in the material database, and a glass found there becomes an
  :class:`~optiland.materials.agf.AGFMaterial`. A glass the AGF file lacks is
  not looked up in the database's catalog of the same name; it goes on to
  steps 3 and 4, as in OpticStudio.
* A catalog without an AGF file uses the material database as before.
* When the folders hold the same catalog more than once, the first folder
  wins.

``agf_dirs=[]`` switches the AGF lookup off even when ``OPTILAND_AGF_PATH``
is set. Without AGF folders, glasses resolve exactly as described above.
Optiland ships no AGF data.

What changed
------------

Earlier versions fell back to ``Material(name)``, which searches every catalog
with substring and edit-distance matching. Some files now resolve to a
different material:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - ``GLAS`` under ``GCAT SCHOTT``
     - Before
     - Now
   * - ``BAF2``
     - The BaF\ :sub:`2` crystal (``main/BaF2``)
     - Model glass from nd/vd, with a warning
   * - ``SF5``
     - ``N-SF5``, the lead-free replacement
     - Model glass from nd/vd, with a warning
   * - ``K5``
     - ``N-K5``, a different glass
     - Model glass from nd/vd, with a warning

Glasses that a declared catalog contains under their exact name resolve as
before. ``Material(name)`` itself is unchanged; only the Zemax importer
resolves strictly.

Export
------

``save_zemax_file`` writes the catalog of each ``Material`` and
``AGFMaterial`` to ``GCAT`` and records nd/vd on every ``GLAS`` line, so a
saved file reloads to the same glasses. A ``Material`` created without a reference exports the glass catalog
it resolved to.
