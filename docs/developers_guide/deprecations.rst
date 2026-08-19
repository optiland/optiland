.. _deprecations:

Deprecation Removal Tracking
=============================

Optiland's public API is stable: a signature or attribute is never removed silently. When
something is superseded, the old path stays available, emits a ``DeprecationWarning`` pointing
at its replacement, and is tracked here until a maintainer deliberately cuts the release that
removes it. This page is the single place that tracks *what* is deprecated, *what replaces it*,
and *which release* is allowed to remove it — nothing gets removed just because a phase of work
happens to touch the surrounding file.

Current Optiland version: ``0.6.1``.

Pending Removals
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Deprecated API
     - Replacement
     - Target removal
     - Notes
   * - ``Optic.surface_group``
     - ``Optic.surfaces``
     - ``v0.7.0``
     - Property getter/setter both emit ``DeprecationWarning``
       (``optiland/optic/optic.py``). Removal was explicitly scoped to
       ``v0.7.0``; not yet due at the current ``0.6.1`` release.
   * - ``Surface.coating``
     - ``Surface.interaction_model.coating``
     - Not yet committed
     - Getter/setter emit ``DeprecationWarning`` and delegate to
       ``interaction_model.coating`` (``optiland/surfaces/standard_surface.py``).
       No target release has been assigned yet — do not remove until one is
       agreed and this row is updated with it.

How to Use This Table
----------------------

- **Adding a new deprecation:** when you deprecate an API, add a row here in the same PR, with
  either a concrete target release or ``Not yet committed`` if none has been agreed.
- **Cutting a release:** before tagging the release named in the "Target removal" column, grep
  for the deprecated symbol, remove it and its ``DeprecationWarning`` shim, update any docs or
  examples still referencing it, and delete the row from this table in the same PR as the
  removal.
- **Never remove opportunistically.** A refactor that happens to touch a file containing a
  deprecated shim is not authorization to delete that shim — only the release named in this
  table is.
