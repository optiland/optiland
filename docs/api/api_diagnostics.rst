Diagnostics
===========

This section covers Optiland's system diagnostics utilities. The
:func:`~optiland.diagnostics.check_system` function inspects an
:class:`~optiland.optic.optic.Optic` and reports the failure modes that are most
common when setting up a new system - a missing wavelength, an undefined
aperture, a stop that was never marked, and similar configuration issues -
each paired with the offending object and a runnable fix.

Diagnostics are read-only: running :func:`~optiland.diagnostics.check_system`
never modifies the optical system, and the ``diagnostics`` subpackage is not
imported by ``optiland.optic``.

.. code-block:: python

   from optiland.diagnostics import check_system
   from optiland.optic import Optic

   lens = Optic()
   report = check_system(lens)
   if not report.ok:
       print(report)

.. autosummary::
   :toctree: diagnostics/
   :caption: Diagnostics Modules

   diagnostics.checks
   diagnostics.report
