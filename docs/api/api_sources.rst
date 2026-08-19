Sources
=======

This section covers the extended source modeling modules of the Optiland package.
Extended sources allow users to define spatially and angularly extended light sources
(such as single-mode fiber outputs) and trace them through optical systems.

The :class:`sources.base.BaseSource` class defines the abstract interface that all
source implementations must follow. Concrete implementations, such as
:class:`sources.smf.SMFSource`, provide specific source types with their own
spatial and angular distributions.

.. autosummary::
   :toctree: sources/
   :caption: Sources Modules

   sources.base
   sources.smf
   sources.visualization
