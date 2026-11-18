ISO 10110 Drawings
==================

This section covers the ISO 10110 fabrication-drawing generator of the Optiland package.
Starting from an :class:`~optiland.optic.optic.Optic`, it infers the lens elements,
lets you attach surface- and element-level fabrication tolerances (form error, centration,
surface imperfections, coatings, glass quality, title-block data) and renders one
drawing sheet per element to PDF, PNG or DXF.

PDF and PNG output need no extra dependencies. DXF output requires the optional
``ezdxf`` package, available through the ``manufacturing`` extra
(``pip install optiland[manufacturing]``).

.. note::

   Element inference groups surfaces by glass/air transitions. Reflective
   (mirror) surfaces and images immersed in a non-air medium are not supported
   yet and emit a ``UserWarning``.

.. autosummary::
   :toctree: iso10110/
   :caption: ISO 10110 Modules

   iso10110.spec
   iso10110.report
   iso10110.drawing
   iso10110.elements
   iso10110.notation
   iso10110.style
