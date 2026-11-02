Physical Apertures
==================

This section provides an overview of the physical apertures available in the
Optiland package. Physical apertures are used to define boundaries of the
individual optical surface, i.e., where rays may interact on the surface. For
example, a primary telescope mirror that has a central hole may be defined by a
physical aperture that is circular with a hole in the center (i.e., `RadialAperture`).

.. autosummary::
   :toctree: physical_apertures/
   :caption: Physical Aperture Modules

   physical_apertures.base
   physical_apertures.radial
   physical_apertures.rectangular
   physical_apertures.elliptical
   physical_apertures.offset_radial
   physical_apertures.polygon
   physical_apertures.rotated
   physical_apertures.spider
   physical_apertures.unclipped

Transforms and drawing boundaries
----------------------------------------

``RotatedAperture`` wraps any physical aperture and rotates its membership test
and enclosing drawing bounds. Its angle is in radians, counterclockwise about
the surface's positive z-axis; the optional pivot is in surface coordinates.
Scaling changes the child shape and pivot together, while retaining the angle.
The OSLO reader uses this wrapper for rotated ellipses and rectangles, with the
shape's centroid as pivot.

``UnclippedAperture`` retains a child shape for drawing while leaving ray
intensities unchanged when the surface calls ``clip``. Its ``contains`` method
still describes the drawing shape; it does not mean that rays outside that
shape are blocked. Use it around the complete aperture assigned to a surface,
after composing any Boolean shapes. The OSLO reader uses it for unchecked
ordinary apertures and disabled aperture checking. The OSLO writer preserves
this distinction for supported circular boundaries.

Both wrappers use the existing ``BaseAperture`` serialization dispatch and
survive native ``Optic`` save/reload. Surface tracing calls ``clip``; layout
renderers use ``extent`` and ``contains``. Neither wrapper changes refraction,
reflection, material selection or polarization interactions.

For example, this rectangle is rotated around its displaced centroid::

   import math
   from optiland.physical_apertures import RectangularAperture, RotatedAperture

   aperture = RotatedAperture(
       RectangularAperture(2, 6, 1, 3), math.pi / 2, center_x=4, center_y=2
   )
   # aperture.extent is approximately (3, 5, 0, 4).

Spiders
-------

``SpiderAperture`` blocks the rays that fall on the vanes of a telescope's
spider: ``num_arms`` straight arms of full width ``width``, running outward
from a common centre at equal angles, the first along the surface's positive
x-axis. Every other ray passes, with no outer limit, so a spider is usually
paired with a separate aperture or obscuration elsewhere in the system. The
optional ``offset_x`` and ``offset_y`` move the centre. The Zemax reader builds
it from ``SPID`` and ``OBDC``, and the writer emits them back.
