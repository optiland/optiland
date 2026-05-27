Paraxial
========

This section provides an overview of the paraxial optics module in Optiland.
This module enables the user to perform paraxial analysis of the optical system.

Conventions
---------------------------

Location quantities follow Optiland's reference-surface convention, where
object-space quantities are measured relative to the first physical surface
(index 1) and image-space quantities relative to the image surface:

- :py:meth:`~paraxial.Paraxial.EPL` (entrance pupil location) is relative to
  the first physical surface (surface 1).
- :py:meth:`~paraxial.Paraxial.XPL` (exit pupil location) is relative to the
  image surface.
- The focal, principal, nodal, and anti-nodal planes (``F1``/``F2``,
  ``P1``/``P2``, ``N1``/``N2``, ``P1anti``/``P2anti``, ``N1anti``/``N2anti``)
  follow the same per-surface convention, where ``1`` denotes object space and
  ``2`` denotes image space.

When a **global** z coordinate is needed for the entrance pupil — for instance
to compare it against object or surface positions — use
:py:meth:`~paraxial.Paraxial.entrance_pupil_z`, which converts ``EPL()`` to the
global frame in one place. Internally, every ray-aiming, ray-tracing, and
aperture consumer that mixes the pupil location with global coordinates routes
through this helper.

.. autosummary::
   :toctree: paraxial/
   :caption: Paraxial Modules

   paraxial
