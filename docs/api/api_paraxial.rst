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

When the entrance pupil is needed on the same axial coordinate as surface
positions — for instance to compare it against object or surface positions —
use :py:meth:`~paraxial.Paraxial.entrance_pupil_axial_position`, which
re-anchors ``EPL()`` in one place (``entrance_pupil_z`` is its deprecated
legacy alias). Internally, every ray-aiming, ray-tracing, and aperture
consumer that mixes the pupil location with axial coordinates routes through
this helper. For the pupil's real-space location in a folded system, use
:py:meth:`~paraxial.Paraxial.entrance_pupil_point_gcs` and
:py:meth:`~paraxial.Paraxial.exit_pupil_point_gcs`.

The scalar first-order model itself is written on the **signed unfolded
axial coordinate** carried by :py:class:`~paraxial_path.ParaxialPath` (see
the developer's guide page :doc:`../developers_guide/folded_systems`).

.. autosummary::
   :toctree: paraxial/
   :caption: Paraxial Modules

   paraxial
   paraxial_path
