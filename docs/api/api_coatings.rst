Coatings
========

This section covers coating functionality of the Optiland package.
The coating module is used to define the coating properties of the optical surfaces.
For example, the user can define the reflectance and transmittance of the surface,
or apply a Fresnel-based coating to a surface.

.. autosummary::
   :toctree: coatings/
   :caption: Coating Modules

   coatings

Cartesian polarization elements
--------------------------------

``CartesianJonesCoating(j00, j01, j10, j11)`` applies four finite complex
coefficients to electric-field X/Y components in the surface's local Cartesian
frame. The local Z component is unchanged. This differs from coatings whose
Jones matrix is evaluated in the ray's incidence-plane s/p basis. The surface
first performs geometrical refraction or reflection; the Cartesian element then
modifies the transported field. It does not add Fresnel coefficients.

For example, ``CartesianJonesCoating(1, 0, 0, 1j)`` adds a quarter-cycle phase
to local Y, and ``CartesianJonesCoating(1, 0, 0, 0)`` transmits local X while
removing local Y. Scalar ``RealRays`` ignore the element, matching the other
polarized coatings. Select an explicit ``PolarizationState`` to trace fields.

Coefficients are field amplitudes, so an amplitude of 0.5 gives an intensity
factor of 0.25 for that component. Arbitrary finite matrices are accepted; the
class does not enforce passivity or normalize their gain. Native serialization
stores exactly four real/imaginary pairs and uses ``BaseCoating`` dispatch.

At oblique incidence an arbitrary X/Y matrix can create a component parallel to
the outgoing ray. This ideal Cartesian convention does not enforce transverse
fields; use an incidence-plane s/p coating for that physical constraint.

File readers can reuse this native class without adding a file-format dependency
to the coating. See :ref:`polarization_transport` for
field frames, scalar losses and the responsibilities of each caller.
