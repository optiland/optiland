.. _api_physical_optics:

Physical Optics
===============

.. toctree::
   :maxdepth: 1

   Gaussian beam propagation example <../examples/gaussian_beam_propagation>

Boundary occupancy diagnostic
-----------------------------

Angular-spectrum propagation uses a periodic FFT grid.
To report the fraction of the current field intensity in a boundary band,
call the optional diagnostic after propagation::

   from optiland.physical_optics import boundary_diagnostic, gaussian_field

   field = gaussian_field((128, 192), dx=0.01, wavelength=0.0005,
                          waist_radius=0.1)
   propagated = field.propagate(10.0)
   diagnostic = boundary_diagnostic(propagated, edge_width=4,
                                    threshold=0.01, warn=True)
   fraction = diagnostic.boundary_fraction

``edge_width`` is the number of samples at each edge, at most half the smaller
grid dimension.
The boundary is the union of these bands, with each corner counted once.
``threshold`` must be finite and positive; the flag and optional warning use
a strict greater-than comparison.
A zero field returns a zero fraction.
Nonfinite amplitudes are rejected.
The diagnostic never modifies, pads, or filters the field.

The fraction retains PyTorch gradients, dtype, and device for nonzero fields.
Validation and the Python threshold decision synchronize device execution,
including when warnings are disabled.
The zero-field gradient is defined as zero; the normalized fraction has no
direction-independent limit there.
A small boundary fraction does not prove alias-free propagation: content may
already have wrapped into the interior, and frequency sampling is not tested.
Choose the band width and threshold for the field and propagation distances
being studied, and check grid convergence separately.

Paraxial Gaussian beams
-----------------------

:class:`~optiland.physical_optics.GaussianBeam` describes a rotationally
symmetric Gaussian beam at one plane by its complex beam parameter *q*.
The real part of *q* is the distance from the waist to the plane (negative
before the waist), and the imaginary part is the Rayleigh range.
Lengths are in millimeters and wavelengths in micrometers, as in ``Optic``, and
beam radii are 1/e² intensity radii.

``GaussianBeamAnalysis`` (see :doc:`analysis/analysis.gaussian_beam`) carries a
beam through an optical system with its ray-transfer (ABCD) matrices, including
systems folded by plane mirrors. Because the beam is rotationally symmetric,
surfaces with different radii in x and y (toroidal, biconic) raise an error::

   from optiland.analysis import GaussianBeamAnalysis
   from optiland.physical_optics import GaussianBeam
   from optiland.samples.objectives import CookeTriplet

   lens = CookeTriplet()
   beam = GaussianBeam.from_waist(1.0, wavelength=0.55)  # 1 mm waist on surface 1
   analysis = GaussianBeamAnalysis(lens, beam)
   focus = analysis.beam_at(-1, "before")  # arriving at the image plane
   print(focus)
   # GaussianBeam(w=0.00965934 mm, w0=0.0087533 mm, waist at +0.204216 mm,
   #              wavelength=0.55 µm, m2=1.0)

``GaussianBeam.from_round_trip`` gives the eigenmode of a stable resonator, and
``GaussianBeam.overlap`` the mode-matching efficiency between two beams.
:doc:`../examples/Tutorial_1f_Paraxial_Gaussian_Beams` shows both, including the
arm cavity of ETpathfinder.

.. automodule:: optiland.physical_optics
   :members:
   :undoc-members:
   :show-inheritance:
