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

Wraparound bounds
-----------------

Each plane-wave component of the angular spectrum shifts laterally by a fixed
amount over the propagation distance.
To bound the power that crosses the periodic grid edge, call the (optional)
diagnostic on the field *before* propagation::

   from optiland.physical_optics import gaussian_field, wraparound_diagnostic

   field = gaussian_field((128, 192), dx=0.01, wavelength=0.0005,
                          waist_radius=0.1)
   diagnostic = wraparound_diagnostic(field, 400.0, support_threshold=1e-6,
                                      threshold=0.01, warn=True)
   diagnostic.lower_fraction  # 2.5e-05
   diagnostic.upper_fraction  # 0.24

``lower_fraction`` is the power that shifts by at least the grid width.
It wraps from any start position, including when it lands back in the
interior with no power at the boundary.
It is the power outside the non-aliasing band of Matsushima and Shimobaba
(2009), Eq. (17), for the grid zero padded to twice its size.
``upper_fraction`` is the power that shifts further than the empty margin
between the field's support box and the grid edge.
The support box spans the samples whose intensity is at least
``support_threshold`` times the peak.
Power near an edge can wrap after a small shift, which only the upper bound
detects.
The flag and optional warning compare the upper bound with ``threshold``,
using a strict greater-than comparison.

With twofold zero padding, the fraction of power that wraps into the cropped
window lies between :math:`0` and ``lower_fraction`` of the unpadded field.
Let the field have :math:`N_x \times N_y` samples on a window of width
:math:`S_x = N_x \Delta x` and height :math:`S_y = N_y \Delta y`.
Over a distance :math:`z`, the component with spatial frequencies
:math:`(f_x, f_y)` shifts laterally by

.. math::

   s_x = \frac{z (\lambda/n) f_x}{\sqrt{1 - (\lambda/n)^2 (f_x^2 + f_y^2)}},
   \qquad
   s_y = \frac{z (\lambda/n) f_y}{\sqrt{1 - (\lambda/n)^2 (f_x^2 + f_y^2)}},

where :math:`\lambda` is the vacuum wavelength and :math:`n` the refractive
index of the field.
Pad the field with zeros to at least :math:`2N_x \times 2N_y` samples,
propagate, and crop the :math:`N_x \times N_y` samples that held the field.
A component with :math:`|s_x| < S_x` and :math:`|s_y| < S_y` then wraps only
into the padding, which the crop discards.
This band is the non-aliasing condition of Matsushima and Shimobaba (2009),
Eq. (17), the two-dimensional form of Eq. (13), for the padded frequency
spacings :math:`\Delta f_x = 1/(2S_x)` and :math:`\Delta f_y = 1/(2S_y)`.
Only components with :math:`|s_x| \ge S_x` or :math:`|s_y| \ge S_y` can wrap
into the cropped window, and ``lower_fraction`` of the unpadded field is
exactly their share of the power.
If the field has no power outside the band, ``lower_fraction`` is
:math:`0`, so the lower and upper bounds on wraparound in the cropped window
are both :math:`0`.
Evaluate the diagnostic before padding: on the padded grid it reports power
that crosses the padded edge, which the crop discards.

Both bounds, and the zero-padding result above, assume a ray model: each
spatial-frequency component carries its power along a single lateral shift,
from anywhere in the support box.
This misses diffraction, the sinc-like tails of the sampled propagation kernel,
and content below ``support_threshold``, so the wrapped power can fall slightly
outside the bounds.
Evanescent components do not shift.
The fractions retain PyTorch gradients with respect to the field, so a design
that generates the field, such as a phase mask, can be optimized to reduce
wraparound.
Penalize ``upper_fraction``: ``lower_fraction`` can reach zero while power
near an edge still wraps.
A distance tensor receives no gradient, because the bounds are piecewise
constant in distance.

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
