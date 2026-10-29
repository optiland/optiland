.. _pyxel_radiometry:

Radiometry and Pyxel handoff
============================

Optiland's radiometry helpers convert a calibrated monochromatic scalar field
into expected photons per detector pixel per second.
They use NumPy or PyTorch arrays and require no detector package.
ESA's optional ``pyxel-sim`` package can consume the exported NumPy arrays.
The handoff below is checked against Pyxel 3.0.2.

Calibration and sampling
------------------------

``ScalarField.intensity`` is squared scalar amplitude, with no implied SI
electric-field calibration.
``field_to_irradiance`` requires a finite, nonnegative scale in W/mm² per unit
of squared amplitude.
An amplitude explicitly normalized in sqrt(W/mm²) uses a scale of one.
For a complex peak electric-field phasor in V/m, a normally propagating wave
in a lossless, nonmagnetic medium instead uses
``n * epsilon_0 * c / (2 * 1e6)``.
That factor does not apply to power-normalized amplitudes or RMS phasors.

Keep this calibration fixed when comparing incident, attenuated, or cropped
fields so that lost optical power remains lost.
The conversion never normalizes the captured power.
All field geometry must use one spatial unit, but the photon-rate converter
requires explicit cell widths in mm and vacuum wavelength in nm.
It does not infer those units from the field.

The input irradiance is a nonempty, finite, nonnegative floating-point array
ordered as ``(y, x)``.
Each value represents a cell-mean irradiance; point samples approximate this
mean by rectangle quadrature and require a sampling-convergence check.
Each cell's photon rate is
``irradiance * dx_mm * dy_mm * wavelength_nm * 1e-9 / (h * c)``,
using SciPy's exact SI Planck constant and vacuum speed of light.

``binning=(by, bx)`` sums rates over aligned rectangular blocks of intensities,
not coherent amplitudes.
The positive integer factors must divide the input dimensions exactly.
The output shape is ``(ny // by, nx // bx)`` and its pixel pitch is
``(by * dy_mm, bx * dx_mm)`` in ``(y, x)`` order.
There is no interpolation, automatic crop, or recovery of uncaptured light.
The operations preserve the active backend, dtype, device, and PyTorch gradient
connectivity; input validation can synchronize device execution.

Export a photon-rate map
------------------------

This example uses mm for field geometry and an explicitly chosen irradiance
calibration of 1 pW/mm² per unit of squared amplitude:

.. code-block:: python

   import numpy as np
   import optiland.backend as be
   from optiland.physical_optics import gaussian_field
   from optiland.physical_optics.radiometry import (
       field_to_irradiance,
       irradiance_to_photon_rate,
   )

   field = gaussian_field(
       (96, 128), dx=0.01, dy=0.02,
       wavelength=0.0005, waist_radius=0.1,
   )
   irradiance = field_to_irradiance(
       field, irradiance_scale_w_per_mm2=1e-12,
   )
   rate = irradiance_to_photon_rate(
       irradiance, dx_mm=0.01, dy_mm=0.02,
       wavelength_nm=500.0, binning=(2, 2),
   )
   np.save("photon_rate.npy", be.to_numpy(rate), allow_pickle=False)

The exported array has 48 rows and 64 columns, with vertical and horizontal
pixel pitches of 40 µm and 20 µm, respectively.
NumPy export is an explicit host conversion and ends the differentiable path.
The file contains the array only: record wavelength, calibration, pitch,
orientation, and placement with the detector configuration.

Use this entry in an existing Pyxel pipeline's ``photon_collection`` group:

.. code-block:: yaml

   - name: load_rate
     func: pyxel.models.photon_collection.load_image
     enabled: true
     arguments:
       image_file: photon_rate.npy
       convert_to_photons: false
       time_scale: 1.0

Configure matching detector dimensions and pixel pitches.
Pyxel's ``load_image`` multiplies these rates by the current readout interval;
do not multiply by exposure duration before export.
For destructive readouts at 0.01 s, 0.035 s, and 0.06 s, the intervals are
0.01 s, 0.025 s, and 0.025 s.
Keep the loader inside the timed pipeline because exposure initialization
clears preloaded detector buckets.
Quantum efficiency belongs in the subsequent charge-generation model.
With deterministic ``simple_conversion`` and ``binomial_sampling=False``,
the expected electron count is photon rate times interval times QE.
Shot noise, broadband response, and ADU conversion are outside this contract.
See Pyxel's `load_image reference`_.

Export a PSF shape
------------------

``normalized_psf(detector_sampled_intensity)`` returns a unit-sum kernel from
an already detector-sampled, nonnegative intensity array with positive finite
sum.
Export it with ``np.save`` and ``be.to_numpy`` as above, then use Pyxel's
``load_psf(filename="psf.npy", normalize_kernel=False)``.
This normalization retains only the captured PSF shape; throughput and
uncaptured tails must be accounted for separately.
The helper neither resamples nor recenters the kernel.

In Pyxel 3.0.2, the monochromatic PSF convolution fills beyond the image boundary
with the image mean, so whole-image flux need not be conserved near the edges.
Its native kernel origin is index ``N // 2`` on each axis.
An even-sized Optiland grid has its physical origin between samples, producing
a native displacement of minus half a pixel on each even axis.
Odd-sized centered kernels avoid this registration mismatch.
Real off-axis PSF displacement is retained; no centroid correction is applied.
See Pyxel's `load_psf reference`_.

.. _load_image reference: https://esa.gitlab.io/pyxel/doc/stable/references/model_groups/photon_collection/illumination.html#load-image
.. _load_psf reference: https://esa.gitlab.io/pyxel/doc/stable/references/model_groups/photon_collection/optics.html#load-psf
