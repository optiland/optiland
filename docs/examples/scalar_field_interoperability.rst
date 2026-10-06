Sampled fields, optical prescriptions, and detector simulation
==============================================================

The physical-optics package can accept an arbitrary two-dimensional complex scalar array.
HCIPy source generation is optional: ``ScalarField(data, dx, wavelength, dy=dy)`` already accepts an array on the active NumPy or Torch backend.
The first dimension is y and the last is x.
All spatial metadata use the same unit; the prescription adapter below specifically requires millimeters.

Package responsibilities
------------------------

HCIPy owns source-wavefront generation, its field/grid representation, and explicit export of backend data.
Optiland owns the optional conversion into its field coordinates and its optical prescription.
ESA Pyxel owns detector quantum efficiency, photon-to-charge conversion, exposure timing, and noise.
Neither HCIPy nor Pyxel is a mandatory dependency for Optiland's propagation or radiometric conversion.
ESA's detector package is distributed as ``pyxel-sim`` and imported as ``pyxel``; the unrelated game engine has the same import name.

The optional ``from_hcipy`` and ``to_hcipy`` functions live in ``optiland.physical_optics.interoperability``.
They preserve complex phase, rectangular axis order, sample pitch, grid center, vacuum wavelength, and integrated squared amplitude.
There is no transpose, FFT shift, or phase conjugation.
An explicit host copy separates the packages: importing into Torch creates a new graph, and exporting detaches and downloads it.
This is not a zero-copy GPU or cross-package autograd bridge.
HCIPy backend fields require a working ``Field.to_dict()`` implementation for explicit export.

HCIPy coordinates are not unit-tagged.
The default import multiplies lengths by 1000, converting meters to millimeters, and divides amplitudes by 1000 so that the squared-amplitude area integral is unchanged.
The reverse default multiplies lengths by 0.001 and divides amplitudes by 0.001.
Use a different ``length_scale`` when the input convention differs.
Only scalar regular Cartesian grids with rectangular sample-area weights are supported; vector fields, nonuniform grids, and custom quadrature are rejected rather than reinterpreted.
HCIPy Wavefront does not store refractive index, so supply the incident index explicitly when it is not one, and retain it separately on export.

Prescription model and its limits
----------------------------------

``ScalarOpticalTrain.from_optic(optic).propagate(field)`` applies a bounded scalar phase-screen model.
The incoming plane is immediately before the selected start vertex, and the outgoing plane is immediately after the inclusive end vertex.
The object at surface zero is excluded; no infinite object distance or trailing thickness is propagated.
Gaps come from actual vertex coordinates.
Material wavelengths are converted from millimeters to micrometers for existing material lookup.

At a coaxial interface, the added optical path is ``(n_before - n_after) * sag``.
The resulting factor is ``exp(1j * 2*pi/wavelength * optical_path)`` under the existing positive forward-propagation convention.
The outgoing homogeneous medium sets the angular-spectrum wavenumber for the next vertex gap.
Physical apertures clip amplitude and retain the resulting power loss.
Only explicit ``surface.aperture`` objects define physical stops in this workflow.
The optic's EPD, F-number, or object-NA setting controls ray-launch sampling; the adapter does not turn it into a guessed physical stop for an already supplied field.
Define the required stop apertures explicitly before propagation.

This is a paraxial surface approximation, not exact field remapping onto curved interfaces or general high-NA propagation.
Native coaxial planes, spheres/conics, even aspheres, homogeneous lossless media, and radial/rectangular/elliptical apertures are supported.
Even-asphere coefficients follow the native sag convention: ``coefficients[j]`` multiplies ``r**(2*(j+1))`` and has units ``mm**(1-2*(j+1))``.
For spheres/conics and even aspheres, an infinite radius is a flat conic base; asphere coefficient gradients remain supported, but radius and conic are constant metadata in that limit.
Native planar thin lenses add their paraxial quadratic phase; their focal parameter is inverse reduced optical power, so collimated light focuses at ``n_after*f`` in a non-air outgoing medium.
Native planar ConstantPhaseProfile and RadialPhaseProfile screens supply phase in radians, with amplitude multiplied by the square root of their efficiency.
Rotations/decenters, mirrors/folds, negative gaps, GRIN, gain, coatings/scattering, unaudited phase profiles, curved thin-lens/phase interactions, and custom geometries are currently rejected.
Fresnel reflection, vector polarization, obliquity factors, and curved-interface Jacobians are not included.
Field, supported geometry, asphere coefficients, thin-lens focal parameter, phase coefficients, and vertex-gap Torch gradients are retained; material-index and extinction-coefficient gradients are not supported.
An ``ImageSurface`` is only a planar marker with no material-index or extinction change.
Selected surfaces are revalidated before propagation, including sampled sag on transmitted coordinates.
The input field and native optic/material state are not modified.
Custom material callbacks must not mutate shared or external state; their arbitrary side effects cannot be rolled back by the adapter.

Catalog glasses often have small nonzero extinction coefficients.
The default ``absorption="reject"`` rejects all nonzero values rather than silently treating the glass as lossless.
For a uniform axial-length approximation, explicitly select ``ScalarOpticalTrain.from_optic(optic, absorption="axial")``.
This permits passive extinction ``kappa >= 0`` and multiplies amplitude once per outgoing vertex gap by ``exp(-2*pi*kappa*gap/wavelength)``.
The corresponding power transmission is ``exp(-4*pi*kappa*gap/wavelength)``; both gap and vacuum wavelength are in millimeters, with no extra refractive-index factor.
Extinction is evaluated at the micrometer material-lookup wavelength.
This approximation uses vertex distances, not curved/sag-dependent layer thickness or angle-dependent path lengths; it does not implement complex-index ASM, interface flux, or Fresnel losses.
Neither the incident-to-start path nor a trailing end-surface thickness is included.
Keep radiometric calibration fixed so absorption, like aperture loss, remains in the exported detector rate.

The grid is periodic under FFT propagation.
Resolve surface phase slopes and aperture edges, provide sufficient window/padding, and check sampling/window convergence.
``field.pad(64)`` explicitly adds 64 zero samples at each edge through the existing backend padding operation.
For unequal edges, use ``field.pad(((y_before,y_after),(x_before,x_after)))``.
Padding preserves pitch, wavelength, sample coordinates, and captured power; asymmetric padding changes the grid center without displacing the beam.
It does not interpolate, restore truncated input tails, or guarantee freedom from aliasing.
``boundary_diagnostic`` reports occupancy near the boundary but cannot prove alias-free propagation or recover content that has already wrapped.

Calibrated Pyxel input
----------------------

ScalarField amplitude is not automatically a physical electric field in V/m, and its power integral is not automatically watts.
Use an explicitly calibrated power-density convention or supply the irradiance scale corresponding to one unit of squared amplitude::

   from optiland.physical_optics.radiometry import (
       field_to_irradiance, irradiance_to_photon_rate, normalized_psf,
   )
   import numpy as np
   import optiland.backend as be

   irradiance = field_to_irradiance(output, irradiance_scale_w_per_mm2=1.0)
   rate = irradiance_to_photon_rate(
       irradiance, dx_mm=output.dx, dy_mm=output.dy,
       wavelength_nm=output.wavelength * 1e6, binning=(2, 2),
   )
   np.save("photon_rate.npy", be.to_numpy(rate))

The scale of one above is valid only if amplitude is already in ``sqrt(W/mm²)``.
For arbitrary units, calibrate the represented input-plane power once; do not renormalize each output, which would erase losses.
The conversion uses vacuum photon energy ``h*c/lambda`` and integrates intensity, not complex amplitude, over aligned detector cells.
``binning=(by,bx)`` requires exactly divisible dimensions and sums fine cells; it does not interpolate, pad, or restore uncaptured light.
This is rectangular quadrature when the optical data are point samples, so detector integration also needs sampling convergence.

Pyxel's existing ``load_image`` model multiplies this rate by its current readout interval.
Set ``time_scale=1.0`` and ``convert_to_photons=False``; the latter option otherwise converts ADU, not irradiance.
Do not multiply by exposure time or quantum efficiency in Optiland and then apply them again in Pyxel.
Both packages use rows=y and columns=x; plotting origin conventions are not a reason to flip the array.
Configure detector pitch in micrometers and environment wavelength in nanometers separately: NumPy files do not encode those quantities.

For the runnable example below, the detector map has 95 rows and 127 columns, with vertical/horizontal pitches of 16/12 micrometers and a wavelength of 500 nm.
The relevant Pyxel configuration is::

   ccd_detector:
     geometry:
       row: 95
       col: 127
       pixel_vert_size: 16.0
       pixel_horz_size: 12.0
     environment:
       wavelength: 500.0
     characteristics:
       quantum_efficiency: 0.4
   exposure:
     readout:
       times: [0.01, 0.025]
       non_destructive: false
   pipeline:
     photon_collection:
       - name: imported_optical_illumination
         func: pyxel.models.photon_collection.load_image
         enabled: true
         arguments:
           image_file: photon_rate.npy
           time_scale: 1.0
           convert_to_photons: false
     charge_generation:
       - name: deterministic_quantum_efficiency
         func: pyxel.models.charge_generation.simple_conversion
         enabled: true
         arguments:
           binomial_sampling: false

Readout times are absolute sampling times: these two intervals are 0.01 and 0.015 seconds, not two independent exposures of 0.01 and 0.025 seconds.
This deterministic example leaves shot noise disabled; add the appropriate Pyxel noise models only when needed.
Import illumination through the pipeline rather than preassigning a bucket that the exposure runner will clear.
Save the configuration as a YAML file and replace ``image_file`` with the absolute path to the exported photon-rate file.
With ESA ``pyxel-sim`` installed, load and run that configuration through its existing API::

   import pyxel
   config = pyxel.load("/path/to/example.yaml")
   results = pyxel.run_mode(config)

For optical blur instead of absolute illumination, ``normalized_psf`` produces a unit-sum kernel from an already detector-sampled nonnegative intensity array.
Save it with ``numpy.save`` and use Pyxel ``load_psf`` after initializing scene photons.
A conditional normalized kernel does not encode optical throughput or lost tails.
``normalized_psf`` changes only normalization: it does not register the physical optical-axis origin or resample detector pitch.
Pyxel's native convolution places the kernel origin at integer index ``(Ny//2,Nx//2)``.
A centered even-sized Optiland kernel straddles that origin and would shift the impulse response by minus half a detector pixel on each even axis.
Use matched detector sampling with physical zero at the native origin, normally an odd-sized centered kernel.
Do not repair an even grid by cropping or relabeling coordinates; existing samples are still half a pixel away from the required origin.
Preserve real PSF asymmetry rather than forcing the intensity centroid to zero.
The example's two-by-two binning deliberately produces odd dimensions and a central detector cell on the optical axis.
Direct photon-rate illumination through ``load_image`` has no convolution-origin requirement.
Only monochromatic two-dimensional interchange is covered here; wavelength-dependent detector cubes and spectral PSF convolution require separate validation.

See the `ESA Pyxel illumination models <https://esa.gitlab.io/pyxel/doc/stable/references/model_groups/photon_collection/illumination.html>`_
and `pixel coordinate conventions <https://esa.gitlab.io/pyxel/doc/stable/background/pixel_coordinate_conventions.html>`_.

Runnable example
----------------

Install HCIPy in the example environment and run from the repository root::

   python -m docs.examples.scalar_field_interoperability --output-dir /path/to/results
   python -m docs.examples.scalar_field_interoperability --output-dir /path/to/torch-results --backend torch

The example normalizes a Gaussian waist source to 1 nW of captured power, applies a clipped native thin-lens prescription, and exports photon rates and a conditional PSF.
The source power is assigned only once, before clipping and propagation.
ESA Pyxel is not needed to generate the files.

.. literalinclude:: scalar_field_interoperability.py
   :language: python
