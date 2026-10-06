.. _api_physical_optics:

Physical Optics
===============

.. toctree::
   :maxdepth: 1

   Gaussian beam propagation example <../examples/gaussian_beam_propagation>
   Sampled-field interoperability <../examples/scalar_field_interoperability>

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

.. automodule:: optiland.physical_optics
   :members:
   :undoc-members:
   :show-inheritance:

Optional HCIPy interchange
--------------------------

.. automodule:: optiland.physical_optics.interoperability
   :members: from_hcipy, to_hcipy
