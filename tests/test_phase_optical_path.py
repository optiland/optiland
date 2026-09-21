from __future__ import annotations

import numpy as np

from optiland import backend as be
from optiland.optic import Optic
from optiland.phase import RadialPhaseProfile
from optiland.psf import FFTPSF
from optiland.wavefront import Wavefront


def test_focusing_phase_has_constant_optical_path(set_test_backend):
    """An f/5 phase lens must focus both rays and their optical phases."""
    f = 100.0
    wavelength = 10.0
    k = 2 * np.pi / (wavelength * 1e-3)
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1, radius=be.inf, thickness=f, is_stop=True,
        phase_profile=RadialPhaseProfile(
            coefficients=[-k / (2 * f), k / (8 * f**3), -k / (16 * f**5)]
        ),
    )
    lens.surfaces.add(index=2)
    lens.set_aperture("EPD", 20.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=wavelength, is_primary=True)

    py = be.linspace(0.0, 1.0, 6)
    zero = be.zeros_like(py)
    rays = lens.trace_generic(zero, zero, zero, py, wavelength)
    path = be.to_numpy(rays.opd) / (wavelength * 1e-3)
    # The r^6 expansion leaves a small, analytically bounded r^8 residual.
    assert np.max(np.abs(be.to_numpy(rays.y))) < 4e-6
    assert np.ptp(path) < 1e-4
    data = Wavefront(lens, num_rays=8).get_data((0, 0), wavelength)
    assert np.std(be.to_numpy(data.opd)) < 1e-4
    psf = FFTPSF(lens, (0, 0), wavelength, num_rays=64)
    assert float(psf.strehl_ratio()) > 0.99
