from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.analysis.jones_pupil import JonesPupil
from optiland.samples.objectives import CookeTriplet


def test_jones_pupil_initialization(set_test_backend):
    optic = CookeTriplet()
    optic.updater.set_polarization("ignore")  # Default state for test
    jp = JonesPupil(optic)
    assert jp.optic == optic
    assert jp.grid_size == 65
    assert jp.field == (0, 0)
    assert jp.wavelengths is not None


def test_jones_pupil_generate_data(set_test_backend):
    optic = CookeTriplet()
    optic.updater.set_polarization("ignore")  # Default state for test
    jp = JonesPupil(optic, grid_size=5)
    data = jp.data

    # Check structure: list of wavelengths -> dict
    assert isinstance(data, list)
    assert len(data) == len(jp.wavelengths)

    # Check data for first wavelength
    single_data = data[0]
    assert isinstance(single_data, dict)
    assert "Px" in single_data
    assert "Py" in single_data
    assert "J" in single_data

    # Check shapes
    num_rays = 5 * 5
    assert single_data["Px"].shape == (num_rays,)
    assert single_data["J"].shape == (num_rays, 2, 2)

    J = single_data["J"]
    Jxx = J[:, 0, 0]
    Jxy = J[:, 0, 1]

    # Center ray (index 12 for 5x5 grid)
    center_idx = num_rays // 2

    # Use backend agnostic checks
    val_xx = be.to_numpy(Jxx[center_idx])
    val_xy = be.to_numpy(Jxy[center_idx])

    assert np.abs(val_xx) > 0.5
    assert np.abs(val_xy) < 0.1


def test_jones_pupil_polarization_handling(set_test_backend):
    optic = CookeTriplet()
    optic.updater.set_polarization("ignore")  # Default state for test
    # Ensure it works even if optic polarization is 'ignore'
    optic.updater.set_polarization("ignore")
    jp = JonesPupil(optic, grid_size=3)
    # Trace happens in __init__ / first access to data

    # Ensure optic state is restored
    assert optic.polarization == "ignore"

    # Data should be valid
    assert jp.data is not None


def test_jones_pupil_view(set_test_backend):
    optic = CookeTriplet()
    optic.updater.set_polarization("ignore")  # Default state for test
    jp = JonesPupil(optic, grid_size=5)
    fig, axs = jp.view()

    assert fig is not None
    # 2 rows, 4 columns = 8 axes + 8 colorbars = 16 axes
    assert len(axs) == 16

    # Clean up
    plt.close(fig)


def test_jones_pupil_view_custom_field(set_test_backend):
    optic = CookeTriplet()
    optic.updater.set_polarization("ignore")  # Default state for test
    # Test with off-axis field
    jp = JonesPupil(optic, field=(0, 1.0), grid_size=5)
    data = jp.data
    assert len(data) == len(jp.wavelengths)

    # Just verify valid execution
    single_data = data[0]
    assert "J" in single_data
