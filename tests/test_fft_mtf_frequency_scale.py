"""Independent geometric and Zemax regressions for FFT frequency calibration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import Material
from optiland.mtf.fft import ScalarFFTMTF
from optiland.rays import RealRays
from optiland.samples.objectives import CookeTriplet
from optiland.samples.telescopes import HubbleTelescope
from tests.utils import assert_allclose


def _synthetic_mtf(theta, tilt, index=1.0, frame_rotation=0.0):
    """Use an anamorphic cone with independently known angular bandwidths."""
    a, b = 0.1, 0.2
    zero = theta * 0
    L = be.stack([zero, zero, zero, zero + np.sin(b), zero - np.sin(b)])
    M = be.stack(
        [
            be.sin(theta),
            be.sin(theta + a),
            be.sin(theta - a),
            be.cos(b) * be.sin(theta),
            be.cos(b) * be.sin(theta),
        ]
    )
    N = be.stack(
        [
            be.cos(theta),
            be.cos(theta + a),
            be.cos(theta - a),
            be.cos(b) * be.cos(theta),
            be.cos(b) * be.cos(theta),
        ]
    )
    rays = RealRays(be.zeros(5), be.zeros(5), be.zeros(5), L, M, N, 1, 0.55)
    cs = CoordinateSystem(z=10, rx=frame_rotation)
    cs.globalize(rays)

    def normal(local_rays):
        assert_allclose(local_rays.z, be.zeros(5), atol=1e-12)
        return (be.zeros(5), be.ones(5) * be.sin(tilt), be.ones(5) * be.cos(tilt))

    m = ScalarFFTMTF.__new__(ScalarFFTMTF)
    m.num_rays = 129
    m.resolved_wavelength = 0.55
    m.optic = SimpleNamespace(
        trace_generic=lambda *args, **kwargs: rays,
        image_surface=SimpleNamespace(
            geometry=SimpleNamespace(cs=cs, surface_normal=normal),
            material_post=SimpleNamespace(n=lambda wavelength: index),
        ),
    )
    return m


@pytest.mark.parametrize("tilt", [0.0, 0.25, -0.4])
@pytest.mark.parametrize("frame_rotation", [0.0, 0.7])
def test_directional_bandwidth(set_test_backend, tilt, frame_rotation):
    """Project each cone independently, including a rotated coordinate frame."""
    theta = be.array(0.3)
    m = _synthetic_mtf(theta, be.array(tilt), 1.5, frame_rotation)
    dt, ds = m._get_mtf_frequency_steps((0.0, 1.0))
    denominator = 128 * 0.55e-3
    expected_t = 1.5 * 2 * np.sin(0.1) * np.cos(0.3 - tilt) / denominator
    expected_s = 1.5 * 2 * np.sin(0.2) / denominator
    assert_allclose(dt, expected_t, rtol=1e-12)
    assert_allclose(ds, expected_s, rtol=1e-12)


def test_frequency_scale_keeps_torch_gradient(set_test_backend):
    """The geometry projection must not detach trainable optical parameters."""
    if be.get_backend() != "torch":
        pytest.skip("Autograd is specific to PyTorch.")
    import torch

    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    m = _synthetic_mtf(theta, be.array(0.2))
    dt, _ = m._get_mtf_frequency_steps((0.0, 1.0))
    dt.backward()
    expected = -2 * np.sin(0.1) * np.sin(0.1) / (128 * 0.55e-3)
    assert theta.grad.item() == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", ["hubble", "cooke_hikari_f2"])
def test_off_axis_zemax_frequency_regression(set_test_backend, name):
    """ZOS-API 2025 R1 FFT MTF, 4096 pupil grid, unpolarized modulation.

    Hubble uses the canonical sample. Cooke deliberately uses Hikari F2 on
    both sides to remove the same-name Schott/Hikari catalogue ambiguity.
    No external Zemax installation is needed to run this regression.
    """
    if name == "hubble":
        optic = HubbleTelescope()
        frequency, axis, expected = 10.0, "T", 0.4679959101098332
    else:
        optic = CookeTriplet()
        optic.surfaces[3].material_post = Material("F2", "hikari")
        frequency, axis, expected = 20.0, "S", 0.5386639208575651
    m = ScalarFFTMTF(optic, fields=[(0, 1)], num_rays=256, grid_size=512)
    freq = m.freq_tang[0] if axis == "T" else m.freq_sag[0]
    data = m.mtf[0][0 if axis == "T" else 1]
    actual = np.interp(frequency, be.to_numpy(freq), be.to_numpy(data))
    assert actual == pytest.approx(expected, abs=0.002)
