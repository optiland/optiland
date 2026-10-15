"""Electric-field transport must follow changes of the ray coordinate frame."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import PolarizedRays
from tests.utils import assert_allclose


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_rotation_transports_complex_output_field(axis, set_test_backend):
    rays = PolarizedRays(*[be.array([v]) for v in (0, 0, 0, 0, 0, 1, 1, 0.55)])
    rays.p = be.array([[[1, 0, 0], [0, 2, 0], [0, 0, 1]]]) + 1j * be.array(
        [[[0, 0, 0], [1, 0, 0], [0, 0, 0]]]
    )
    before = be.to_numpy(rays.p).copy()
    getattr(rays, f"rotate_{axis}")(0.4)
    expected = Rotation.from_euler(axis, 0.4).as_matrix() @ before
    assert_allclose(rays.p, expected)


def test_local_frame_matrix_returns_to_global_frame(set_test_backend):
    rays = PolarizedRays(*[be.array([v]) for v in (1, 2, 3, 0, 0, 1, 1, 0.55)])
    cs = CoordinateSystem(x=1, y=2, z=3, rx=0.2, ry=-0.3, rz=0.4)
    matrix = np.diag([1, 1j, 1])
    rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    cs.localize(rays)
    rays.p = be.matmul(be.array(matrix.real) + 1j * be.array(matrix.imag), rays.p)
    cs.globalize(rays)
    assert_allclose(rays.p[0], rotation @ matrix @ rotation.T)
    assert_allclose([rays.x, rays.y, rays.z], [[1], [2], [3]])
    assert_allclose([rays.L, rays.M, rays.N], [[0], [0], [1]])


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_each_ray_can_have_its_own_frame_rotation(axis, set_test_backend):
    angles = np.array([0.3, -0.7])
    rays = PolarizedRays(*[be.array([v, v]) for v in (0, 0, 0, 0, 0, 1, 1, 0.55)])
    getattr(rays, f"rotate_{axis}")(be.array(angles))
    assert_allclose(rays.p, Rotation.from_euler(axis, angles).as_matrix())


@pytest.mark.parametrize("axis,index", [("x", 1), ("y", 0), ("z", 0)])
def test_frame_rotation_preserves_angle_gradients(axis, index, set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Autograd is a Torch capability")
    import torch

    angle = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    rays = PolarizedRays(*[be.array([v]) for v in (0, 0, 0, 0, 0, 1, 1, 0.55)])
    getattr(rays, f"rotate_{axis}")(angle)
    rays.p[0, index, index].real.backward()
    assert_allclose(angle.grad, -np.sin(0.4))
