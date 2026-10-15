# ruff: noqa: I002

from types import SimpleNamespace

import numpy as np
import pytest

import optiland.backend as be
from optiland.wavefront.reference_geometry import PlanarReference
from tests.utils import assert_allclose


@pytest.fixture
def float32_backend(set_test_backend):
    """Temporarily lower configured precision without narrowing explicit data."""
    previous_precision = be.get_precision()
    be.set_precision("float32")
    try:
        yield
    finally:
        be.set_precision(f"float{previous_precision}")


def make_rays(x, y, z, L, M, N):
    """Build the subset of the ray interface used by PlanarReference."""
    return SimpleNamespace(
        x=be.array(x),
        y=be.array(y),
        z=be.array(z),
        L=be.array(L),
        M=be.array(M),
        N=be.array(N),
    )


def axial_rays(z, normal_direction):
    """Build rays whose plane denominator depends only on N."""
    z = be.array(z)
    normal_direction = be.array(normal_direction)
    zeros = be.zeros_like(z)
    return SimpleNamespace(
        x=zeros,
        y=zeros,
        z=z,
        L=be.sqrt(1 - normal_direction**2),
        M=zeros,
        N=normal_direction,
    )


def test_signed_intersections_and_refractive_scaling(set_test_backend):
    reference = PlanarReference((0, 0, 0), (0, 0, 1))
    rays = axial_rays([2, -2, 0], [1, 1, -1])

    result = reference.path_length(rays, 1.5)

    assert_allclose(result, be.array([3, -3, 0]), rtol=0, atol=0)
    assert result.shape == rays.z.shape
    assert result.dtype == rays.z.dtype


def test_tilted_translated_plane_intersection_residual(set_test_backend):
    reference = PlanarReference((2, -3, 4), (2, -1, 3))
    rays = make_rays(
        [3, 1],
        [-1, 2],
        [7, -3],
        [0, 0.6],
        [0, 0],
        [1, 0.8],
    )

    distance = reference.path_length(rays, 1.7) / 1.7
    hit_x = rays.x - distance * rays.L
    hit_y = rays.y - distance * rays.M
    hit_z = rays.z - distance * rays.N
    residual = (
        (hit_x - reference.point[0]) * reference.normal[0]
        + (hit_y - reference.point[1]) * reference.normal[1]
        + (hit_z - reference.point[2]) * reference.normal[2]
    )

    assert_allclose(distance, be.array([3, -70 / 9]), rtol=1e-12, atol=0)
    assert_allclose(residual, be.zeros_like(residual), rtol=0, atol=1e-12)


def test_exact_parallel_and_nonfinite_contract(set_test_backend):
    reference = PlanarReference((0, 0, 0), (0, 0, 1))
    rays = axial_rays(
        [1, 0, float("nan"), 1, 1],
        [0, 0, 1, 1, 1],
    )
    # Inject nonfinite components after constructing otherwise unit directions.
    rays.N = be.array([0, 0, 1, float("inf"), float("nan")])

    with be.errstate(divide="raise", invalid="raise"):
        result = reference.path_length(rays, 1.0)

    assert result[1] == 0
    assert be.isnan(result[0])
    assert be.all(be.isnan(result[2:]))
    assert result.shape == rays.z.shape
    assert result.dtype == rays.z.dtype


def test_tiny_off_plane_parallel_offset_is_not_coplanar(set_test_backend):
    result = PlanarReference((0, 0, 0), (0, 0, 1)).path_length(
        axial_rays([1e-20], [0]),
        1.0,
    )

    assert be.all(be.isnan(result))


@pytest.mark.parametrize(
    "normal_direction",
    [-2e-12, -1e-12, -5e-13, -1e-15, 1e-15, 5e-13, 1e-12, 2e-12],
)
def test_all_finite_nonzero_denominators_are_preserved(
    set_test_backend, normal_direction
):
    result = PlanarReference((0, 0, 0), (0, 0, 1)).path_length(
        axial_rays([1], [normal_direction]),
        1.0,
    )

    assert_allclose(
        result,
        be.array([1 / normal_direction]),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize("scale", [1, -1, 1e-20, -1e-20, 1e20, -1e20])
def test_normal_rescaling_reversal_and_tiny_normal(set_test_backend, scale):
    rays = axial_rays([-1, 1, 1, 0], [5e-13, -5e-13, 0, 0])

    result = PlanarReference((0, 0, 0), (0, 0, scale)).path_length(rays, 1.0)

    assert_allclose(result[[0, 1, 3]], be.array([-2e12, -2e12, 0]), rtol=1e-12)
    assert be.isnan(result[2])


@pytest.mark.parametrize(
    ("point", "normal"),
    [
        ((0, 0), (0, 0, 1)),
        ((0, 0, 0, 0), (0, 0, 1)),
        ((0, [0], 0), (0, 0, 1)),
        ((0, "0", 0), (0, 0, 1)),
        ((float("nan"), 0, 0), (0, 0, 1)),
        ((float("inf"), 0, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 0)),
        ((0, 0, 0), (0, 0, 0, 1)),
        ((0, 0, 0), (0, [0], 1)),
        ((0, 0, 0), (0, "0", 1)),
        ((0, 0, 0), (0, float("nan"), 1)),
        ((0, 0, 0), (0, float("inf"), 1)),
        ((0, 0, 0), (0, 0, 0)),
    ],
)
def test_invalid_point_or_normal_is_rejected(set_test_backend, point, normal):
    with pytest.raises(ValueError):
        PlanarReference(point, normal)


@pytest.mark.parametrize("vector", ["point", "normal"])
@pytest.mark.parametrize("value", [None, 1.0])
def test_noniterable_plane_vector_reports_value_error(set_test_backend, vector, value):
    arguments = {"point": (0.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)}
    arguments[vector] = value

    with pytest.raises(ValueError, match=f"Plane {vector} must contain") as error:
        PlanarReference(**arguments)

    assert isinstance(error.value.__cause__, TypeError)


@pytest.mark.parametrize("vector", ["point", "normal"])
@pytest.mark.parametrize("dtype", [str, object], ids=["string", "object"])
def test_nonnumeric_scalar_array_reports_value_error(
    set_test_backend, vector, dtype
):
    # String scalar arrays can arrive from imported tabular data. Backend dtype
    # or finiteness errors must become the constructor's documented ValueError.
    component = np.asarray("not a coordinate", dtype=dtype)
    arguments = {"point": (0.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)}
    arguments[vector] = (0.0, 0.0, component)

    with pytest.raises(ValueError, match=f"Plane {vector} must contain") as error:
        PlanarReference(**arguments)

    assert isinstance(error.value.__cause__, TypeError)


def test_mixed_array_shape_and_dtype_are_preserved(set_test_backend):
    reference = PlanarReference((0, 0, 0), (0, 0, 1))
    rays = axial_rays([[1, -1], [0, 2]], [[1, 1], [0, 0]])

    result = reference.path_length(rays, 2.0)

    assert_allclose(result[0], be.array([2, -2]), rtol=0, atol=0)
    assert result[1, 0] == 0
    assert be.isnan(result[1, 1])
    assert result.shape == rays.z.shape
    assert result.dtype == rays.z.dtype


def test_empty_input_preserves_shape_and_dtype(set_test_backend):
    rays = axial_rays([], [])

    result = PlanarReference((0, 0, 0), (0, 0, 1)).path_length(rays, 1.0)

    assert result.shape == (0,)
    assert result.dtype == rays.z.dtype


def test_explicit_float32_dtype_survives_float64_backend_precision(set_test_backend):
    be.set_precision("float64")
    if be.get_backend() == "numpy":
        z = np.asarray([2, -2, 0, 1], dtype=np.float32)
        normal_direction = np.asarray([1, 1, 0, 0], dtype=np.float32)
    else:
        torch = pytest.importorskip("torch")
        z = torch.tensor([2, -2, 0, 1], dtype=torch.float32)
        normal_direction = torch.tensor([1, 1, 0, 0], dtype=torch.float32)

    zeros = z * 0
    rays = SimpleNamespace(
        x=zeros,
        y=zeros,
        z=z,
        L=be.sqrt(1 - normal_direction**2),
        M=zeros,
        N=normal_direction,
    )

    result = PlanarReference((0, 0, 0), (0, 0, 1)).path_length(rays, 1.5)

    assert result.dtype == rays.z.dtype
    assert_allclose(result[:3], np.asarray([3, -3, 0], dtype=np.float32), atol=0)
    assert be.isnan(result[3])


@pytest.mark.parametrize("component_kind", ["python", "scalar_array"])
@pytest.mark.parametrize(
    ("point", "normal"),
    [
        ((1e40, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1e40)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1e-50)),
    ],
    ids=["large_point", "large_normal", "tiny_normal"],
)
def test_float64_plane_operands_survive_float32_configuration(
    float32_backend, component_kind, point, normal
):
    if component_kind == "scalar_array":
        point = tuple(be.asarray(value, dtype=be.float64) for value in point)
        normal = tuple(be.asarray(value, dtype=be.float64) for value in normal)
    rays = SimpleNamespace(
        **{
            name: be.asarray(values, dtype=be.float64)
            for name, values in {
                "x": [0.0, 0.0, 0.0],
                "y": [0.0, 0.0, 0.0],
                "z": [2.0, -2.0, 0.0],
                "L": [0.0, 0.0, 0.0],
                "M": [0.0, 0.0, 0.0],
                "N": [1.0, 1.0, 1.0],
            }.items()
        }
    )

    reference = PlanarReference(point, normal)
    result = reference.path_length(rays, 1.5)

    for retained, original in zip(
        (*reference.point, *reference.normal), (*point, *normal), strict=True
    ):
        assert retained is original
    assert result.dtype == rays.z.dtype
    assert result.shape == rays.z.shape
    assert_allclose(result, be.asarray([3.0, -3.0, 0.0], dtype=be.float64), atol=0)
    assert be.get_precision() == 32


@pytest.mark.parametrize("component_kind", ["python", "scalar_array"])
@pytest.mark.parametrize("vector", ["point", "normal"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_plane_operands_rejected_under_float32_configuration(
    float32_backend, component_kind, vector, value
):
    if component_kind == "scalar_array":
        value = be.asarray(value, dtype=be.float64)
    point = (value, 0.0, 0.0) if vector == "point" else (0.0, 0.0, 0.0)
    normal = (0.0, 0.0, value) if vector == "normal" else (0.0, 0.0, 1.0)

    with pytest.raises(ValueError, match=f"Plane {vector}"):
        PlanarReference(point, normal)


@pytest.mark.parametrize("vector", ["point", "normal"])
@pytest.mark.parametrize("value", [[1.0], [[1.0]], 1.0 + 2.0j])
def test_nonscalar_or_complex_array_plane_components_rejected(
    float32_backend, vector, value
):
    dtype = np.complex128 if isinstance(value, complex) else be.float64
    component = be.asarray(value, dtype=dtype)
    point = (component, 0.0, 0.0) if vector == "point" else (0.0, 0.0, 0.0)
    normal = (0.0, 0.0, component) if vector == "normal" else (0.0, 0.0, 1.0)

    with pytest.raises(ValueError, match=f"Plane {vector}"):
        PlanarReference(point, normal)


def test_float64_plane_scalar_tensors_retain_gradients(float32_backend):
    if be.get_backend() != "torch":
        pytest.skip("Requires Torch autograd.")
    point_x = be.asarray(1e40, dtype=be.float64).requires_grad_(True)
    point_z = be.asarray(0.0, dtype=be.float64).requires_grad_(True)
    normal_x = be.asarray(0.0, dtype=be.float64).requires_grad_(True)
    normal_z = be.asarray(1e40, dtype=be.float64).requires_grad_(True)
    rays = SimpleNamespace(
        x=be.asarray([0.0], dtype=be.float64),
        y=be.asarray([0.0], dtype=be.float64),
        z=be.asarray([2.0], dtype=be.float64),
        L=be.asarray([0.0], dtype=be.float64),
        M=be.asarray([0.0], dtype=be.float64),
        N=be.asarray([1.0], dtype=be.float64),
    )

    reference = PlanarReference((point_x, 0.0, point_z), (normal_x, 0.0, normal_z))
    assert reference.point[0] is point_x
    assert reference.point[2] is point_z
    assert reference.normal[0] is normal_x
    assert reference.normal[2] is normal_z
    result = reference.path_length(rays, 1.5)
    result.sum().backward()

    assert_allclose(result, 3.0, rtol=1e-12, atol=0)
    assert_allclose(point_x.grad, 0.0, rtol=0, atol=0)
    assert_allclose(point_z.grad, -1.5, rtol=1e-12, atol=0)
    assert_allclose(normal_x.grad, -1.5, rtol=1e-12, atol=0)
    assert_allclose(normal_z.grad, 0.0, rtol=0, atol=1e-54)
    assert be.get_precision() == 32


def test_torch_gradients_with_invalid_lane_and_differentiable_medium(
    set_test_backend,
):
    if be.get_backend() != "torch":
        pytest.skip("Requires Torch autograd.")

    z = be.array([1.0, 1.0, 0.0, -1.0]).requires_grad_(True)
    normal_direction = be.array([1.0, 0.0, 1.0, 5e-13]).requires_grad_(True)
    medium = be.array(1.5).requires_grad_(True)
    zeros = be.zeros_like(z)
    rays = SimpleNamespace(
        x=zeros,
        y=zeros,
        z=z,
        # Keep direction components independent to avoid sqrt's derivative
        # singularity at N=1 while supplying unit directions at evaluation.
        L=be.array([0.0, 1.0, 0.0, 1.0]),
        M=zeros,
        N=normal_direction,
    )

    result = PlanarReference((0, 0, 0), (0, 0, 1)).path_length(rays, medium)
    be.sum(result[be.isfinite(result)]).backward()

    assert_allclose(z.grad, be.array([1.5, 0, 1.5, 3e12]), rtol=1e-12)
    assert_allclose(
        normal_direction.grad,
        be.array([-1.5, 0, 0, 6e24]),
        rtol=1e-12,
    )
    assert_allclose(medium.grad, be.array(1 - 2e12), rtol=1e-12)
    assert be.all(be.isfinite(z.grad))
    assert be.all(be.isfinite(normal_direction.grad))
    assert be.isfinite(medium.grad)
