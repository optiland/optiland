"""Unit tests for the GRINPropagation model."""

from __future__ import annotations

import numpy as np
import pytest

from optiland import backend as be
from optiland.materials import BaseMaterial, GradientMaterial, IdealMaterial
from optiland.optic import Optic
from optiland.propagation.grin import GRINPropagation
from optiland.rays.real_rays import RealRays

from ..utils import assert_allclose

N0 = 1.6
A = 0.01  # n = n0 (1 - A r² / 2): paraxial rays oscillate with period 2π/√A
QUARTER_PITCH = np.pi / 2 / np.sqrt(A)


def _optic(material, thickness, radii=(np.inf, np.inf), epd=1.0, field=0.0):
    """Object at infinity, one medium between two surfaces, then an image."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    optic.surfaces.add(
        index=1, radius=radii[0], thickness=thickness, material=material, is_stop=True
    )
    optic.surfaces.add(index=2, radius=radii[1], thickness=10.0, material="air")
    optic.surfaces.add(index=3, radius=np.inf)
    optic.set_aperture(aperture_type="EPD", value=epd)
    optic.fields.set_type("angle")
    optic.fields.add(y=field)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _rod(**model):
    material = GradientMaterial(
        n0=N0, nr2=-N0 * A / 2, propagation_model=GRINPropagation(**model)
    )
    return _optic(material, QUARTER_PITCH)


def test_grin_propagation_raises_not_implemented_error():
    """Verify that GRINPropagation.propagate raises NotImplementedError."""
    model = GRINPropagation()
    rays = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1], wavelength=[0.5]
    )

    with pytest.raises(NotImplementedError):
        model.propagate(rays, t=10.0)


@pytest.mark.parametrize("kwargs", [{"step_size": 0.0}, {"max_steps": 0}])
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        GRINPropagation(**kwargs)


def test_uniform_medium_matches_ideal_material(set_test_backend):
    """Zero gradients must reproduce homogeneous tracing on curved faces."""
    Px = be.array([0.0, 0.5, 0.6])
    Py = be.array([0.3, -0.7, 0.7])
    traces = []
    for material in (IdealMaterial(1.6), GradientMaterial(n0=1.6)):
        optic = _optic(material, 5.0, radii=(20.0, -30.0), epd=6.0, field=5.0)
        traces.append(optic.trace_generic(0.0, 1.0, Px, Py, 0.55))
    ideal, grin = traces
    for name in ("x", "y", "z", "L", "M", "N", "opd", "i"):
        assert_allclose(getattr(grin, name), getattr(ideal, name), atol=1e-9)


def test_quarter_pitch_rod_focuses_collimated_ray(set_test_backend):
    """A paraxial ray y(z) = y0 cos(√A z) crosses the axis at a quarter pitch."""
    y0 = 0.01
    optic = _rod()
    optic.trace_generic(0.0, 0.0, be.array([0.0]), be.array([2 * y0]), 0.55)
    exit_face = optic.surfaces[2]

    # Residuals are the O(A y0²) ≈ 1e-6 paraxial approximation, not RK4 error.
    assert_allclose(exit_face.z, be.array([QUARTER_PITCH]), atol=1e-9)
    assert_allclose(exit_face.y, be.array([0.0]), atol=1e-7)
    # In air after the flat exit face: sin = n0 · dy/dz = -n0 · y0 · √A.
    assert_allclose(exit_face.M, be.array([-N0 * y0 * np.sqrt(A)]), rtol=1e-5)


def test_axial_gradient_optical_path(set_test_backend):
    """On axis, the optical path is ∫ n dz."""
    thickness = 10.0
    material = GradientMaterial(n0=1.5, nz1=0.02, nz2=-0.001, nz3=2e-5)
    optic = _optic(material, thickness)
    optic.trace_generic(0.0, 0.0, be.array([0.0]), be.array([0.0]), 0.55)

    expected = 1.5 * 10 + 0.02 * 10**2 / 2 - 0.001 * 10**3 / 3 + 2e-5 * 10**4 / 4
    entrance_opd = optic.surfaces[1].opd
    assert_allclose(optic.surfaces[2].opd - entrance_opd, be.array([expected]))


def test_rays_land_on_curved_exit_face(set_test_backend):
    material = GradientMaterial(n0=1.6, nr2=-0.004, nz1=0.01)
    optic = _optic(material, 6.0, radii=(25.0, -15.0), epd=8.0)
    optic.trace(0.0, 0.0, 0.55, num_rays=12, distribution="hexapolar")
    exit_face = optic.surfaces[2]
    vertex_z = exit_face.geometry.cs.z
    sag = exit_face.geometry.sag(exit_face.x, exit_face.y)
    assert_allclose(exit_face.z - vertex_z, sag, atol=1e-9)


def test_entrance_refraction_uses_local_index(set_test_backend):
    """Snell's law at a flat face holds with n at the hit point, not n0."""
    angle = 10.0
    material = GradientMaterial(n0=N0, nr2=-0.02)
    optic = _optic(material, 5.0, epd=4.0, field=angle)
    optic.trace_generic(0.0, 1.0, be.array([0.0]), be.array([1.0]), 0.55)
    entrance = optic.surfaces[1]
    n_hit = N0 - 0.02 * (entrance.x**2 + entrance.y**2)
    sine = be.array([np.sin(np.radians(angle))])
    assert_allclose(n_hit * entrance.M, sine)
    assert be.to_numpy(n_hit)[0] < N0 - 1e-3


def test_medium_moves_with_tilted_decentered_faces(set_test_backend):
    """The index is defined in the entrance frame, so a rigid move is invisible."""
    tilt = 0.2
    material = GradientMaterial(n0=N0, nr2=-N0 * A / 2)
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, z=-np.inf)
    optic.surfaces.add(
        index=1, radius=np.inf, x=0.3, y=0.5, z=0.0, rx=tilt, material=material
    )
    optic.surfaces.add(
        index=2,
        radius=np.inf,
        x=0.3,
        y=0.5 - QUARTER_PITCH * np.sin(tilt),
        z=QUARTER_PITCH * np.cos(tilt),
        rx=tilt,
        material="air",
    )
    entrance, exit_face = optic.surfaces[1], optic.surfaces[2]

    rays = RealRays(
        x=[0.0], y=[0.01], z=[0.0], L=[0.0], M=[0.0], N=[1.0],
        intensity=[1.0], wavelength=[0.55],
    )  # fmt: skip
    entrance.geometry.globalize(rays)
    exit_face.geometry.localize(rays)
    material.propagation_model.propagate_to_surface(rays, exit_face)

    assert_allclose(rays.z, be.array([0.0]), atol=1e-9)
    assert_allclose(rays.y, be.array([0.0]), atol=1e-7)
    assert_allclose(rays.M, be.array([-0.01 * np.sqrt(A)]), rtol=1e-5)


def test_rays_that_do_not_reach_the_exit_are_vignetted(set_test_backend):
    optic = _rod(max_steps=5)
    rays = optic.trace_generic(0.0, 0.0, be.array([0.0]), be.array([0.5]), 0.55)
    assert_allclose(rays.i, be.array([0.0]))


def test_non_gradient_medium_is_rejected(set_test_backend):
    material = IdealMaterial(1.5, propagation_model=GRINPropagation())
    optic = _optic(material, 5.0)
    with pytest.raises(TypeError, match="index_and_gradient"):
        optic.trace_generic(0.0, 0.0, be.array([0.0]), be.array([0.5]), 0.55)


def test_paraxial_tracing_is_rejected(set_test_backend):
    optic = _rod()
    with pytest.raises(NotImplementedError, match="Paraxial"):
        optic.paraxial.f2()


def test_serialization_round_trip():
    material = GradientMaterial(
        n0=1.7, nr2=-0.01, nz1=0.002, propagation_model=GRINPropagation(0.05, 500)
    )
    restored = BaseMaterial.from_dict(material.to_dict())

    assert isinstance(restored, GradientMaterial)
    assert restored.to_dict() == material.to_dict()
    assert restored.propagation_model.step_size == 0.05
    assert restored.propagation_model.max_steps == 500


def test_display_path_samples_the_curved_ray(set_test_backend):
    optic = _rod()
    optic.trace_generic(0.0, 0.0, be.array([0.0, 0.0]), be.array([0.4, -0.8]), 0.55)
    entrance, exit_face = optic.surfaces[1], optic.surfaces[2]
    rays = RealRays(
        entrance.x, entrance.y, entrance.z, entrance.L, entrance.M, entrance.N,
        entrance.intensity, be.full_like(entrance.x, 0.55),
    )  # fmt: skip

    path = be.to_numpy(GRINPropagation().display_path(rays, exit_face))

    assert path.shape[1:] == (2, 3)
    assert path.shape[0] > 100  # ~16 mm in 0.1 mm steps
    for ray in range(2):
        points = path[:, ray][np.isfinite(path[:, ray]).all(axis=1)]
        assert_allclose(points[-1], np.column_stack(
            [be.to_numpy(c) for c in (exit_face.x, exit_face.y, exit_face.z)]
        )[ray])  # fmt: skip
        # The path bends: y = y0 cos(√A z) sits 0.21 y0 off the chord at L/2.
        mid = points[len(points) // 2]
        chord_mid = (points[0][1] + points[-1][1]) / 2
        assert abs(mid[1] - chord_mid) > 0.15 * abs(points[0][1])


def test_torch_gradient_flows_to_index_coefficient():
    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    be.set_precision("float64")
    try:
        nr2 = torch.tensor(-N0 * A / 2, dtype=torch.float64, requires_grad=True)
        optic = _optic(GradientMaterial(n0=N0, nr2=nr2), QUARTER_PITCH)
        rays = optic.trace_generic(0.0, 0.0, be.array([0.0]), be.array([0.4]), 0.55)
        rays.y.sum().backward()
        assert nr2.grad is not None
        assert torch.isfinite(nr2.grad).all()
        assert nr2.grad.abs().item() > 0
    finally:
        be.set_backend("numpy")
