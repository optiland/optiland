"""Jones pupil normalization agrees with physical exit-field evaluation."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.analysis.jones_pupil import JonesPupil
from optiland.apodization import GaussianApodization
from optiland.coatings import CartesianJonesCoating, SimpleCoating
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.rays import PolarizedRays, create_polarization
from tests.utils import assert_allclose


def planar_optic(angle=0, coating=None):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=1, is_stop=True, coating=coating)
    optic.surfaces.add(index=2)
    optic.set_aperture(aperture_type="EPD", value=1)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(x=angle, y=0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    return optic


@pytest.mark.parametrize("angle", [0, 30, -30])
def test_lossless_oblique_pupil_uses_transverse_source_basis(angle, set_test_backend):
    optic = planar_optic(angle)
    pupil = JonesPupil(optic, field=(1, 0), grid_size=3)
    assert_allclose(pupil.data[0]["J"], be.tile(be.eye(2), (9, 1, 1)), atol=1e-12)
    assert optic.polarization == "ignore"


def test_pupil_retains_apodization_and_scalar_loss(set_test_backend):
    optic = planar_optic(coating=SimpleCoating(0.25))
    optic.updater.set_apodization(GaussianApodization(sigma=0.7))
    data = JonesPupil(optic, grid_size=3).data[0]
    expected = 0.5 * be.exp(-(data["Px"] ** 2 + data["Py"] ** 2) / (4 * 0.7**2))
    assert_allclose(data["J"], expected[:, None, None] * be.eye(2), atol=1e-12)


def test_pupil_includes_bulk_absorption(set_test_backend):
    optic = planar_optic()
    optic.surfaces[1].material_post = IdealMaterial(1, 1e-6)
    data = JonesPupil(optic, grid_size=3).data[0]
    amplitude = be.exp(be.array(-2 * be.pi * 1e-6 / 0.55 * 1000))
    assert_allclose(data["J"], amplitude * be.tile(be.eye(2), (9, 1, 1)))


@pytest.mark.parametrize("source", ["H", "V", "RCP", "unpolarized"])
def test_pupil_columns_are_independent_of_selected_source(source, set_test_backend):
    # H is completely blocked; evaluating it must not erase the V column.
    optic = planar_optic(coating=CartesianJonesCoating(0, 0.3j, 0, 0.4))
    state = create_polarization(source)
    optic.updater.set_polarization(state)
    first = JonesPupil(optic, grid_size=3).data[0]["J"]
    expected = be.to_complex(be.array([[0, 0], [0, 0.4]]))
    expected[0, 1] = 0.3j
    assert_allclose(first, be.tile(expected, (9, 1, 1)))
    assert_allclose(JonesPupil(optic, grid_size=3).data[0]["J"], first)
    assert optic.polarization_state is state


def test_clipped_pupil_rows_are_zero(set_test_backend):
    optic = planar_optic(coating=SimpleCoating(0.25))
    optic.surfaces[1].aperture = RadialAperture(r_max=0.1)
    matrix = JonesPupil(optic, grid_size=3).data[0]["J"]
    assert_allclose(matrix[4], 0.5 * be.eye(2))
    assert_allclose(matrix[be.arange(9) != 4], 0)


@pytest.mark.parametrize("clipped", [False, True])
def test_invalid_output_geometry_is_masked_only_for_dark_fields(
    clipped, monkeypatch, set_test_backend
):
    optic = planar_optic()
    rays = PolarizedRays(*[be.full(9, float(v)) for v in (0, 0, 0, 0, 0, 1, 1, 0.55)])
    if clipped:
        rays.clip(be.ones(9))
    rays.L = rays.M = rays.N = be.full(9, float("nan"))
    rays.p = be.full((9, 3, 3), float("nan"))
    monkeypatch.setattr(optic, "trace_generic", lambda **kwargs: rays)
    matrix = JonesPupil(optic, grid_size=3).data[0]["J"]
    if clipped:
        assert_allclose(matrix, 0)
    else:
        assert be.all(be.isnan(matrix))


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("z", [0.0, 1e-18, 1e-8])
def test_output_parallel_to_x_preserves_transverse_field_power(
    sign, z, monkeypatch, set_test_backend
):
    optic = planar_optic()
    rays = PolarizedRays(*[be.full(9, float(v)) for v in (0, 0, 0, 0, 0, 1, 1, 0.55)])
    rays.L0, rays.M0, rays.N0 = be.copy(rays.L), be.copy(rays.M), be.copy(rays.N)
    rays.L = be.full(9, sign * (1 - z**2) ** 0.5)
    rays.M, rays.N = be.zeros(9), be.full(9, z)
    rays.update()
    monkeypatch.setattr(optic, "trace_generic", lambda **kwargs: rays)
    matrix = JonesPupil(optic, grid_size=3).data[0]["J"]
    # A Z-to-X bend transports H to -sign*Z and leaves V along Y.
    # With Y as the reference axis, the output basis is (Y, sign*Z).
    expected = be.array([[0.0, 1.0], [-1.0, 0.0]]) if z == 0 else be.eye(2)
    assert_allclose(matrix, be.tile(expected, (9, 1, 1)), atol=1e-14)

