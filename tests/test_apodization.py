from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.apodization import (
    BaseApodization,
    GaussianApodization,
    UniformApodization,
)
from optiland.samples.objectives import CookeTriplet


def test_uniform_apodization_get_intensity(set_test_backend):
    apod = UniformApodization()
    px = be.array([0.5])
    py = be.array([0.5])
    intensity = apod.get_intensity(px, py)
    assert be.all(intensity == be.array([1.0])), (
        "Intensity should be 1.0 for UniformApodization"
    )

    px_array = be.array([-1.0, 0.0, 1.0])
    py_array = be.array([-1.0, 0.0, 1.0])
    intensity_array = apod.get_intensity(px_array, py_array)
    assert be.all(intensity_array == be.ones_like(px_array)), (
        "Intensity should be all ones for UniformApodization with array inputs"
    )


def test_gaussian_apodization_get_intensity(set_test_backend):
    sigma = 0.5
    apod = GaussianApodization(sigma=sigma)
    px = be.array([0.0])
    py = be.array([0.0])
    # Expected: exp(-(0^2 + 0^2) / (2 * 0.5^2)) = exp(0) = 1.0
    intensity_center = apod.get_intensity(px, py)
    assert be.allclose(intensity_center, be.array([1.0])), (
        "Intensity at center should be 1.0 for GaussianApodization"
    )

    px_edge = be.array([sigma])
    py_edge = be.array([0.0])
    # Expected: exp(-(sigma^2 + 0^2) / (2 * sigma^2)) = exp(-1/2)
    intensity_edge = apod.get_intensity(px_edge, py_edge)
    expected_intensity_edge = be.exp(be.array([-0.5]))
    assert be.allclose(intensity_edge, expected_intensity_edge), (
        "Intensity at sigma distance not as expected"
    )

    # Test with array inputs
    px_array = be.array([-sigma, 0.0, sigma])
    py_array = be.array([0.0, sigma, 0.0])
    # Expected: exp(-([-s,0,s]^2 + [0,s,0]^2) / (2s^2)) = exp(-([s^2,s^2,s^2])/(2s^2)) = exp(-0.5)
    intensity_array = apod.get_intensity(px_array, py_array)
    expected_intensity_array = be.exp(be.array([-0.5, -0.5, -0.5]))
    assert be.allclose(intensity_array, expected_intensity_array), (
        "Intensity array not as expected for GaussianApodization"
    )


def test_apodization_to_dict_uniform(set_test_backend):
    apod = UniformApodization()
    data = apod.to_dict()
    assert data == {"type": "UniformApodization"}


def test_apodization_to_dict_gaussian(set_test_backend):
    sigma = 0.75
    apod = GaussianApodization(sigma=sigma)
    data = apod.to_dict()
    assert data == {"type": "GaussianApodization", "sigma": sigma}


def test_apodization_from_dict_uniform(set_test_backend):
    data = {"type": "UniformApodization"}
    apod = BaseApodization.from_dict(data)
    assert isinstance(apod, UniformApodization)


def test_apodization_from_dict_gaussian(set_test_backend):
    sigma = 0.6
    data = {"type": "GaussianApodization", "sigma": sigma}
    apod = BaseApodization.from_dict(data)
    assert isinstance(apod, GaussianApodization)
    assert apod.sigma == sigma


def test_apodization_from_dict_gaussian_default_sigma(set_test_backend):
    data = {"type": "GaussianApodization"}  # Missing sigma
    apod = BaseApodization.from_dict(data)
    assert isinstance(apod, GaussianApodization)
    assert apod.sigma == 1.0  # Default sigma


def test_apodization_from_dict_unknown(set_test_backend):
    data = {"type": "UnknownApodization"}
    with pytest.raises(ValueError):
        BaseApodization.from_dict(data)


def test_set_apodization_uniform(set_test_backend):
    apod = UniformApodization()
    lens = CookeTriplet()
    lens.updater.set_apodization(apod)
    assert lens.apodization == apod, "Aperture should be set to UniformApodization"


def test_set_apodization_gaussian(set_test_backend):
    apod = GaussianApodization()
    lens = CookeTriplet()
    lens.updater.set_apodization(apod)
    assert lens.apodization == apod, "Aperture should be set to GaussianApodization"


def test_set_apodization_invalid_type(set_test_backend):
    lens = CookeTriplet()
    with pytest.raises(TypeError):
        lens.updater.set_apodization(123)


def test_set_apodization_none(set_test_backend):
    lens = CookeTriplet()
    lens.updater.set_apodization(None)
    assert lens.apodization is None, "Aperture should be set to None"


def test_trace_apodization(set_test_backend):
    lens = CookeTriplet()
    lens.updater.set_apodization(GaussianApodization(sigma=0.1))
    rays = lens.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=256, distribution="uniform")

    x = lens.surfaces.x[4, :]
    y = lens.surfaces.y[4, :]
    r = be.sqrt(x**2 + y**2)
    center_intensities = rays.i[r <= 0.5]
    edge_intensities = rays.i[r > 0.5]
    print(be.sum(center_intensities), be.sum(edge_intensities))
    assert be.sum(center_intensities) > be.sum(edge_intensities)
