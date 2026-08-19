from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.apodization import (
    BaseApodization,
    CosineSquaredApodization,
    GaussianApodization,
    HannApodization,
    PolynomialApodization,
    SuperGaussianApodization,
    TukeyApodization,
)
from optiland.optic import Optic


@pytest.fixture
def Px_Py():
    """Provides a standard set of pupil coordinates for testing."""
    px, py = be.meshgrid(be.linspace(-1.5, 1.5, 5), be.linspace(-1.5, 1.5, 5))
    return px.ravel(), py.ravel()


def test_supergaussian_apodization(set_test_backend, Px_Py):
    """Test the SuperGaussianApodization class."""
    Px, Py = Px_Py
    apod = SuperGaussianApodization(w=1.0, n=4)
    intensity = apod.get_intensity(Px, Py)
    assert be.isclose(intensity[12], be.array(1.0))  # Center of the pupil
    assert intensity[0] < 1e-6  # Edge of the pupil

    # Test parameter validation
    with pytest.raises(ValueError):
        SuperGaussianApodization(w=0, n=4)
    with pytest.raises(ValueError):
        SuperGaussianApodization(w=1, n=1)


def test_supergaussian_to_from_dict(set_test_backend):
    """Test the to_dict and from_dict methods for SuperGaussianApodization."""
    apod = SuperGaussianApodization(w=0.8, n=6)
    data = apod.to_dict()
    assert data["type"] == "SuperGaussianApodization"
    assert data["w"] == 0.8
    assert data["n"] == 6

    new_apod = BaseApodization.from_dict(data)
    assert isinstance(new_apod, SuperGaussianApodization)
    assert new_apod.w == 0.8
    assert new_apod.n == 6


def test_cosine_squared_apodization(set_test_backend, Px_Py):
    """Test the CosineSquaredApodization class."""
    Px, Py = Px_Py
    apod = CosineSquaredApodization(R=1.0)
    intensity = apod.get_intensity(Px, Py)
    r = (Px**2 + Py**2) ** 0.5
    assert be.isclose(intensity[12], be.array(1.0))  # Center
    assert be.all(intensity[r >= 1.0] == 0.0)  # Outside radius

    with pytest.raises(ValueError):
        CosineSquaredApodization(R=0)


def test_cosine_squared_to_from_dict(set_test_backend):
    """Test serialization for CosineSquaredApodization."""
    apod = CosineSquaredApodization(R=1.2)
    data = apod.to_dict()
    assert data["type"] == "CosineSquaredApodization"
    assert data["R"] == 1.2
    new_apod = BaseApodization.from_dict(data)
    assert isinstance(new_apod, CosineSquaredApodization)
    assert new_apod.R == 1.2


def test_tukey_apodization(set_test_backend, Px_Py):
    """Test the TukeyApodization class."""
    Px, Py = Px_Py
    apod = TukeyApodization(R=1.0, alpha=0.5)
    intensity = apod.get_intensity(Px, Py)
    r = (Px**2 + Py**2) ** 0.5
    flat_region_end = 1.0 * (1 - 0.5 / 2)
    assert be.all(intensity[r <= flat_region_end] == 1.0)  # Flat top
    assert be.all(intensity[r >= 1.0] == 0.0)  # Outside radius

    with pytest.raises(ValueError):
        TukeyApodization(R=0)
    with pytest.raises(ValueError):
        TukeyApodization(alpha=-0.1)


def test_tukey_to_from_dict(set_test_backend):
    """Test serialization for TukeyApodization."""
    apod = TukeyApodization(R=1.5, alpha=0.8)
    data = apod.to_dict()
    assert data["type"] == "TukeyApodization"
    assert data["R"] == 1.5
    assert data["alpha"] == 0.8
    new_apod = BaseApodization.from_dict(data)
    assert isinstance(new_apod, TukeyApodization)
    assert new_apod.R == 1.5
    assert new_apod.alpha == 0.8


def test_polynomial_apodization(set_test_backend, Px_Py):
    """Test the PolynomialApodization class."""
    Px, Py = Px_Py
    apod = PolynomialApodization(R=1.0, p=2)
    intensity = apod.get_intensity(Px, Py)
    r = (Px**2 + Py**2) ** 0.5
    assert be.isclose(intensity[12], be.array(1.0))  # Center
    assert be.all(intensity[r >= 1.0] == 0.0)  # Outside radius

    with pytest.raises(ValueError):
        PolynomialApodization(R=0)
    with pytest.raises(ValueError):
        PolynomialApodization(p=-1)


def test_polynomial_to_from_dict(set_test_backend):
    """Test serialization for PolynomialApodization."""
    apod = PolynomialApodization(R=2.0, p=3)
    data = apod.to_dict()
    assert data["type"] == "PolynomialApodization"
    assert data["R"] == 2.0
    assert data["p"] == 3
    new_apod = BaseApodization.from_dict(data)
    assert isinstance(new_apod, PolynomialApodization)
    assert new_apod.R == 2.0
    assert new_apod.p == 3


def test_hann_apodization(set_test_backend, Px_Py):
    """Test the HannApodization class."""
    Px, Py = Px_Py
    apod = HannApodization(D=2.0)
    intensity = apod.get_intensity(Px, Py)
    r = (Px**2 + Py**2) ** 0.5
    assert be.isclose(
        intensity[12], be.array(0.0)
    )  # Center should be 0.5 * (1 - cos(0)) = 0
    assert be.all(intensity[r >= 1.0] == 0.0)  # Outside radius

    with pytest.raises(ValueError):
        HannApodization(D=0)


def test_hann_to_from_dict(set_test_backend):
    """Test serialization for HannApodization."""
    apod = HannApodization(D=3.0)
    data = apod.to_dict()
    assert data["type"] == "HannApodization"
    assert data["D"] == 3.0
    new_apod = BaseApodization.from_dict(data)
    assert isinstance(new_apod, HannApodization)
    assert new_apod.D == 3.0


def test_set_apodization_by_string(set_test_backend):
    """Test setting apodization using a string identifier."""
    optic = Optic()
    optic.updater.set_apodization("GaussianApodization", sigma=0.8)
    assert isinstance(optic.apodization, GaussianApodization)
    assert optic.apodization.sigma == 0.8

    optic.updater.set_apodization("TukeyApodization", R=1.2, alpha=0.3)
    assert isinstance(optic.apodization, TukeyApodization)
    assert optic.apodization.R == 1.2
    assert optic.apodization.alpha == 0.3


def test_set_apodization_by_dict(set_test_backend):
    """Test setting apodization using a dictionary."""
    optic = Optic()
    apod_dict = {"type": "PolynomialApodization", "R": 1.5, "p": 2}
    optic.updater.set_apodization(apod_dict)
    assert isinstance(optic.apodization, PolynomialApodization)
    assert optic.apodization.R == 1.5
    assert optic.apodization.p == 2


def test_set_apodization_invalid_string(set_test_backend):
    """Test setting apodization with an invalid string identifier."""
    optic = Optic()
    with pytest.raises(ValueError):
        optic.updater.set_apodization("InvalidApodizationName")


def test_set_apodization_invalid_type(set_test_backend):
    """Test setting apodization with an invalid type."""
    optic = Optic()
    with pytest.raises(TypeError):
        optic.updater.set_apodization(123)
