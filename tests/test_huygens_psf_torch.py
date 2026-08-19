from __future__ import annotations

import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import optiland.backend as be
from optiland.psf.huygens_fresnel import HuygensPSF
from optiland.samples.objectives import CookeTriplet


@pytest.fixture(autouse=True)
def set_torch_backend():
    """Ensure the torch backend is used for all tests in this module, and restore numpy after."""
    if TORCH_AVAILABLE:
        be.set_backend("torch")
        yield
        be.set_backend("numpy")
    else:
        yield


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
class TestHuygensPSFTorch:
    WAVELENGTH_GREEN = 0.550
    NUM_RAYS_LOW = 32
    IMAGE_SIZE_LOW = 32

    @pytest.fixture()
    def cooke_triplet_optic(self):
        return CookeTriplet()

    def test_huygens_psf_torch_initialization(self, cooke_triplet_optic):
        """
        Tests the initialization of HuygensPSF with the torch backend.
        """
        psf_instance = HuygensPSF(
            optic=cooke_triplet_optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )

        assert psf_instance.psf is not None
        assert be.is_torch_tensor(psf_instance.psf)
        assert psf_instance.psf.shape == (self.IMAGE_SIZE_LOW, self.IMAGE_SIZE_LOW)

    def test_huygens_psf_torch_strehl_ratio(self, cooke_triplet_optic):
        """
        Tests the Strehl ratio calculation with the torch backend.
        """
        psf_instance = HuygensPSF(
            optic=cooke_triplet_optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )
        sr = psf_instance.strehl_ratio()
        assert isinstance(sr, float)
        assert 0 < sr <= 1.005

    def test_torch_and_numpy_psf_consistency(self, cooke_triplet_optic):
        """
        Tests that the PSF calculated with torch is consistent with numpy.
        """
        # Set torch to double precision for a fair comparison with numpy
        be.set_backend("torch")
        be.set_precision("float64")
        torch_optic = CookeTriplet()
        psf_torch_instance = HuygensPSF(
            optic=torch_optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )
        psf_torch = be.to_numpy(psf_torch_instance.psf)

        # Calculate PSF with numpy backend
        be.set_backend("numpy")
        numpy_optic = CookeTriplet()
        psf_numpy_instance = HuygensPSF(
            optic=numpy_optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )
        psf_numpy = psf_numpy_instance.psf

        # Compare the results
        assert psf_torch.shape == psf_numpy.shape
        assert be.allclose(psf_torch, psf_numpy, atol=1e-5)
