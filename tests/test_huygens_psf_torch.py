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


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
class TestImageVertexGradients:
    """The Huygens ideal-normalization point must stay differentiable.

    The normalization anchors at the full 3-D image-surface vertex; its
    construction must not sever trainable vertex coordinates from the
    autograd graph (be.full extracts a Python scalar from a tensor fill
    value on torch, which silently detached them).
    """

    @staticmethod
    def _finite_singlet(dx=0.0, dy=0.0, dz=0.0):
        """Small nonsymmetric finite-conjugate singlet for gradient tests."""
        from optiland.optic import Optic

        optic = Optic(name="huygens-grad-probe")
        optic.surfaces.add(index=0, radius=be.inf, thickness=60.0)
        optic.surfaces.add(
            index=1, radius=25.84, thickness=4.0, material="N-BK7", is_stop=True
        )
        optic.surfaces.add(index=2, radius=be.inf, thickness=46.0)
        optic.surfaces.add(index=3)
        optic.set_aperture(aperture_type="EPD", value=10.0)
        optic.fields.set_type("angle")
        optic.fields.add(y=0.0)
        optic.wavelengths.add(value=0.55, is_primary=True)
        if dx or dy or dz:
            for surf in optic.surfaces:
                cs = surf.geometry.cs
                for name, delta in (("x", dx), ("y", dy), ("z", dz)):
                    value = getattr(cs, name)
                    if bool(be.all(be.isfinite(value))):
                        setattr(cs, name, value + delta)
        return optic

    def test_image_vertex_grid_preserves_grad(self):
        """Unit check: the shared helper keeps requires_grad tensors live."""
        from optiland.psf.huygens_fresnel import image_vertex_grid

        be.set_precision("float64")
        be.grad_mode.enable()
        try:
            optic = self._finite_singlet(dx=1.5, dy=-2.0, dz=0.75)
            cs = optic.surfaces[-1].geometry.cs
            cs.x = cs.x.detach().clone().requires_grad_(True)
            cs.y = cs.y.detach().clone().requires_grad_(True)
            cs.z = cs.z.detach().clone().requires_grad_(True)

            image_x, image_y, image_z = image_vertex_grid(optic)
            assert image_x.requires_grad
            assert image_y.requires_grad
            assert image_z.requires_grad
            total = image_x.sum() + image_y.sum() + image_z.sum()
            total.backward()
            for leaf in (cs.x, cs.y, cs.z):
                assert leaf.grad is not None
                assert float(leaf.grad) == pytest.approx(1.0)
        finally:
            be.grad_mode.disable()

    def test_vertex_with_nonzero_xyz_feeds_normalization(self):
        """The normalization uses the real 3-D vertex (all of x, y and z
        nonzero): rigid transverse translation leaves the ideal
        normalization invariant, because the reference point follows the
        vertex."""
        from optiland.psf.huygens_fresnel import ScalarHuygensPSF

        be.set_precision("float64")
        ref = ScalarHuygensPSF(
            self._finite_singlet(),
            field=(0, 0),
            wavelength=0.55,
            num_rays=32,
            image_size=8,
        )
        moved = ScalarHuygensPSF(
            self._finite_singlet(dx=4.0, dy=-2.5),
            field=(0, 0),
            wavelength=0.55,
            num_rays=32,
            image_size=8,
        )
        vx, vy, vz = moved.optic.surfaces[-1].geometry.cs.position_in_gcs
        assert float(vx) != 0.0
        assert float(vy) != 0.0
        assert float(vz) != 0.0
        norm_ref = float(be.to_numpy(ref._get_normalization()))
        norm_moved = float(be.to_numpy(moved._get_normalization()))
        assert norm_moved == pytest.approx(norm_ref, rel=1e-3)

    def test_normalization_matches_numpy_forward(self):
        """Torch forward normalization equals the numpy value."""
        from optiland.psf.huygens_fresnel import ScalarHuygensPSF

        be.set_precision("float64")
        torch_psf = ScalarHuygensPSF(
            self._finite_singlet(dx=1.0, dy=0.5),
            field=(0, 0),
            wavelength=0.55,
            num_rays=32,
            image_size=8,
        )
        norm_torch = float(be.to_numpy(torch_psf._get_normalization()))

        be.set_backend("numpy")
        numpy_psf = ScalarHuygensPSF(
            self._finite_singlet(dx=1.0, dy=0.5),
            field=(0, 0),
            wavelength=0.55,
            num_rays=32,
            image_size=8,
        )
        norm_numpy = float(be.to_numpy(numpy_psf._get_normalization()))
        assert norm_torch == pytest.approx(norm_numpy, rel=1e-8)

    def test_end_to_end_normalization_gradient_exists(self):
        """d(normalization)/d(image vertex z) through the full pipeline is
        finite and nonzero -- before the fix it was None/zero because
        be.full severed the vertex from the graph."""
        from optiland.psf.huygens_fresnel import ScalarHuygensPSF

        be.set_precision("float64")
        be.grad_mode.enable()
        try:
            optic = self._finite_singlet(dx=1.5, dy=-2.0)
            cs = optic.surfaces[-1].geometry.cs
            cs.z = cs.z.detach().clone().requires_grad_(True)
            psf = ScalarHuygensPSF(
                optic,
                field=(0, 0),
                wavelength=0.55,
                num_rays=16,
                image_size=8,
            )
            norm = psf._get_normalization()
            norm.backward()
            grad_ad = float(be.to_numpy(cs.z.grad))
        finally:
            be.grad_mode.disable()

        assert torch.isfinite(torch.tensor(grad_ad))
        assert grad_ad != 0.0

    def test_vertex_gradient_matches_central_fd(self):
        """AD through the image-vertex normalization geometry agrees with a
        central finite difference.

        The pupil data is held fixed and only the image vertex varies, so
        this isolates exactly the path the gradient fix owns. (The full
        pipeline's total derivative additionally includes the reference
        sphere radius, which the wavefront pipeline stores as a plain float
        by pre-existing design.)
        """
        from optiland.psf.huygens_fresnel import ScalarHuygensPSF, image_vertex_grid

        be.set_precision("float64")
        optic = self._finite_singlet(dx=1.5, dy=-2.0)
        psf = ScalarHuygensPSF(
            optic,
            field=(0, 0),
            wavelength=0.55,
            num_rays=16,
            image_size=8,
        )
        data = psf.get_data((0, 0), 0.55)
        pupil_opd_ideal = be.zeros_like(data.opd)
        cs = optic.surfaces[-1].geometry.cs
        z0 = float(be.to_numpy(cs.z))

        def norm_at(z_value):
            cs.z = z_value
            image_x, image_y, image_z = image_vertex_grid(optic)
            peak = psf._summation_strategy.compute(
                image_x,
                image_y,
                image_z,
                data.pupil_x,
                data.pupil_y,
                data.pupil_z,
                be.ones_like(data.intensity),
                pupil_opd_ideal,
                0.55 * 1e-3,
                data.radius,
            )
            return peak[0, 0]

        be.grad_mode.enable()
        try:
            z_leaf = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
            norm = norm_at(z_leaf)
            norm.backward()
            grad_ad = float(be.to_numpy(z_leaf.grad))
        finally:
            be.grad_mode.disable()
            cs.z = be.array(z0)

        assert torch.isfinite(torch.tensor(grad_ad))
        assert grad_ad != 0.0

        # The summation phase varies on the defocus scale (~4 lambda FNO^2,
        # tens of micrometres here), so the central-difference step must sit
        # well below it; h = 1e-5 mm puts the truncation error near 1e-7
        # relative while staying far above double-precision round-off.
        h = 1e-5
        grad_fd = (
            float(be.to_numpy(norm_at(be.array(z0 + h))))
            - float(be.to_numpy(norm_at(be.array(z0 - h))))
        ) / (2 * h)
        cs.z = be.array(z0)
        assert grad_ad == pytest.approx(grad_fd, rel=1e-5)

    def test_scalar_and_vectorial_share_vertex_construction(self):
        """Both Huygens variants must use the one shared helper, so the two
        normalization paths cannot drift."""
        from optiland.psf import huygens_fresnel, vectorial_huygens

        assert vectorial_huygens.image_vertex_grid is huygens_fresnel.image_vertex_grid
