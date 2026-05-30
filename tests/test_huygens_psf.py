from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland.psf.huygens_fresnel import HuygensPSF
from optiland.samples.objectives import CookeTriplet, DoubleGauss, ReverseTelephoto

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture(autouse=True)
def set_numpy_backend():
    """Ensure the numpy backend is used for all tests in this module and restore after."""
    original_backend = be.get_backend()
    be.set_backend("numpy")
    yield
    be.set_backend(original_backend)


@pytest.fixture(scope="module")
def cooke_triplet_optic():
    """Provides a CookeTriplet instance."""
    return CookeTriplet()


@pytest.fixture(scope="module")
def double_gauss_optic():
    """Provides a DoubleGauss instance."""
    return DoubleGauss()


@pytest.fixture(scope="module")
def reverse_telephoto_optic():
    """Provides a ReverseTelephoto instance."""
    return ReverseTelephoto()


class TestHuygensPSF:
    """
    Test suite for the HuygensPSF class.
    """

    # Default parameters for faster tests; can be overridden for specific tests
    WAVELENGTH_GREEN = 0.550  # microns (e.g., green light)
    NUM_RAYS_LOW = 32  # Lower number of rays for speed in most tests
    IMAGE_SIZE_LOW = 32  # Smaller image size for speed in most tests
    NUM_RAYS_HIGH = 128  # Higher number of rays for more accurate Strehl/PSF
    IMAGE_SIZE_HIGH = 128  # Larger image size for more accurate Strehl/PSF

    OPTIC_FIXTURES = [
        "cooke_triplet_optic",
        "double_gauss_optic",
        "reverse_telephoto_optic",
    ]
    FIELDS_TO_TEST = [(0, 0), (0.5, 0.0), (0.0, 0.7)]  # On-axis and off-axis

    @pytest.mark.parametrize("optic_fixture_name", OPTIC_FIXTURES)
    @pytest.mark.parametrize("field", FIELDS_TO_TEST)
    def test_huygens_psf_initialization(self, optic_fixture_name, field, request):
        """
        Tests the initialization of HuygensPSF instance, checking attributes
        and that the PSF is computed.
        """
        optic = request.getfixturevalue(optic_fixture_name)
        psf_instance = HuygensPSF(
            optic=optic,
            field=field,
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )

        assert psf_instance.optic == optic
        assert [fp.coord for fp in psf_instance.fields] == [
            field
        ]  # BasePSF stores fields as a list
        assert [wp.value for wp in psf_instance.wavelengths] == [
            self.WAVELENGTH_GREEN
        ]  # BasePSF stores wavelengths as a list
        assert psf_instance.num_rays == self.NUM_RAYS_LOW
        assert psf_instance.image_size == self.IMAGE_SIZE_LOW  #

        assert psf_instance.psf is not None, "PSF should be computed on init"  #
        assert isinstance(psf_instance.psf, np.ndarray), "PSF should be a numpy array"
        assert psf_instance.psf.shape == (self.IMAGE_SIZE_LOW, self.IMAGE_SIZE_LOW)  #

        assert psf_instance.cx is not None, "Image center x (cx) not set"  #
        assert psf_instance.cy is not None, "Image center y (cy) not set"  #
        assert psf_instance.pixel_pitch is not None, "Pixel pitch not set"  #
        assert psf_instance.pixel_pitch > 0, "Pixel pitch must be positive"

    @patch("optiland.backend.get_backend", return_value="cupy")
    def test_huygens_psf_unsupported_backend_check(
        self, mock_get_backend, cooke_triplet_optic
    ):
        """
        Tests that HuygensPSF raises ValueError for an unsupported backend.
        """
        with pytest.raises(
            ValueError, match="Unsupported backend for HuygensPSF: cupy"
        ):
            HuygensPSF(
                optic=cooke_triplet_optic,
                field=(0, 0),
                wavelength=self.WAVELENGTH_GREEN,
                num_rays=self.NUM_RAYS_LOW,
                image_size=self.IMAGE_SIZE_LOW,
            )

    @pytest.mark.parametrize("optic_fixture_name", OPTIC_FIXTURES)
    def test_get_image_extent(self, optic_fixture_name, request):
        """
        Tests the _get_image_extent method's effects (cx, cy, pixel_pitch) and return
        values.
        """
        optic = request.getfixturevalue(optic_fixture_name)
        # Initialize instance (this calls _get_image_extent via _get_image_coordinates
        # via _compute_psf)
        psf_instance = HuygensPSF(
            optic=optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )

        # Check attributes set by _get_image_extent (via constructor path)
        assert isinstance(psf_instance.cx, float)
        assert isinstance(psf_instance.cy, float)
        assert (
            isinstance(psf_instance.pixel_pitch, float) and psf_instance.pixel_pitch > 0
        )  #

        # Call directly to check return values
        xmin, xmax, ymin, ymax = psf_instance._get_image_extent()  #
        assert isinstance(xmin, float)
        assert isinstance(xmax, float)
        assert isinstance(ymin, float)
        assert isinstance(ymax, float)
        assert xmin < xmax, "xmin should be less than xmax"
        assert ymin < ymax, "ymin should be less than ymax"

    @pytest.mark.parametrize("optic_fixture_name", OPTIC_FIXTURES)
    def test_get_image_coordinates(self, optic_fixture_name, request):
        """
        Tests the _get_image_coordinates method for correct shape and type of output.
        """
        optic = request.getfixturevalue(optic_fixture_name)
        psf_instance = HuygensPSF(
            optic=optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )
        image_x, image_y, image_z = psf_instance._get_image_coordinates()

        assert isinstance(image_x, np.ndarray)
        assert isinstance(image_y, np.ndarray)
        assert isinstance(image_z, np.ndarray)

        expected_shape = (self.IMAGE_SIZE_LOW, self.IMAGE_SIZE_LOW)
        assert image_x.shape == expected_shape, "Image_x shape mismatch"
        assert image_y.shape == expected_shape, "Image_y shape mismatch"
        assert image_z.shape == expected_shape, "Image_z shape mismatch"

    @pytest.mark.parametrize("optic_fixture_name", OPTIC_FIXTURES)
    def test_compute_psf_properties_and_normalization(
        self, optic_fixture_name, request
    ):
        """
        Tests properties of the computed PSF: non-negativity and normalization
        (peak value). Uses higher resolution for a more stable peak for
        normalization check.
        """
        optic = request.getfixturevalue(optic_fixture_name)
        field_on_axis = (0, 0)
        psf_instance = HuygensPSF(
            optic=optic,
            field=field_on_axis,  # Test normalization with on-axis field
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_HIGH,  # More rays for better PSF
            image_size=self.IMAGE_SIZE_HIGH,  # Larger image for better peak capture
        )
        psf = psf_instance.psf

        assert isinstance(psf, np.ndarray)
        assert psf.shape == (self.IMAGE_SIZE_HIGH, self.IMAGE_SIZE_HIGH)
        assert np.all(psf >= 0), "PSF values must be non-negative"

        # The PSF is normalized such that an ideal diffraction-limited system's
        # peak would be 100.0.
        # For real systems, the peak will be Strehl Ratio * 100.0.
        # Thus, the peak value should be <= 100.0 (allowing for small numerical margin
        # if SR > 1 due to artifacts).
        max_psf_value = np.max(psf)
        assert 0 < max_psf_value <= 100.5, (
            f"PSF peak {max_psf_value} out of expected range (0, 100.5]"
        )

        # The peak of the PSF should correspond to Strehl * 100
        # Strehl is peak of actual PSF / peak of diffraction-limited PSF.
        # Here, psf is already normalized. The value at the center (approx peak) should
        # be SR*100. This is more thoroughly checked in strehl ratio tests.

    @pytest.mark.parametrize("optic_fixture_name", OPTIC_FIXTURES)
    @pytest.mark.parametrize(
        "field, expected_strehl_min",
        [
            ((0, 0), 0.05),
            ((0.7, 0.0), 0.005),
        ],
    )
    def test_strehl_ratio_general(
        self, optic_fixture_name, field, expected_strehl_min, request
    ):
        """
        Tests the Strehl ratio calculation for various optics and fields.
        Uses higher resolution for more accurate Strehl.
        """
        # For this general test, we use a placeholder minimum.
        # The `test_strehl_ratio_specific_values` is for more precise checks.
        optic = request.getfixturevalue(optic_fixture_name)
        print(
            f"Testing Strehl for {optic.__class__.__name__} at field {field}. "
            f"Expected min: {expected_strehl_min}"
        )

        psf_instance = HuygensPSF(
            optic=optic,
            field=field,
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_HIGH,
            image_size=self.IMAGE_SIZE_HIGH,
        )
        sr = psf_instance.strehl_ratio()  #

        assert isinstance(sr, float), "Strehl ratio must be a float"
        assert expected_strehl_min <= sr <= 1.005, (
            f"Strehl ratio {sr:.4f} for {optic.__class__.__name__} at field {field} "
            f"out of expected range [{expected_strehl_min}, 1.005]. "
        )

        # Verify Strehl calculation from BasePSF: peak at center / 100
        center_x = psf_instance.psf.shape[0] // 2
        center_y = psf_instance.psf.shape[1] // 2
        expected_sr_from_psf_center = psf_instance.psf[center_x, center_y] / 100.0
        assert np.isclose(sr, expected_sr_from_psf_center), (
            "Strehl ratio mismatch with definition"
        )

    def test_strehl_ratio_specific_values(
        self, cooke_triplet_optic, double_gauss_optic, reverse_telephoto_optic
    ):
        EXPECTED_STREHL_VALUES = {
            "CookeTriplet": {
                (0, 0): 0.3023159962682067,
                (
                    0.7,
                    0.0,
                ): 0.019029390469840174,
            },
            "DoubleGauss": {
                (
                    0,
                    0,
                ): 0.07405715702199461,
                (
                    0.7,
                    0.0,
                ): 0.0063032279399868095,
            },
            "ReverseTelephoto": {
                (
                    0,
                    0,
                ): 0.9785343625747402,
                (
                    0.7,
                    0.0,
                ): 0.881275354472487,
            },
        }
        # Tolerance for Strehl comparison
        strehl_tolerance = 0.001

        optics_map = {
            "CookeTriplet": cooke_triplet_optic,
            "DoubleGauss": double_gauss_optic,
            "ReverseTelephoto": reverse_telephoto_optic,
        }

        for optic_name, field_strehl_map in EXPECTED_STREHL_VALUES.items():
            optic_under_test = optics_map[optic_name]
            for field, expected_sr in field_strehl_map.items():
                print(
                    f"Testing specific Strehl for {optic_name} at {field}. "
                    f"Expected: ~{expected_sr}"
                )
                psf_instance = HuygensPSF(
                    optic=optic_under_test,
                    field=field,
                    wavelength=self.WAVELENGTH_GREEN,
                    num_rays=self.NUM_RAYS_HIGH,  # Use high resolution
                    image_size=self.IMAGE_SIZE_HIGH,
                )
                actual_sr = psf_instance.strehl_ratio()
                assert np.isclose(actual_sr, expected_sr, atol=strehl_tolerance), (
                    f"Strehl for {optic_name} at {field}: expected ~{expected_sr}, "
                    f"got {actual_sr:.4f}"
                )
                print(
                    f"  Actual Strehl for {optic_name} at {field}: {actual_sr:.4f} "
                    f"(Expected ~{expected_sr})"
                )

    @pytest.mark.parametrize("optic_fixture_name", OPTIC_FIXTURES)
    def test_get_psf_units(self, optic_fixture_name, request):
        """
        Tests the _get_psf_units method for correct calculation of physical extent.
        """
        optic = request.getfixturevalue(optic_fixture_name)
        psf_instance = HuygensPSF(
            optic=optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )

        # Create a dummy image (e.g., a zoomed portion of PSF)
        # Its shape determines the extent to be calculated
        dummy_image_shape_x = self.IMAGE_SIZE_LOW // 2
        dummy_image_shape_y = self.IMAGE_SIZE_LOW // 2
        dummy_image_np = np.zeros((dummy_image_shape_x, dummy_image_shape_y))

        # Method under test
        x_extent_um, y_extent_um = psf_instance._get_psf_units(dummy_image_np)  #

        assert isinstance(x_extent_um.item(), (np.float64, float))
        assert isinstance(y_extent_um.item(), (np.float64, float))
        assert x_extent_um > 0, "X extent must be positive"
        assert y_extent_um > 0, "Y extent must be positive"

        # Verify calculation: num_pixels * pixel_pitch_mm * 1000 um/mm
        expected_x_extent_um = dummy_image_shape_x * psf_instance.pixel_pitch * 1e3
        expected_y_extent_um = dummy_image_shape_y * psf_instance.pixel_pitch * 1e3

        assert np.isclose(x_extent_um, expected_x_extent_um), (
            "Calculated X extent is incorrect"
        )
        assert np.isclose(y_extent_um, expected_y_extent_um), (
            "Calculated Y extent is incorrect"
        )

    @pytest.mark.parametrize("optic_fixture_name", OPTIC_FIXTURES)
    @pytest.mark.parametrize("projection", ["2d", "3d"])
    @pytest.mark.parametrize("log_scale", [True, False])
    def test_view_runs_without_error(
        self, optic_fixture_name, projection, log_scale, request
    ):
        """
        Tests that the `view` method (inherited from BasePSF) runs without raising
        errors and calls `plt.show()`.
        """
        optic = request.getfixturevalue(optic_fixture_name)
        psf_instance = HuygensPSF(
            optic=optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,  # Low res for speed
            image_size=self.IMAGE_SIZE_LOW,
        )

        try:
            fig, ax = psf_instance.view(
                projection=projection, log=log_scale, num_points=32
            )  # Use fewer points for interpolation
        except Exception as e:
            pytest.fail(f"view() raised an exception: {e}")

        assert fig is not None
        assert ax is not None
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(fig)

    def test_view_invalid_projection(self, cooke_triplet_optic):
        """
        Tests that `view` raises ValueError for an invalid projection type.
        """
        psf_instance = HuygensPSF(
            optic=cooke_triplet_optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )
        with pytest.raises(ValueError, match='Projection must be "2d" or "3d".'):  #
            psf_instance.view(projection="invalid_projection_type")

    @pytest.mark.parametrize(
        "projection",
        [
            "2d",
            "3d",
        ],
    )
    @patch("matplotlib.figure.Figure.text")
    def test_view_annotate_sampling(self, mock_text, projection, cooke_triplet_optic):
        psf_instance = HuygensPSF(
            optic=cooke_triplet_optic,
            field=(0, 1),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )
        psf_instance.view(projection=projection, num_points=32)

        mock_text.assert_called_once()

        plt.close("all")

    @pytest.mark.parametrize(
        "projection",
        [
            "2d",
            "3d",
        ],
    )
    def test_view_oversampling(self, projection, cooke_triplet_optic):
        psf_instance = HuygensPSF(
            optic=cooke_triplet_optic,
            field=(0, 1),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )

        with pytest.warns(
            UserWarning, match="The PSF view has a high oversampling factor"
        ):
            psf_instance.view(projection=projection, log=False, num_points=128)

    @pytest.mark.parametrize("optic_fixture_name", OPTIC_FIXTURES)
    def test_get_normalization_value_positive(self, optic_fixture_name, request):
        """
        Tests that _get_normalization returns a positive value.
        The exact value is complex to predict, but it must be positive.
        """
        optic = request.getfixturevalue(optic_fixture_name)
        psf_instance = HuygensPSF(
            optic=optic,
            field=(0, 0),  # Normalization is based on on-axis ideal case
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )
        # _get_normalization is called during __init__ via _compute_psf.
        # We can call it again if we want to isolate its test, or trust it was called.
        # For this test, let's retrieve it by re-calling:
        norm_factor = psf_instance._get_normalization()

        assert isinstance(norm_factor, float), "Normalization factor must be a float"
        assert norm_factor > 0, "Normalization factor must be positive"

    # The following tests address cases that are hard/impossible to trigger in
    #  HuygensPSF because self.psf is always set in __init__. They are more relevant
    # for direct BasePSF users.
    def test_view_psf_not_computed_error_case_not_directly_triggerable(
        self, cooke_triplet_optic
    ):
        """
        This RuntimeError from BasePSF is not directly triggerable in HuygensPSF
        as `psf` is computed in `__init__`. This documents the expected BasePSF
        behavior.
        """
        psf_instance = HuygensPSF(
            optic=cooke_triplet_optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )
        # To test this, we'd have to manually set psf_instance.psf = None
        psf_instance.psf = None  # Manually override
        with pytest.raises(RuntimeError, match="PSF has not been computed."):
            psf_instance.view()
        # Restore psf for any subsequent internal state dependent tests if object is
        # reused (not typical in pytest fixtures per test)
        # psf_instance.psf = psf_instance._compute_psf()

    def test_strehl_ratio_psf_not_computed_error_case_not_directly_triggerable(
        self, cooke_triplet_optic
    ):
        """
        This RuntimeError from BasePSF is not directly triggerable in HuygensPSF.
        Documents the expected BasePSF behavior.
        """
        psf_instance = HuygensPSF(
            optic=cooke_triplet_optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=self.IMAGE_SIZE_LOW,
        )
        psf_instance.psf = None  # Manually override
        with pytest.raises(RuntimeError, match="PSF has not been computed."):
            psf_instance.strehl_ratio()

    def test_pixel_pitch_override(self, cooke_triplet_optic):
        """
        Tests that providing a pixel_pitch in the constructor overrides the
        automatic calculation.
        """
        custom_pixel_pitch = 0.005  # 5 um in mm
        image_size = self.IMAGE_SIZE_LOW

        psf_instance = HuygensPSF(
            optic=cooke_triplet_optic,
            field=(0, 0),
            wavelength=self.WAVELENGTH_GREEN,
            num_rays=self.NUM_RAYS_LOW,
            image_size=image_size,
            pixel_pitch=custom_pixel_pitch,
        )

        # 1. Check if the instance's pixel_pitch is set correctly
        assert psf_instance.pixel_pitch == custom_pixel_pitch

        # 2. Check if the image extent is calculated based on this pixel_pitch
        # The total extent is image_size * pixel_pitch. Half-extent is half of that.
        expected_extent = 0.5 * image_size * custom_pixel_pitch
        xmin, xmax, ymin, ymax = psf_instance._get_image_extent()

        # The extent is centered around (cx, cy)
        assert np.isclose(xmax - xmin, 2 * expected_extent)
        assert np.isclose(ymax - ymin, 2 * expected_extent)
