from __future__ import annotations

from unittest.mock import patch

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    ChebyshevPolynomialGeometry,
    ForbesQ2dGeometry,
    ForbesQbfsGeometry,
    ForbesSurfaceConfig,
    PolynomialGeometry,
    ZernikePolynomialGeometry,
)
from optiland.materials.abbe import AbbeMaterial
from optiland.optic import Optic
from optiland.optimization import OptimizationProblem, OptimizerGeneric, variable
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable import (
    AsphereCoeffVariable,
    DecenterVariable,
    TiltVariable,
    Variable,
    ZernikeCoeffVariable,
)
from optiland.optimization.variable.material import MaterialVariable
from optiland.samples.microscopes import Objective60x, UVReflectingMicroscope
from optiland.samples.simple import AsphericSinglet, Edmund_49_847

from .utils import assert_allclose


class TestRadiusVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.radius_var = variable.RadiusVariable(self.optic, 1)

    def test_get_value(self, set_test_backend):
        assert_allclose(self.radius_var.get_value(), 553.260)

    def test_update_value(self, set_test_backend):
        self.radius_var.update_value(5.0)
        assert_allclose(self.radius_var.get_value(), 5.0)


class TestReciprocalRadiusVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Edmund_49_847()
        self.reciprocal_radius_var = variable.ReciprocalRadiusVariable(self.optic, 1)
        self.radius_var = variable.RadiusVariable(
            self.optic, 1, scaler=IdentityScaler()
        )
        self.problem = OptimizationProblem()
        input_data = {
            "optic": self.optic,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 5,
            "wavelength": self.optic.wavelengths.primary_wavelength.value,
            "distribution": "hexapolar",
            "surface_number": 3,
        }
        self.problem.add_operand(
            operand_type="rms_spot_size", target=0, weight=1, input_data=input_data
        )

    def test_get_value(self, set_test_backend):
        expected = 1.0 / self.radius_var.get_value()
        assert_allclose(self.reciprocal_radius_var.get_value(), expected, atol=1e-4)

    def test_update_value(self, set_test_backend):
        self.reciprocal_radius_var.update_value(0.008)
        expected_radius = 1.0 / 0.008
        assert_allclose(self.radius_var.get_value(), expected_radius, atol=1e-4)
        assert_allclose(self.reciprocal_radius_var.get_value(), 0.008, atol=1e-4)

    def test_get_value_infinity(self, set_test_backend):
        self.radius_var.update_value(0.0)
        val = self.reciprocal_radius_var.get_value()
        assert val == be.inf

    def test_update_value_zero(self, set_test_backend):
        self.reciprocal_radius_var.update_value(0.0)
        assert self.radius_var.get_value() == be.inf

    def test_optimization(self):
        self.problem.add_variable(self.optic, "reciprocal_radius", surface_number=1)
        optimizer = OptimizerGeneric(self.problem)
        self.optic.updater.set_radius(22.0, 1)
        optimizer.optimize(tol=1e-9)
        expected_radius = 19.93
        optimized_radius = self.optic.surfaces.radii[1]
        assert_allclose(optimized_radius, expected_radius, atol=5)

    def test_optimization_with_flat_surface(self):
        self.problem.add_variable(self.optic, "reciprocal_radius", surface_number=1)
        self.optic.updater.set_radius(-be.inf, 1)
        optimizer = OptimizerGeneric(self.problem)
        optimizer.optimize(tol=1e-9)
        optimizer.optimize(tol=1e-9)
        expected_radius = 19.93
        optimized_radius = self.optic.surfaces.radii[1]
        assert_allclose(optimized_radius, expected_radius, atol=5)


class TestConicVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = UVReflectingMicroscope()
        self.conic_var = variable.ConicVariable(self.optic, 1)

    def test_get_value(self, set_test_backend):
        assert_allclose(self.conic_var.get_value(), 0.0)

    def test_update_value(self, set_test_backend):
        self.conic_var.update_value(-0.5)
        assert_allclose(self.conic_var.get_value(), -0.5)

    def test_string_representation(self, set_test_backend):
        assert str(self.conic_var) == "Conic Constant, Surface 1"


class TestThicknessVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.thickness_var = variable.ThicknessVariable(self.optic, 2)

    def test_get_value(self, set_test_backend):
        assert_allclose(self.thickness_var.get_value(), 4.4)

    def test_update_value(self, set_test_backend):
        self.thickness_var.update_value(5.0)
        assert_allclose(self.thickness_var.get_value(), 5.0)

    def test_get_value_no_scaling(self, set_test_backend):
        self.optic = Objective60x()
        self.thickness_var = variable.ThicknessVariable(
            self.optic,
            2,
            scaler=IdentityScaler(),
        )
        assert_allclose(self.thickness_var.get_value(), 4.4)


class TestIndexVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.index_var = variable.IndexVariable(self.optic, 1, 0.55)

    def test_get_value(self, set_test_backend):
        assert_allclose(self.index_var.get_value(), 1.4877935552990422)

    def test_update_value(self, set_test_backend):
        self.index_var.update_value(1.6)
        assert_allclose(self.index_var.get_value(), 1.6)

    def test_get_value_no_scaling(self, set_test_backend):
        self.optic = Objective60x()
        self.index_var = variable.IndexVariable(
            self.optic,
            1,
            0.55,
            scaler=IdentityScaler(),
        )
        assert_allclose(self.index_var.get_value(), 1.4877935552990422)

    def test_string_representation(self, set_test_backend):
        assert str(self.index_var) == "Refractive Index, Surface 1"


class TestMaterialVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.surface_number = 1
        self.glass_selection = ["N-BK7", "N-SSK2", "N-SK2", "N-SK16"]
        self.material_var = variable.MaterialVariable(
            optic=self.optic,
            surface_number=self.surface_number,
            glass_selection=self.glass_selection,
        )

    def test_get_value(self, set_test_backend):
        assert self.material_var.get_value() == "N-FK51"

    def test_update_value(self, set_test_backend):
        self.material_var.update_value("F5")
        assert self.material_var.get_value() == "F5"

    def test_string_representation(self, set_test_backend):
        assert str(self.material_var) == "Material, Surface 1"

    def test_init_with_abbe_material(self):
        abbe = AbbeMaterial(n=(1.5168,), abbe=(64.17,), model="polynomial")
        self.optic.surfaces[self.surface_number].material_post = abbe

        with (
            patch(
                "optiland.materials.material_utils.find_closest_glass",
                return_value="N-BK7",
            ),
            patch(
                "optiland.materials.material_utils.get_nd_vd",
                return_value=(1.5168, 64.17),
            ),
            patch("builtins.print") as mock_print,
        ):
            mat_var = MaterialVariable(
                optic=self.optic,
                surface_number=self.surface_number,
                glass_selection=self.glass_selection,
            )
            assert mat_var.get_value() == "N-BK7"
            mock_print.assert_called()
            printed = mock_print.call_args[0][0]
            assert "AbbeMaterial" in printed
            assert "N-BK7" in printed


class TestAsphereCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        self.asphere_var = variable.AsphereCoeffVariable(self.optic, 1, 0)

    def test_get_value(self, set_test_backend):
        assert_allclose(self.asphere_var.get_value(), -0.0002248851)

    def test_update_value(self, set_test_backend):
        self.asphere_var.update_value(-2.0)
        assert_allclose(self.asphere_var.get_value(), -2.0)

    def test_get_value_no_scaling(self, set_test_backend):
        self.optic = AsphericSinglet()
        self.asphere_var = variable.AsphereCoeffVariable(
            self.optic,
            1,
            0,
            scaler=IdentityScaler(),
        )
        assert_allclose(self.asphere_var.get_value(), -0.0002248851)

    def test_string_representation(self, set_test_backend):
        assert str(self.asphere_var) == "Asphere Coeff. 0, Surface 1"


class TestPolynomialCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(
            CoordinateSystem(),
            100,
            coefficients=be.zeros((3, 3)),
        )
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1))

    def test_get_value(self, set_test_backend):
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self, set_test_backend):
        self.poly_var.update_value(1.0)
        assert_allclose(self.poly_var.get_value(), 1.0)

    def test_get_value_index_error(self, set_test_backend):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1))
        assert self.poly_var.get_value() == 0.0

    def test_update_value_index_error(self, set_test_backend):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1))
        self.poly_var.update_value(1.0)
        assert_allclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self, set_test_backend):
        assert str(self.poly_var) == "Poly. Coeff. (1, 1), Surface 0"

    def test_get_value_no_scaling(self, set_test_backend):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(
            CoordinateSystem(),
            100,
            coefficients=be.zeros((3, 3)),
        )
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(
            self.optic,
            0,
            (1, 1),
            scaler=IdentityScaler(),
        )
        assert self.poly_var.get_value() == 0.0


class TestChebyshevCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = ChebyshevPolynomialGeometry(
            CoordinateSystem(),
            100,
            coefficients=be.zeros((3, 3)),
        )
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ChebyshevCoeffVariable(self.optic, 0, (1, 1))

    def test_get_value(self, set_test_backend):
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self, set_test_backend):
        self.poly_var.update_value(1.0)
        assert_allclose(self.poly_var.get_value(), 1.0)

    def test_get_value_index_error(self, set_test_backend):
        self.optic = AsphericSinglet()
        poly_geo = ChebyshevPolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ChebyshevCoeffVariable(self.optic, 0, (1, 1))
        assert self.poly_var.get_value() == 0.0

    def test_update_value_index_error(self, set_test_backend):
        self.optic = AsphericSinglet()
        poly_geo = ChebyshevPolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ChebyshevCoeffVariable(self.optic, 0, (1, 1))
        self.poly_var.update_value(1.0)
        assert_allclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self, set_test_backend):
        assert str(self.poly_var) == "Chebyshev Coeff. (1, 1), Surface 0"


class TestZernikeCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = ZernikePolynomialGeometry(
            CoordinateSystem(),
            100,
            coefficients=be.zeros(3),
        )
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ZernikeCoeffVariable(self.optic, 0, 1)

    def test_get_value(self, set_test_backend):
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self, set_test_backend):
        self.poly_var.update_value(1.0)
        assert_allclose(self.poly_var.get_value(), 1.0)

    def test_get_value_index_error(self, set_test_backend):
        self.optic = AsphericSinglet()
        poly_geo = ZernikePolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ZernikeCoeffVariable(self.optic, 0, 1)
        assert self.poly_var.get_value() == 0.0

    def test_update_value_index_error(self, set_test_backend):
        self.optic = AsphericSinglet()
        poly_geo = ZernikePolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ZernikeCoeffVariable(self.optic, 0, 1)
        self.poly_var.update_value(1.0)
        assert_allclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self, set_test_backend):
        assert str(self.poly_var) == "Zernike Coeff. 1, Surface 0"


class TestVariable:
    def test_get_value(self, set_test_backend):
        optic = Objective60x()
        radius_var = variable.Variable(optic, "radius", surface_number=1)
        assert_allclose(radius_var.value, 4.5326)

    def test_unrecognized_attribute(self, capsys):
        optic = Objective60x()
        radius_var = variable.Variable(
            optic,
            "radius",
            surface_number=1,
            unrecognized_attribute=1,
        )
        assert_allclose(radius_var.value, 4.5326)
        captured = capsys.readouterr()
        assert (
            "Warning: unrecognized_attribute is not a recognized attribute"
            in captured.out
        )

    def test_invalid_type(self, set_test_backend):
        optic = Objective60x()
        with pytest.raises(ValueError):
            variable.Variable(optic, "invalid", surface_number=1)


class TestTiltVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.tilt_var_x = variable.TiltVariable(self.optic, 1, "x")
        self.tilt_var_y = variable.TiltVariable(self.optic, 1, "y")

    def test_get_value_x(self, set_test_backend):
        assert_allclose(self.tilt_var_x.get_value(), 0.0)

    def test_get_value_y(self, set_test_backend):
        assert_allclose(self.tilt_var_y.get_value(), 0.0)

    def test_update_value_x(self, set_test_backend):
        self.tilt_var_x.update_value(5.0)
        assert_allclose(self.tilt_var_x.get_value(), 5.0)

    def test_update_value_y(self, set_test_backend):
        self.tilt_var_y.update_value(5.0)
        assert_allclose(self.tilt_var_y.get_value(), 5.0)

    def test_invalid_axis(self, set_test_backend):
        with pytest.raises(ValueError):
            variable.TiltVariable(self.optic, 1, "z")

    def test_str(self, set_test_backend):
        assert str(self.tilt_var_x) == "Tilt X, Surface 1"
        assert str(self.tilt_var_y) == "Tilt Y, Surface 1"

    def test_get_value_no_scaling(self, set_test_backend):
        self.optic = Objective60x()
        self.tilt_var_x = variable.TiltVariable(
            self.optic, 1, "x", scaler=IdentityScaler()
        )
        assert_allclose(self.tilt_var_x.get_value(), 0.0)


class TestDecenterVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.decenter_var_x = variable.DecenterVariable(self.optic, 1, "x")
        self.decenter_var_y = variable.DecenterVariable(self.optic, 1, "y")

    def test_get_value_x(self, set_test_backend):
        assert_allclose(self.decenter_var_x.get_value(), 0.0)

    def test_get_value_y(self, set_test_backend):
        assert_allclose(self.decenter_var_y.get_value(), 0.0)

    def test_update_value_x(self, set_test_backend):
        self.decenter_var_x.update_value(5.0)
        assert_allclose(self.decenter_var_x.get_value(), 5.0)

    def test_update_value_y(self, set_test_backend):
        self.decenter_var_y.update_value(5.0)
        assert_allclose(self.decenter_var_y.get_value(), 5.0)

    #    def test_invalid_axis(self, set_test_backend):
    #        with pytest.raises(ValueError):
    #            variable.DecenterVariable(self.optic, 1, "z")

    def test_str(self, set_test_backend):
        assert str(self.decenter_var_x) == "Decenter X, Surface 1"
        assert str(self.decenter_var_y) == "Decenter Y, Surface 1"

    def test_get_value_no_scaling(self, set_test_backend):
        self.optic = Objective60x()
        self.decenter_var_x = variable.DecenterVariable(
            self.optic,
            1,
            "x",
            scaler=IdentityScaler(),
        )
        assert_allclose(self.decenter_var_x.get_value(), 0.0)


class TestNormalizationRadiusVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        zernike_geo = ZernikePolynomialGeometry(
            CoordinateSystem(), 100, coefficients=be.zeros(3), norm_radius=10.0
        )
        self.optic.surfaces[1].geometry = zernike_geo
        self.norm_radius_var = variable.NormalizationRadiusVariable(self.optic, 1)

    def test_get_value(self, set_test_backend):
        assert_allclose(self.norm_radius_var.get_value(), 10.0)

    def test_update_value(self, set_test_backend):
        self.norm_radius_var.update_value(12.0)
        assert_allclose(self.norm_radius_var.get_value(), 12.0)

    def test_string_representation(self, set_test_backend):
        assert str(self.norm_radius_var) == "Normalization Radius, Surface 1"


class TestForbesQbfsCoeffVariable:
    pytestmark = [
        pytest.mark.filterwarnings(
            "ignore:ForbesQbfsGeometry is deprecated:DeprecationWarning"
        ),
        pytest.mark.filterwarnings(
            "ignore:ForbesQbfsCoeffVariable is deprecated:DeprecationWarning"
        ),
    ]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()

        surface_config = ForbesSurfaceConfig(
            radius=100, terms={1: 0.1, 2: 0.2, 3: 0.3}, norm_radius=15.0
        )
        forbes_geo = ForbesQbfsGeometry(
            CoordinateSystem(), surface_config=surface_config
        )
        self.optic.surfaces[1].geometry = forbes_geo
        self.forbes_var = variable.ForbesQbfsCoeffVariable(
            self.optic, 1, 1, scaler=IdentityScaler()
        )

    def test_get_value(self, set_test_backend):
        assert_allclose(self.forbes_var.get_value(), 0.1)

    def test_update_value(self, set_test_backend):
        self.forbes_var.update_value(0.5)
        assert_allclose(self.forbes_var.get_value(), 0.5)
        assert_allclose(self.optic.surfaces[1].geometry.coeffs_c[1], 0.5)

    def test_update_value_out_of_bounds(self, set_test_backend):
        forbes_var_new = variable.ForbesQbfsCoeffVariable(
            self.optic, 1, 4, scaler=IdentityScaler()
        )
        forbes_var_new.update_value(0.9)
        assert_allclose(forbes_var_new.get_value(), 0.9)
        assert len(self.optic.surfaces[1].geometry.coeffs_c) == 5

    def test_string_representation(self, set_test_backend):
        assert str(self.forbes_var) == "Forbes Q-bfs Coeff n=1, Surface 1"

    def test_get_value_nonexistent(self, set_test_backend):
        var_n0 = variable.ForbesQbfsCoeffVariable(
            self.optic, 1, 0, scaler=IdentityScaler()
        )
        assert_allclose(var_n0.get_value(), 0.0)

        var_n5 = variable.ForbesQbfsCoeffVariable(
            self.optic, 1, 5, scaler=IdentityScaler()
        )
        assert_allclose(var_n5.get_value(), 0.0)

    def test_scaling(self, set_test_backend):
        from optiland.optimization.scaling.linear import LinearScaler

        var_scaled = variable.ForbesQbfsCoeffVariable(
            self.optic, 1, 2, scaler=LinearScaler(factor=10.0)
        )

        assert_allclose(var_scaled.get_value(), 0.2)


class TestForbesQ2dCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()

        freeform_coeffs = {
            ("a", 1, 1): 0.1,
            ("a", 2, 2): 0.2,
            ("b", 1, 1): 0.3,
        }
        surface_config = ForbesSurfaceConfig(
            radius=100, conic=0.0, terms=freeform_coeffs, norm_radius=15.0
        )
        forbes_geo = ForbesQ2dGeometry(
            CoordinateSystem(),
            surface_config=surface_config,
        )
        self.optic.surfaces[1].geometry = forbes_geo

        self.forbes_var = variable.ForbesQ2dCoeffVariable(
            self.optic, 1, ("a", 2, 2), scaler=IdentityScaler()
        )

    def test_get_value(self, set_test_backend):
        assert_allclose(self.forbes_var.get_value(), 0.2)

    def test_get_value_nonexistent(self, set_test_backend):
        forbes_var_new = variable.ForbesQ2dCoeffVariable(
            self.optic, 1, ("a", 1, 3), scaler=IdentityScaler()
        )
        assert_allclose(forbes_var_new.get_value(), 0.0)

    def test_update_value_existing(self, set_test_backend):
        self.forbes_var.update_value(0.5)
        assert_allclose(self.forbes_var.get_value(), 0.5)

        assert_allclose(
            self.optic.surfaces[1].geometry.freeform_coeffs[("a", 2, 2)],
            0.5,
        )

    def test_update_value__new(self, set_test_backend):
        key = ("a", 1, 3)
        forbes_var_new = variable.ForbesQ2dCoeffVariable(
            self.optic, 1, key, scaler=IdentityScaler()
        )
        forbes_var_new.update_value(0.9)
        assert_allclose(forbes_var_new.get_value(), 0.9)
        assert key in self.optic.surfaces[1].geometry.freeform_coeffs
        assert_allclose(self.optic.surfaces[1].geometry.freeform_coeffs[key], 0.9)

    def test_string_representation(self, set_test_backend):
        assert str(self.forbes_var) == "Forbes Q-2D Coeff (n=2, m=2, cos), Surface 1"

    def test_invalid_coeff_tuple(self, set_test_backend):
        with pytest.raises(ValueError):
            variable.ForbesQ2dCoeffVariable(self.optic, 1, (1))

    def test_get_and_update_sine_term(self, set_test_backend):
        key = ("b", 1, 1)
        var_sin = variable.ForbesQ2dCoeffVariable(
            self.optic, 1, key, scaler=IdentityScaler()
        )
        assert_allclose(var_sin.get_value(), 0.3)

        var_sin.update_value(-0.5)
        assert_allclose(var_sin.get_value(), -0.5)
        assert_allclose(
            self.optic.surfaces[1].geometry.freeform_coeffs[key],
            -0.5,
        )

    def test_string_representation_sine(self, set_test_backend):
        """Test the string representation for a sine term."""

        var_sin = variable.ForbesQ2dCoeffVariable(self.optic, 1, ("b", 1, 1))
        assert str(var_sin) == "Forbes Q-2D Coeff (n=1, m=1, sin), Surface 1"


class TestVariableManager:
    def test_add(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        assert len(var_manager) == 1

    def test_clear(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager.clear()
        assert len(var_manager) == 0

    def test_iter(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager.add(optic, "radius", surface_number=2)
        for var in var_manager:
            assert isinstance(var, variable.Variable)

    def test_getitem(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        assert isinstance(var_manager[0], variable.Variable)

    def test_setitem(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager[0] = variable.Variable(optic, "radius", surface_number=2)
        assert isinstance(var_manager[0], variable.Variable)
        assert var_manager[0].variable.surface_number == 2
        assert len(var_manager) == 1

    def test_len(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager.add(optic, "radius", surface_number=2)
        assert len(var_manager) == 2

    def test_getitem_index_error(self, set_test_backend):
        var_manager = variable.VariableManager()
        with pytest.raises(IndexError):
            var_manager[0]

    def test_setitem_index_error(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        with pytest.raises(IndexError):
            var_manager[0] = variable.Variable(optic, "radius", surface_number=2)

    def test_setitem_invalid_type(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        with pytest.raises(ValueError):
            var_manager[0] = "invalid"

    def test_iterable(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager.add(optic, "radius", surface_number=2)
        for i, var in enumerate(var_manager):
            assert isinstance(var, variable.Variable)
            assert var.variable.surface_number == i + 1
        assert i == 1

    def test_delitem(self, set_test_backend):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        del var_manager[0]
        assert len(var_manager) == 0


class TestVariablesExtended:
    @pytest.fixture
    def optic(self):
        optic = Optic()
        optic.surfaces.add(index=0, surface_type="plane")
        optic.surfaces.add(index=1, surface_type="standard", radius=100)
        optic.surfaces.add(
            index=2, surface_type="even_asphere", radius=50, coefficients=[0.0, 0.1]
        )
        optic.surfaces.add(
            index=3, surface_type="zernike", radius=200, coefficients=[0.0] * 5
        )
        return optic

    def test_decenter_variable(self, optic):
        # Surface 1 is standard
        var = DecenterVariable(optic, surface_number=1, axis="x")
        assert var.get_value() == 0.0

        var.update_value(0.5)
        assert optic.surfaces[1].geometry.cs.x == 0.5
        assert var.get_value() == 0.5

        # Test y and z
        var_y = DecenterVariable(optic, surface_number=1, axis="y")
        var_y.update_value(-0.2)
        assert optic.surfaces[1].geometry.cs.y == -0.2

        var_z = DecenterVariable(optic, surface_number=1, axis="z")
        var_z.update_value(1.0)
        assert optic.surfaces[1].geometry.cs.z == 1.0

    def test_decenter_variable_invalid_axis(self, optic):
        with pytest.raises(ValueError, match='Invalid axis "r"'):
            DecenterVariable(optic, surface_number=1, axis="r")

        var = DecenterVariable(optic, surface_number=1, axis="x")
        var.axis = "r"
        with pytest.raises(ValueError, match='Invalid axis "r"'):
            var.get_value()
        with pytest.raises(ValueError, match='Invalid axis "r"'):
            var.update_value(1.0)

    def test_tilt_variable(self, optic):
        var = TiltVariable(optic, surface_number=1, axis="x")
        var.update_value(0.1)  # radians
        assert optic.surfaces[1].geometry.cs.rx == 0.1

        var_y = TiltVariable(optic, surface_number=1, axis="y")
        var_y.update_value(0.2)
        assert optic.surfaces[1].geometry.cs.ry == 0.2

    def test_tilt_variable_invalid_axis(self, optic):
        with pytest.raises(ValueError, match='Invalid axis "r"'):
            TiltVariable(optic, surface_number=1, axis="r")

    def test_asphere_coeff_variable_get_padding(self, optic):
        # Surface 2 is even_asphere with 2 coeffs
        var = AsphereCoeffVariable(optic, surface_number=2, coeff_number=5)
        # Should return 0 and pad
        val = var.get_value()
        assert val == 0.0
        # Check padding happened
        assert len(optic.surfaces[2].geometry.coefficients) >= 6

    def test_asphere_coeff_variable_update(self, optic):
        var = AsphereCoeffVariable(optic, surface_number=2, coeff_number=1)
        var.update_value(0.5)
        assert optic.surfaces[2].geometry.coefficients[1] == 0.5

    def test_zernike_coeff_variable_padding(self, optic):
        # Surface 3 is zernike with 5 coeffs (indices 0-4)
        var = ZernikeCoeffVariable(optic, surface_number=3, coeff_index=10)

        # Test update padding
        var.update_value(1.5)
        coeffs = optic.surfaces[3].geometry.coefficients
        assert len(coeffs) >= 11
        assert coeffs[10] == 1.5

        # Test get padding
        var2 = ZernikeCoeffVariable(optic, surface_number=3, coeff_index=15)
        val = var2.get_value()
        assert val == 0.0
        assert len(optic.surfaces[3].geometry.coefficients) >= 16

    def test_variable_wrapper(self, optic):
        # Test generic Variable wrapper
        v = Variable(optic, type_name="decenter", surface_number=1, axis="x")
        v.update(1.0)
        assert v.value == 1.0
        assert optic.surfaces[1].geometry.cs.x == 1.0

    def test_variable_wrapper_invalid_type(self, optic):
        with pytest.raises(ValueError, match='Invalid variable type "invalid"'):
            Variable(optic, type_name="invalid")
