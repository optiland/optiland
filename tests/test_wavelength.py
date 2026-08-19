from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.wavelength import Wavelength, WavelengthGroup, add_wavelengths
from tests.utils import assert_allclose


class TestWavelengths:
    def test_wavelength_initialization(self, set_test_backend):
        wl = Wavelength(500, unit="nm")
        assert wl.value == 0.5
        assert wl.unit == "um"

    def test_wavelength_conversion(self, set_test_backend):
        wl_nm = Wavelength(500, unit="nm")
        wl_um = Wavelength(0.5, unit="um")
        wl_mm = Wavelength(0.0005, unit="mm")
        wl_cm = Wavelength(0.00005, unit="cm")
        wl_m = Wavelength(0.0000005, unit="m")

        assert wl_nm.value == 0.5
        assert wl_um.value == 0.5
        assert wl_mm.value == 0.5
        assert wl_cm.value == 0.5
        assert wl_m.value == 0.5

    def test_invalid_unit(self, set_test_backend):
        with pytest.raises(ValueError):
            Wavelength(500, unit="invalid_unit")

    def test_unit_setter(self, set_test_backend):
        wl = Wavelength(500, unit="nm")
        assert wl.value == 0.5
        wl.unit = "mm"
        assert wl.unit == "um"
        assert wl._unit == "mm"
        assert wl.value == 500000

    def test_to_dict(self, set_test_backend):
        wl = Wavelength(500, unit="nm", weight=2.0)
        assert wl.to_dict() == {
            "value": 500,
            "unit": "nm",
            "is_primary": True,
            "weight": 2.0,
        }

    def test_from_dict(self, set_test_backend):
        wl_dict = {"value": 500, "unit": "nm", "is_primary": True, "weight": 2.0}
        wl = Wavelength.from_dict(wl_dict)
        assert wl.value == 0.5
        assert wl.unit == "um"
        assert wl.is_primary is True
        assert wl.weight == 2.0

    def test_from_dict_default_weight(self, set_test_backend):
        wl_dict = {"value": 500, "unit": "nm", "is_primary": True}
        wl = Wavelength.from_dict(wl_dict)
        assert wl.weight == 1.0

    def test_is_primary(self, set_test_backend):
        wl = Wavelength(500, is_primary=True, unit="nm")
        assert wl.is_primary is True

    def test_is_not_primary(self, set_test_backend):
        wl = Wavelength(500, is_primary=False, unit="nm")
        assert wl.is_primary is False

    def test_is_primary_default(self, set_test_backend):
        wl = Wavelength(500, unit="nm")
        assert wl.is_primary is True

    def test_wavelength_initialization_with_weight(self, set_test_backend):
        wl = Wavelength(500, unit="nm", weight=2.0)
        assert wl.weight == 2.0

    def test_wavelength_default_weight(self, set_test_backend):
        wl = Wavelength(500, unit="nm")
        assert wl.weight == 1.0


class TestWavelengthGroups:
    def test_add_wavelength(self, set_test_backend):
        wg = WavelengthGroup()
        wg.add(500, unit="nm")
        assert wg.num_wavelengths == 1
        assert wg.get_wavelength(0) == 0.5

    def test_remove_wavelength(self, set_test_backend):
        wg = WavelengthGroup()
        wg.add(500, unit="nm")
        wg.add(600, unit="nm")
        wg.add(700, unit="nm")
        assert wg.num_wavelengths == 3
        wg.remove(1)
        assert wg.num_wavelengths == 2
        assert_allclose(wg.get_wavelength(1), 0.7)

    def test_primary_wavelength(self, set_test_backend):
        wg = WavelengthGroup()
        wg.add(500, unit="nm")
        wg.add(600, is_primary=True, unit="nm")
        assert wg.primary_wavelength.value == 0.6

    def test_set_primary_index(self, set_test_backend):
        wg = WavelengthGroup()
        wg.add(500, unit="nm")
        wg.add(600, is_primary=True, unit="nm")
        wg.primary_index = 0
        assert wg.primary_wavelength.value == 0.5

    @pytest.mark.parametrize("index", (-1, 2))
    def test_set_primary_index_raises(self, index, set_test_backend):
        wg = WavelengthGroup()
        wg.add(500, unit="nm")
        wg.add(600, is_primary=True, unit="nm")
        with pytest.raises(ValueError, match="Invalid primary wavelength index"):
            wg.primary_index = index

    def test_multiple_wavelengths(self, set_test_backend):
        wg = WavelengthGroup()
        wg.add(500, unit="nm")
        wg.add(600, unit="nm", is_primary=True)
        assert wg.num_wavelengths == 2
        assert wg.get_wavelength(0) == 0.5
        assert wg.get_wavelength(1) == 0.6

    def test_get_wavelengths(self, set_test_backend):
        wg = WavelengthGroup()
        wg.add(500, unit="nm")
        wg.add(600, unit="nm")
        assert wg.get_wavelengths() == [0.5, 0.6]

    def test_to_dict(self, set_test_backend):
        wg = WavelengthGroup()
        wg.add(500, unit="nm", weight=0.5)
        wg.add(600, unit="nm", weight=1.5, is_primary=True)
        assert wg.to_dict() == {
            "wavelengths": [
                {"value": 500, "unit": "nm", "is_primary": False, "weight": 0.5},
                {"value": 600, "unit": "nm", "is_primary": True, "weight": 1.5},
            ],
        }

    def test_from_dict(self, set_test_backend):
        wg_dict = {
            "wavelengths": [
                {"value": 500, "unit": "nm", "is_primary": False, "weight": 0.5},
                {"value": 600, "unit": "nm", "is_primary": True, "weight": 1.5},
            ],
        }
        wg = WavelengthGroup.from_dict(wg_dict)
        assert wg.num_wavelengths == 2
        assert wg.get_wavelength(0) == 0.5
        assert wg.get_wavelength(1) == 0.6
        assert wg.primary_wavelength.value == 0.6
        assert wg.primary_index == 1
        assert wg.weights == (0.5, 1.5)

    def test_from_dict_invalid_key(self, set_test_backend):
        with pytest.raises(ValueError):
            WavelengthGroup.from_dict({})

    def test_wavelengthgroup_weights_property(self, set_test_backend):
        wg = WavelengthGroup()
        wg.add(500, unit="nm", weight=0.5)
        wg.add(600, unit="nm", weight=1.5)
        wg.add(700, unit="nm", weight=2.5)
        assert wg.weights == (0.5, 1.5, 2.5)

    def test_add_wavelengths(self, set_test_backend):
        # exp-Chebyshev
        wg = WavelengthGroup()
        add_wavelengths(wg, 400, 800, 5, unit="nm", sampling="chebyshev", scale="log")
        assert wg.num_wavelengths == 5
        middle = be.sqrt(0.4 * 0.8)
        assert be.isclose(wg.wavelengths[2].value, middle)
        assert wg.wavelengths[2].is_primary
        assert be.isclose(
            wg.wavelengths[0].value,
            middle * 2 ** (-0.125 * (be.sqrt(10 + 2 * be.sqrt(5)))),
        )
        assert be.isclose(
            wg.wavelengths[1].value,
            middle * 2 ** (-0.125 * (be.sqrt(10 - 2 * be.sqrt(5)))),
        )
        assert be.isclose(
            wg.wavelengths[3].value,
            middle * 2 ** (0.125 * (be.sqrt(10 - 2 * be.sqrt(5)))),
        )
        assert be.isclose(
            wg.wavelengths[4].value,
            middle * 2 ** (0.125 * (be.sqrt(10 + 2 * be.sqrt(5)))),
        )
        # frequency Chebyshev
        wg = WavelengthGroup()
        add_wavelengths(
            wg, 400, 800, 5, unit="nm", sampling="chebyshev", scale="frequency"
        )
        assert wg.num_wavelengths == 5
        middle = be.as_array_1d(2.0 / (1.0 / 0.4 + 1.0 / 0.8))
        assert be.isclose(wg.wavelengths[2].value, middle)
        assert wg.wavelengths[2].is_primary
        # wavelength Chebyshev
        wg = WavelengthGroup()
        add_wavelengths(wg, 0.4, 0.8, 5, sampling="chebyshev", scale="wavelength")
        assert wg.num_wavelengths == 5
        middle = 0.6
        wavelengths_ref = middle + 0.05 * be.as_array_1d(
            [
                -be.sqrt(10 + 2 * be.sqrt(5)),
                -be.sqrt(10 - 2 * be.sqrt(5)),
                0,
                be.sqrt(10 - 2 * be.sqrt(5)),
                be.sqrt(10 + 2 * be.sqrt(5)),
            ]
        )
        assert be.allclose(
            wavelengths_ref, be.as_array_1d([wg.wavelengths[i].value for i in range(5)])
        )
        # exponential
        wg = WavelengthGroup()
        add_wavelengths(wg, 0.25, 1.0, 7, sampling="uniform", scale="log")
        assert wg.num_wavelengths == 7
        assert be.isclose(wg.wavelengths[3].value, be.as_array_1d(0.5))
        assert be.isclose(0.25 / wg.wavelengths[0].value, wg.wavelengths[6].value)
        for i in range(1):
            assert be.isclose(
                wg.wavelengths[i].value / wg.wavelengths[i + 1].value,
                wg.wavelengths[6].value ** 2,
            )
        # uniform (frequency)
        wg = WavelengthGroup()
        add_wavelengths(wg, 0.25, 1.0, 7, sampling="uniform", scale="frequency")
        assert wg.num_wavelengths == 7
        assert be.isclose(wg.wavelengths[3].value, be.as_array_1d(0.4))
        assert be.isclose(
            4 - 1 / wg.wavelengths[0].value, 1 / wg.wavelengths[6].value - 1
        )
        for i in range(1):
            assert be.isclose(
                1 / wg.wavelengths[i].value - 1 / wg.wavelengths[i + 1].value,
                2 * (1 / wg.wavelengths[6].value - 1),
            )
        # uniform (wavelength)
        wg = WavelengthGroup()
        add_wavelengths(wg, 0.25, 1.0, 7, sampling="uniform", scale="wavelength")
        assert wg.num_wavelengths == 7
        assert be.isclose(wg.wavelengths[3].value, be.as_array_1d(0.625))
        assert be.isclose(wg.wavelengths[0].value - 0.25, 1 - wg.wavelengths[6].value)
        for i in range(1):
            assert be.isclose(
                wg.wavelengths[i + 1].value - wg.wavelengths[i].value,
                2 * (1 - wg.wavelengths[6].value),
            )
