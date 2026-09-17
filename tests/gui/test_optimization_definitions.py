"""Dialog definition values reach both owned calculations and value displays."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from optiland_gui.services.optimization_jobs import build_problem, resolve_wavelength
from optiland_gui.services.optimization_service import OptimizationService


def current_value(optic, definition):
    service = SimpleNamespace(_connector=SimpleNamespace(_optic=optic))
    return OptimizationService.get_variable_current_value(service, definition)


@pytest.mark.parametrize(
    "selector,expected",
    [
        ("primary", 0.55),
        ("'primary'", 0.55),
        ('"primary"', 0.55),
        (" [0.4861] ", 0.4861),
        ("0.4861", 0.4861),
        (0.4861, 0.4861),
        ([0.4861], 0.4861),
        ((0.4861,), 0.4861),
        (["primary"], 0.55),
    ],
)
def test_index_variable_uses_dialog_wavelength_in_setup_and_display(
    minimal_optic, selector, expected
):
    definition = {
        "type": "index",
        "surface_number": 1,
        "wavelength": selector,
        "min_val": 1.0,
        "max_val": 2.0,
    }
    before = copy.deepcopy(definition)
    problem = build_problem(
        minimal_optic,
        [definition],
        [{"type": "total_track", "target": 50.0}],
        {},
    )
    variable = problem.variables[0]
    assert variable.variable.wavelength == expected
    assert variable.min_val == 1.0
    assert variable.max_val == 2.0
    index = float(minimal_optic.surfaces.n(expected)[1])
    assert float(variable.variable.get_value()) == pytest.approx(index)
    assert current_value(minimal_optic, definition) == pytest.approx(index)
    assert definition == before


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("selector", ["'primary'", "[0.55]", 0.55, None])
def test_ray_operand_resolves_selector_or_metadata_default(
    minimal_optic, selector, as_json
):
    values = {"surface_number": 3, "Hx": 0.0, "Hy": 0.0, "Px": 0.0, "Py": 0.0}
    if selector is not None:
        values["wavelength"] = selector
    operand = {"type": "real_y_intercept", "target": 0.0}
    if as_json:
        operand["input_data_str"] = json.dumps(values)
    else:
        operand["input_data"] = values
    before = copy.deepcopy(operand)
    problem = build_problem(
        minimal_optic,
        [{"type": "thickness", "surface_number": 1}],
        [operand],
        {"real_y_intercept": {"wavelength": {"default": "primary"}}},
    )
    assert problem.operands[0].input_data["wavelength"] == 0.55
    assert float(problem.operands[0].value) == pytest.approx(0.0)
    assert operand == before


@pytest.mark.parametrize(
    "selector",
    [
        "all",
        "'all'",
        "[0.4861, 0.55]",
        [0.4861, 0.55],
        [],
        True,
        None,
        float("nan"),
        float("inf"),
        "NaN",
        -0.55,
        0.0,
        "__import__('os').getcwd()",
    ],
)
def test_invalid_or_multiple_wavelengths_fail_as_setup_errors(minimal_optic, selector):
    with pytest.raises(ValueError, match="Select one positive wavelength"):
        resolve_wavelength(minimal_optic, selector)


def test_multiple_operand_wavelengths_are_not_silently_truncated(minimal_optic):
    with pytest.raises(ValueError, match="Select one positive wavelength"):
        build_problem(
            minimal_optic,
            [{"type": "thickness", "surface_number": 1}],
            [
                {
                    "type": "real_y_intercept",
                    "target": 0.0,
                    "input_data": {"wavelength": "[0.4861, 0.55]"},
                }
            ],
            {},
        )


@pytest.mark.parametrize(
    "kind,axis,attribute,value",
    [
        ("tilt", "x", "rx", 0.02),
        ("tilt", "y", "ry", -0.03),
        ("decenter", "x", "x", 0.5),
        ("decenter", "y", "y", -0.2),
        ("decenter", "z", "z", 0.8),
    ],
)
def test_variable_current_value_uses_selected_axis(
    minimal_optic, kind, axis, attribute, value
):
    setattr(minimal_optic.surfaces[1].geometry.cs, attribute, value)
    assert current_value(
        minimal_optic, {"type": kind, "surface_number": 1, "axis": axis}
    ) == pytest.approx(value)


@pytest.mark.parametrize(
    "surface_type,variable_type,coefficient_key",
    [
        ("zernike", "zernike_coeff", "coeff_index"),
        ("even_asphere", "asphere_coeff", "coeff_number"),
    ],
)
def test_variable_current_value_uses_selected_coefficient(
    surface_type, variable_type, coefficient_key
):
    from optiland.optic import Optic

    optic = Optic()
    optic.surfaces.add(index=0, surface_type="plane")
    optic.surfaces.add(
        index=1, surface_type=surface_type, radius=50.0, coefficients=[0.01, 0.02]
    )
    assert current_value(
        optic, {"type": variable_type, "surface_number": 1, coefficient_key: 1}
    ) == pytest.approx(0.02)
