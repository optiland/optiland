"""Variable accepts the keyword arguments of each variable type."""

from __future__ import annotations

import inspect

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.optimization.variable import variable as variable_module
from optiland.optimization.variable.base import VariableBehavior
from optiland.optimization.variable.variable import Variable

# Arguments that Variable passes itself, not from the user's keywords.
_OWN = {"self", "optic", "scaler", "type_name"}


def _behavior_classes():
    """The variable types that ``Variable`` can build."""
    return [
        obj
        for obj in vars(variable_module).values()
        if inspect.isclass(obj)
        and issubclass(obj, VariableBehavior)
        and obj is not VariableBehavior
    ]


def test_every_variable_keyword_is_allowed():
    classes = _behavior_classes()
    assert classes
    allowed = Variable.allowed_attributes()
    for cls in classes:
        params = inspect.signature(cls.__init__).parameters.values()
        names = {
            p.name
            for p in params
            if p.name not in _OWN and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        assert names <= allowed, (cls.__name__, names - allowed)


def _q2d_singlet():
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=30.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(
        index=2,
        surface_type="forbes_q2d",
        radius=-40.0,
        conic=-0.5,
        freeform_coeffs={("a", 1, 1): 1e-3},
        norm_radius=9.0,
        thickness=35.0,
    )
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    return optic


def test_forbes_q2d_variable_gives_no_warning(set_test_backend, capsys):
    optic = _q2d_singlet()
    var = Variable(optic, "forbes_q2d_coeff", surface_number=2, coeff_tuple=("a", 1, 1))
    assert "not a recognized attribute" not in capsys.readouterr().out
    assert var.coeff_tuple == ("a", 1, 1)
    assert float(be.to_numpy(var.value)) == pytest.approx(1e-3)
