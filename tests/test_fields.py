from __future__ import annotations

import pytest

import optiland.backend as be
from optiland import fields

from .utils import assert_allclose


@pytest.mark.parametrize(
    "x, y",
    [(0, 0), (5.3, 8.5), (0, 4.2)],
)
def test_field(set_test_backend, x, y):
    f = fields.Field(x, y)

    assert f.x == x
    assert f.y == y


def test_field_group_inputs(set_test_backend):
    input_data = [(0, 0), (5, 0), (0, 6), (7, 9.2)]
    f = fields.FieldGroup()
    for field_data in input_data:
        f.add(x=field_data[0], y=field_data[1])

    assert_allclose(f.x_fields, be.array([0, 5, 0, 7]))
    assert_allclose(f.y_fields, be.array([0, 0, 6, 9.2]))

    assert f.max_x_field == 7
    assert f.max_y_field == 9.2
    assert f.max_field == be.sqrt(be.array(7**2 + 9.2**2))


def test_field_group_getters(set_test_backend):
    input_data = [(0, 0), (2.5, 0), (0, 2), (4, 3)]
    f = fields.FieldGroup()
    for field_data in input_data:
        f.add(x=field_data[0], y=field_data[1])

    assert f.get_field_coords() == [(0, 0), (0.5, 0), (0, 0.4), (0.8, 0.6)]

    assert f.get_field(0).x == 0
    assert f.get_field(0).y == 0
    assert f.get_field(3).x == 4
    assert f.get_field(3).y == 3

    # test case when max field is zero
    f = fields.FieldGroup()
    f.add(x=0, y=0)
    assert f.get_field_coords() == [(0, 0)]


def test_field_group_remove(set_test_backend):
    input_data = [(0, 0), (2.5, 0), (0, 2), (4, 3)]
    f = fields.FieldGroup()
    for field_data in input_data:
        f.add(x=field_data[0], y=field_data[1])

    assert f.num_fields == 4
    f.remove(1)
    assert f.num_fields == 3
    assert f.get_field(1).x == 0
    assert f.get_field(1).y == 2

    f.remove(0)
    assert f.num_fields == 2
    assert f.get_field(0).x == 0
    assert f.get_field(0).y == 2

    with pytest.raises(IndexError):
        f.remove(10)


def test_field_group_get_vig_factor(set_test_backend):
    input_data = [(0, 0)]
    f = fields.FieldGroup()
    for field_data in input_data:
        f.add(x=field_data[0], y=field_data[1])

    vx, vy = f.get_vig_factor(1, 1)
    assert vx == 0.0
    assert vy == 0.0

    input_data = [(0, 0), (0, 7), (0, 10)]
    f = fields.FieldGroup()
    for field_data in input_data:
        f.add(x=field_data[0], y=field_data[1], vx=0.2, vy=0.2)

    vx, vy = f.get_vig_factor(0.5, 0.7)
    assert vx == 0.2
    assert vy == 0.2

    vx, vy = f.get_vig_factor(1, 1)
    assert vx == 0.2
    assert vy == 0.2


def test_field_group_telecentric(set_test_backend):
    f = fields.FieldGroup()
    assert f.telecentric is False

    f.set_telecentric(True)
    assert f.telecentric is True


def test_field_group_set_type(set_test_backend):
    f = fields.FieldGroup()
    f.set_type("angle")
    assert isinstance(f.field_definition, fields.AngleField)

    f.set_type("object_height")
    assert isinstance(f.field_definition, fields.ObjectHeightField)

    f.set_type("paraxial_image_height")
    assert isinstance(f.field_definition, fields.ParaxialImageHeightField)


def test_field_to_dict(set_test_backend):
    f = fields.Field(0, 0)
    assert f.to_dict() == {"x": 0, "y": 0, "vx": 0.0, "vy": 0.0, "weight": 1.0}


def test_field_group_to_dict(set_test_backend):
    input_data = [(0, 0), (2.5, 0), (0, 2), (4, 3)]
    f = fields.FieldGroup()
    for field_data in input_data:
        f.add(x=field_data[0], y=field_data[1])

    data = f.to_dict()
    assert data["fields"] == [
        {"x": 0, "y": 0, "vx": 0.0, "vy": 0.0, "weight": 1.0},
        {"x": 2.5, "y": 0, "vx": 0.0, "vy": 0.0, "weight": 1.0},
        {"x": 0, "y": 2, "vx": 0.0, "vy": 0.0, "weight": 1.0},
        {"x": 4, "y": 3, "vx": 0.0, "vy": 0.0, "weight": 1.0},
    ]
    assert data["telecentric"] is False
    assert data["field_definition"] is None


def test_field_group_to_dict_with_definition(set_test_backend):
    f = fields.FieldGroup()
    f.set_type("angle")
    data = f.to_dict()
    assert "field_definition" in data
    assert data["field_definition"]["field_type"] == "AngleField"


def test_field_from_dict(set_test_backend):
    f = fields.Field.from_dict(
        {"x": 0, "y": 0, "vx": 0, "vy": 0},
    )
    assert f.x == 0
    assert f.y == 0
    assert f.vx == 0
    assert f.vy == 0


def test_field_group_from_dict(set_test_backend):
    f = fields.FieldGroup.from_dict(
        {
            "fields": [
                {"x": 0, "y": 0, "vx": 0, "vy": 0},
                {"x": 2.5, "y": 0, "vx": 0, "vy": 0},
                {"x": 0, "y": 2, "vx": 0, "vy": 0},
                {"x": 4, "y": 3, "vx": 0, "vy": 0},
            ],
            "telecentric": False,
        },
    )
    assert f.get_field(0).x == 0
    assert f.get_field(0).y == 0
    assert f.get_field(3).x == 4
    assert f.get_field(3).y == 3
    assert f.telecentric is False
