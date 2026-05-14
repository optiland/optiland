from __future__ import annotations

import optiland.backend as be
from optiland.optic import Optic
from optiland.samples import objectives

from .utils import assert_allclose


def test_paraxial_image_height_infinite_object(set_test_backend):
    """Test paraxial image height field type for an object at infinity."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(material="N-BK7", radius=50, thickness=5, index=1, is_stop=True)
    optic.surfaces.add(thickness=100, index=2)
    optic.surfaces.add(index=3)
    optic.fields.set_type("paraxial_image_height")
    optic.fields.add(y=10)
    optic.set_aperture("EPD", 10)
    optic.wavelengths.add(0.58756, is_primary=True)

    # trace a chief ray
    y, u = optic.paraxial.chief_ray()

    # verify that the ray's y-coordinate at the image plane matches the
    # expected paraxial image height
    assert_allclose(y[-1], 10, rtol=1e-5)


def test_paraxial_image_height_finite_object(set_test_backend):
    """Test paraxial image height field type for a finite object."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=50)
    optic.surfaces.add(material="N-BK7", radius=50, thickness=5, index=1, is_stop=True)
    optic.surfaces.add(thickness=100, index=2)
    optic.surfaces.add(index=3)
    optic.fields.set_type("paraxial_image_height")
    optic.fields.add(y=10)
    optic.set_aperture("EPD", 10)
    optic.wavelengths.add(0.58756, is_primary=True)

    # trace a chief ray
    y, u = optic.paraxial.chief_ray()

    # verify that the ray's y-coordinate at the image plane matches the
    # expected paraxial image height
    assert_allclose(y[-1], 9.67243803, rtol=1e-5)


def test_field_definition_to_dict(set_test_backend):
    """Test that field definition to_dict method works."""
    from optiland.fields.field_types import (
        AngleField,
        ObjectHeightField,
        ParaxialImageHeightField,
    )

    field_defs = [AngleField(), ObjectHeightField(), ParaxialImageHeightField()]
    for field_def in field_defs:
        d = field_def.to_dict()
        assert d["field_type"] == field_def.__class__.__name__


def test_field_definition_from_dict(set_test_backend):
    """Test that field definition from_dict method works."""
    from optiland.fields.field_types import (
        AngleField,
        BaseFieldDefinition,
        ObjectHeightField,
        ParaxialImageHeightField,
    )

    field_defs = [AngleField(), ObjectHeightField(), ParaxialImageHeightField()]
    for field_def in field_defs:
        d = field_def.to_dict()
        new_field_def = BaseFieldDefinition.from_dict(d)
        assert isinstance(new_field_def, field_def.__class__)


def test_paraxial_image_height_cooke_triplet(set_test_backend):
    """Test that paraxial image height field type is equivalent to angle field
    type for a Cooke triplet."""
    # load a Cooke triplet
    optic = objectives.CookeTriplet()

    # compute the chief ray with the default angle field type
    y_chief_angle, u_chief_angle = optic.paraxial.chief_ray()

    # get the paraxial image height for the original system
    paraxial_image_height = y_chief_angle[-1]

    # change the field type to paraxial image height and set the equivalent
    # field value
    optic.fields.set_type("paraxial_image_height")
    optic.fields.fields = []
    optic.fields.add(y=0)
    optic.fields.add(y=paraxial_image_height[0])

    # recompute the chief ray
    y_chief_pih, u_chief_pih = optic.paraxial.chief_ray()

    # check that the two chief rays are allclose
    assert_allclose(y_chief_angle, y_chief_pih)
    assert_allclose(u_chief_angle, u_chief_pih)


def test_real_image_height_get_ray_origins(set_test_backend):
    """Test that real image height field type get_ray_origins method works."""
    # load a Cooke triplet
    optic = objectives.CookeTriplet()

    # change the field type to real image height
    optic.fields.set_type("real_image_height")

    # set the field value
    optic.fields.fields = []
    optic.fields.add(y=0)
    optic.fields.add(y=20.0)

    # get the ray origins
    origins = optic.fields.field_definition.get_ray_origins(
        optic,
        Hx=0,
        Hy=1,
        Px=0,
        Py=0,
        vx=0,
        vy=0,
    )

    # basic structural checks
    assert len(origins) == 3
    assert_allclose(origins[0], [0.0])

    # Change to finite object
    optic = objectives.CookeTriplet()
    optic.object_surface.geometry.cs.z = be.array([-555.0])
    optic.fields.set_type("real_image_height")
    optic.fields.fields = []
    optic.fields.add(y=0)
    optic.fields.add(y=20.0)

    origins = optic.fields.field_definition.get_ray_origins(
        optic,
        Hx=0,
        Hy=1,
        Px=0,
        Py=0,
        vx=0,
        vy=0,
    )

    # stronger checks
    assert len(origins) == 3
    assert_allclose(origins[0], [0.0])
    assert_allclose(origins[2], [-555.0])


def test_paraxial_get_ray_origins(set_test_backend):
    """Test that paraxial image height field type get_ray_origins method
    works."""
    # load a Cooke triplet
    optic = objectives.CookeTriplet()

    # change the field type to paraxial image height
    optic.fields.set_type("paraxial_image_height")

    # set the field value
    optic.fields.fields = []
    optic.fields.add(y=0)
    optic.fields.add(y=20.0)

    # get the ray origins
    origins = optic.fields.field_definition.get_ray_origins(
        optic,
        Hx=0,
        Hy=1,
        Px=0,
        Py=0,
        vx=0,
        vy=0,
    )

    # check that the origins are correct
    assert_allclose(origins[0], [0])
    assert_allclose(origins[1], [-8.63986616])
    assert_allclose(origins[2], [-10])

    # Change to finite object
    optic = objectives.CookeTriplet()
    optic.object_surface.geometry.cs.z = be.array([-555.0])

    # get the ray origins
    origins = optic.fields.field_definition.get_ray_origins(
        optic,
        Hx=0,
        Hy=1,
        Px=0,
        Py=0,
        vx=0,
        vy=0,
    )

    # check that the origins are correct
    assert_allclose(origins[0], [0.0])
    assert_allclose(origins[2], [-555.0])


def test_get_paraxial_object_position(set_test_backend):
    """Test that paraxial image height field type get_paraxial_object_position
    method works."""
    # load a Cooke triplet
    optic = objectives.CookeTriplet()

    # change the field type to paraxial image height
    optic.fields.set_type("paraxial_image_height")

    # set the field value
    optic.fields.fields = []
    optic.fields.add(y=0)
    optic.fields.add(y=20.0)

    # get the paraxial object position
    obj_pos = optic.fields.field_definition.get_paraxial_object_position(
        optic, Hy=1, y1=0.5, EPL=optic.paraxial.EPL()
    )

    # check that the object position is correct (should be at infinity)
    assert_allclose(obj_pos[0], [-4.12359504])
    assert_allclose(obj_pos[1], [0.0])

    # Change to finite object
    optic = objectives.CookeTriplet()
    optic.object_surface.geometry.cs.z = be.array([-555.0])

    # get the paraxial object position
    obj_pos = optic.fields.field_definition.get_paraxial_object_position(
        optic, Hy=1, y1=0.5, EPL=optic.paraxial.EPL()
    )

    # check that the object position is correct
    assert_allclose(obj_pos[0], [-3.69008309])
    assert_allclose(obj_pos[1], [0.0])


def test_get_starting_z_offset(set_test_backend):
    """Test that paraxial image height field type get_starting_z_offset method
    works."""
    # load a Cooke triplet
    optic = objectives.CookeTriplet()

    # change the field type to paraxial image height
    optic.fields.set_type("paraxial_image_height")

    # set the field value
    optic.fields.fields = []
    optic.fields.add(y=0)
    optic.fields.add(y=20.0)

    # get the starting z offset
    z_offset = optic.fields.field_definition._get_starting_z_offset(optic)

    # check that the z offset is correct
    assert_allclose(z_offset, optic.paraxial.EPD())
