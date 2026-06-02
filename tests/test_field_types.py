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


def test_tilted_object_origins(set_test_backend):
    """Test ray origins for a tilted object surface across field types."""
    import numpy as np

    optic = Optic()
    # 45 deg tilt around x-axis
    rx = 45 * np.pi / 180
    optic.surfaces.add(index=0, thickness=60, rx=rx)
    optic.surfaces.add(index=1, thickness=10, is_stop=True)
    optic.surfaces.add(index=2)

    optic.set_aperture("EPD", 10)
    optic.wavelengths.add(0.58756, is_primary=True)

    # 1. Test ObjectHeightField
    optic.fields.set_type("object_height")
    optic.fields.add(y=10)

    origins = optic.fields.field_definition.get_ray_origins(
        optic, Hx=0, Hy=1, Px=0, Py=0, vx=0, vy=0
    )

    # Expected: y_global = 10 * cos(45), z_global = -60 + 10 * sin(45)
    assert_allclose(origins[0], [0.0])
    assert_allclose(origins[1], [10 * np.cos(rx)])
    assert_allclose(origins[2], [-60 + 10 * np.sin(rx)])

    # 2. Test AngleField (finite)
    optic.fields.set_type("angle")
    # dist_to_ep = EPL - pos[0] = 0 - (-60) = 60
    # y_local = -tan(field_y) * 60.
    # To get y_local = 10, we need tan(field_y) = -10/60 => field_y = arctan(-10/60)
    angle_deg = np.degrees(np.arctan(-10/60))
    optic.fields.fields = []
    optic.fields.add(y=angle_deg)

    origins = optic.fields.field_definition.get_ray_origins(
        optic, Hx=0, Hy=-1, Px=0, Py=0, vx=0, vy=0
    )
    # y_local should be 10. y_global = 10 * cos(45), z_global = -60 + 10 * sin(45)
    assert_allclose(origins[1], [10 * np.cos(rx)], atol=1e-7)
    assert_allclose(origins[2], [-60 + 10 * np.sin(rx)], atol=1e-7)

    # 3. Test ParaxialImageHeightField (finite)
    optic.fields.set_type("paraxial_image_height")
    # We set a field height that should correspond to some y_obj.
    optic.fields.fields = []
    optic.fields.add(y=10)

    # Add a lens to make ParaxialImageHeightField happy
    optic.surfaces.add(index=1, radius=50, thickness=5, material="N-BK7", is_stop=True)

    origins = optic.fields.field_definition.get_ray_origins(
        optic, Hx=0, Hy=1, Px=0, Py=0, vx=0, vy=0
    )
    # y0, z0 should reflect the tilt.
    # We verify that (y0, z0) correctly maps back to a point with local z=0
    # after rotating back by 45 degrees.
    y_global = be.to_numpy(origins[1])
    z_global = be.to_numpy(origins[2])
    z_local = -(y_global) * np.sin(rx) + (z_global + 60) * np.cos(rx)
    assert_allclose(z_local, [0.0], atol=1e-7)
