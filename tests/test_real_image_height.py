from __future__ import annotations

import optiland.backend as be
from optiland.optic import Optic


def create_simple_lens(infinite=True):
    optic = Optic()
    # Object Surface (0)
    optic.surfaces.add(surface_type="standard", is_stop=False, index=0)

    # Lens Front (1)
    optic.surfaces.add(surface_type="standard", radius=100.0, thickness=10.0, index=1)
    optic.updater.set_index(1.5, 1)  # N=1.5

    # Lens Back (2) - Aperture Stop
    optic.surfaces.add(
        surface_type="standard", radius=-100.0, thickness=100.0, is_stop=True, index=2
    )

    # Image Surface (3)
    optic.surfaces.add(surface_type="standard", index=3)

    optic.set_aperture("EPD", 10.0)
    optic.wavelengths.add(0.55)

    if infinite:
        optic.object_surface.geometry.cs.z = be.array(-float("inf"))
    else:
        optic.object_surface.geometry.cs.z = be.array(-200.0)

    optic.fields.set_type("real_image_height")
    optic.fields.add(y=5.0, x=0.0)  # Sets max_field to 5.0

    optic.updater.image_solve()  # Solve for paraxial focus
    return optic


def test_infinite_object():
    optic = create_simple_lens(infinite=True)

    # Define fields
    Hx = be.array([0.0, 0.5, 1.0, -0.5])
    Hy = be.array([0.0, 0.5, 1.0, -0.5])

    # Trace GENERIC rays (Px=0, Py=0) to check chief ray intersection
    rays = optic.trace_generic(Hx, Hy, Px=0.0, Py=0.0, wavelength=0.55)

    x_img = rays.x
    y_img = rays.y

    expected_x = Hx * optic.fields.max_field
    expected_y = Hy * optic.fields.max_field

    err_x = be.max(be.abs(x_img - expected_x))
    err_y = be.max(be.abs(y_img - expected_y))

    print(f"Infinite Max Error X: {err_x}")
    print(f"Infinite Max Error Y: {err_y}")

    assert err_x < 1e-6
    assert err_y < 1e-6


def test_finite_object():
    optic = create_simple_lens(infinite=False)

    # Define fields
    Hx = be.array([0.0, 0.5, 1.0, -0.5])
    Hy = be.array([0.0, 0.5, 1.0, -0.5])

    # Trace GENERIC rays (Px=0, Py=0) to check chief ray intersection
    rays = optic.trace_generic(Hx, Hy, Px=0.0, Py=0.0, wavelength=0.55)

    x_img = rays.x
    y_img = rays.y

    expected_x = Hx * optic.fields.max_field
    expected_y = Hy * optic.fields.max_field

    err_x = be.max(be.abs(x_img - expected_x))
    err_y = be.max(be.abs(y_img - expected_y))

    print(f"Finite Max Error X: {err_x}")
    print(f"Finite Max Error Y: {err_y}")

    assert err_x < 1e-6
    assert err_y < 1e-6
