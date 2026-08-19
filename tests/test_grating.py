from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optic import Optic

from .utils import assert_allclose


@pytest.fixture
def flat_transmission_grating():
    """flat transmission grating with 3 fields and 1 wavelength"""
    lens = Optic()

    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, radius=be.inf, thickness=10)
    lens.surfaces.add(index=2, radius=be.inf, thickness=5, material="N-BK7")
    lens.surfaces.add(
        index=3,
        radius=be.inf,
        thickness=30,
        surface_type="grating",
        grating_order=-1,
        grating_period=5.0,
        groove_orientation_angle=0.0,
        is_stop=True,
    )
    lens.surfaces.add(index=4)

    # add aperture
    lens.set_aperture(aperture_type="EPD", value=15)

    # add field
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=10)
    lens.fields.add(y=0, x=10)

    # add wavelength
    lens.wavelengths.add(value=0.587, is_primary=True)

    lens.updater.update_paraxial()

    return lens


@pytest.fixture
def curved_transmission_grating():
    """curved transmission grating with 3 fields and 1 wavelength"""
    lens = Optic()

    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, radius=be.inf, thickness=10)
    lens.surfaces.add(index=2, radius=be.inf, thickness=5, material="N-BK7")
    lens.surfaces.add(
        index=3,
        radius=50.0,
        thickness=30,
        conic=1.0,
        surface_type="grating",
        grating_order=-1,
        grating_period=5.0,
        groove_orientation_angle=0.0,
        is_stop=True,
    )
    lens.surfaces.add(index=4)

    # add aperture
    lens.set_aperture(aperture_type="EPD", value=15)

    # add field
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=10)
    lens.fields.add(y=0, x=10)

    # add wavelength
    lens.wavelengths.add(value=0.587, is_primary=True)

    lens.updater.update_paraxial()

    return lens


@pytest.fixture
def curved_reflective_grating():
    """curved reflective grating with 3 fields and 1 wavelength"""
    lens = Optic()

    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=70,
        thickness=-30,
        material="mirror",
        surface_type="grating",
        is_stop=True,
        grating_period=5.0,
        grating_order=1,
        groove_orientation_angle=0.0,
    )
    lens.surfaces.add(index=2)

    # add aperture
    lens.set_aperture(aperture_type="EPD", value=15)

    # add field
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=10)
    lens.fields.add(y=0, x=10)

    # add wavelength
    lens.wavelengths.add(value=0.587, is_primary=True)

    lens.updater.update_paraxial()

    return lens


def test_flat_grating_transmission(set_test_backend, flat_transmission_grating):
    lens = flat_transmission_grating
    wv = 0.587
    Px = 0.0
    Py = 0.0
    Hx = 0.0
    Hy = 0.0
    # axial ray central field
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.1174, 0.9930847094])
    # marginal ray central field
    Px = 0.0
    Py = 1.0
    Hx = 0.0
    Hy = 0.0
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.1174, 0.9930847094])
    # generic ray
    Px = -0.15
    Py = 0.7
    Hx = 0.2
    Hy = 0.8
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [0.0345602649, 0.0216899611, 0.9991672201]
    )


def test_curved_grating_transmission(set_test_backend, curved_transmission_grating):
    lens = curved_transmission_grating
    wv = 0.587
    Px = 0.0
    Py = 0.0
    Hx = 0.0
    Hy = 0.0
    # axial ray central field
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.1174, 0.9930847094])
    # marginal ray central field
    Px = 0.0
    Py = 1.0
    Hx = 0.0
    Hy = 0.0
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.0379603895, 0.9992792447])
    # generic ray
    Px = -0.15
    Py = 0.7
    Hx = 0.2
    Hy = 0.8
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [0.0229384233, 0.0764682608, 0.9968081229]
    )


def test_curved_grating_reflection(set_test_backend, curved_reflective_grating):
    lens = curved_reflective_grating
    wv = 0.587
    # generic ray
    Px = -0.15
    Py = 0.7
    Hx = 0.2
    Hy = 0.8
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [-0.0040370331, -0.4006582284, 0.9162186892]
    )


def test_paraxial_flat_grating_transmission(
    set_test_backend, flat_transmission_grating
):
    lens = flat_transmission_grating
    wv = 0.587
    Hy = 0.0
    Py = 0.0
    lens.paraxial.trace(Hy=Hy, Py=Py, wavelength=wv)
    u = lens.surfaces.u[-1].item()
    y = lens.surfaces.y[-1].item()
    assert_allclose([u, y], [0.1174, 3.522])
    Hy = 0.0
    Py = 1.0
    lens.paraxial.trace(Hy=Hy, Py=Py, wavelength=wv)
    u = lens.surfaces.u[-1].item()
    y = lens.surfaces.y[-1].item()
    assert_allclose([u, y], [0.1174, 11.022])
    Hy = 0.8
    Py = 0.8
    lens.paraxial.trace(Hy=Hy, Py=Py, wavelength=wv)
    u = lens.surfaces.u[-1].item()
    y = lens.surfaces.y[-1].item()
    assert_allclose([u, y], [0.25794083, 13.73822504])
