from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland.analysis import GaussianBeamAnalysis
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.paraxial_path import (
    ParaxialDomainWarning,
    UnsupportedParaxialGeometryError,
)
from optiland.physical_optics import GaussianBeam
from optiland.samples.objectives import CookeTriplet
from optiland.samples.telescopes import HubbleTelescope
from optiland.sources import SMFSource
from tests.test_folded_paraxial import folded, periscope, retro, straight
from tests.test_folded_paraxial_hardening import (
    folded_tilted_lens,
    oblique_powered_mirror,
)
from tests.utils import assert_allclose

matplotlib.use("Agg")

WAVELENGTH = 1.064  # µm
LAM = WAVELENGTH * 1e-3  # mm


def _finish(optic, aperture=("EPD", 10.0), field_type="angle", wavelength=WAVELENGTH):
    optic.set_aperture(aperture_type=aperture[0], value=aperture[1])
    optic.fields.set_type(field_type)
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=wavelength, is_primary=True)
    return optic


def thin_lens(image_distance=100.0, object_distance=be.inf):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=object_distance)
    optic.surfaces.add(
        index=1,
        surface_type="paraxial",
        f=100.0,
        thickness=image_distance,
        is_stop=True,
    )
    optic.surfaces.add(index=2)
    if object_distance == be.inf:
        return _finish(optic)
    return _finish(optic, aperture=("objectNA", 0.01), field_type="object_height")


def unfolded_twin():
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, thickness=10.0, is_stop=True)
    optic.surfaces.add(index=2, radius=25.84, thickness=4.0, material="N-BK7")
    optic.surfaces.add(index=3, radius=be.inf, thickness=46.0)
    optic.surfaces.add(index=4)
    return _finish(optic)


def lens_after_mirror():
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=be.inf, material="mirror", thickness=-10.0, is_stop=True
    )
    optic.surfaces.add(index=2, radius=-25.84, thickness=-4.0, material="N-BK7")
    optic.surfaces.add(index=3, radius=be.inf, thickness=-46.0)
    optic.surfaces.add(index=4)
    return _finish(optic)


def test_thin_lens_matches_self(set_test_backend):
    beam = GaussianBeam.from_waist(1.0, -50.0, wavelength=WAVELENGTH)
    analysis = GaussianBeamAnalysis(thin_lens(), beam)
    out = analysis.beam_at(1, "after")
    assert_allclose(out.waist_position, 99.942664, atol=1e-6)
    assert_allclose(out.waist_radius * 1e3, 33.8633, atol=1e-4)
    # At the image plane, 100 mm after the lens, the waist is 0.057 mm behind.
    assert_allclose(analysis.beam_at(-1, "before").waist_position, -0.057336, atol=1e-6)


def test_glass_block_moves_waist_to_n_d(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=50.0, material="N-BK7", is_stop=True)
    optic.surfaces.add(index=2, thickness=10.0)
    optic.surfaces.add(index=3)
    _finish(optic)
    n_glass = be.ravel(optic.surfaces.n(WAVELENGTH))[1]
    beam = GaussianBeam.from_waist(0.05, 20.0, wavelength=WAVELENGTH)
    out = GaussianBeamAnalysis(optic, beam).beam_at(1, "after")
    assert_allclose(out.waist_position, n_glass * 20.0)
    assert_allclose(out.waist_position, 30.132696, atol=1e-6)
    assert_allclose(out.waist_radius, 0.05)
    assert_allclose(out.refractive_index, n_glass)


def test_surface_by_surface_equals_system_matrix(set_test_backend):
    optic = CookeTriplet()
    last = optic.surfaces.num_surfaces - 1
    beam = GaussianBeam.from_waist(2.0, -30.0, wavelength=WAVELENGTH)
    analysis = GaussianBeamAnalysis(optic, beam)
    matrix = optic.paraxial.ray_transfer_matrix(1, last, WAVELENGTH)
    expected = beam.transform(matrix)
    out = analysis.beam_at(-1, "after")
    assert_allclose(out.q_real, expected.q_real, rtol=1e-9, atol=1e-9)
    assert_allclose(out.q_imag, expected.q_imag, rtol=1e-9, atol=1e-12)


def test_collimated_beam_focuses_to_lambda_f_over_pi_w(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=25.0, thickness=6.0, material="N-BK7", is_stop=True
    )
    optic.surfaces.add(index=2, radius=-75.0, thickness=34.0)
    optic.surfaces.add(index=3)
    _finish(optic, aperture=("EPD", 20.0))
    beam = GaussianBeam.from_waist(2.0, 0.0, wavelength=WAVELENGTH)
    out = GaussianBeamAnalysis(optic, beam).beam_at(2, "after")
    f2 = optic.paraxial.f2()
    assert_allclose(out.waist_radius, LAM * f2 / (math.pi * 2.0), rtol=1e-4)
    assert_allclose(out.waist_radius * 1e3, 6.396, atol=1e-3)


@pytest.mark.parametrize("make", [retro, folded, periscope])
def test_folded_systems_match_straight(set_test_backend, make):
    beam = GaussianBeam.from_waist(1.0, -30.0, wavelength=WAVELENGTH)
    reference = GaussianBeamAnalysis(straight(), beam)
    analysis = GaussianBeamAnalysis(make(), beam)
    expected = reference.beam_at(-1, "before")
    out = analysis.beam_at(-1, "before")
    assert_allclose(out.q_real, expected.q_real, rtol=1e-9, atol=1e-9)
    assert_allclose(out.q_imag, expected.q_imag, rtol=1e-9, atol=1e-9)
    assert_allclose(
        analysis.data["gouy_phase"][-1], reference.data["gouy_phase"][-1], atol=1e-9
    )
    assert_allclose(out.q_real, -2.342017515, atol=1e-8)
    assert_allclose(out.q_imag, 0.880977387, atol=1e-8)


def test_lens_on_reversed_leg_matches_unfolded_twin(set_test_backend):
    beam = GaussianBeam.from_waist(1.0, -30.0, wavelength=WAVELENGTH)
    expected = GaussianBeamAnalysis(unfolded_twin(), beam).beam_at(-1)
    analysis = GaussianBeamAnalysis(lens_after_mirror(), beam)
    out = analysis.beam_at(-1)
    assert_allclose(out.q_real, expected.q_real, rtol=1e-9, atol=1e-9)
    assert_allclose(out.q_imag, expected.q_imag, rtol=1e-9, atol=1e-9)
    for side in ("before", "after"):
        for b in analysis.data[side]:
            assert float(be.to_numpy(b.q_imag)) > 0


def test_concave_mirror_matches_self(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=-200.0, material="mirror", thickness=-100.0, is_stop=True
    )
    optic.surfaces.add(index=2)
    _finish(optic)
    beam = GaussianBeam.from_waist(2.0, 0.0, wavelength=WAVELENGTH)
    out = GaussianBeamAnalysis(optic, beam).beam_at(1, "after")
    z_r = math.pi * 4.0 / LAM
    mag2 = 1 / (1 + (z_r / 100.0) ** 2)
    assert_allclose(out.waist_position, 100.0 * (1 - mag2))
    assert_allclose(out.waist_radius, 2.0 * math.sqrt(mag2))
    assert_allclose(out.waist_position, 99.992831, atol=1e-6)


def etpathfinder_arm(length=9220.0, radius=14500.0):
    # ETpathfinder arm cavity at 1550 nm (Utina et al. 2022, Class. Quantum
    # Grav. 39 215008, Sec. 2.2): mirrors 9.22 m apart with 14.5 m radii of
    # curvature, quoted as a 1.8 mm waist and 2.2 mm on the mirrors. Surface 1
    # is the input mirror's reflective face, heading to the end mirror.
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=length, is_stop=True)
    optic.surfaces.add(index=2, radius=-radius, material="mirror", thickness=-length)
    optic.surfaces.add(index=3, radius=radius, material="mirror", thickness=0.0)
    optic.surfaces.add(index=4)
    return _finish(optic, wavelength=1.55)


def test_etpathfinder_arm_cavity_eigenmode(set_test_backend):
    length, radius, lam = 9220.0, 14500.0, 1.55e-3
    optic = etpathfinder_arm(length, radius)
    matrix = optic.paraxial.ray_transfer_matrix(1, 3, 1.55)
    mode = GaussianBeam.from_round_trip(matrix, wavelength=1.55)
    analysis = GaussianBeamAnalysis(optic, mode)

    again = analysis.beam_at(3, "after")
    assert_allclose(again.q_real, mode.q_real, atol=1e-8)
    assert_allclose(again.q_imag, mode.q_imag)

    g = 1 - length / radius
    z_r = math.sqrt(length * (2 * radius - length)) / 2
    w_mirror = math.sqrt(lam * length / (math.pi * math.sqrt(1 - g**2)))
    assert_allclose(mode.waist_position, length / 2)
    assert_allclose(mode.waist_radius, math.sqrt(lam * z_r / math.pi))
    assert_allclose(mode.beam_radius, w_mirror)
    assert_allclose(analysis.beam_at(2, "before").beam_radius, w_mirror)
    assert_allclose(analysis.data["gouy_phase"][-1], 2 * math.acos(g))
    assert_allclose(mode.waist_radius, 1.825220, atol=1e-6)
    assert_allclose(mode.beam_radius, 2.210045, atol=1e-6)


def test_hubble_focus(set_test_backend):
    optic = HubbleTelescope()
    w_in = be.to_numpy(optic.paraxial.EPD()).item() / 4
    beam = GaussianBeam.from_waist(w_in, 0.0, wavelength=WAVELENGTH)
    analysis = GaussianBeamAnalysis(optic, beam)
    for side in ("before", "after"):
        for b in analysis.data[side]:
            assert float(be.to_numpy(b.q_imag)) > 0
    f2 = optic.paraxial.f2()
    out = analysis.beam_at(-1)
    assert_allclose(out.waist_radius, LAM * f2 / (math.pi * w_in), rtol=1e-4)


SOBOL_SEED = 2026


def test_matches_second_moments_of_smf_source_rays(set_test_backend, monkeypatch):
    # A Gaussian beam's phase-space distribution is Gaussian, and its second
    # moments follow the same ABCD law, so real rays sampled by SMFSource must
    # have an rms height of w / 2 at any plane.
    #
    # SMFSource scrambles its Sobol sequence randomly. On PyTorch the samples
    # are drawn in float32, where a sample close to 1 can round to exactly 1.0
    # and erfinv turns it into an infinite ray (in about 0.7% of 2**16-ray
    # draws). A fixed seed keeps this test deterministic on both backends.
    backend_class = type(be.sobol_sampler.__self__)
    sobol_sampler = backend_class.sobol_sampler

    def seeded_sobol_sampler(self, dim, num_samples, scramble=True, seed=None):
        return sobol_sampler(
            self, dim, num_samples, scramble=scramble, seed=SOBOL_SEED
        )

    monkeypatch.setattr(backend_class, "sobol_sampler", seeded_sobol_sampler)

    beam = GaussianBeam.from_waist(1.0, -50.0, wavelength=WAVELENGTH)
    for image_distance in (99.942664, 300.0):
        optic = thin_lens(image_distance, object_distance=50.0)
        w = GaussianBeamAnalysis(optic, beam).beam_at(-1).beam_radius
        source = SMFSource(
            mfd_um=2000.0, wavelength_um=WAVELENGTH, position=(0.0, 0.0, -50.0)
        )
        optic.surfaces.trace(source.generate_rays(2**16))
        y = be.to_numpy(optic.surfaces.y[-1]).ravel()
        assert_allclose(2 * np.sqrt(np.mean(y**2)), w, rtol=1e-3)


def test_unsupported_geometry_raises(set_test_backend):
    beam = GaussianBeam.from_waist(1.0, wavelength=WAVELENGTH)
    with pytest.raises(
        UnsupportedParaxialGeometryError, match="TILTED_REFRACTIVE_SURFACE"
    ):
        GaussianBeamAnalysis(folded_tilted_lens(), beam)


def test_astigmatic_fold_warns(set_test_backend):
    # Same policy as the paraxial API: approximate, so a warning, not an error.
    beam = GaussianBeam.from_waist(1.0, wavelength=WAVELENGTH)
    with pytest.warns(ParaxialDomainWarning, match="OBLIQUE_POWERED_MIRROR"):
        GaussianBeamAnalysis(oblique_powered_mirror(), beam)


def singlet(surface_type="standard", **front):
    """N-BK7 singlet whose front surface is spherical, toroidal or biconic."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        surface_type=surface_type,
        thickness=5.0,
        material="N-BK7",
        is_stop=True,
        **front,
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=100.0)
    optic.surfaces.add(index=3)
    return _finish(optic)


@pytest.mark.parametrize("surface_type", ["toroidal", "biconic"])
@pytest.mark.parametrize(
    ("radius_x", "radius_y"),
    [(51.68, 60.0), (51.68, be.inf), (be.inf, 51.68)],
)
def test_surface_with_different_xy_radii_raises(
    set_test_backend, surface_type, radius_x, radius_y
):
    # The first-order engine keeps one of the two radii, so a rotationally
    # symmetric beam would silently describe one plane only.
    beam = GaussianBeam.from_waist(1.0, wavelength=WAVELENGTH)
    optic = singlet(surface_type, radius_x=radius_x, radius_y=radius_y)
    with pytest.raises(UnsupportedParaxialGeometryError, match="ANAMORPHIC_SURFACE"):
        GaussianBeamAnalysis(optic, beam)


@pytest.mark.parametrize("surface_type", ["toroidal", "biconic"])
def test_surface_with_equal_xy_radii_matches_sphere(set_test_backend, surface_type):
    # Equal radii make the surface a sphere, which the analysis accepts.
    beam = GaussianBeam.from_waist(1.0, wavelength=WAVELENGTH)
    sphere = GaussianBeamAnalysis(singlet(radius=51.68), beam).beam_at(-1, "before")
    out = GaussianBeamAnalysis(
        singlet(surface_type, radius_x=51.68, radius_y=51.68), beam
    ).beam_at(-1, "before")
    assert_allclose(out.q_real, sphere.q_real, rtol=1e-12, atol=1e-12)
    assert_allclose(out.q_imag, sphere.q_imag, rtol=1e-12, atol=1e-12)


def test_beam_at_validates_arguments(set_test_backend):
    beam = GaussianBeam.from_waist(1.0, -50.0, wavelength=WAVELENGTH)
    analysis = GaussianBeamAnalysis(thin_lens(), beam)
    assert analysis.beam_at(-1) is analysis.beam_at(2)
    for bad in (0, 3, -3):
        with pytest.raises(ValueError):
            analysis.beam_at(bad)
    with pytest.raises(ValueError):
        analysis.beam_at(1, side="inside")


def test_view(set_test_backend):
    beam = GaussianBeam.from_waist(1.0, -50.0, wavelength=WAVELENGTH)
    fig, ax = GaussianBeamAnalysis(thin_lens(), beam).view(show=False)
    assert fig is not None
    assert len(ax.lines) >= 2
    plt.close(fig)


def test_view_draws_into_an_existing_figure(set_test_backend):
    beam = GaussianBeam.from_waist(1.0, -50.0, wavelength=WAVELENGTH)
    existing = plt.figure()
    fig, ax = GaussianBeamAnalysis(thin_lens(), beam).view(fig_to_plot_on=existing)
    assert fig is existing
    assert len(ax.lines) >= 2
    plt.close(fig)


def test_view_shows_a_new_figure(set_test_backend, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    beam = GaussianBeam.from_waist(1.0, -50.0, wavelength=WAVELENGTH)
    fig, _ = GaussianBeamAnalysis(thin_lens(), beam).view()
    assert shown == [True]
    plt.close(fig)


def test_view_without_legs_draws_an_empty_envelope(set_test_backend):
    # Object and image only: no leg between surfaces, so nothing to sample.
    optic = Optic()
    optic.surfaces.add(index=0, thickness=10.0)
    optic.surfaces.add(index=1, is_stop=True)
    beam = GaussianBeam.from_waist(1.0, wavelength=WAVELENGTH)
    fig, ax = GaussianBeamAnalysis(_finish(optic), beam).view(show=False)
    assert all(len(line.get_xdata()) == 0 for line in ax.lines[:2])
    plt.close(fig)


def test_input_beam_is_reexpressed_in_object_space_medium(set_test_backend):
    # A beam given in air entering water object space keeps its waist; its
    # Rayleigh range grows by the refractive index.
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf, material=IdealMaterial(n=1.33))
    optic.surfaces.add(index=1, radius=be.inf, thickness=10.0, is_stop=True)
    optic.surfaces.add(index=2)
    beam = GaussianBeam.from_waist(1.0, 0.0, wavelength=WAVELENGTH)
    inside = GaussianBeamAnalysis(_finish(optic), beam).beam_at(1, "before")
    assert_allclose(inside.refractive_index, 1.33)
    assert_allclose(inside.waist_radius, 1.0)
    assert_allclose(inside.waist_position, 0.0, atol=1e-12)
    assert_allclose(inside.rayleigh_range, 1.33 * beam.rayleigh_range)


def _two_lens(separation):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(
        index=1, surface_type="paraxial", f=100.0, thickness=150.0, is_stop=True
    )
    optic.surfaces.add(index=2, surface_type="paraxial", f=50.0, thickness=60.0)
    optic.surfaces.add(index=3)
    _finish(optic)
    optic.updater.set_thickness(separation, 1)
    return optic


def _image_waist_position(optic):
    beam = GaussianBeam.from_waist(1.0, -50.0, wavelength=WAVELENGTH)
    return GaussianBeamAnalysis(optic, beam).beam_at(-1).waist_position


def test_gradient_of_waist_position_wrt_thickness(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Autodiff test requires the torch backend.")
    be.grad_mode.enable()
    separation = be.array(150.0)
    separation.requires_grad_(True)
    value = _image_waist_position(_two_lens(separation))
    value.backward()
    grad = float(be.to_numpy(separation.grad))

    h = 1e-3
    plus = _image_waist_position(_two_lens(150.0 + h))
    minus = _image_waist_position(_two_lens(150.0 - h))
    grad_fd = float(be.to_numpy((plus - minus) / (2 * h)))
    assert math.isfinite(grad)
    assert_allclose(grad, grad_fd, rtol=1e-5)


@pytest.mark.parametrize("make", [straight, retro, periscope])
def test_gradient_of_waist_radius_wrt_lens_radius(set_test_backend, make):
    if be.get_backend() != "torch":
        pytest.skip("Autodiff test requires the torch backend.")
    be.grad_mode.enable()
    beam = GaussianBeam.from_waist(1.0, -30.0, wavelength=WAVELENGTH)

    def waist_radius(radius):
        optic = make()
        optic.surfaces[1].geometry.radius = radius
        return GaussianBeamAnalysis(optic, beam).beam_at(-1).waist_radius

    radius = be.array(25.84)
    radius.requires_grad_(True)
    waist_radius(radius).backward()
    grad = float(be.to_numpy(radius.grad))

    h = 1e-4
    plus = waist_radius(be.array(25.84 + h))
    minus = waist_radius(be.array(25.84 - h))
    grad_fd = float(be.to_numpy((plus - minus) / (2 * h)))
    assert grad != 0.0
    assert_allclose(grad, grad_fd, rtol=1e-5)


def test_gradient_of_cavity_mode_wrt_mirror_radius(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Autodiff test requires the torch backend.")
    be.grad_mode.enable()

    def spot_on_mirrors(radius):
        optic = etpathfinder_arm()
        optic.surfaces[2].geometry.radius = -radius
        optic.surfaces[3].geometry.radius = radius
        matrix = optic.paraxial.ray_transfer_matrix(1, 3, 1.55)
        return GaussianBeam.from_round_trip(matrix, wavelength=1.55).beam_radius

    radius = be.array(14500.0)
    radius.requires_grad_(True)
    spot_on_mirrors(radius).backward()
    grad = float(be.to_numpy(radius.grad))

    h = 1e-1
    plus = spot_on_mirrors(be.array(14500.0 + h))
    minus = spot_on_mirrors(be.array(14500.0 - h))
    grad_fd = float(be.to_numpy((plus - minus) / (2 * h)))
    # With g > 0, flatter mirrors give a larger spot.
    assert grad > 0.0
    assert_allclose(grad, grad_fd, rtol=1e-5)
