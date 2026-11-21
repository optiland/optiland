from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.analysis import GhostAnalysis
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.samples.objectives import CookeTriplet
from optiland.sequences import SequencedOptic
from tests.utils import assert_allclose

matplotlib.use("Agg")

R1, R2, THICKNESS, INDEX, IMAGE_DISTANCE = 50.0, -50.0, 5.0, 1.5, 40.0


def singlet(epd=1.0):
    """Biconvex singlet of constant index at infinite conjugate."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=R1,
        thickness=THICKNESS,
        material=IdealMaterial(n=INDEX),
        is_stop=True,
    )
    lens.surfaces.add(index=2, radius=R2, thickness=IMAGE_DISTANCE)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=epd)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def paraxial_ghost_focus_shift():
    """Independent y-nu trace of the 1-2 ghost of the singlet.

    A reflection is a refraction into index -n, followed by a negative
    thickness. Returns the ghost focus measured from the image surface.
    """

    def refract(y, nu, n1, n2, radius):
        return y, nu - y * (n2 - n1) / radius

    y, nu = 1e-3, 0.0
    y, nu = refract(y, nu, 1.0, INDEX, R1)
    y = y + nu / INDEX * THICKNESS
    y, nu = refract(y, nu, INDEX, -INDEX, R2)
    y = y + nu / -INDEX * -THICKNESS
    y, nu = refract(y, nu, -INDEX, INDEX, R1)
    y = y + nu / INDEX * THICKNESS
    y, nu = refract(y, nu, INDEX, 1.0, R2)
    return -y / nu - IMAGE_DISTANCE


def test_singlet_ghost_matches_paraxial_trace(set_test_backend):
    ghost = GhostAnalysis(singlet(), num_rings=4)

    assert ghost.data["surfaces"] == [(1, 2)]
    fresnel = ((INDEX - 1) / (INDEX + 1)) ** 2
    assert_allclose(ghost.data["reflectance"], [fresnel**2])
    assert_allclose(ghost.data["transmitted_fraction"], [1.0])
    assert_allclose(ghost.data["energy_fraction"], [fresnel**2])
    # Real rays with a 1 mm pupil: within 0.01 mm of the paraxial focus.
    assert_allclose(
        ghost.data["focus_shift"], [paraxial_ghost_focus_shift()], atol=1e-2
    )


def test_singlet_irradiance_from_spot_radius(set_test_backend):
    ghost = GhostAnalysis(singlet(), num_rings=4)
    radius = ghost.data["rms_radius"]
    assert_allclose(
        ghost.data["irradiance"], ghost.data["energy_fraction"] / (np.pi * radius**2)
    )


def test_cooke_triplet_enumerates_every_pair(set_test_backend):
    ghost = GhostAnalysis(CookeTriplet(), num_rings=4)

    expected = {(i, j) for j in range(2, 7) for i in range(1, j)}
    assert set(ghost.data["surfaces"]) == expected
    assert len(ghost.data["surfaces"]) == len(expected)
    for key in ("reflectance", "focus_shift", "irradiance", "rms_radius"):
        assert be.to_numpy(ghost.data[key]).shape == (len(expected),)


def test_ghosts_are_sorted_brightest_first(set_test_backend):
    ghost = GhostAnalysis(CookeTriplet(), num_rings=4)
    irradiance = be.to_numpy(ghost.data["irradiance"])
    assert np.all(np.diff(irradiance) <= 0)


def test_scalar_reflectance_applies_to_every_surface(set_test_backend):
    ghost = GhostAnalysis(CookeTriplet(), num_rings=4, reflectance=0.01)
    assert_allclose(ghost.data["reflectance"], np.full(15, 1e-4))


def test_dict_reflectance_selects_surfaces_and_allows_sensor(set_test_backend):
    lens = singlet()
    ghost = GhostAnalysis(lens, num_rings=4, reflectance={2: 0.01, 3: 0.2})

    assert ghost.data["surfaces"] == [(2, 3)]
    assert_allclose(ghost.data["reflectance"], [0.002])


def test_zero_reflectance_surfaces_are_skipped(set_test_backend):
    lens = singlet()
    ghost = GhostAnalysis(lens, num_rings=4, reflectance={1: 0.0, 2: 0.01, 3: 0.2})
    assert ghost.data["surfaces"] == [(2, 3)]


def test_no_candidate_pairs_gives_empty_data(set_test_backend):
    ghost = GhostAnalysis(singlet(), num_rings=4, reflectance={1: 0.01})
    assert ghost.data["surfaces"] == []
    assert be.to_numpy(ghost.data["irradiance"]).shape == (0,)


def test_analysis_does_not_register_sequences():
    lens = singlet()
    GhostAnalysis(lens, num_rings=4)
    assert lens.sequences == {}


def test_sequence_returns_the_ghost_path():
    ghost = GhostAnalysis(CookeTriplet(), num_rings=4)
    sequence = ghost.sequence(2, 5)

    assert isinstance(sequence, SequencedOptic)
    assert sequence.raw_steps == [
        0,
        1,
        2,
        3,
        4,
        (5, "reflect"),
        4,
        3,
        (2, "reflect"),
        3,
        4,
        5,
        6,
        7,
    ]


@pytest.mark.parametrize("i, j", [(0, 2), (2, 2), (3, 2), (1, 8)])
def test_sequence_rejects_invalid_indices(i, j):
    ghost = GhostAnalysis(CookeTriplet(), num_rings=4)
    with pytest.raises(ValueError, match="0 < i < j"):
        ghost.sequence(i, j)


def test_mirror_is_rejected():
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1, radius=-200.0, thickness=-100.0, material="mirror", is_stop=True
    )
    lens.surfaces.add(index=2)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.55, is_primary=True)

    with pytest.raises(ValueError, match="reflective"):
        GhostAnalysis(lens)


@pytest.mark.parametrize("reflectance", [-0.1, 1.5, {1: 2.0}])
def test_reflectance_out_of_range_is_rejected(reflectance):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        GhostAnalysis(singlet(), reflectance=reflectance)


@pytest.mark.parametrize("index", [0, 4])
def test_reflectance_index_out_of_range_is_rejected(index):
    with pytest.raises(ValueError, match="out of range"):
        GhostAnalysis(singlet(), reflectance={index: 0.1, 1: 0.1})


def test_explicit_wavelength(set_test_backend):
    lens = CookeTriplet()
    ghost = GhostAnalysis(lens, wavelength=0.6, num_rings=4)
    assert ghost.wavelengths[0].value == pytest.approx(0.6)


def test_view_returns_figure_and_two_axes(set_test_backend):
    ghost = GhostAnalysis(CookeTriplet(), num_rings=4)
    fig, axes = ghost.view(show=False, max_ghosts=5)

    assert isinstance(fig, Figure)
    assert len(axes) == 2
    assert len(axes[0].patches) == 5
    plt.close(fig)


def test_view_on_existing_figure(set_test_backend):
    ghost = GhostAnalysis(CookeTriplet(), num_rings=4)
    existing = plt.figure()
    fig, _ = ghost.view(fig_to_plot_on=existing)
    assert fig is existing
    plt.close(fig)


def test_off_axis_field_and_fully_clipped_ghost(set_test_backend):
    # At full field, the apertures clip all but at most one ray of a Cooke
    # triplet ghost, too few for a spot size or a focus.
    off_axis = GhostAnalysis(CookeTriplet(), field=(0.0, 1.0), num_rings=4)
    data = {k: be.to_numpy(v) for k, v in off_axis.data.items() if k != "surfaces"}

    survives = np.isfinite(data["focus_shift"])
    assert 0 < np.sum(survives) < len(survives)
    num_rays = 1 + 3 * 4 * 5  # hexapolar, 4 rings
    assert np.all(data["transmitted_fraction"][~survives] < 2 / num_rays)
    assert np.all(np.isfinite(data["focus_shift"][survives]))
    assert np.all(np.isnan(data["focus_shift"][~survives]))
    assert np.all(np.isnan(data["rms_radius"][~survives]))
    assert np.all(data["irradiance"][~survives] == 0)
    # Clipped ghosts sort last.
    assert not np.any(survives[np.argmin(survives) :])
