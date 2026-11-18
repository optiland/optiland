from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.analysis import ChromaticFocalShift
from optiland.materials import Material
from optiland.optic import Optic
from optiland.samples.objectives import CookeTriplet
from tests.utils import assert_allclose

matplotlib.use("Agg")

F_LINE, D_LINE, C_LINE = 0.4861327, 0.5875618, 0.6562725
R1, R2, THICKNESS = 50.0, -50.0, 0.5


def nbk7():
    return Material("N-BK7", catalog="schott")


def singlet(wavelengths=(F_LINE, D_LINE, C_LINE), primary=D_LINE, object_distance=None):
    """Equiconvex N-BK7 singlet, at infinite conjugate unless an object
    distance is given."""
    lens = Optic()
    lens.surfaces.add(
        index=0, thickness=be.inf if object_distance is None else object_distance
    )
    lens.surfaces.add(index=1, radius=R1, thickness=THICKNESS, material=nbk7())
    lens.surfaces.add(index=2, radius=R2, thickness=48.0, is_stop=True)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=5.0)
    lens.fields.set_type("angle" if object_distance is None else "object_height")
    lens.fields.add(y=0.0)
    for w in wavelengths:
        lens.wavelengths.add(value=w, is_primary=(w == primary))
    return lens


def back_focal_distance(n):
    """Thick-lens back focal distance of the singlet in air, from the last
    vertex."""
    power = (n - 1) * (1 / R1 - 1 / R2 + (n - 1) * THICKNESS / (n * R1 * R2))
    f = 1 / power
    return f * (1 - (n - 1) * THICKNESS / (n * R1))


def mirror():
    """A concave spherical mirror at infinite conjugate: no refraction."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1, radius=-200.0, thickness=-100.0, material="mirror", is_stop=True
    )
    lens.surfaces.add(index=2)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.45)
    lens.wavelengths.add(value=0.55, is_primary=True)
    lens.wavelengths.add(value=0.70)
    return lens


def test_singlet_f_to_c_matches_thin_lens(set_test_backend):
    """The F-to-C shift of a thin singlet is f / V within 2 %."""
    lens = singlet()
    shift = ChromaticFocalShift(lens).data["defined_focal_shift"]
    shift_f, shift_c = be.to_numpy(shift[0]), be.to_numpy(shift[2])

    m = nbk7()
    n_f, n_d, n_c = (
        float(be.to_numpy(m.n(w)).item()) for w in (F_LINE, D_LINE, C_LINE)
    )
    abbe = (n_d - 1) / (n_f - n_c)
    f = 1 / ((n_d - 1) * (1 / R1 - 1 / R2))

    # the C focus is farther from the lens than the F focus, by f / V
    assert abs((shift_c - shift_f) / (f / abbe) - 1) < 0.02


def test_singlet_curve_matches_thick_lens(set_test_backend):
    """Every point of the curve is the change of the thick-lens back focal
    distance from its value at the primary wavelength."""
    lens = singlet()
    cfs = ChromaticFocalShift(lens, wavelength_range=(0.45, 0.70), num_points=11)
    m = nbk7()
    bfd_d = back_focal_distance(m.n(D_LINE))
    expected = [
        back_focal_distance(m.n(float(w))) - bfd_d
        for w in be.to_numpy(cfs.data["wavelengths"])
    ]
    assert_allclose(
        cfs.data["focal_shift"], be.stack([be.ravel(e)[0] for e in expected])
    )


def test_finite_conjugate_matches_thick_lens(set_test_backend):
    """With the object 200 mm in front of the singlet, the shift is the change
    of the thick-lens image distance."""
    object_distance = 200.0
    lens = singlet(object_distance=object_distance)
    m = nbk7()

    def image_distance(n):
        # ray from the axial object point through the two surfaces
        u = 1.0 / object_distance
        y = 1.0
        u = (u - y * (n - 1) / R1) / n
        y = y + u * THICKNESS
        u = n * u - y * (1 - n) / R2
        return -y / u

    cfs = ChromaticFocalShift(lens, wavelength_range=(0.45, 0.70), num_points=11)
    v_d = image_distance(m.n(D_LINE))
    expected = [
        image_distance(m.n(float(w))) - v_d
        for w in be.to_numpy(cfs.data["wavelengths"])
    ]
    assert_allclose(
        cfs.data["focal_shift"], be.stack([be.ravel(e)[0] for e in expected])
    )


def test_all_reflective_shift_is_zero(set_test_backend):
    cfs = ChromaticFocalShift(mirror())
    assert_allclose(cfs.data["focal_shift"], be.zeros(128), atol=1e-12)
    assert_allclose(cfs.data["defined_focal_shift"], be.zeros(3), atol=1e-12)
    assert cfs.max_focal_shift() < 1e-12


def test_shift_at_primary_is_exactly_zero(set_test_backend):
    lens = CookeTriplet()
    cfs = ChromaticFocalShift(lens)
    primary = lens.wavelengths.primary_index
    assert be.to_numpy(cfs.data["defined_focal_shift"][primary]) == 0.0


def test_default_range_and_grid(set_test_backend):
    lens = CookeTriplet()
    cfs = ChromaticFocalShift(lens, num_points=64)
    values = [w.value for w in lens.wavelengths.wavelengths]
    assert cfs.wavelength_range == (min(values), max(values))
    assert be.size(cfs.data["wavelengths"]) == 64
    assert be.size(cfs.data["focal_shift"]) == 64
    assert_allclose(cfs.data["wavelengths"][0], min(values))
    assert_allclose(cfs.data["wavelengths"][-1], max(values))


def test_explicit_range(set_test_backend):
    cfs = ChromaticFocalShift(CookeTriplet(), wavelength_range=(0.4, 0.8))
    assert_allclose(cfs.data["wavelengths"][0], 0.4)
    assert_allclose(cfs.data["wavelengths"][-1], 0.8)


def test_max_focal_shift(set_test_backend):
    cfs = ChromaticFocalShift(CookeTriplet())
    shift = be.to_numpy(cfs.data["focal_shift"])
    assert cfs.max_focal_shift() == pytest.approx(shift.max() - shift.min())
    assert cfs.max_focal_shift() > 0


def test_empty_range_raises(set_test_backend):
    lens = singlet(wavelengths=(D_LINE,))
    with pytest.raises(ValueError, match="wavelength range is empty"):
        ChromaticFocalShift(lens)
    with pytest.raises(ValueError, match="wavelength range is empty"):
        ChromaticFocalShift(CookeTriplet(), wavelength_range=(0.6, 0.5))
    cfs = ChromaticFocalShift(lens, wavelength_range=(0.5, 0.6))
    assert cfs.max_focal_shift() > 0


def test_view(set_test_backend):
    cfs = ChromaticFocalShift(CookeTriplet())
    fig, ax = cfs.view(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # the curve, in µm
    assert np.allclose(
        ax.lines[0].get_ydata(), be.to_numpy(cfs.data["focal_shift"]) * 1e3
    )
    plt.close(fig)


def test_view_on_existing_figure(set_test_backend):
    cfs = ChromaticFocalShift(CookeTriplet())
    existing = plt.figure()
    fig, ax = cfs.view(fig_to_plot_on=existing)
    assert fig is existing
    assert ax in fig.axes
    plt.close(fig)


@patch("matplotlib.pyplot.show")
def test_view_shows_by_default(mock_show, set_test_backend):
    cfs = ChromaticFocalShift(CookeTriplet())
    fig, _ = cfs.view()
    mock_show.assert_called_once()
    plt.close(fig)
