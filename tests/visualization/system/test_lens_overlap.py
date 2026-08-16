from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.visualization.system import OpticViewer3D
from optiland.visualization.system.lens import Lens2D, Lens3D

matplotlib.use("Agg")


def _create_singlet(
    r1: float,
    r2: float,
    thickness: float,
    epd: float = 20.0,
    material: str = "N-BK7",
    asphere: bool = False,
) -> Optic:
    optic = Optic()
    optic.wavelengths.add(value=0.55, is_primary=True)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)

    if asphere:
        optic.surfaces.add(
            index=1,
            surface_type="even_asphere",
            radius=r1,
            conic=-1.0,
            coefficients=[1e-5, 1e-8],
            thickness=thickness,
            material=material,
            is_stop=True,
        )
    else:
        optic.surfaces.add(
            index=1,
            radius=r1,
            thickness=thickness,
            material=material,
            is_stop=True,
        )

    optic.surfaces.add(index=2, radius=r2, thickness=50.0)
    optic.surfaces.add(index=3, radius=be.inf)
    optic.set_aperture(aperture_type="EPD", value=epd)
    return optic


def test_overlapping_biconvex_lens_warns(set_test_backend):
    """Test that a biconvex lens with overlapping edges emits UserWarning."""
    # R1=50, R2=-50, EPD=20 (r=10): sag1 ~ 1.01, sag2 ~ -1.01 -> min thickness < 0
    optic = _create_singlet(r1=50.0, r2=-50.0, thickness=2.0, epd=20.0)

    with pytest.warns(UserWarning, match="Lens surfaces overlap."):
        fig, ax = optic.draw()
        plt.close(fig)


def test_valid_biconvex_lens_does_not_warn(set_test_backend):
    """Test that a nearby valid biconvex lens does not emit UserWarning."""
    # R1=50, R2=-50, EPD=20 (r=10): thickness=3.0 -> min thickness ~ 0.98 > 0
    optic = _create_singlet(r1=50.0, r2=-50.0, thickness=3.0, epd=20.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, ax = optic.draw()
        plt.close(fig)


def test_overlapping_meniscus_lens_warns(set_test_backend):
    """Test that a steep meniscus lens with overlapping edges emits UserWarning."""
    # R1=30, R2=50, EPD=20: sag1(10) ~ 1.72, sag2(10) ~ 1.01 -> delta sag ~ 0.71 > 0.5
    optic = _create_singlet(r1=30.0, r2=50.0, thickness=0.5, epd=20.0)

    with pytest.warns(UserWarning, match="Lens surfaces overlap."):
        fig, ax = optic.draw()
        plt.close(fig)


def test_valid_meniscus_lens_does_not_warn(set_test_backend):
    """Test that a valid meniscus lens does not emit UserWarning."""
    optic = _create_singlet(r1=30.0, r2=50.0, thickness=1.5, epd=20.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, ax = optic.draw()
        plt.close(fig)


def test_plano_concave_lens_overlap_and_valid(set_test_backend):
    """Test planar front surface with concave rear surface."""
    # Overlapping: R1=inf, R2=-50, t=0.5, EPD=20 (sag2 ~ -1.01 -> t + sag2 ~ -0.51 < 0)
    optic_overlap = _create_singlet(r1=be.inf, r2=-50.0, thickness=0.5, epd=20.0)
    with pytest.warns(UserWarning, match="Lens surfaces overlap."):
        fig, ax = optic_overlap.draw()
        plt.close(fig)

    # Valid: R1=inf, R2=-50, t=1.5, EPD=20
    optic_valid = _create_singlet(r1=be.inf, r2=-50.0, thickness=1.5, epd=20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, ax = optic_valid.draw()
        plt.close(fig)


def test_convex_plano_lens_overlap_and_valid(set_test_backend):
    """Test convex front surface with planar rear surface."""
    # Overlapping: R1=50, R2=inf, t=0.5, EPD=20 (sag1 ~ 1.01 -> t - sag1 ~ -0.51 < 0)
    optic_overlap = _create_singlet(r1=50.0, r2=be.inf, thickness=0.5, epd=20.0)
    with pytest.warns(UserWarning, match="Lens surfaces overlap."):
        fig, ax = optic_overlap.draw()
        plt.close(fig)

    # Valid: R1=50, R2=inf, t=1.5, EPD=20
    optic_valid = _create_singlet(r1=50.0, r2=be.inf, thickness=1.5, epd=20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, ax = optic_valid.draw()
        plt.close(fig)


def test_zero_thickness_singlet_warns(set_test_backend):
    """Test that a lens with zero center thickness emits UserWarning."""
    optic = _create_singlet(r1=50.0, r2=50.0, thickness=0.0, epd=10.0)
    with pytest.warns(UserWarning, match="Lens surfaces overlap."):
        fig, ax = optic.draw()
        plt.close(fig)


def test_aspheric_lens_overlap_and_valid(set_test_backend):
    """Test that an aspheric surface overlap correctly raises UserWarning."""
    optic_overlap = _create_singlet(
        r1=50.0, r2=-50.0, thickness=1.5, epd=20.0, asphere=True
    )
    with pytest.warns(UserWarning, match="Lens surfaces overlap."):
        fig, ax = optic_overlap.draw()
        plt.close(fig)

    optic_valid = _create_singlet(
        r1=50.0, r2=-50.0, thickness=4.0, epd=20.0, asphere=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, ax = optic_valid.draw()
        plt.close(fig)


def test_optic_viewer_3d_overlap_warns(set_test_backend):
    """Test that 3D visualization also triggers the overlap warning."""
    optic = _create_singlet(r1=50.0, r2=-50.0, thickness=2.0, epd=20.0)
    viewer = OpticViewer3D(optic)

    with (
        patch.object(viewer.iren, "Start"),
        patch.object(viewer.ren_win, "Render"),
        pytest.warns(UserWarning, match="Lens surfaces overlap."),
    ):
        viewer.view()


def test_lens2d_and_lens3d_empty_or_single_surface():
    """Test Lens2D and Lens3D with empty or single surface does not error or warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Lens2D([])
        Lens3D([])
        s = MagicMock()
        Lens2D([s])
        Lens3D([s])


def test_to_float_helper():
    """Test _to_float helper with various input types and fallback."""
    from optiland.visualization.system.lens import _to_float

    assert _to_float(3.14) == 3.14
    assert _to_float(42) == 42.0
    assert _to_float(be.array(1.5)) == 1.5
    assert _to_float(be.array([2.5])) == 2.5
    assert _to_float("invalid_string", default=99.0) == 99.0
    assert _to_float(None, default=0.0) == 0.0


def test_lens2d_surfaces_missing_attributes():
    """Test that surfaces without surf or geometry attributes are skipped safely."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Objects without surf attribute
        s1 = object()
        s2 = object()
        Lens2D([s1, s2])

        # Objects with surf but without geometry
        s3 = MagicMock(spec=[])
        s4 = MagicMock(spec=[])
        s3.surf = object()
        s4.surf = object()
        Lens2D([s3, s4])


def test_surfaces_overlap_zero_extent(set_test_backend):
    """Test surface overlap behavior when aperture extent is zero."""
    # Valid lens with epd=0 (zero extent) and thickness > 0
    optic_valid = _create_singlet(r1=50.0, r2=-50.0, thickness=2.0, epd=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, ax = optic_valid.draw()
        plt.close(fig)

    # Overlapping lens with epd=0 and thickness <= 0
    optic_zero_t = _create_singlet(r1=50.0, r2=-50.0, thickness=0.0, epd=0.0)
    with pytest.warns(UserWarning, match="Lens surfaces overlap."):
        fig, ax = optic_zero_t.draw()
        plt.close(fig)


def test_surfaces_overlap_non_finite_fallback(set_test_backend):
    """Test fallback when surface sags return all non-finite (NaN/inf) values."""
    optic = _create_singlet(r1=50.0, r2=-50.0, thickness=2.0, epd=20.0)
    surf1 = optic.surfaces.surfaces[1]
    surf2 = optic.surfaces.surfaces[2]

    lens = Lens2D([])
    with patch.object(surf1.geometry, "sag", return_value=be.array([be.nan])):
        # thickness = 2.0 > 0 -> returns False (no overlap)
        assert not lens._surfaces_overlap(
            MagicMock(surf=surf1, extent=10.0),
            MagicMock(surf=surf2, extent=10.0),
        )

    # thickness <= 0 -> returns True (overlap)
    with (
        patch.object(surf1.geometry, "sag", return_value=be.array([be.nan])),
        patch.object(surf1, "thickness", 0.0),
    ):
        assert lens._surfaces_overlap(
            MagicMock(surf=surf1, extent=10.0),
            MagicMock(surf=surf2, extent=10.0),
        )

