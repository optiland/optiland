from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from unittest.mock import MagicMock

import matplotlib
import numpy as np
from matplotlib.patches import Polygon

matplotlib.use("Agg")

sys.modules.setdefault("vtk", MagicMock())

import matplotlib.pyplot as plt
from optiland import optic
from optiland.materials import IdealMaterial
from optiland.physical_apertures import RadialAperture


def _build_annular_lens(*, single_lens_group: bool = False) -> optic.Optic:
    lens = optic.Optic()

    lens.surfaces.add(index=0, radius=np.inf, thickness=5)
    lens.surfaces.add(
        index=1,
        radius=7.72,
        thickness=0.55,
        material=IdealMaterial(n=1.369),
        aperture=RadialAperture(r_max=5.5),
        is_stop=True,
    )
    lens.surfaces.add(
        index=2,
        radius=6.5,
        thickness=0.25,
        material=IdealMaterial(n=1.332 if single_lens_group else 1.0),
        aperture=RadialAperture(r_max=5.5),
    )
    lens.surfaces.add(
        index=3,
        radius=11.5,
        thickness=0.01,
        material=IdealMaterial(n=1.38),
        aperture=RadialAperture(r_max=11.0, r_min=5.0),
    )
    lens.surfaces.add(
        index=4,
        radius=11.5,
        thickness=10,
        material=IdealMaterial(n=1.332 if single_lens_group else 1.0),
        aperture=RadialAperture(r_max=11.0, r_min=5.0),
    )
    lens.surfaces.add(index=5)

    lens.set_aperture(aperture_type="EPD", value=3.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def test_annular_lens_polygons_do_not_include_nan_vertices(set_test_backend):
    lens = _build_annular_lens()

    fig, ax = lens.draw()
    lens_patches = [patch for patch in ax.patches if isinstance(patch, Polygon)]

    assert len(lens_patches) >= 3
    for patch in lens_patches:
        assert np.isfinite(patch.get_xy()).all()
    plt.close(fig)


def test_annular_gap_is_split_into_separate_lens_patches(set_test_backend):
    lens = _build_annular_lens(single_lens_group=True)

    fig, ax = lens.draw()
    lens_patches = [patch for patch in ax.patches if isinstance(patch, Polygon)]

    assert len(lens_patches) > 3
    for patch in lens_patches:
        assert np.isfinite(patch.get_xy()).all()
    plt.close(fig)
