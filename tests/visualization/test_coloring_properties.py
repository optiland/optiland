"""Tests for the glass-property value functions used by PropertyColorizer."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import AbbeMaterialE, IdealMaterial
from optiland.optic import Optic
from optiland.samples.simple import CementedAchromat
from optiland.samples.telescopes import HubbleTelescope
from optiland.visualization.coloring import (
    ColorScale,
    ComponentRef,
    PropertyColorizer,
    abbe_number,
    refractive_index,
)

matplotlib.use("Agg")

VIRIDIS = matplotlib.colormaps["viridis"]

# CementedAchromat: N-BAK1 crown on surfaces (1, 2), SF2 flint on (2, 3).
CROWN = ComponentRef("lens_element", (1, 2))
FLINT = ComponentRef("lens_element", (2, 3))
# The single element of _singlet().
LENS = ComponentRef("lens_element", (1, 2))

# Schott catalog values.
N_BAK1_ND, N_BAK1_VD = 1.57250, 57.55
SF2_ND, SF2_VD = 1.64769, 33.82


def _singlet(material) -> Optic:
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=20.0, thickness=3.0, is_stop=True, material=material
    )
    optic.surfaces.add(index=2, radius=-20.0, thickness=30.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=5.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=0.5875618, is_primary=True)
    return optic


class TestAbbeNumber:
    def test_matches_catalog_for_achromat(self, set_test_backend):
        optic = CementedAchromat()
        assert abbe_number(optic, CROWN) == pytest.approx(N_BAK1_VD, abs=0.05)
        assert abbe_number(optic, FLINT) == pytest.approx(SF2_VD, abs=0.05)

    def test_returns_python_float(self, set_test_backend):
        assert type(abbe_number(CementedAchromat(), CROWN)) is float

    def test_crown_gets_high_end_of_scale(self, set_test_backend):
        optic = CementedAchromat()
        colorizer = PropertyColorizer(abbe_number)
        colorizer.prepare(optic, [CROWN, FLINT])
        np.testing.assert_allclose(colorizer.color_for(CROWN), VIRIDIS(1.0))
        np.testing.assert_allclose(colorizer.color_for(FLINT), VIRIDIS(0.0))
        assert colorizer.scale.vmin == pytest.approx(SF2_VD, abs=0.05)
        assert colorizer.scale.vmax == pytest.approx(N_BAK1_VD, abs=0.05)

    def test_draw_colors_elements_and_adds_colorbar(self, set_test_backend):
        colorizer = PropertyColorizer(
            abbe_number, ColorScale(cmap="coolwarm_r", label="Abbe number")
        )
        fig, ax = CementedAchromat().draw(
            show=False, show_apertures=False, component_colorizer=colorizer
        )
        coolwarm_r = matplotlib.colormaps["coolwarm_r"]
        faces = [p.get_facecolor() for p in ax.patches]
        np.testing.assert_allclose(faces, [coolwarm_r(1.0), coolwarm_r(0.0)])
        assert fig.axes[1].get_ylabel() == "Abbe number"
        plt.close(fig)

    def test_abbe_e_material_reports_vd_not_stored_ve(self, set_test_backend):
        material = AbbeMaterialE(1.5749, 57.27)
        n_d, n_f, n_c = (
            float(be.to_numpy(material.n(wl)).item())
            for wl in (0.5875618, 0.4861327, 0.6562725)
        )
        value = abbe_number(_singlet(material), LENS)
        assert value == pytest.approx((n_d - 1) / (n_f - n_c))
        assert value != pytest.approx(57.27)

    def test_non_dispersive_material_returns_none(self, set_test_backend):
        assert abbe_number(_singlet(IdealMaterial(1.5)), LENS) is None

    def test_mirrors_return_none(self, set_test_backend):
        optic = HubbleTelescope()
        mirrors = [
            ComponentRef("mirror", (k,))
            for k, s in enumerate(optic.surfaces)
            if s.interaction_model.is_reflective
        ]
        assert mirrors
        assert all(abbe_number(optic, ref) is None for ref in mirrors)

    def test_standalone_surfaces_return_none(self, set_test_backend):
        optic = CementedAchromat()
        assert abbe_number(optic, ComponentRef("surface", (0,))) is None
        assert abbe_number(optic, ComponentRef("surface", (4,))) is None


class TestRefractiveIndex:
    def test_d_line_matches_catalog(self, set_test_backend):
        optic = CementedAchromat()
        n_d = refractive_index(0.5876)
        assert n_d(optic, CROWN) == pytest.approx(N_BAK1_ND, abs=1e-4)
        assert n_d(optic, FLINT) == pytest.approx(SF2_ND, abs=1e-4)

    def test_default_is_primary_wavelength(self, set_test_backend):
        optic = CementedAchromat()
        explicit = refractive_index(optic.primary_wavelength)
        default = refractive_index()
        for ref in (CROWN, FLINT):
            assert default(optic, ref) == pytest.approx(explicit(optic, ref))

    def test_follows_wavelength(self, set_test_backend):
        optic = CementedAchromat()
        n_f = refractive_index(0.4861327)(optic, CROWN)
        n_c = refractive_index(0.6562725)(optic, CROWN)
        assert n_f > N_BAK1_ND > n_c

    def test_returns_python_float(self, set_test_backend):
        assert type(refractive_index()(CementedAchromat(), CROWN)) is float

    def test_ideal_material_returns_its_index(self, set_test_backend):
        optic = _singlet(IdealMaterial(1.5))
        assert refractive_index()(optic, LENS) == pytest.approx(1.5)

    def test_mirrors_and_surfaces_return_none(self, set_test_backend):
        optic = HubbleTelescope()
        value_fn = refractive_index()
        refs = [ComponentRef("surface", (0,))] + [
            ComponentRef("mirror", (k,))
            for k, s in enumerate(optic.surfaces)
            if s.interaction_model.is_reflective
        ]
        assert all(value_fn(optic, ref) is None for ref in refs)
