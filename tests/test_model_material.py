from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import ModelMaterial
from optiland.materials.model import ModelMaterial as ModelMaterialFromModule

from .utils import assert_allclose


LAMBDA_D = 0.5875618
LAMBDA_F = 0.4861327
LAMBDA_C = 0.6562725
LAMBDA_G = 0.4358343


def _assert_visible_anchors(material: ModelMaterial, nd: float, vd: float, dpgf: float):
    """Assert the exact visible anchors enforced by the coefficient solve."""
    n_d = material.n(LAMBDA_D)
    n_f = material.n(LAMBDA_F)
    n_c = material.n(LAMBDA_C)
    n_g = material.n(LAMBDA_G)

    dn_fc = (nd - 1.0) / vd
    pgf = (n_g - n_f) / (n_f - n_c)
    pgf_expected = 0.6438 - 0.001682 * vd + dpgf

    assert_allclose(n_d, nd, atol=1e-9)
    assert_allclose(n_f - n_c, dn_fc, atol=1e-9)
    assert_allclose(pgf, pgf_expected, atol=1e-9)


class TestModelMaterial:
    def test_import_export(self):
        assert ModelMaterial is ModelMaterialFromModule

    def test_visible_route_anchors(self, set_test_backend):
        nd = 1.5168
        vd = 64.17
        dpgf = 0.0
        material = ModelMaterial(nd=nd, vd=vd, dPgF=dpgf)

        _assert_visible_anchors(material, nd, vd, dpgf)

    def test_ir_route_anchors(self, set_test_backend):
        nd = 1.5168
        vd = 64.17
        dpgf = 0.0
        lambda_ref = 1.55

        visible_material = ModelMaterial(nd=nd, vd=vd, dPgF=dpgf)
        p_ref = visible_material.P_ref(lambda_ref)
        material = ModelMaterial(
            nd=nd,
            vd=vd,
            dPgF=dpgf,
            P_ref=float(be.to_numpy(p_ref)),
            lambda_ref=lambda_ref,
        )

        _assert_visible_anchors(material, nd, vd, dpgf)
        assert_allclose(material.P_ref(lambda_ref), p_ref, atol=1e-9)

    def test_array_shape_and_extinction(self, set_test_backend):
        material = ModelMaterial(nd=1.5168, vd=64.17, dPgF=0.0)
        wavelengths = be.array([LAMBDA_F, LAMBDA_D, LAMBDA_C])

        n = material.n(wavelengths)
        k = material.k(wavelengths)

        n_np = be.to_numpy(n)
        assert n_np.shape == (3,)
        assert np.all(np.isfinite(n_np))
        assert be.to_numpy(k).shape == (3,)
        assert_allclose(k, be.zeros_like(wavelengths))

    @pytest.mark.parametrize("vd", [9.999, 100.001])
    def test_invalid_abbe_number_raises(self, vd, set_test_backend):
        with pytest.raises(ValueError, match="Abbe number"):
            ModelMaterial(nd=1.5168, vd=vd, dPgF=0.0)

    @pytest.mark.parametrize("dpgf", [-1.001, 1.001])
    def test_invalid_partial_dispersion_deviation_raises(
        self, dpgf, set_test_backend
    ):
        with pytest.raises(ValueError, match="dPgF"):
            ModelMaterial(nd=1.5168, vd=64.17, dPgF=dpgf)

    @pytest.mark.parametrize("wavelength", [0.3649, 2.3001])
    def test_wavelength_out_of_range_raises(self, wavelength, set_test_backend):
        material = ModelMaterial(nd=1.5168, vd=64.17, dPgF=0.0)

        with pytest.raises(ValueError, match="Wavelength out of range"):
            material.n(wavelength)

    def test_to_dict_from_dict_round_trip(self, set_test_backend):
        material = ModelMaterial(
            nd=1.5168,
            vd=64.17,
            dPgF=0.002,
            P_ref=0.3,
            lambda_ref=1.55,
        )

        data = material.to_dict()
        restored = ModelMaterial.from_dict(data)

        assert data["type"] == "ModelMaterial"
        assert restored.nd == material.nd
        assert restored.vd == material.vd
        assert restored.dPgf == material.dPgf
        assert restored.p_ref == material.p_ref
        assert restored.lambda_ref == material.lambda_ref
