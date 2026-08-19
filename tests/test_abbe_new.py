from __future__ import annotations

import pytest

from optiland.materials import AbbeMaterial, AbbeMaterialE
from optiland.materials.abbe import AbbePolynomialModel, BuchdahlDModel, BuchdahlEModel

from .utils import assert_allclose


class TestAbbePolynomialModel:
    def test_initialization(self, set_test_backend):
        model = AbbePolynomialModel(index=1.5, abbe=50.0)
        assert model.index == 1.5
        assert model.abbe == 50.0
        assert model._p is not None

    def test_predict_n(self, set_test_backend):
        # Known legacy behavior check
        model = AbbePolynomialModel(1.5, 64.17)  # Roughly N-BK7
        n_d = model.predict_n(0.5875618)
        assert_allclose(n_d, 1.5, atol=1e-3)  # Polynomial approximation

    def test_predict_k(self, set_test_backend):
        model = AbbePolynomialModel(1.5, 64.17)
        assert model.predict_k(0.5) == 0


class TestBuchdahlDModel:
    def test_initialization(self, set_test_backend):
        model = BuchdahlDModel(index=1.5, abbe=50.0)
        assert model.index == 1.5
        assert model.abbe == 50.0
        # Coefficients should be calculated
        assert model.v1 is not None
        assert model.v2 is not None
        assert model.v3 is not None
        assert model.WAVE_REF == 0.5875618

    def test_predict_n_at_d_line(self, set_test_backend):
        # By definition, n(lambda_d) should be close to nd
        nd_input = 1.5168
        vd_input = 64.17
        model = BuchdahlDModel(nd_input, vd_input)
        n_pred = model.predict_n(0.5875618)  # lambda_d
        assert_allclose(n_pred, nd_input, atol=1e-6)

    def test_predict_k(self, set_test_backend):
        model = BuchdahlDModel(1.5, 64.17)
        assert model.predict_k(0.5) == 0


class TestBuchdahlEModel:
    def test_initialization(self, set_test_backend):
        # Test values roughly corresponding to N-BK7 e-line
        # N-BK7: nd=1.5168, vd=64.17
        # ne ~ 1.5187, ve ~ 63.96 (approximate conversion for testing init)
        ne = 1.51872
        ve = 63.96
        model = BuchdahlEModel(index=ne, abbe=ve)
        assert model.index == ne
        assert model.abbe == ve
        assert model.WAVE_REF == 0.546074
        assert model.v1 is not None

    def test_predict_n_at_e_line(self, set_test_backend):
        ne = 1.51872
        ve = 63.96
        model = BuchdahlEModel(ne, ve)
        n_pred = model.predict_n(0.546074)  # lambda_e
        assert_allclose(n_pred, ne, atol=1e-6)


class TestAbbeMaterialWrapper:
    def test_default_legacy_model(self, set_test_backend):
        # Should raise warning and use polynomial model
        with pytest.warns(
            FutureWarning, match="default model for AbbeMaterial will change"
        ):
            mat = AbbeMaterial(1.5, 64.17)
        assert isinstance(mat.model, AbbePolynomialModel)
        assert mat.model_name == "polynomial"

    def test_explicit_polynomial_model(self, set_test_backend):
        # Should NOT raise warning
        import warnings

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter(
                "always"
            )  # Cause all warnings to always be triggered.
            mat = AbbeMaterial(1.5, 64.17, model="polynomial")
            # Check that no FutureWarning was raised related to AbbeMaterial
            future_warnings = [
                w
                for w in record
                if issubclass(w.category, FutureWarning)
                and "AbbeMaterial" in str(w.message)
            ]
            assert len(future_warnings) == 0
        assert isinstance(mat.model, AbbePolynomialModel)

    def test_buchdahl_model(self, set_test_backend):
        # Should NOT raise warning
        import warnings

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter(
                "always"
            )  # Cause all warnings to always be triggered.
            mat = AbbeMaterial(1.5, 64.17, model="buchdahl")
            # Check that no FutureWarning was raised related to AbbeMaterial
            future_warnings = [
                w
                for w in record
                if issubclass(w.category, FutureWarning)
                and "AbbeMaterial" in str(w.message)
            ]
            assert len(future_warnings) == 0
        assert isinstance(mat.model, BuchdahlDModel)

    def test_unknown_model(self, set_test_backend):
        with pytest.raises(ValueError, match="Unknown model"):
            AbbeMaterial(1.5, 64.17, model="invalid")

    def test_to_dict_polynomial(self, set_test_backend):
        mat = AbbeMaterial(1.5, 64.17, model="polynomial")
        d = mat.to_dict()
        assert d["type"] == "AbbeMaterial"
        assert d["index"] == 1.5
        assert d["abbe"] == 64.17
        assert d["model"] == "polynomial"

    def test_to_dict_buchdahl(self, set_test_backend):
        mat = AbbeMaterial(1.5, 64.17, model="buchdahl")
        d = mat.to_dict()
        assert d["model"] == "buchdahl"

    def test_from_dict_legacy_data(self, set_test_backend):
        # Legacy data won't have "model" key, should default to polynomial (and warn ideally, but init handles warning)
        data = {"type": "AbbeMaterial", "index": 1.5, "abbe": 64.17}
        with pytest.warns(FutureWarning):
            mat = AbbeMaterial.from_dict(data)
        assert isinstance(mat.model, AbbePolynomialModel)

    def test_from_dict_explicit_model(self, set_test_backend):
        data = {
            "type": "AbbeMaterial",
            "index": 1.5,
            "abbe": 64.17,
            "model": "buchdahl",
        }
        mat = AbbeMaterial.from_dict(data)
        assert isinstance(mat.model, BuchdahlDModel)


class TestAbbeMaterialE:
    def test_initialization(self, set_test_backend):
        ne = 1.51872
        ve = 63.96
        mat = AbbeMaterialE(ne, ve)
        assert isinstance(mat.model, BuchdahlEModel)
        assert mat.index == ne
        assert mat.abbe == ve

    def test_prediction(self, set_test_backend):
        ne = 1.51872
        ve = 63.96
        mat = AbbeMaterialE(ne, ve)
        n_pred = mat.n(0.546074)
        assert_allclose(n_pred, ne, atol=1e-6)

    def test_to_dict(self, set_test_backend):
        mat = AbbeMaterialE(1.5, 60.0)
        d = mat.to_dict()
        assert d["type"] == "AbbeMaterialE"
        assert d["index"] == 1.5
        assert d["abbe"] == 60.0

    def test_from_dict(self, set_test_backend):
        data = {"type": "AbbeMaterialE", "index": 1.5, "abbe": 60.0}
        mat = AbbeMaterialE.from_dict(data)
        assert mat.index == 1.5
        assert mat.abbe == 60.0
        assert isinstance(mat.model, BuchdahlEModel)
