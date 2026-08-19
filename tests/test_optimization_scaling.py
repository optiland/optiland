from __future__ import annotations

import optiland.backend as be
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.scaling.log import LogScaler
from optiland.optimization.scaling.power import PowerScaler
from optiland.optimization.scaling.reciprocal import ReciprocalScaler

from .utils import assert_allclose


class TestIdentityScaler:
    def test_scale(self, set_test_backend):
        scaler = IdentityScaler()
        assert_allclose(scaler.scale(10.0), 10.0)

    def test_inverse_scale(self, set_test_backend):
        scaler = IdentityScaler()
        assert_allclose(scaler.inverse_scale(10.0), 10.0)


class TestLinearScaler:
    def test_scale(self, set_test_backend):
        scaler = LinearScaler(factor=2.0, offset=1.0)
        assert_allclose(scaler.scale(10.0), 21.0)

    def test_inverse_scale(self, set_test_backend):
        scaler = LinearScaler(factor=2.0, offset=1.0)
        assert_allclose(scaler.inverse_scale(21.0), 10.0)


class TestReciprocalScaler:
    def test_scale(self, set_test_backend):
        scaler = ReciprocalScaler()
        assert_allclose(scaler.scale(2.0), 0.5)

    def test_inverse_scale(self, set_test_backend):
        scaler = ReciprocalScaler()
        assert_allclose(scaler.inverse_scale(0.5), 2.0)

    def test_scale_zero(self, set_test_backend):
        scaler = ReciprocalScaler()
        assert scaler.scale(0.0) == be.inf

    def test_inverse_scale_zero(self, set_test_backend):
        scaler = ReciprocalScaler()
        assert scaler.inverse_scale(0.0) == be.inf


class TestPowerScaler:
    def test_scale(self, set_test_backend):
        scaler = PowerScaler(power=2.0)
        assert_allclose(scaler.scale(10.0), 100.0)

    def test_inverse_scale(self, set_test_backend):
        scaler = PowerScaler(power=2.0)
        assert_allclose(scaler.inverse_scale(100.0), 10.0)


class TestLogScaler:
    def test_scale(self, set_test_backend):
        scaler = LogScaler(base=10.0)
        assert_allclose(scaler.scale(100.0), 2.0)

    def test_inverse_scale(self, set_test_backend):
        scaler = LogScaler(base=10.0)
        assert_allclose(scaler.inverse_scale(2.0), 100.0)
