"""Regression tests for Forbes coefficient caching / cache invalidation.

The Forbes Q / Q2D geometries cache prepared (trimmed) coefficient
containers and rebuild them only when invalidated -- on construction, on
``scale``, and on optimizer coefficient updates -- rather than on every
``sag`` / ``_surface_normal`` call (issue #617 performance work). These
tests pin the *correctness* of that cache: a coefficient change must be
reflected in the very next evaluation, and the cached path must match a
freshly constructed surface. Autograd through both coefficient tensors and
``(x, y)`` must also survive the cached, ``_no_trim`` hot path.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    ForbesQ2dGeometry,
    ForbesQNormalSlopeGeometry,
    ForbesSurfaceConfig,
)
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable import (
    ForbesQ2dCoeffVariable,
    ForbesQNormalSlopeCoeffVariable,
)
from optiland.samples.simple import AsphericSinglet

from .utils import assert_allclose


def _abs_diff(a, b):
    """Backend- and autograd-safe ``|a - b|`` as a Python float."""
    return float(be.to_numpy(be.abs(a - b)))


def _single_surface_optic(geometry):
    """Optic exposing ``geometry`` on surface index 1.

    The coefficient variables resolve the geometry through
    ``optic.surfaces[surface_number].geometry``; replacing the geometry on a
    ready-made singlet satisfies that lookup without rebuilding an optic.
    """
    optic = AsphericSinglet()
    optic.surfaces[1].geometry = geometry
    return optic


def _qbfs_geom():
    cfg = ForbesSurfaceConfig(
        radius=25.0,
        conic=-0.5,
        norm_radius=8.0,
        terms={0: 1.0e-3, 1: -5.0e-4, 2: 2.0e-4, 3: -8.0e-5},
    )
    return ForbesQNormalSlopeGeometry(CoordinateSystem(), cfg)


def _q2d_geom():
    cfg = ForbesSurfaceConfig(
        radius=25.0,
        conic=-0.5,
        norm_radius=8.0,
        terms={
            ("a", 0, 0): 5.0e-4,
            ("a", 0, 1): -2.0e-4,
            ("a", 1, 0): 1.0e-4,
            ("b", 1, 0): 8.0e-5,
        },
    )
    return ForbesQ2dGeometry(CoordinateSystem(), cfg)


# ---------------------------------------------------------------------------
# Direct in-place dict edits must invalidate the cache (no variable system).
# ---------------------------------------------------------------------------


class TestDirectDictEditInvalidation:
    """A raw ``geom.radial_terms[k] = v`` poke (bypassing the variable system
    and ``scale``) must still be reflected by the next evaluation, matching the
    pre-cache behavior. Guarded by the ``_CoeffCacheDict`` wrapper."""

    def test_qbfs_direct_radial_terms_edit(self, set_test_backend):
        geom = _qbfs_geom()
        x, y = be.array(3.0), be.array(2.0)
        before = be.copy(geom.sag(x, y))  # primes the cache (dirty -> clean)

        geom.radial_terms[1] = be.array(0.05)  # raw in-place edit
        after = geom.sag(x, y)

        assert _abs_diff(after, before) > 1e-9
        fresh_cfg = ForbesSurfaceConfig(
            radius=25.0,
            conic=-0.5,
            norm_radius=8.0,
            terms={0: 1.0e-3, 1: 0.05, 2: 2.0e-4, 3: -8.0e-5},
        )
        fresh = ForbesQNormalSlopeGeometry(CoordinateSystem(), fresh_cfg)
        assert_allclose(after, fresh.sag(x, y))

    def test_q2d_direct_freeform_coeffs_edit(self, set_test_backend):
        geom = _q2d_geom()
        x, y = be.array(3.0), be.array(2.0)
        before = be.copy(geom.sag(x, y))

        geom.freeform_coeffs[("a", 1, 0)] = be.array(0.03)  # raw in-place edit
        after = geom.sag(x, y)

        assert _abs_diff(after, before) > 1e-9
        new_terms = {
            ("a", 0, 0): 5.0e-4,
            ("a", 0, 1): -2.0e-4,
            ("a", 1, 0): 0.03,
            ("b", 1, 0): 8.0e-5,
        }
        fresh = ForbesQ2dGeometry(
            CoordinateSystem(),
            ForbesSurfaceConfig(
                radius=25.0, conic=-0.5, norm_radius=8.0, terms=new_terms
            ),
        )
        assert_allclose(after, fresh.sag(x, y))


# ---------------------------------------------------------------------------
# Cache invalidation: a coefficient change must be reflected immediately.
# ---------------------------------------------------------------------------


class TestQbfsCacheInvalidation:
    def test_repeated_sag_is_stable(self, set_test_backend):
        """Two evaluations without a change must agree (cache not corrupted)."""
        geom = _qbfs_geom()
        x, y = be.array(3.0), be.array(2.0)
        first = geom.sag(x, y)
        second = geom.sag(x, y)
        assert_allclose(first, second)

    def test_update_value_changes_sag_and_matches_fresh(self, set_test_backend):
        geom = _qbfs_geom()
        optic = _single_surface_optic(geom)
        x, y = be.array(3.0), be.array(2.0)

        before = be.copy(geom.sag(x, y))

        var = ForbesQNormalSlopeCoeffVariable(optic, 1, 1, scaler=IdentityScaler())
        var.update_value(0.05)
        after = geom.sag(x, y)

        # The cached prepared coefficients must have been invalidated.
        assert _abs_diff(after, before) > 1e-9

        # ... and the result must match a fresh surface built with the new
        # coefficient (proves the rebuild is correct, not just different).
        fresh_cfg = ForbesSurfaceConfig(
            radius=25.0,
            conic=-0.5,
            norm_radius=8.0,
            terms={0: 1.0e-3, 1: 0.05, 2: 2.0e-4, 3: -8.0e-5},
        )
        fresh = ForbesQNormalSlopeGeometry(CoordinateSystem(), fresh_cfg)
        assert_allclose(after, fresh.sag(x, y))

    def test_update_value_changes_surface_normal(self, set_test_backend):
        geom = _qbfs_geom()
        optic = _single_surface_optic(geom)
        x, y = be.array(3.0), be.array(2.0)

        nx0, ny0, nz0 = (be.copy(c) for c in geom._surface_normal(x, y))

        var = ForbesQNormalSlopeCoeffVariable(optic, 1, 2, scaler=IdentityScaler())
        var.update_value(0.02)
        nx1, ny1, nz1 = geom._surface_normal(x, y)

        assert _abs_diff(nx1, nx0) > 1e-9

        fresh_cfg = ForbesSurfaceConfig(
            radius=25.0,
            conic=-0.5,
            norm_radius=8.0,
            terms={0: 1.0e-3, 1: -5.0e-4, 2: 0.02, 3: -8.0e-5},
        )
        fresh = ForbesQNormalSlopeGeometry(CoordinateSystem(), fresh_cfg)
        fnx, fny, fnz = fresh._surface_normal(x, y)
        assert_allclose(nx1, fnx)
        assert_allclose(ny1, fny)
        assert_allclose(nz1, fnz)


class TestQ2dCacheInvalidation:
    @pytest.mark.parametrize("key", [("a", 1, 0), ("b", 1, 0)])
    def test_update_value_changes_sag_and_matches_fresh(self, set_test_backend, key):
        geom = _q2d_geom()
        optic = _single_surface_optic(geom)
        x, y = be.array(3.0), be.array(2.0)

        before = be.copy(geom.sag(x, y))

        var = ForbesQ2dCoeffVariable(optic, 1, key, scaler=IdentityScaler())
        var.update_value(0.03)
        after = geom.sag(x, y)

        assert _abs_diff(after, before) > 1e-9

        new_terms = {
            ("a", 0, 0): 5.0e-4,
            ("a", 0, 1): -2.0e-4,
            ("a", 1, 0): 1.0e-4,
            ("b", 1, 0): 8.0e-5,
        }
        new_terms[key] = 0.03
        fresh_cfg = ForbesSurfaceConfig(
            radius=25.0, conic=-0.5, norm_radius=8.0, terms=new_terms
        )
        fresh = ForbesQ2dGeometry(CoordinateSystem(), fresh_cfg)
        assert_allclose(after, fresh.sag(x, y))

    def test_update_value_changes_surface_normal(self, set_test_backend):
        geom = _q2d_geom()
        optic = _single_surface_optic(geom)
        x, y = be.array(3.0), be.array(2.0)

        nx0 = be.copy(geom._surface_normal(x, y)[0])

        var = ForbesQ2dCoeffVariable(optic, 1, ("a", 1, 0), scaler=IdentityScaler())
        var.update_value(0.03)
        nx1 = geom._surface_normal(x, y)[0]
        assert _abs_diff(nx1, nx0) > 1e-9


# ---------------------------------------------------------------------------
# scale() must invalidate and produce the correctly scaled behavior.
# ---------------------------------------------------------------------------


class TestScaleInvalidation:
    def test_qbfs_scale_changes_sag(self, set_test_backend):
        geom = _qbfs_geom()
        x, y = be.array(3.0), be.array(2.0)
        before = be.copy(geom.sag(x, y))  # primes the cache

        geom.scale(2.0)
        after = geom.sag(be.array(6.0), be.array(4.0))

        # A geometry scaled by k satisfies z_scaled(k*r) = k * z(r).
        assert_allclose(after, 2.0 * before, rtol=1e-6, atol=1e-9)

    def test_q2d_scale_changes_sag(self, set_test_backend):
        geom = _q2d_geom()
        x, y = be.array(3.0), be.array(2.0)
        before = be.copy(geom.sag(x, y))

        geom.scale(2.0)
        after = geom.sag(be.array(6.0), be.array(4.0))
        assert_allclose(after, 2.0 * before, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Autograd must survive the cached, _no_trim hot path.
# ---------------------------------------------------------------------------


class TestAutogradThroughCache:
    def test_qbfs_grad_through_coeffs_and_xy(self, set_test_backend):
        if be.get_backend() != "torch":
            pytest.skip("autograd path is torch-only")
        import torch

        c1 = torch.tensor(-5.0e-4, dtype=torch.float64, requires_grad=True)
        cfg = ForbesSurfaceConfig(
            radius=25.0, conic=-0.5, norm_radius=8.0, terms={0: 1.0e-3, 2: 2.0e-4}
        )
        geom = ForbesQNormalSlopeGeometry(CoordinateSystem(), cfg)
        # Inject a grad-requiring coefficient and invalidate the cache so the
        # next evaluation rebuilds the prepared containers from it.
        geom.radial_terms[1] = c1
        geom._coeffs_dirty = True

        x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        z = geom.sag(x, y)
        z.backward()

        for g in (c1.grad, x.grad, y.grad):
            assert g is not None
            assert torch.isfinite(g).all()

    def test_q2d_grad_through_coeffs_and_xy(self, set_test_backend):
        if be.get_backend() != "torch":
            pytest.skip("autograd path is torch-only")
        import torch

        a10 = torch.tensor(1.0e-4, dtype=torch.float64, requires_grad=True)
        cfg = ForbesSurfaceConfig(
            radius=25.0,
            conic=-0.5,
            norm_radius=8.0,
            terms={("a", 0, 0): 5.0e-4, ("a", 1, 0): 2.0e-4, ("b", 1, 0): 8.0e-5},
        )
        geom = ForbesQ2dGeometry(CoordinateSystem(), cfg)
        geom.freeform_coeffs[("a", 1, 0)] = a10
        geom._coeffs_dirty = True

        x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        # Sum sag and the Cartesian normal so both hot paths are exercised.
        z = geom.sag(x, y)
        dfdx, dfdy = geom._surface_normal_analytical_cartesian(x, y)
        (z + dfdx + dfdy).backward()

        for g in (a10.grad, x.grad, y.grad):
            assert g is not None
            assert torch.isfinite(g).all()
