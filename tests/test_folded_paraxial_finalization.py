"""Finalization tests for the folded-paraxial merge gates (PR #729).

Covers the merge-hardening workstreams that close the remaining integration
defects:

- the ray-transfer-matrix API shares the validated, orientation-aware scalar
  sequence with the explicit paraxial tracer (matrix and trace can never
  disagree on the same prescription);
- physically equivalent centered surfaces authored with the local axis
  reversed are normalized to the same effective scalar power on straight
  paths, not only folded ones;
- z-bound geometry mutations reject folded/off-axis prescriptions before
  touching any state (preflight atomicity).

Every folded assertion compares against an exact unfolding reference (the
classic z-authored retro/trombone chain) or an equivalent-authoring pair,
never a hard-coded number.
"""

from __future__ import annotations

import math
import warnings

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.paraxial_path import (
    ParaxialDomainWarning,
    UnsupportedParaxialGeometryError,
)
from optiland.rays import RealRays

from .test_folded_paraxial import _finish, folded, retro, straight
from .test_folded_paraxial_hardening import (
    fold60,
    folded_powered_lens,
    folded_powered_mirror,
    folded_tilted_lens,
    oblique_powered_mirror,
    periscope_out_of_plane,
    trombone_lens,
    trombone_mirror,
)
from .utils import assert_allclose

# Float64 acceptance for scalar trace/matrix equivalence (spec section 14).
RTOL = 1e-10
ATOL = 1e-10


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def straight_reversed_lens():
    """The ``straight`` singlet with surface 1 authored on a reversed axis.

    Physically identical glass: the front face is the same sphere, described
    in a local frame rotated by pi about x, so the authored radius flips
    sign. The scalar model must normalize it to the canonical effective
    power.
    """
    optic = Optic(name="straight-reversed")
    optic.surfaces.add(index=0, x=0.0, y=0.0, z=-be.inf, radius=be.inf)
    optic.surfaces.add(
        index=1,
        x=0.0,
        y=0.0,
        z=0.0,
        rx=math.pi,
        radius=-25.84,
        material="N-BK7",
        is_stop=True,
    )
    optic.surfaces.add(index=2, x=0.0, y=0.0, z=4.0, radius=be.inf)
    optic.surfaces.add(index=3, x=0.0, y=0.0, z=50.0)
    return _finish(optic)


def straight_canonical_mirror():
    """Concave f=40 mirror at normal incidence on a straight retro chain."""
    optic = Optic(name="straight-mirror-canonical")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True, thickness=24.0)
    optic.surfaces.add(index=2, radius=-80.0, material="mirror", thickness=-40.0)
    optic.surfaces.add(index=3)
    return _finish(optic)


def straight_reversed_mirror():
    """The same retro chain with the powered mirror on a reversed local axis."""
    optic = Optic(name="straight-mirror-reversed")
    optic.surfaces.add(index=0, x=0.0, y=0.0, z=-be.inf, radius=be.inf)
    optic.surfaces.add(index=1, x=0.0, y=0.0, z=0.0, radius=be.inf, is_stop=True)
    optic.surfaces.add(
        index=2,
        x=0.0,
        y=0.0,
        z=24.0,
        rx=math.pi,
        radius=80.0,
        material="mirror",
    )
    optic.surfaces.add(index=3, x=0.0, y=0.0, z=-16.0, rx=math.pi)
    return _finish(optic)


def straight_paraxial_lens(authoring="canonical", f0=60.0):
    """Ideal thin-lens singlet, canonically or reversed-axis authored.

    The reversed authoring describes the same physical element in a local
    frame rotated by pi about x, so the authored focal length flips sign
    exactly as a geometric radius does.
    """
    optic = Optic(name=f"straight-paraxial-{authoring}")
    optic.surfaces.add(index=0, x=0.0, y=0.0, z=-be.inf, radius=be.inf)
    if authoring == "canonical":
        optic.surfaces.add(
            index=1,
            x=0.0,
            y=0.0,
            z=0.0,
            surface_type="paraxial",
            f=f0,
            is_stop=True,
        )
    else:
        optic.surfaces.add(
            index=1,
            x=0.0,
            y=0.0,
            z=0.0,
            rx=math.pi,
            surface_type="paraxial",
            f=-f0,
            is_stop=True,
        )
    optic.surfaces.add(index=2, x=0.0, y=0.0, z=f0)
    return _finish(optic)


def trombone_paraxial_lens(f0=40.0):
    """Classic z-authored retro chain with an ideal lens on the return leg.

    In the reduced (unfolded) representation, spacings after the mirror are
    negative and a converging element on the return leg carries a negative
    reduced focal length -- the same sign bookkeeping as a geometric radius.
    """
    optic = Optic(name="trombone-paraxial")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True, thickness=24.0)
    optic.surfaces.add(index=2, radius=be.inf, material="mirror", thickness=-20.0)
    optic.surfaces.add(index=3, surface_type="paraxial", f=-f0, thickness=-30.0)
    optic.surfaces.add(index=4)
    return _finish(optic)


def folded_paraxial_lens(authoring, f0=40.0):
    """90-degree fold into +y with an ideal thin lens on the folded arm."""
    optic = Optic(name=f"folded-paraxial-{authoring}")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, is_stop=True)
    optic.surfaces.add(index=2, x=0.0, y=0.0, z=24.0, rx=math.pi / 4, material="mirror")
    if authoring == "along":
        # Local +z along the incoming +y beam; converging lens has f = +f0
        # in its own frame.
        optic.surfaces.add(
            index=3,
            x=0.0,
            y=20.0,
            z=24.0,
            rx=-math.pi / 2,
            surface_type="paraxial",
            f=f0,
        )
    else:
        # Local +z against the beam: the same element authored in the
        # flipped frame carries the opposite focal length.
        optic.surfaces.add(
            index=3,
            x=0.0,
            y=20.0,
            z=24.0,
            rx=math.pi / 2,
            surface_type="paraxial",
            f=-f0,
        )
    optic.surfaces.add(index=4, x=0.0, y=50.0, z=24.0, rx=math.pi / 2)
    return _finish(optic)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def basis_ray_matrix(optic, start, end, wavelength=None):
    """Reference ABCD matrix from two explicit basis-ray traces.

    Launches (y=1, u=0) and (y=0, u=1) immediately before surface ``start``
    and reads the state immediately after surface ``end`` -- the same
    boundary convention ``ray_transfer_matrix`` documents.
    """
    if wavelength is None:
        wavelength = optic.primary_wavelength
    path = optic.surfaces.build_paraxial_path()
    pos = be.ravel(path.axial_positions)
    z0 = pos[start]
    ya, ua = optic.paraxial.trace_generic(
        1.0, 0.0, z0, wavelength, skip=start, path=path
    )
    yb, ub = optic.paraxial.trace_generic(
        0.0, 1.0, z0, wavelength, skip=start, path=path
    )
    i = end - start

    def scalar(value):
        return float(be.to_numpy(value).reshape(-1)[0])

    return (
        (scalar(ya[i]), scalar(yb[i])),
        (scalar(ua[i]), scalar(ub[i])),
    )


def matrix_to_rows(matrix):
    m = be.to_numpy(matrix)
    return ((float(m[0, 0]), float(m[0, 1])), (float(m[1, 0]), float(m[1, 1])))


def assert_matrix_close(actual, reference, rtol=RTOL, atol=ATOL):
    for row_a, row_r in zip(actual, reference, strict=True):
        for a, r in zip(row_a, row_r, strict=True):
            assert_allclose(a, r, rtol=rtol, atol=atol)


def trace_probe_rays(optic, offsets=(0.0, 1.0, 2.5)):
    """Real marginal-ish rays through the system; returns final states."""
    n = len(offsets)
    rays = RealRays(
        x=be.zeros(n),
        y=be.array(list(offsets)),
        z=be.full((n,), -10.0),
        L=be.zeros(n),
        M=be.zeros(n),
        N=be.ones(n),
        intensity=be.ones(n),
        wavelength=be.ones(n) * 0.55,
    )
    optic.surfaces.trace(rays)
    return tuple(
        be.to_numpy(component).copy()
        for component in (rays.x, rays.y, rays.z, rays.L, rays.M, rays.N)
    )


# ---------------------------------------------------------------------------
# Workstream A: matrix and explicit trace share one scalar sequence
# ---------------------------------------------------------------------------


class TestMatrixTraceUnification:
    @pytest.mark.parametrize(
        "builder", [straight, retro, folded, fold60, periscope_out_of_plane]
    )
    def test_full_system_matrix_matches_basis_rays(self, set_test_backend, builder):
        optic = builder()
        last = optic.surfaces.num_surfaces - 1
        matrix = matrix_to_rows(optic.paraxial.ray_transfer_matrix(1, last))
        reference = basis_ray_matrix(optic, 1, last)
        assert_matrix_close(matrix, reference)

    @pytest.mark.parametrize("builder", [trombone_lens, folded_powered_lens])
    def test_interior_range_matrix_matches_basis_rays(self, set_test_backend, builder):
        optic = builder("along") if builder is folded_powered_lens else builder()
        matrix = matrix_to_rows(optic.paraxial.ray_transfer_matrix(2, 4))
        reference = basis_ray_matrix(optic, 2, 4)
        assert_matrix_close(matrix, reference)

    @pytest.mark.parametrize("builder", [straight, folded])
    def test_matrix_composition(self, set_test_backend, builder):
        """M(a, c) == M(b+1, c) @ T(b -> b+1) @ M(a, b).

        With inclusive surface ranges, composing two sub-ranges requires the
        propagation between surface b and surface b+1 to be inserted
        explicitly -- that is the boundary convention the matrix documents
        (no propagation before the first or after the last surface of a
        range).
        """
        optic = builder()
        wl = optic.primary_wavelength
        last = optic.surfaces.num_surfaces - 1
        a, b, c = 1, 2, last
        path = optic.surfaces.build_paraxial_path()
        pos = be.ravel(path.axial_positions)

        m_ab = optic.paraxial.ray_transfer_matrix(a, b, wl, path=path)
        m_bc = optic.paraxial.ray_transfer_matrix(b + 1, c, wl, path=path)
        t = pos[b + 1] - pos[b]
        transfer = be.stack(
            [
                be.stack([be.array(1.0), t]),
                be.stack([be.array(0.0), be.array(1.0)]),
            ]
        )
        composed = be.matmul(m_bc, be.matmul(transfer, m_ab))
        full = optic.paraxial.ray_transfer_matrix(a, c, wl, path=path)
        assert_matrix_close(matrix_to_rows(composed), matrix_to_rows(full))

    @pytest.mark.parametrize("builder", [straight, folded, retro])
    def test_whole_system_matrix_gives_f2(self, set_test_backend, builder):
        optic = builder()
        last = optic.surfaces.num_surfaces - 1
        matrix = optic.paraxial.ray_transfer_matrix(1, last)
        efl = -1.0 / float(be.to_numpy(matrix[1, 0]))
        assert_allclose(efl, optic.paraxial.f2(), rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("builder", [straight, folded])
    def test_f2_range_matches_basis_rays(self, set_test_backend, builder):
        optic = builder()
        last = optic.surfaces.num_surfaces - 1
        reference = basis_ray_matrix(optic, 1, last)
        efl_ref = -1.0 / reference[1][0]
        assert_allclose(optic.paraxial.f2_range(1, last), efl_ref, rtol=RTOL, atol=ATOL)


class TestOddParityAuthorings:
    """Matrix equality for equivalent local-axis authorings on folded arms."""

    @pytest.mark.parametrize("authoring", ["along", "against"])
    def test_powered_mirror_matrix_matches_trombone(self, set_test_backend, authoring):
        optic = folded_powered_mirror(authoring)
        reference = trombone_mirror()
        last = optic.surfaces.num_surfaces - 1
        matrix = matrix_to_rows(optic.paraxial.ray_transfer_matrix(1, last))
        ref = matrix_to_rows(reference.paraxial.ray_transfer_matrix(1, last))
        assert_matrix_close(matrix, ref)

    @pytest.mark.parametrize("authoring", ["along", "against"])
    def test_powered_lens_matrix_matches_trombone(self, set_test_backend, authoring):
        optic = folded_powered_lens(authoring)
        reference = trombone_lens()
        last = optic.surfaces.num_surfaces - 1
        matrix = matrix_to_rows(optic.paraxial.ray_transfer_matrix(1, last))
        ref = matrix_to_rows(reference.paraxial.ray_transfer_matrix(1, last))
        assert_matrix_close(matrix, ref)

    @pytest.mark.parametrize("authoring", ["along", "against"])
    def test_paraxial_thin_lens_matrix_matches_trombone(
        self, set_test_backend, authoring
    ):
        optic = folded_paraxial_lens(authoring)
        reference = trombone_paraxial_lens()
        last = optic.surfaces.num_surfaces - 1
        matrix = matrix_to_rows(optic.paraxial.ray_transfer_matrix(1, last))
        ref = matrix_to_rows(reference.paraxial.ray_transfer_matrix(1, last))
        assert_matrix_close(matrix, ref)

    def test_paraxial_thin_lens_trace_matches_between_authorings(
        self, set_test_backend
    ):
        along = folded_paraxial_lens("along")
        against = folded_paraxial_lens("against")
        ya, ua = along.paraxial.marginal_ray()
        yb, ub = against.paraxial.marginal_ray()
        assert_allclose(ya, yb, rtol=RTOL, atol=ATOL)
        assert_allclose(ua, ub, rtol=RTOL, atol=ATOL)
        assert_allclose(along.paraxial.f2(), against.paraxial.f2(), rtol=RTOL)


class TestMatrixDomainValidation:
    def test_oblique_powered_mirror_matrix_raises(self, set_test_backend):
        optic = oblique_powered_mirror()
        with pytest.raises(UnsupportedParaxialGeometryError):
            optic.paraxial.ray_transfer_matrix(1, optic.surfaces.num_surfaces - 1)

    def test_tilted_powered_refractive_matrix_raises(self, set_test_backend):
        optic = folded_tilted_lens()
        with pytest.raises(UnsupportedParaxialGeometryError):
            optic.paraxial.ray_transfer_matrix(1, optic.surfaces.num_surfaces - 1)

    def test_f2_range_rejects_out_of_domain_geometry(self, set_test_backend):
        optic = oblique_powered_mirror()
        with pytest.raises(UnsupportedParaxialGeometryError):
            optic.paraxial.f2_range(1, optic.surfaces.num_surfaces - 1)

    def test_straight_tilted_lens_matrix_warns_advisory(self, set_test_backend):
        optic = Optic(name="straight-tilted-advisory")
        optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        optic.surfaces.add(
            index=1,
            radius=25.84,
            thickness=4.0,
            material="N-BK7",
            is_stop=True,
            rx=0.05,
        )
        optic.surfaces.add(index=2, radius=be.inf, thickness=46.0)
        optic.surfaces.add(index=3)
        optic = _finish(optic)
        with pytest.warns(ParaxialDomainWarning):
            optic.paraxial.ray_transfer_matrix(1, 3)


# ---------------------------------------------------------------------------
# Workstream B: straight-path authoring independence
# ---------------------------------------------------------------------------


class TestStraightReversedAxisAuthoring:
    @pytest.mark.parametrize(
        "canonical_builder,reversed_builder",
        [
            (straight, straight_reversed_lens),
            (straight_canonical_mirror, straight_reversed_mirror),
            (straight_paraxial_lens, lambda: straight_paraxial_lens("reversed")),
        ],
        ids=["refractive", "mirror", "paraxial"],
    )
    def test_scalar_trace_matrix_and_pupils_match(
        self, set_test_backend, canonical_builder, reversed_builder
    ):
        canonical = canonical_builder()
        reversed_ = reversed_builder()
        last = canonical.surfaces.num_surfaces - 1

        # Scalar trace equivalence.
        assert_allclose(
            reversed_.paraxial.f2(), canonical.paraxial.f2(), rtol=RTOL, atol=ATOL
        )
        assert_allclose(
            reversed_.paraxial.f1(), canonical.paraxial.f1(), rtol=RTOL, atol=ATOL
        )
        # Pupil equivalence.
        assert_allclose(
            reversed_.paraxial.EPL(), canonical.paraxial.EPL(), rtol=RTOL, atol=ATOL
        )
        assert_allclose(
            reversed_.paraxial.XPL(), canonical.paraxial.XPL(), rtol=RTOL, atol=ATOL
        )
        # Matrix equivalence.
        assert_matrix_close(
            matrix_to_rows(reversed_.paraxial.ray_transfer_matrix(1, last)),
            matrix_to_rows(canonical.paraxial.ray_transfer_matrix(1, last)),
        )

    def test_real_rays_equivalent_refractive(self, set_test_backend):
        reference = trace_probe_rays(straight())
        probe = trace_probe_rays(straight_reversed_lens())
        for ref, got in zip(reference, probe, strict=True):
            assert_allclose(got, ref, rtol=1e-9, atol=1e-9)

    def test_real_rays_equivalent_mirror(self, set_test_backend):
        reference = trace_probe_rays(straight_canonical_mirror())
        probe = trace_probe_rays(straight_reversed_mirror())
        for ref, got in zip(reference, probe, strict=True):
            assert_allclose(got, ref, rtol=1e-9, atol=1e-9)

    def test_authored_geometry_and_serialization_unchanged(self, set_test_backend):
        optic = straight_reversed_lens()
        before = optic.surfaces.to_dict()
        # Run every first-order computation that applies orientation signs.
        optic.paraxial.f2()
        optic.paraxial.ray_transfer_matrix(1, 3)
        optic.paraxial.marginal_ray()
        after = optic.surfaces.to_dict()
        assert before == after
        assert_allclose(optic.surfaces[1].geometry.radius, -25.84)

    def test_canonical_values_bit_for_bit(self, set_test_backend):
        """The canonical +z authoring must not change at all: every
        orientation sign is +1 and the sign application is skipped."""
        optic = straight()
        path = optic.surfaces.build_paraxial_path()
        assert all(s == 1.0 for s in path.effective_orientation_signs())
        radii = optic.surfaces.radii
        sequence = optic.paraxial._ray_tracer.prepare_scalar_sequence(
            optic.primary_wavelength, path=path
        )
        assert be.to_numpy(sequence.radii).tolist() == (be.to_numpy(radii).tolist())

    def test_straight_oblique_surface_keeps_historical_raw_power(
        self, set_test_backend
    ):
        """A genuinely oblique powered surface on a straight system keeps
        the historical raw value (sign +1) -- never a heuristic sign."""
        optic = Optic(name="straight-tilted-keeps-raw")
        optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        optic.surfaces.add(
            index=1,
            radius=25.84,
            thickness=4.0,
            material="N-BK7",
            is_stop=True,
            rx=0.3,
        )
        optic.surfaces.add(index=2, radius=be.inf, thickness=46.0)
        optic.surfaces.add(index=3)
        optic = _finish(optic)
        path = optic.surfaces.build_paraxial_path()
        signs = path.effective_orientation_signs()
        assert signs[1] == 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ParaxialDomainWarning)
            f2_tilted = float(be.to_numpy(optic.paraxial.f2()))
        f2_straight = float(be.to_numpy(straight().paraxial.f2()))
        # Historical behavior: the tilt is ignored, values unchanged.
        assert_allclose(f2_tilted, f2_straight, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Backend parity and gradients through the matrix path
# ---------------------------------------------------------------------------


class TestMatrixBackendParity:
    @pytest.mark.parametrize("authoring", ["along", "against"])
    def test_matrix_matches_reference_on_all_backends(
        self, set_test_backend, authoring
    ):
        """Parametrized over numpy/torch: the folded matrix equals the
        trombone reference on every backend, which pins backend parity."""
        optic = folded_powered_lens(authoring)
        reference = trombone_lens()
        last = optic.surfaces.num_surfaces - 1
        assert_matrix_close(
            matrix_to_rows(optic.paraxial.ray_transfer_matrix(1, last)),
            matrix_to_rows(reference.paraxial.ray_transfer_matrix(1, last)),
        )

    def test_gradient_through_folded_radius_in_matrix(self, set_test_backend):
        """d(f2_range)/d(radius) through the folded matrix path is finite
        and matches a central finite difference."""
        if be.get_backend() != "torch":
            pytest.skip("Autodiff test requires the torch backend.")
        be.grad_mode.enable()

        optic = folded_powered_lens("against")
        radius = optic.surfaces[3].geometry.radius
        radius.requires_grad_(True)
        last = optic.surfaces.num_surfaces - 1
        efl = optic.paraxial.f2_range(1, last)
        efl.backward()
        grad = float(be.to_numpy(radius.grad))
        assert math.isfinite(grad)
        assert grad != 0.0

        h = 1e-4
        efl_vals = []
        for delta in (+h, -h):
            probe = folded_powered_lens("against")
            probe.surfaces[3].geometry.radius = be.array(-25.84 + delta)
            efl_vals.append(float(be.to_numpy(probe.paraxial.f2_range(1, last))))
        grad_fd = (efl_vals[0] - efl_vals[1]) / (2 * h)
        assert_allclose(grad, grad_fd, rtol=1e-5)
