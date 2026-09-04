"""Local launch parameterization for iterative ray aiming.

The Newton/Broyden aiming core drives exactly two scalar parameters per ray.
Historically those were global launch ``(x, y)`` for infinite conjugates and
global direction ``(L, M)`` (with ``N`` held fixed) for finite ones -- both
of which silently assume the beam enters along global +z. For a system
entered along any other direction one of those parameters points along the
beam (a vanishing Jacobian column), and varying ``L, M`` at fixed ``N``
neither preserves the unit norm nor spans the transverse plane.

:class:`LaunchParameterization` replaces both with two true transverse
degrees of freedom ``(xi, eta)``:

- infinite conjugate: the launch point moves in the entry-frame transverse
  plane, ``r(xi, eta) = r_seed + xi * u + eta * v``, while the field
  direction stays fixed;
- finite conjugate: the object point stays fixed and the direction rotates
  in a per-ray orthonormal tangent basis around the seed direction,
  ``k(xi, eta) = normalize(k_seed + xi * e1 + eta * e2)`` with
  ``e1, e2 perpendicular to k_seed``, so every trial and line-search
  candidate is a unit direction by construction.

For the canonical +z entry the infinite-conjugate map reduces exactly to
the historical ``(x, y)`` offsets (``u = +x``, ``v = +y``).

The bases are stored as plain Python floats. Ray aiming is an iterative
solve whose result is polished against real traces, so it never carries
gradient information (same design as ``pupil_map.py``).

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.utils import machine_eps

if TYPE_CHECKING:
    from optiland.optic import Optic


def _degenerate_projection_tolerance() -> float:
    """Norm below which a projected basis vector is degenerate, dtype-aware.

    The rejection ``u - (u . k) k`` of two nearly parallel unit vectors
    cancels catastrophically: with eps-level errors in the components, a
    rejection norm below ~sqrt(eps) carries no reliable direction (fewer
    than half the significant digits survive), so the secondary basis
    vector is used instead. ``sqrt(eps)`` is ~1.5e-8 for float64 --
    matching the historical fixed 1e-8 -- and ~3.5e-4 for float32, where a
    fixed 1e-8 sat below round-off and could never trigger.
    """
    return float(machine_eps(be.zeros(1))) ** 0.5


def _to_float(value: Any) -> float:
    return float(be.to_numpy(be.array(value)).reshape(-1)[0])


@dataclass(frozen=True)
class SolveReport:
    """Outcome of one Newton/Broyden aiming solve.

    A returned finite ray is not evidence of convergence; consult this
    report. Residuals are max-abs stop-plane errors in the stop surface's
    local transverse coordinates (mm).

    Attributes:
        seed_residual: Residual of the initial (seed) launch.
        final_residual: Residual of the returned launch.
        converged: Whether every ray met the solver tolerance.
        iterations: Newton/Broyden iterations executed.
        num_rays: Number of rays in the solve.
        num_converged: Number of rays that met tolerance.
        fallback_used: True when the Newton core substituted the
            sign-preserving paraxial diagonal for at least one ray's
            Jacobian -- at initialization because the central finite
            difference was unusable (a perturbed ray missed a surface), or
            during iteration because a refreshed finite-difference Jacobian
            remained ill-conditioned. Populated by ``_solve_core`` from the
            actual conditioning path taken.
        jacobian_refreshes: Number of iteration-time central-difference
            Jacobian refreshes triggered by an ill-conditioned (round-off
            level reciprocal-condition) Newton solve.
    """

    seed_residual: float
    final_residual: float
    converged: bool
    iterations: int
    num_rays: int
    num_converged: int
    fallback_used: bool = False
    jacobian_refreshes: int = 0


@dataclass(frozen=True)
class LaunchParameterization:
    """Two-parameter transverse launch model in the entry frame.

    Attributes:
        is_infinite: Whether the object is at infinity (position DOF) or
            finite (direction DOF).
        u: First entry-frame transverse basis vector (unit, floats).
        v: Second entry-frame transverse basis vector (unit, floats).
    """

    is_infinite: bool
    u: tuple[float, float, float]
    v: tuple[float, float, float]

    @classmethod
    def for_optic(cls, optic: Optic, is_infinite: bool) -> LaunchParameterization:
        """Build the parameterization from the optic's entry frame."""
        path = optic.surfaces.build_paraxial_path()
        u = tuple(_to_float(c) for c in path.entry_u)
        v = tuple(_to_float(c) for c in path.entry_v)
        return cls(is_infinite=is_infinite, u=u, v=v)

    def bind(self, x, y, z, L, M, N) -> BoundLaunch:
        """Bind per-ray seed launch states, yielding a (xi, eta) map."""
        return BoundLaunch(self, x, y, z, L, M, N)


class BoundLaunch:
    """A :class:`LaunchParameterization` bound to per-ray seed states.

    ``launch(0, 0)`` reproduces the seeds exactly. For finite conjugates the
    per-ray tangent basis ``(e1, e2)`` is built perpendicular to each seed
    direction; the basis is refreshed by re-binding whenever the seeds
    change meaningfully (each solve binds once).
    """

    def __init__(self, param: LaunchParameterization, x, y, z, L, M, N) -> None:
        self.param = param
        self.x0 = be.copy(be.as_array_1d(x))
        self.y0 = be.copy(be.as_array_1d(y))
        self.z0 = be.copy(be.as_array_1d(z))
        self.L0 = be.copy(be.as_array_1d(L))
        self.M0 = be.copy(be.as_array_1d(M))
        self.N0 = be.copy(be.as_array_1d(N))

        if not param.is_infinite:
            degenerate_tol = _degenerate_projection_tolerance()
            # Normalize the seed direction, then build the per-ray
            # orthonormal tangent basis perpendicular to it.
            norm = be.sqrt(self.L0**2 + self.M0**2 + self.N0**2)
            norm = be.where(norm < degenerate_tol, 1.0, norm)
            k0 = (self.L0 / norm, self.M0 / norm, self.N0 / norm)
            self.k0 = k0

            u, v = param.u, param.v
            e1 = self._reject(u, k0)
            n1 = be.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
            # Near-parallel seed and u (a field at ~90 deg from the entry
            # axis): fall back to v, which is then guaranteed transverse.
            alt = self._reject(v, k0)
            n_alt = be.sqrt(alt[0] ** 2 + alt[1] ** 2 + alt[2] ** 2)
            use_alt = n1 < degenerate_tol
            n1_safe = be.where(use_alt, be.ones_like(n1), n1)
            n_alt_safe = be.where(n_alt < degenerate_tol, 1.0, n_alt)
            self.e1 = tuple(
                be.where(use_alt, alt[i] / n_alt_safe, e1[i] / n1_safe)
                for i in range(3)
            )
            self.e2 = (
                k0[1] * self.e1[2] - k0[2] * self.e1[1],
                k0[2] * self.e1[0] - k0[0] * self.e1[2],
                k0[0] * self.e1[1] - k0[1] * self.e1[0],
            )

    @staticmethod
    def _reject(vec: tuple[float, float, float], k0: tuple) -> tuple:
        """Component of a constant vector perpendicular to per-ray k0."""
        dot = vec[0] * k0[0] + vec[1] * k0[1] + vec[2] * k0[2]
        return (vec[0] - dot * k0[0], vec[1] - dot * k0[1], vec[2] - dot * k0[2])

    def launch(self, xi, eta) -> tuple:
        """Physical launch states ``(x, y, z, L, M, N)`` for parameters.

        Infinite conjugate: seed positions displaced by ``xi*u + eta*v``
        (fixed directions). Finite conjugate: seed positions kept, unit
        directions ``normalize(k0 + xi*e1 + eta*e2)``.
        """
        if self.param.is_infinite:
            u, v = self.param.u, self.param.v
            x = self.x0 + xi * u[0] + eta * v[0]
            y = self.y0 + xi * u[1] + eta * v[1]
            z = self.z0 + xi * u[2] + eta * v[2]
            return x, y, z, self.L0, self.M0, self.N0

        kx = self.k0[0] + xi * self.e1[0] + eta * self.e2[0]
        ky = self.k0[1] + xi * self.e1[1] + eta * self.e2[1]
        kz = self.k0[2] + xi * self.e1[2] + eta * self.e2[2]
        norm = be.sqrt(kx**2 + ky**2 + kz**2)
        return self.x0, self.y0, self.z0, kx / norm, ky / norm, kz / norm

    def project(self, x, y, z, L, M, N) -> tuple:
        """Inverse map: parameters ``(xi, eta)`` of a physical launch state.

        Exact inverse of :meth:`launch` (used for warm starts from cached
        or previously solved launches). For finite conjugates the identity
        ``xi = (k . e1) / (k . k0)`` inverts the normalized tangent update.
        """
        if self.param.is_infinite:
            u, v = self.param.u, self.param.v
            dx = be.as_array_1d(x) - self.x0
            dy = be.as_array_1d(y) - self.y0
            dz = be.as_array_1d(z) - self.z0
            xi = dx * u[0] + dy * u[1] + dz * u[2]
            eta = dx * v[0] + dy * v[1] + dz * v[2]
            return xi, eta

        L = be.as_array_1d(L)
        M = be.as_array_1d(M)
        N = be.as_array_1d(N)
        along = L * self.k0[0] + M * self.k0[1] + N * self.k0[2]
        along = be.where(be.abs(along) < _degenerate_projection_tolerance(), 1.0, along)
        xi = (L * self.e1[0] + M * self.e1[1] + N * self.e1[2]) / along
        eta = (L * self.e2[0] + M * self.e2[1] + N * self.e2[2]) / along
        return xi, eta
