"""Forbes Polynomial Geometries for Optical Surfaces.

This module provides implementations for optical surfaces described by
Forbes polynomials, which are superimposed on a base conic section.
Forbes polynomials offer a modern alternative to standard power-series
aspheres.

The implementation is based on the work of G. W. Forbes. See, for example,
G. W. Forbes, "Manufacturability estimates for optical aspheres," Opt. Express (2011).

This module implements two types of Forbes surfaces:
    1.  **ForbesQNormalSlopeGeometry** (slope-orthogonal Q polynomials):
        This class represents rotationally symmetric aspheric surfaces using
        Forbes Q polynomials (historically called Q^bfs in Forbes 2007). These
        polynomials are orthogonal with respect to the normal-departure
        slope metric, and the implementation supports a general conic
        reference surface (Forbes 2011 generalization). The deprecated alias
        ``ForbesQbfsGeometry`` is retained for backward compatibility.
    2.  **ForbesQ2dGeometry (Q-type, 2D)**:
        This class represents non-rotationally symmetric, or "freeform," surfaces.
        The Q2D polynomials extend the Q-type formalism to two dimensions, allowing
        for the description of complex, freeform optical surfaces that lack
        rotational symmetry.
"""

from __future__ import annotations

import operator
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import optiland.backend as be
from optiland._host_values import host
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries._derivatives import is_transformed
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from optiland.geometries.standard import _is_radius_infinite

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

from ._zero_terms import (
    KEEP,
    REVERSE,
    q2d_cartesian_weights,
    q2d_polar_weights,
    q2d_sum_weights,
    q2d_vertex_weights,
    sum_weights,
    zero_terms,
)
from .qpoly import (
    _q2d_cartesian_eval,
    _trim_trailing_zeros,
    clenshaw_q2d,
    clenshaw_qbfs,
    compute_z_q2d,
    compute_z_zprime_q2d,
    compute_z_zprime_qbfs,
    q2d_nm_coeffs_to_ams_bms,
    q2d_sum_from_alphas,
)

_EPSILON = 1e-12
# The state of a coefficient tensor that shows a change in place (#1175).
_STATE = operator.attrgetter("_version", "requires_grad")


def _watchable(value):
    """Return True for a tensor whose state can show a change in place.

    Args:
        value: A coefficient as the dict holds it.

    Returns:
        bool: False for numbers, NumPy arrays and tensors of a ``torch.func``
        transform.
    """
    return hasattr(value, "requires_grad") and not is_transformed(value)


def _is_radius_zero(radius):
    """Return True if the radius is zero, from the host copy in a CUDA trace.

    Args:
        radius (float or array_like): The vertex radius of curvature.

    Returns:
        bool: True for a zero radius.
    """
    value = host(radius)
    return bool((radius if value is None else value) == 0)


def _radial_sums(coefficients, basis, usq, slope):
    """Return (S,), or (S, dS/d(u**2)) with ``slope``, of radial coefficients.

    The Q-BFS sum of ``clenshaw_qbfs`` and ``compute_z_zprime_qbfs``, for the
    trimmed coefficients of a Q surface or for all of them.

    Args:
        coefficients (list): The coefficients, at least one.
        basis (list or None): Their change of basis, or None to compute it.
        usq (array_like): The squared normalized radius u**2.
        slope (bool): Also return dS/d(u**2).

    Returns:
        tuple: S, and dS/d(u**2) with ``slope``.
    """
    if slope:
        return compute_z_zprime_qbfs(coefficients, None, usq, _no_trim=True, bs=basis)
    return (clenshaw_qbfs(coefficients, usq, _no_trim=True, bs=basis),)


def _conic_roots(r2, c2, k, numerator=True, denominator=True):
    """Return the roots N and D of the conic correction factor phi = N / D.

    The operations of ``_conic_correction``, with its floors for a negative
    argument.

    Args:
        r2 (array_like): The squared radial coordinate (rho^2).
        c2 (float or array_like): The squared curvature 1 / R**2.
        k (float or array_like): The conic constant.
        numerator (bool): Compute N. Without it, N is None.
        denominator (bool): Compute D. Without it, D is None.

    Returns:
        tuple: N = sqrt(1 - k c**2 r**2) and D = sqrt(1 - (k + 1) c**2 r**2).
    """
    N = D = None
    if numerator:
        num_arg = 1 - k * c2 * r2
        N = be.sqrt(be.where(num_arg > 0, num_arg, 1e-12))
    if denominator:
        den_arg = 1 - (k + 1) * c2 * r2
        D = be.sqrt(be.where(den_arg > 0, den_arg, 1e-12))
    return N, D


def _zero_departure(saved, gradient):
    """Return the values that the derivative of a zero departure needs.

    The departure of a Q surface is u**2 (1 - u**2) phi S for the sum S of
    the Q polynomials, and zero outside u**2 = 1. See ``_sag_weights`` and
    ``_slope_factor_weights``.

    Args:
        saved (tuple): r**2, Rn**2, c**2, k and the root D of the evaluation
            (None if it computed no D), without a graph; c**2 is None for a
            plane.
        gradient (torch.Tensor): The gradient of the output.

    Returns:
        tuple: u**2, the envelope u**2 (1 - u**2), the conic factor phi,
        d(phi)/d(r**2) times 2, and the gradient, zero outside u**2 = 1.
    """
    r2, rn2, c2, k, D = saved
    usq = r2 / rn2
    if c2 is None:
        factor, factor_slope = 1.0, 0.0
    else:
        N, D_new = _conic_roots(r2, c2, k, denominator=D is None)
        D = D_new if D is None else D
        factor, factor_slope = N / D, c2 / (N * D**3)
    gradient = torch.where(usq > 1, 0.0, gradient)
    return usq, usq * (1 - usq), factor, factor_slope, gradient


def _sag_weights(saved, gradients, families):
    """Weights of the zero terms for the sag of a Q surface without terms.

    If the trimming keeps no coefficient, the departure u**2 (1 - u**2) phi S
    is zero, and ``ForbesQNormalSlopeGeometry._sag_with_terms`` leaves it
    out. Its derivative with respect to a coefficient c_n is
    u**2 (1 - u**2) phi Q_n. This function gives the weights of it for
    ``ZeroTerms.attach``; saved (r**2, Rn**2, c**2, k), with c**2 None for a
    plane.

    Args:
        saved (tuple): The saved positions and parameters, without a graph.
        gradients (tuple): The gradient of the sag.
        families (tuple): The one m = 0 family.

    Returns:
        tuple: The variable u**2 of the basis and the weights of the family.
    """
    usq, envelope, factor, _, gradient = _zero_departure(saved, gradients[0])
    return usq, [(gradient * (envelope * factor), None)]


def _slope_factor_weights(saved, gradients, families):
    """Weights of the zero terms for the slope factor of a Q surface.

    ``ForbesQNormalSlopeGeometry._slopes_from_terms`` gives d(sag)/dx =
    x F and d(sag)/dy = y F with F = c / D + where(u**2 > 1, 0, a S + b S'),
    a = 2 (1 - 2 u**2) phi / Rn**2 + u**2 (1 - u**2) c**2 / (N D**3) and
    b = 2 u**2 (1 - u**2) phi / Rn**2, for the sum S of the Q polynomials
    and S' = dS/d(u**2). If the trimming keeps no coefficient, S is zero and
    F = c / D. This function gives the weights of the derivative of F for
    ``ZeroTerms.attach``; saved (r**2, Rn**2, c**2, k), with c**2 None for a
    plane.

    Args:
        saved (tuple): The saved positions and parameters, without a graph.
        gradients (tuple): The gradient of F.
        families (tuple): The one m = 0 family.

    Returns:
        tuple: The variable u**2 of the basis and the weights of the family.
    """
    usq, envelope, factor, factor_slope, gradient = _zero_departure(saved, gradients[0])
    rn2 = saved[1]
    a = 2 * (1 - 2 * usq) * factor / rn2 + envelope * factor_slope
    b = 2 * envelope * factor / rn2
    return usq, [(gradient * a, gradient * b)]


def _conic_correction(r2, radius, k):
    """Return the Forbes conic correction factor and its derivative.

    See ``ForbesGeometryBase._conic_correction_factor``.

    Args:
        r2 (float or array_like): The squared radial coordinate (rho^2).
        radius (float or array_like): The vertex radius of curvature.
        k (float or array_like): The conic constant.

    Returns:
        tuple: The factor and its derivative with respect to rho.
    """
    if _is_radius_infinite(radius):
        return 1.0, 0.0

    c2 = (1.0 / radius) ** 2
    rho = be.sqrt(r2)
    num_arg = 1 - k * c2 * r2
    den_arg = 1 - (k + 1) * c2 * r2

    safe_num_arg = be.where(num_arg > 0, num_arg, 1e-12)
    safe_den_arg = be.where(den_arg > 0, den_arg, 1e-12)
    N = be.sqrt(safe_num_arg)
    D = be.sqrt(safe_den_arg)

    factor = N / D
    derivative = (c2 * rho) / (N * D**3)
    return factor, derivative


class _CoeffCacheDict(dict):
    """Coefficient dict that invalidates its owner's prepared-coefficient cache.

    The Forbes geometries cache trimmed/stacked coefficient containers and
    rebuild them lazily (see ``_ensure_coeffs``). The optimizer and ``scale``
    invalidate that cache explicitly, but a user may also poke a coefficient in
    place (``geom.radial_terms[n] = value``). This dict catches such in-place
    edits and marks the owner dirty, so the next ``sag`` / ``_surface_normal``
    picks them up -- matching the pre-cache behavior where the coefficients
    were rebuilt on every call.

    The standard ``dict`` constructor signature is kept intact so that
    ``dataclasses.asdict`` (used by ``to_dict``) can reconstruct the mapping via
    ``type(obj)(...)``. The owner is attached afterwards with :meth:`_bind`; an
    unbound instance simply behaves like a plain ``dict``.
    """

    _owner = None

    def _bind(self, owner) -> _CoeffCacheDict:
        self._owner = owner
        return self

    def _invalidate(self) -> None:
        if self._owner is not None:
            self._owner._coeffs_dirty = True

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self._invalidate()

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        self._invalidate()

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._invalidate()

    def setdefault(self, key, default=None):
        result = super().setdefault(key, default)
        self._invalidate()
        return result

    def pop(self, *args, **kwargs):
        result = super().pop(*args, **kwargs)
        self._invalidate()
        return result

    def popitem(self):
        result = super().popitem()
        self._invalidate()
        return result

    def clear(self) -> None:
        super().clear()
        self._invalidate()


@dataclass
class ForbesSolverConfig:
    """Configuration for the Newton-Raphson numerical solver.

    Attributes:
        tol (float): Tolerance for the iterative solver.
        max_iter (int): Maximum number of iterations for the solver.
    """

    tol: float = 1e-10
    max_iter: int = 100


@dataclass
class ForbesSurfaceConfig:
    """Configuration for a surface's core geometric properties.

    Attributes:
        radius (float): The vertex radius of curvature of the base surface.
        conic (float): The conic constant of the base surface.
        norm_radius (float, optional): The normalization radius for the
            polynomial terms. If None, the radius scales automatically
            during paraxial updates. Defaults to None.
        terms (dict, optional): A dictionary of polynomial coefficients.
        The key format depends on the
            specific geometry type (e.g., `radial_terms` for Qbfs,
            `freeform_coeffs` for Q2d).
    """

    radius: float
    conic: float = 0.0
    norm_radius: float | None = None
    # either radial_terms or freeform_coeffs
    terms: dict[Any, float] | None = None


class ForbesGeometryBase(NewtonRaphsonGeometry):
    """Base class for Forbes geometries to share common mathematical logic."""

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        surface_config: ForbesSurfaceConfig,
        solver_config: ForbesSolverConfig = None,
    ):
        """Initializes the base Forbes geometry.

        Args:
            coordinate_system (CoordinateSystem): The local coordinate system of
                                                    the surface.
            surface_config (ForbesSurfaceConfig): An object containing the
                                                    core geometric parameters.
            solver_config (ForbesSolverConfig, optional): An object containing
                                            parameters for the numerical solver.
                If None, defaults are used.
        """
        if solver_config is None:
            solver_config = ForbesSolverConfig()

        super().__init__(
            coordinate_system,
            surface_config.radius,
            surface_config.conic,
            solver_config.tol,
            solver_config.max_iter,
        )
        self.surface_config = surface_config
        self.solver_config = solver_config
        # Coefficient cache: ``_prepare_coeffs`` is expensive (dict -> stacked
        # tensors + per-family trimming, each ``bool(v == 0)`` forcing a CUDA
        # sync). It is rebuilt only when invalidated -- on construction, on
        # ``scale``, and on optimizer coefficient updates -- not on every
        # ``sag``/``_surface_normal`` call. ``norm_radius`` is applied at
        # evaluation time and does not enter coefficient preparation, so
        # ``update_normalization`` does not invalidate this cache.
        self._coeffs_dirty = True
        # The trimmed zero coefficients that can carry a derivative (see
        # ``_zero_terms.py``). ``_prepare_coeffs`` sets it.
        self._zero_terms = None
        # The coefficient tensors of the last preparation and their state
        # (see ``_watch_coeffs``).
        self._watched = ()
        self._watched_state = ()

    def _watch_coeffs(self, values) -> None:
        """Remember the coefficient tensors of a preparation and their state.

        The prepared containers are copies (a stack of the coefficients). A
        change of a coefficient tensor in place, for example by a
        ``torch.optim`` step, does not reach them, and neither does a later
        ``requires_grad_()``. The state of each tensor, its version counter
        and its requires_grad flag, shows such a change (issue #1175).
        ``_ensure_coeffs`` then prepares again.

        Args:
            values (Iterable): The coefficients as the dict holds them.
                Numbers and NumPy arrays have no state to watch, and neither
                has a tensor of a ``torch.func`` transform.
        """
        watched = tuple(value for value in values if _watchable(value))
        self._watched = watched
        self._watched_state = tuple(map(_STATE, watched))

    def _coeffs_changed(self) -> bool:
        """Return True if a watched coefficient tensor changed its state.

        The check reads attributes of a few tensors and no tensor values, so
        it costs little in each evaluation and never waits for a GPU.

        Returns:
            bool: True if a version counter or a requires_grad flag changed.
        """
        return tuple(map(_STATE, self._watched)) != self._watched_state

    def _zero_term_mode(self):
        """Return how the trimmed zero terms get their derivative now.

        Returns:
            str or None: ``REVERSE``, ``KEEP``, or None if no derivative with
            respect to a trimmed coefficient can be requested now. See
            ``ZeroTerms.mode``.
        """
        terms = self._zero_terms
        return None if terms is None else terms.mode()

    def _ensure_coeffs(self) -> None:
        """Rebuild cached coefficient containers only when invalidated.

        The Newton-Raphson ``distance()`` solve may trigger the *first*
        (cache-building) call from inside a ``torch.no_grad()`` block (the
        primal solve is intentionally untracked). Coefficient tensors
        stacked while grad recording is disabled become permanently
        detached -- ``requires_grad_(True)`` called on the originals
        afterwards cannot reconnect them. Force grad recording on while
        (re)building the cache so it stays differentiable regardless of the
        ambient grad context of the caller.

        A coefficient tensor that changed in place, or that requires grad
        only now, also starts a new preparation (see ``_watch_coeffs``).
        """
        if self._coeffs_dirty or (self._watched and self._coeffs_changed()):
            if torch is not None and be.get_backend() == "torch":
                with torch.enable_grad():
                    self._prepare_coeffs()
            else:
                self._prepare_coeffs()

    def _flip_terms(self, attr: str) -> None:
        """Flip the radius and the polynomial terms; keep the config in step.

        Shared by the Forbes ``flip`` methods. The coefficient dict named by
        ``attr`` is never mutated: it may be shared with another geometry (a
        pickup assigns the source's dict to the target). A new dict holding
        the negated values is built, bound to this geometry, and assigned to
        the attribute, and the coefficient cache is marked dirty. The radius
        is written to ``surface_config`` as a plain float. Under NumPy,
        ``surface_config.terms`` is then the new dict itself (as the
        constructor sets it up), so later edits reach ``to_dict``; under
        torch it is a separate copy of the terms as plain floats.

        Args:
            attr: Name of the coefficient dict attribute, ``"radial_terms"``
                or ``"freeform_coeffs"``.
        """
        super().flip()
        old = getattr(self, attr)
        new = _CoeffCacheDict({k: -v for k, v in old.items()})._bind(self)
        setattr(self, attr, new)
        self._coeffs_dirty = True

        self.surface_config.radius = float(be.to_numpy(self.radius))
        if be.get_backend() != "torch":
            self.surface_config.terms = new
        else:
            self.surface_config.terms = {
                k: float(be.to_numpy(v)) for k, v in new.items()
            }

    def _base_sag(self, r2, plane=None):
        """Calculates the sag of the base conic surface.

        Args:
            r2 (float or array_like): The squared radial coordinate (rho^2).
            plane (bool, optional): True if the radius is infinite, if the
                caller has checked it. Defaults to None: check it.

        Returns:
            float or array_like: The sag of the base conic.
        """
        if _is_radius_infinite(self.radius) if plane is None else plane:
            return be.zeros_like(r2)

        sqrt_arg = 1 - (1 + self.k) * r2 / self.radius**2
        safe_sqrt_arg = be.where(sqrt_arg < 0, 0, sqrt_arg)
        return r2 / (self.radius * (1 + be.sqrt(safe_sqrt_arg)))

    def _base_sag_derivative(self, rho, r2):
        """Calculates the derivative of the base conic sag w.r.t. rho.

        Args:
            rho (float or array_like): The radial coordinate (rho).
            r2 (float or array_like): The squared radial coordinate (rho^2).

        Returns:
            float or array_like: The derivative dz_base/drho.
        """
        if _is_radius_infinite(self.radius) or _is_radius_zero(self.radius):
            return be.zeros_like(rho)

        c = 1.0 / self.radius
        sqrt_arg_base = 1 - (self.k + 1) * c**2 * r2
        safe_sqrt_base = be.sqrt(be.where(sqrt_arg_base > 0, sqrt_arg_base, 1e-12))
        return c * rho / safe_sqrt_base

    def _conic_correction_factor(self, r2):
        """Calculates the Forbes conic correction factor and its derivative.

        This factor projects the normal departure from the base conic onto the
        sag axis.

        Args:
            r2 (float or array_like): The squared radial coordinate (rho^2).

        Returns:
            tuple[float or array_like, float or array_like]: A tuple containing:
                - factor (float or array_like): The unitless conic correction factor.
                - derivative (float or array_like): The derivative of the
                                                        factor with respect to rho.
        """
        return _conic_correction(r2, self.radius, self.k)

    def _conic_terms(self, r2, factor=True, root=True):
        """Return the conic correction factor, and the values of its slope.

        The operations of ``_conic_correction`` for the factor, without the
        derivative: the sag does not need it, and the slopes compute it from
        the values here (see ``_slope_parts``).

        Args:
            r2 (array_like): The squared radial coordinate (rho^2).
            factor (bool): Compute the factor. Without it, the factor and N
                are None, and only the root D of the base slope is computed.
            root (bool): Without the factor, compute D. Without both, the
                values are c and c**2 only.

        Returns:
            tuple: The factor N / D, and (c, c**2, N, D) with c = 1 / R, or
            None for a plane, where the factor is 1.0.
        """
        if _is_radius_infinite(self.radius):
            return 1.0, None
        c = 1.0 / self.radius
        c2 = c**2
        N, D = _conic_roots(r2, c2, self.k, numerator=factor, denominator=root)
        return (N / D if factor else None), (c, c2, N, D)

    def _slope_parts(self, conic):
        """Return the slope factors of the base conic and of the conic factor.

        The base sag has d(z_base)/dx = x c / D, and the conic factor
        phi = N / D has d(phi)/dx = 2 x d(phi)/d(r**2) = x c**2 / (N D**3).
        Both slopes vanish at the vertex, without a division by the radial
        coordinate.

        Args:
            conic (tuple or None): The values of ``_conic_terms``.

        Returns:
            tuple: c / D and c**2 / (N D**3), or 0.0 and None for a plane.
        """
        if conic is None:
            return 0.0, None
        c, c2, N, D = conic
        base = 0.0 if _is_radius_zero(self.radius) else c / D
        return base, c2 / (N * D**3)


class ForbesQNormalSlopeGeometry(ForbesGeometryBase):
    r"""Represents a Forbes Q surface using slope-orthogonal polynomials.

    This surface uses the Forbes Q polynomials that are orthogonal with respect
    to the normal-departure slope metric. These were historically called
    Q^bfs ("best-fit sphere") in Forbes 2007, but the implementation here
    supports a general conic reference surface following the Forbes 2011
    generalization.

    The surface sag is defined as:
    $z(\rho) = z_{base}(\rho) + \phi(\rho)
    \left[ u^2(1-u^2) \sum_{m=0}^{M} a_m Q_m(u^2) \right]$

    where:
        - $z_{base}(\rho) = \frac{c\rho^2}{1 + \sqrt{1 - (1+k)c^2\rho^2}}$
          is the base conic.
        - $c = 1/R$ is the curvature, $k$ is the conic constant.
        - $u = \rho/\rho_{max}$ is the normalized radial coordinate.
        - $Q_m(u^2)$ are the Forbes slope-orthogonal polynomials.
        - $a_m$ are the polynomial coefficients.
        - $\phi(\rho) = \sqrt{\frac{1 - kc^2\rho^2}{1 - (1+k)c^2\rho^2}}$ is the
          conic correction factor that projects the normal departure onto the
          sag axis.

    Note:
        The internal polynomial basis functions (named ``*_qbfs`` in the code)
        retain the historical "qbfs" identifier for stability. This does not
        imply a spherical reference; the conic constant $k$ can be nonzero.

    Args:
        coordinate_system (CoordinateSystem): The local coordinate system of the
                                                surface.
        surface_config (ForbesSurfaceConfig): An object containing the core
            geometric parameters. The `terms` dictionary should be provided
            as `radial_terms`.
        solver_config (ForbesSolverConfig, optional): An object containing
            parameters for the numerical solver.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        surface_config: ForbesSurfaceConfig,
        solver_config: ForbesSolverConfig = None,
    ):
        super().__init__(coordinate_system, surface_config, solver_config)
        if be.get_backend() == "torch":
            self.radial_terms = _CoeffCacheDict(
                {k: be.array(v) for k, v in (self.surface_config.terms or {}).items()}
            )._bind(self)
        else:
            # Preserve the historical numpy aliasing: ``radial_terms`` and
            # ``surface_config.terms`` are the same object, so in-place edits
            # propagate to serialization.
            self.radial_terms = _CoeffCacheDict(self.surface_config.terms or {})._bind(
                self
            )
            self.surface_config.terms = self.radial_terms

        if self.surface_config.norm_radius is not None:
            self.normalization_mode = "manual"
            self.norm_radius = be.array(self.surface_config.norm_radius)
        else:
            self.normalization_mode = "auto"
            self.norm_radius = be.array(1.0)

        self.is_symmetric = True

    def _prepare_coeffs(self):
        """Prepares the internal coefficient lists from the radial_terms dictionary.

        Also builds the trimmed ``_prepared_radial_coeffs`` container and the
        ``_all_coeffs_zero`` flag consumed by the hot ``sag``/``_surface_normal``
        paths, so that trailing-zero trimming (and its CUDA synchronization)
        happens here -- once per coefficient change -- rather than on every
        evaluation. The trimmed coefficients that can carry a derivative go
        to ``_zero_terms``, and ``_full_radial_coeffs`` keeps the trailing
        zeros for evaluations in forward mode and in ``torch.func``
        transforms. Clears the dirty flag.
        """
        if not self.radial_terms:
            self.coeffs_n, self.coeffs_c = [], be.array([])
            self._prepared_radial_coeffs = []
            self._all_coeffs_zero = True
            self._zero_terms = None
            self._full_radial_coeffs = []
            self._watch_coeffs(())
            self._coeffs_dirty = False
            return

        max_n = max(self.radial_terms.keys())
        if max_n >= 0:
            terms_list = [
                self.radial_terms.get(n, be.array(0.0)) for n in range(max_n + 1)
            ]
            self.coeffs_c = be.stack(terms_list)

            self.coeffs_n = [(n, 0) for n in range(max_n + 1)]
        else:
            self.coeffs_n, self.coeffs_c = [], be.array([])

        self._prepared_radial_coeffs = _trim_trailing_zeros(self.coeffs_c)
        self._all_coeffs_zero = len(self._prepared_radial_coeffs) == 0
        # An absent coefficient (None) has no derivative to carry.
        sources = [self.radial_terms.get(n) for n in range(len(self.coeffs_n))]
        self._zero_terms = zero_terms(
            [("m0", 0, sources, len(self._prepared_radial_coeffs))], self.coeffs_c
        )
        self._full_radial_coeffs = (
            self._prepared_radial_coeffs
            if self._zero_terms is None
            else [self.coeffs_c[n] for n in range(len(self.coeffs_n))]
        )
        self._watch_coeffs(self.radial_terms.values())
        self._coeffs_dirty = False

    def _radial_coeffs(self, mode):
        """Return the radial coefficients for an evaluation in ``mode``.

        Args:
            mode (str or None): The result of ``_zero_term_mode``.

        Returns:
            list: The coefficients with the trailing zeros for ``KEEP``, and
            the trimmed coefficients otherwise.
        """
        if mode == KEEP:
            return self._full_radial_coeffs
        return self._prepared_radial_coeffs

    def _radial(self, usq, slope):
        """Return the radial sum S at u**2, and dS/d(u**2) with ``slope``.

        The evaluation runs on the trimmed coefficients. In reverse mode,
        ``ZeroTerms`` adds the derivative of the trimmed zero coefficients.
        In forward mode and in ``torch.func`` transforms, the evaluation
        takes all coefficients if a trimmed zero coefficient can carry a
        derivative.

        Args:
            usq (array_like): The squared normalized radius u**2.
            slope (bool): Also return dS/d(u**2).

        Returns:
            tuple or None: S, and dS/d(u**2) with ``slope``. None if the
            trimming keeps no coefficient: then the departure is zero, and
            the sag and the slopes leave it out (``_sag_with_terms`` and
            ``_base_slopes``).
        """
        mode = self._zero_term_mode()
        coefficients = self._radial_coeffs(mode)
        if not coefficients:
            return None
        sums = _radial_sums(coefficients, None, usq, slope)
        if mode == REVERSE:
            sums = self._zero_terms.attach(sum_weights, (usq,), *sums)
        return sums

    def _radial_terms(self, x, y, slope):
        """Return the values that the sag and its slopes share at (x, y).

        Args:
            x (array_like): The x-coordinates.
            y (array_like): The y-coordinates.
            slope (bool): Also evaluate dS/d(u**2), for the slopes.

        Returns:
            tuple: r**2, u**2, the squared normalization radius, the
            envelope u**2 (1 - u**2), the sums of ``_radial``, the conic
            correction factor, and the values of ``_conic_terms``. Without
            sums, the envelope and the factor are None.
        """
        self._ensure_coeffs()
        r2 = x**2 + y**2
        rn2 = self.norm_radius**2
        usq = r2 / rn2
        sums = self._radial(usq, slope)
        if sums is None:
            # Only the base conic: its slope needs the root D, no factor, and
            # the sag alone needs neither.
            _, conic = self._conic_terms(r2, factor=False, root=slope)
            return r2, usq, rn2, None, None, None, conic
        factor, conic = self._conic_terms(r2)
        return r2, usq, rn2, usq * (1 - usq), sums, factor, conic

    def _sag_with_terms(self, x, y, slope):
        """Return the sag at (x, y) and the terms of ``_radial_terms``."""
        terms = self._radial_terms(x, y, slope)
        r2, usq, rn2, envelope, sums, factor, conic = terms
        z_base = self._base_sag(r2, plane=conic is None)
        if sums is not None:
            departure = envelope * factor * sums[0]
            return z_base + be.where(usq > 1, 0.0, departure), terms
        # No term stays: the departure is zero. u**2 * 0 gives the zero (and
        # a NaN for a NaN u**2) that the departure of before gave.
        sag = z_base + be.where(usq > 1, 0.0, usq * 0.0)
        if self._zero_term_mode() == REVERSE:
            (sag,) = self._zero_terms.attach(
                _sag_weights, self._zero_departure_saved(r2, rn2, conic), sag
            )
        return sag, terms

    def sag(self, x=0, y=0):
        """Calculate the sag of the Forbes Q (slope-orthogonal) surface.

        Args:
            x (int, optional): x-coordinate. Defaults to 0.
            y (int, optional): y-coordinate. Defaults to 0.

        Returns:
            The sag of the Forbes Q (slope-orthogonal) surface.
        """
        x, y = be.array(x), be.array(y)
        return self._sag_with_terms(x, y, slope=False)[0]

    def _sag_and_terms(self, x, y):
        """Return the sag at (x, y), and the terms that its slopes share.

        The evaluation gives S and dS/d(u**2) together, so the slopes in the
        Newton-Raphson loop need no second evaluation of the polynomial. See
        :meth:`NewtonRaphsonGeometry._sag_and_terms`.
        """
        return self._sag_with_terms(x, y, slope=True)

    def _slopes_from_terms(self, x, y, terms):
        """Return d(sag)/dx and d(sag)/dy from the terms of ``_radial_terms``.

        The sag is a function of r**2, so d(sag)/dx = 2 x d(sag)/d(r**2), and
        the same factor serves both slopes. There is no division by r, and
        the slopes at the vertex need no special case. The departure and its
        slope are zero for u**2 > 1, the rule of the sag.

        See :meth:`NewtonRaphsonGeometry._slopes_from_terms`.
        """
        r2, usq, rn2, envelope, sums, factor, conic = terms
        if sums is None:
            return self._base_slopes(x, y, r2, rn2, conic)
        S, S_slope = sums
        base, factor_slope = self._slope_parts(conic)
        # 2 d/d(r**2) [u**2 (1 - u**2) phi S] with u**2 = r**2 / Rn**2.
        radial = ((1 - 2 * usq) * S + envelope * S_slope) * factor * (2 / rn2)
        if factor_slope is not None:
            radial = radial + envelope * S * factor_slope
        shared = base + be.where(usq > 1, 0.0, radial)
        return x * shared, y * shared

    def _base_slopes(self, x, y, r2, rn2, conic):
        """Return the slopes of a surface whose trimming keeps no coefficient.

        The departure is zero, so the slopes are those of the base conic,
        x c / D and y c / D. If a derivative with respect to a trimmed zero
        coefficient can be requested in reverse mode, ``ZeroTerms`` attaches
        it to the factor c / D (``_slope_factor_weights``).

        Args:
            x (array_like): The x-coordinates.
            y (array_like): The y-coordinates.
            r2 (array_like): x**2 + y**2.
            rn2 (array_like): The squared normalization radius.
            conic (tuple or None): The values of ``_conic_terms``.

        Returns:
            tuple: d(sag)/dx and d(sag)/dy.
        """
        if conic is None:
            shared = 0.0
        else:
            c, _, _, D = conic
            shared = 0.0 if _is_radius_zero(self.radius) else c / D
        if self._zero_term_mode() == REVERSE:
            if not hasattr(shared, "shape"):
                shared = be.full_like(r2, shared)
            (shared,) = self._zero_terms.attach(
                _slope_factor_weights,
                self._zero_departure_saved(r2, rn2, conic),
                shared,
            )
        return x * shared, y * shared

    def _zero_departure_saved(self, r2, rn2, conic):
        """Return the values of ``_zero_departure`` for ``ZeroTerms.attach``.

        Args:
            r2 (array_like): x**2 + y**2.
            rn2 (array_like): The squared normalization radius.
            conic (tuple or None): The values of ``_conic_terms``.

        Returns:
            tuple: r**2, Rn**2, c**2, k and the root D of the evaluation (None
            if it computed no D); c**2, k and D None for a plane.
        """
        if conic is None:
            return r2, rn2, None, None, None
        return r2, rn2, conic[1], self.k, conic[3]

    def _surface_normal(self, x, y):
        """Calculates the unit vector normal to the surface.

        Args:
            x (float or array_like): X coordinate(s).
            y (float or array_like): Y coordinate(s).

        Returns:
            tuple[float or array_like, float or array_like, float or array_like]:
                Components of the unit normal vector (nx, ny, nz).
        """
        x_in, y_in = be.array(x), be.array(y)

        df_dx, df_dy = self._surface_normal_analytical(x_in, y_in)

        mag = be.sqrt(df_dx**2 + df_dy**2 + 1)
        safe_mag = be.where(mag < _EPSILON, 1.0, mag)
        return df_dx / safe_mag, df_dy / safe_mag, -1 / safe_mag

    def _surface_normal_analytical(self, x, y):
        """Computes the analytical surface derivatives.

        Args:
            x (float or array_like): X coordinate(s).
            y (float or array_like): Y coordinate(s).

        Returns:
            tuple[float or array_like, float or array_like]:
                The partial derivatives (df_dx, df_dy).
        """
        return self._slopes_from_terms(x, y, self._radial_terms(x, y, slope=True))

    def _surface_normal_analytical_vertex(self):
        """Computes the stable analytical derivative exactly at the vertex.

        For rotationally symmetric surfaces, the gradient at the vertex
        (x=0, y=0) should be (0, 0) due to symmetry.

        Returns:
            tuple[float, float]: The partial derivatives (df_dx, df_dy) at the vertex.
        """
        # For rotationally symmetric surfaces, the gradient at the vertex is (0, 0)
        # due to symmetry - there's no preferred direction for the slope
        return 0.0, 0.0

    def to_dict(self):
        """Serializes the geometry to a dictionary.

        Returns:
            dict: A dictionary representation of the geometry.
        """
        return {
            "type": self.__class__.__name__,
            "cs": self.cs.to_dict(),
            "surface_config": asdict(self.surface_config),
            "solver_config": asdict(self.solver_config),
        }

    @classmethod
    def from_dict(cls, data):
        """Creates an instance from a dictionary.

        Accepts both "ForbesQNormalSlopeGeometry" and legacy "ForbesQbfsGeometry"
        type identifiers for backward compatibility.

        Args:
            data (dict): A dictionary representation of the geometry.

        Returns:
            ForbesQNormalSlopeGeometry: An instance of the class.
        """
        cs = CoordinateSystem.from_dict(data["cs"])
        surface_config = ForbesSurfaceConfig(**data["surface_config"])
        solver_config = ForbesSolverConfig(**data.get("solver_config", {}))
        # Always return the canonical class, regardless of whether the
        # serialized type was "ForbesQbfsGeometry" or "ForbesQNormalSlopeGeometry"
        return ForbesQNormalSlopeGeometry(cs, surface_config, solver_config)

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature and of every Q (radial)
        coefficient, and updates ``surface_config`` to match. The conic
        constant and the normalization radius remain unchanged.
        """
        self._flip_terms("radial_terms")

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Scales the radius, normalization radius, and radial coefficients.
        The polynomial coefficients scale linearly with the sag when the
        normalization radius is also scaled.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        self.surface_config.radius = self.radius
        if self.surface_config.norm_radius is not None:
            self.surface_config.norm_radius *= scale_factor
        self.norm_radius = self.norm_radius * scale_factor

        for key in self.radial_terms:
            self.radial_terms[key] = self.radial_terms[key] * scale_factor

        # Coefficients changed: invalidate the cache so the next sag/normal
        # rebuilds the prepared (trimmed) containers.
        self._coeffs_dirty = True

    def update_normalization(self, semi_aperture: float) -> None:
        if self.normalization_mode == "auto":
            self.norm_radius = be.array(semi_aperture * 1.25)
            self.surface_config.norm_radius = float(self.norm_radius)

    def __str__(self):
        return "ForbesQNormalSlope"


class ForbesQ2dGeometry(ForbesGeometryBase):
    """Forbes Q2D (freeform) surface geometry."""

    # Default path: direct-Cartesian (harmonic powers), for the sag and
    # the slopes. Regular at the origin with finite autograd gradients;
    # the legacy polar path's `1/r` artefact produced NaN gradients at the
    # vertex. The sag and the slopes use the same r**2, without an
    # epsilon, so they describe the same surface. The polar path with its
    # `_surface_normal_analytical_vertex` `where`-blend remains available
    # as a fallback by setting ``_use_cartesian_normal = False``
    # per-instance; with it, the sag and the normal are those of before
    # (the sag with rho = sqrt(r**2 + 1e-12)). It is kept for one release
    # as a safety net per the PR-series guidance, and may be removed in a
    # follow-up once Cartesian has bedded in. See issue #617.
    _use_cartesian_normal: bool = True

    r"""Forbes Q2D freeform surface.

    The Q2D surface is defined by a departure $\delta(u, \theta)$ from a base conic:
    $z(\rho, \theta) = z_{base}(\rho) + \frac{1}{\sigma(\rho)} \delta(u, \theta)$

    The departure term $\delta(u, \theta)$ is given by:
    $\delta(u, \theta) = u^2(1-u^2)\sum_{n=0}^{N} a_n^0 Q_n^0(u^2) +
    \sum_{m=1}^{M} u^m \sum_{n=0}^{N} [a_n^m \cos(m\theta) +
    b_n^m \sin(m\theta)] Q_n^m(u^2)$

    Args:
        coordinate_system (CoordinateSystem): The local coordinate system of the
                                                surface.
        surface_config (ForbesSurfaceConfig): An object containing the core geometric
            parameters. The `terms` dictionary should be provided as `freeform_coeffs`.
        solver_config (ForbesSolverConfig, optional): An object containing parameters
        for the numerical solver.

    Notes:
        The `freeform_coeffs` dictionary keys follow the Zemax convention to ensure
        intuitive use for optical designers.
        - Cosine term $a_n^m$: `('a', m, n)`
        - Sine term $b_n^m$: `('b', m, n)`
        Note the index order `(m, n)` matches the Zemax UI, where `m` is the
        azimuthal frequency and `n` is the radial order. The code handles the
        translation to the mathematical convention internally.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        surface_config: ForbesSurfaceConfig,
        solver_config: ForbesSolverConfig = None,
    ):
        super().__init__(coordinate_system, surface_config, solver_config)
        self.c = 1 / self.radius if self.radius != 0 else 0
        if be.get_backend() == "torch":
            self.freeform_coeffs = _CoeffCacheDict(
                {k: be.array(v) for k, v in (self.surface_config.terms or {}).items()}
            )._bind(self)
        else:
            # Preserve the historical numpy aliasing (see ForbesQNormalSlope).
            self.freeform_coeffs = _CoeffCacheDict(
                self.surface_config.terms or {}
            )._bind(self)
            self.surface_config.terms = self.freeform_coeffs

        if self.surface_config.norm_radius is not None:
            self.normalization_mode = "manual"
            self.norm_radius = be.array(self.surface_config.norm_radius)
        else:
            self.normalization_mode = "auto"
            self.norm_radius = be.array(1.0)

        self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = [], [], []
        # Prepare now, so that the coefficient lists exist after construction.
        # _ensure_coeffs prepares under torch.enable_grad(): a surface built
        # inside torch.no_grad() keeps the graph of its coefficient tensors.
        self._ensure_coeffs()

    def _prepare_coeffs(self):
        """
        Translates the user-facing Zemax-style `freeform_coeffs` dictionary
        into the internal coefficient lists required by the `qpoly` backend.

        Also builds the trimmed ``_prepared_*`` containers consumed by the hot
        ``sag``/``_surface_normal`` paths, so trailing-zero trimming (and its
        CUDA synchronization) happens here -- once per coefficient change --
        rather than on every evaluation. Clears the dirty flag.
        """
        if not self.freeform_coeffs:
            self.coeffs_n, self.coeffs_c = [], be.array([])
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = [], [], []
            self._coeff_sources = []
            self._refresh_prepared_coeffs()
            self._watch_coeffs(())
            self._coeffs_dirty = False
            return

        internal_coeffs = {}
        for key, value in self.freeform_coeffs.items():
            term_type, idx1, idx2 = key
            if term_type.lower() == "a":
                n, m = idx2, idx1
                internal_coeffs[(n, m)] = value
            elif term_type.lower() == "b":
                n, m = idx2, idx1
                internal_coeffs[(n, m, "sin")] = value

        sorted_keys = sorted(
            internal_coeffs.keys(),
            key=lambda k: (k[0], abs(k[1]), 0 if len(k) == 2 else 1),
        )

        coeffs_n, coeffs_c = [], []
        for key in sorted_keys:
            value = internal_coeffs[key]
            m_val = -key[1] if len(key) == 3 and key[2].lower() == "sin" else key[1]
            coeffs_n.append((key[0], m_val))
            coeffs_c.append(value)

        self.coeffs_n, self.coeffs_c = (
            coeffs_n,
            be.stack(coeffs_c) if coeffs_c else be.array([]),
        )
        # The coefficients as the dict holds them, in the order of coeffs_n.
        self._coeff_sources = coeffs_c
        if self.coeffs_n:
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = (
                q2d_nm_coeffs_to_ams_bms(self.coeffs_n, self.coeffs_c)
            )

        self._refresh_prepared_coeffs()
        self._watch_coeffs(self.freeform_coeffs.values())
        self._coeffs_dirty = False

    def _refresh_prepared_coeffs(self) -> None:
        """Rebuild the trimmed ``_prepared_*`` coefficient containers.

        Trailing exact-zero trimming is done once here (the ``bool(v == 0)``
        checks force a CUDA sync) so the per-evaluation Clenshaw passes can
        run with ``_no_trim=True``. The trimmed coefficients that can carry a
        derivative go to ``_zero_terms``.
        """
        self._prepared_cm0_coeffs = _trim_trailing_zeros(self.cm0_coeffs)
        self._prepared_ams_coeffs = [_trim_trailing_zeros(a) for a in self.ams_coeffs]
        self._prepared_bms_coeffs = [_trim_trailing_zeros(b) for b in self.bms_coeffs]
        self._zero_terms = None
        sources = getattr(self, "_coeff_sources", [])
        if sources:
            # The same families as cm0_coeffs, ams_coeffs and bms_coeffs, with
            # the coefficients as the dict holds them. An absent coefficient
            # is a float zero and has no derivative to carry.
            cm0, ams, bms = q2d_nm_coeffs_to_ams_bms(self.coeffs_n, sources)
            families = [("m0", 0, cm0, len(self._prepared_cm0_coeffs))]
            for role, groups, prepared in (
                ("cos", ams, self._prepared_ams_coeffs),
                ("sin", bms, self._prepared_bms_coeffs),
            ):
                families.extend(
                    (role, m, family, len(kept))
                    for m, (family, kept) in enumerate(
                        zip(groups, prepared, strict=True), start=1
                    )
                )
            self._zero_terms = zero_terms(families, self.coeffs_c)

    def _q2d_coeffs(self, mode):
        """Return the (cm0, ams, bms) coefficients for an evaluation in ``mode``.

        Args:
            mode (str or None): The result of ``_zero_term_mode``.

        Returns:
            tuple: The m = 0, cosine and sine coefficient families, with the
            trailing zeros for ``KEEP`` and trimmed otherwise.
        """
        if mode == KEEP:
            return self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs
        return (
            self._prepared_cm0_coeffs,
            self._prepared_ams_coeffs,
            self._prepared_bms_coeffs,
        )

    def _polynomial(self, X, Y, usq, slopes):
        """Return P, and dP/dX and dP/dY with ``slopes``, of ``_q2d_cartesian_eval``.

        The evaluation runs on the trimmed families. In reverse mode,
        ``ZeroTerms`` adds the derivative of the trimmed zero coefficients.
        In forward mode and in ``torch.func`` transforms, the evaluation
        takes all coefficients if a trimmed zero coefficient can carry a
        derivative.

        Args:
            X (array_like): The normalized x-coordinates x / Rn.
            Y (array_like): The normalized y-coordinates y / Rn.
            usq (array_like): X**2 + Y**2.
            slopes (bool): Also return dP/dX and dP/dY.

        Returns:
            tuple: P, and dP/dX and dP/dY with ``slopes``.
        """
        mode = self._zero_term_mode()
        values = _q2d_cartesian_eval(
            X, Y, *self._q2d_coeffs(mode), _no_trim=True, slopes=slopes, usq=usq
        )
        values = values if slopes else (values,)
        if mode == REVERSE:
            values = self._zero_terms.attach(
                q2d_cartesian_weights, (X, Y, usq), *values
            )
        return values

    def _q2d_terms(self, x, y, slopes):
        """Return the values that the sag and its slopes share at (x, y).

        Args:
            x (array_like): The x-coordinates.
            y (array_like): The y-coordinates.
            slopes (bool): Also evaluate dP/dX and dP/dY, for the slopes.

        Returns:
            tuple: r**2, u**2 = r**2 / Rn**2, the values of ``_polynomial``,
            the conic correction factor, and the values of ``_conic_terms``.
        """
        self._ensure_coeffs()
        r2 = x**2 + y**2
        usq = r2 / self.norm_radius**2
        polynomial = self._polynomial(
            x / self.norm_radius, y / self.norm_radius, usq, slopes
        )
        factor, conic = self._conic_terms(r2)
        return r2, usq, polynomial, factor, conic

    def _sag_with_terms(self, x, y, slopes):
        """Return the sag at (x, y) and the terms of ``_q2d_terms``."""
        terms = self._q2d_terms(x, y, slopes)
        r2, usq, polynomial, factor, conic = terms
        departure = factor * polynomial[0]
        z_base = self._base_sag(r2, plane=conic is None)
        return z_base + be.where(usq > 1, 0.0, departure), terms

    def sag(self, x, y):
        """Calculate the sag of the Forbes Q2D freeform surface.

        Args:
            x (float or array_like): x-coordinate
            y (float or array_like): y-coordinate

        Returns:
            float or array_like: The sag of the Forbes Q2D freeform surface.
        """
        x, y = be.array(x), be.array(y)
        if not self._use_cartesian_normal:
            return self._sag_polar(x, y)
        return self._sag_with_terms(x, y, slopes=False)[0]

    def _sag_and_terms(self, x, y):
        """Return the sag at (x, y), and the terms that its slopes share.

        The evaluation gives P, dP/dX and dP/dY together, so the slopes in
        the Newton-Raphson loop need no second evaluation of the polynomial.
        See :meth:`NewtonRaphsonGeometry._sag_and_terms`.
        """
        return self._sag_with_terms(x, y, slopes=True)

    def _slopes_from_terms(self, x, y, terms):
        """Return d(sag)/dx and d(sag)/dy from the terms of ``_q2d_terms``.

        The sag is z = z_base + phi P, so
        d(z)/dx = x (c / D + c**2 / (N D**3) P) + phi dP/dX / Rn (see
        ``_slope_parts``), and the same for y. There is no division by r,
        and the slopes at the vertex need no special case. The departure and
        its slope are zero for u**2 > 1, the rule of the sag.

        See :meth:`NewtonRaphsonGeometry._slopes_from_terms`.
        """
        _, usq, (P, dP_dX, dP_dY), factor, conic = terms
        base, factor_slope = self._slope_parts(conic)
        outside = usq > 1
        radial = base
        if factor_slope is not None:
            radial = base + be.where(outside, 0.0, factor_slope * P)
        scale = be.where(outside, 0.0, factor / self.norm_radius)
        return x * radial + scale * dP_dX, y * radial + scale * dP_dY

    def _has_terms(self):
        """Return False for the polar fallback; see ``_use_cartesian_normal``.

        The Newton-Raphson loop then calls ``sag`` and ``_surface_normal``.
        """
        return self._use_cartesian_normal

    def _sag_polar(self, x, y):
        """Return the sag of before, for the polar fallback.

        Args:
            x (array_like): The x-coordinates.
            y (array_like): The y-coordinates.

        Returns:
            array_like: The sag, with rho = sqrt(r**2 + 1e-12).
        """
        self._ensure_coeffs()
        r2 = x**2 + y**2
        z_base = self._base_sag(r2)

        rho = be.sqrt(r2 + _EPSILON)

        u = rho / self.norm_radius
        safe_x = be.where(rho < _EPSILON, x + 1e-12, x)
        theta = be.arctan2(y, safe_x)

        mode = self._zero_term_mode()
        poly_sum_m0, poly_sum_m_gt0 = compute_z_q2d(
            *self._q2d_coeffs(mode), u, theta, _no_trim=True
        )
        if mode == REVERSE:
            poly_sum_m0, poly_sum_m_gt0 = self._zero_terms.attach(
                q2d_sum_weights, (u, y, safe_x), poly_sum_m0, poly_sum_m_gt0
            )
        conic_correction_factor, _ = self._conic_correction_factor(r2)
        usq = u**2
        prefactor_m0 = usq * (1 - usq)

        departure_m0 = prefactor_m0 * conic_correction_factor * poly_sum_m0
        departure_m_gt0 = conic_correction_factor * poly_sum_m_gt0
        total_departure = departure_m0 + departure_m_gt0

        S = be.where(u > 1, 0.0, total_departure)
        return z_base + S

    def _surface_normal(self, x, y):
        """Calculates the unit vector normal to the surface.

        Dispatches to an autograd-based method for the torch backend and an
        analytical method for the numpy backend. For torch, it patches the `NaN`
        gradient from autograd at the vertex with the correct analytical value.

        Args:
            x (float or array_like): X coordinate(s).
            y (float or array_like): Y coordinate(s).

        Returns:
            tuple[float or array_like, float or array_like, float or array_like]:
                Components of the unit normal vector (nx, ny, nz).
        """
        self._ensure_coeffs()
        x_in, y_in = be.array(x), be.array(y)

        df_dx, df_dy = self._surface_normal_analytical(x_in, y_in)

        mag = be.sqrt(df_dx**2 + df_dy**2 + 1)
        safe_mag = be.where(mag < _EPSILON, 1.0, mag)
        return df_dx / safe_mag, df_dy / safe_mag, -1.0 / safe_mag

    def _surface_normal_analytical_vertex(self, ams_coeffs=None, bms_coeffs=None):
        """Computes the stable analytical derivative exactly at the vertex.

        Args:
            ams_coeffs (list, optional): The cosine families. Defaults to the
                trimmed families.
            bms_coeffs (list, optional): The sine families. Defaults to the
                trimmed families.

        Returns:
            tuple: The slopes (df_dx, df_dy) at the vertex.
        """
        if ams_coeffs is None:
            ams_coeffs = self._prepared_ams_coeffs
        if bms_coeffs is None:
            bms_coeffs = self._prepared_bms_coeffs
        df_dx_vertex, df_dy_vertex = 0.0, 0.0
        if ams_coeffs and ams_coeffs[0]:
            a_coeffs = ams_coeffs[0]
            alphas_a = clenshaw_q2d(a_coeffs, m=1, usq=0.0, _no_trim=True)
            sum_a1 = q2d_sum_from_alphas(alphas_a, m=1, num_coeffs=len(a_coeffs))
            df_dx_vertex = sum_a1 / self.norm_radius
        if bms_coeffs and bms_coeffs[0]:
            b_coeffs = bms_coeffs[0]
            alphas_b = clenshaw_q2d(b_coeffs, m=1, usq=0.0, _no_trim=True)
            sum_b1 = q2d_sum_from_alphas(alphas_b, m=1, num_coeffs=len(b_coeffs))
            df_dy_vertex = sum_b1 / self.norm_radius
        return df_dx_vertex, df_dy_vertex

    def _vertex_with_zero_terms(self, df_dx_vertex, df_dy_vertex):
        """Return the vertex slopes with the derivative of the trimmed terms.

        At the vertex, only the family m = 1 has a slope. A slope from an
        empty family is the float 0.0; it becomes a zero tensor, which gives
        the same values in ``be.where``.

        Args:
            df_dx_vertex (float or torch.Tensor): The slope dz/dx.
            df_dy_vertex (float or torch.Tensor): The slope dz/dy.

        Returns:
            tuple: The slopes.
        """
        terms = self._zero_terms
        if not any(family.m == 1 for family in terms.families):
            return df_dx_vertex, df_dy_vertex
        zero = self.coeffs_c.new_zeros(())
        slopes = [
            value if hasattr(value, "detach") else zero
            for value in (df_dx_vertex, df_dy_vertex)
        ]
        return terms.attach(q2d_vertex_weights, (self.norm_radius, zero), *slopes)

    def _surface_normal_analytical_cartesian(self, x_in, y_in):
        """Direct-Cartesian surface-derivative path for ``ForbesQ2dGeometry``.

        Uses :func:`_q2d_cartesian_eval` (harmonic powers + per-m radial
        Clenshaw) to obtain the dimensionless polynomial value and its
        Cartesian derivatives without ever forming a ``1/r`` factor. The
        slopes of the base conic and of the conic correction factor are
        functions of r**2 times x or y (``_slope_parts``), so they need no
        safe rho either. These are the slopes of the sag.

        Returns:
            tuple: ``(df_dx, df_dy)``.
        """
        return self._slopes_from_terms(
            x_in, y_in, self._q2d_terms(x_in, y_in, slopes=True)
        )

    def _surface_normal_analytical(self, x_in, y_in):
        """
        Computes the analytical surface derivatives for the numpy backend.

        Dispatches to the direct-Cartesian harmonic-powers path when
        ``_use_cartesian_normal`` is set (regular at the origin); the
        default polar path with explicit vertex ``where``-blend is
        otherwise used. See class-level docstring on the switch.

        This method is fully vectorized and uses a `where` clause to combine
        the stable vertex calculation with the general non-vertex calculation.

        Args:
            x_in (float or array_like): x-coordinate
            y_in (float or array_like): y-coordinate

        Returns:
            tuple[float or array_like, float or array_like]: The analytical surface
                                                derivatives for the numpy backend.
        """
        if self._use_cartesian_normal:
            return self._surface_normal_analytical_cartesian(x_in, y_in)

        mode = self._zero_term_mode()
        cm0_coeffs, ams_coeffs, bms_coeffs = self._q2d_coeffs(mode)
        df_dx_vertex, df_dy_vertex = self._surface_normal_analytical_vertex(
            ams_coeffs, bms_coeffs
        )
        if mode == REVERSE:
            df_dx_vertex, df_dy_vertex = self._vertex_with_zero_terms(
                df_dx_vertex, df_dy_vertex
            )

        r2 = x_in**2 + y_in**2
        rho = be.sqrt(r2)
        is_vertex = rho < _EPSILON

        rho_safe = be.where(is_vertex, _EPSILON, rho)
        u = rho / self.norm_radius
        theta = be.arctan2(y_in, x_in)

        vals = compute_z_zprime_q2d(
            cm0_coeffs, ams_coeffs, bms_coeffs, u, theta, _no_trim=True
        )
        if mode == REVERSE:
            vals = self._zero_terms.attach(q2d_polar_weights, (u, y_in, x_in), *vals)
        poly_sum_m0, d_poly_m0_du, poly_sum_m_gt0, dr_poly_m_gt0_du, dt_poly_m_gt0 = (
            vals
        )

        d_poly_m0_drho = d_poly_m0_du / self.norm_radius
        dr_poly_m_gt0_drho = dr_poly_m_gt0_du / self.norm_radius
        conic_factor, dconic_d_rho = self._conic_correction_factor(r2)

        usq = u**2
        dprefactor_d_rho = (2 * u - 4 * u**3) / self.norm_radius
        ds0_d_rho = (
            dprefactor_d_rho * poly_sum_m0 + (usq - usq**2) * d_poly_m0_drho
        ) * conic_factor + (usq - usq**2) * poly_sum_m0 * dconic_d_rho

        ds_gt0_d_rho = (dconic_d_rho * poly_sum_m_gt0) + (
            conic_factor * dr_poly_m_gt0_drho
        )
        ds_d_rho = be.where(u > 1, 0.0, ds0_d_rho + ds_gt0_d_rho)
        ds_d_theta = be.where(u > 1, 0.0, conic_factor * dt_poly_m_gt0)

        cos_t, sin_t = x_in / rho_safe, y_in / rho_safe
        ds_dx = cos_t * ds_d_rho - (sin_t / rho_safe) * ds_d_theta
        ds_dy = sin_t * ds_d_rho + (cos_t / rho_safe) * ds_d_theta

        d_base_dr = self._base_sag_derivative(rho, r2)
        d_base_dx, d_base_dy = d_base_dr * cos_t, d_base_dr * sin_t

        df_dx_non_vertex = d_base_dx + ds_dx
        df_dy_non_vertex = d_base_dy + ds_dy

        df_dx = be.where(is_vertex, df_dx_vertex, df_dx_non_vertex)
        df_dy = be.where(is_vertex, df_dy_vertex, df_dy_non_vertex)

        return df_dx, df_dy

    def to_dict(self):
        """Serializes the geometry to a dictionary.

        Returns:
            dict: A dictionary representation of the geometry.
        """
        return {
            "type": self.__class__.__name__,
            "cs": self.cs.to_dict(),
            "surface_config": asdict(self.surface_config),
            "solver_config": asdict(self.solver_config),
        }

    @classmethod
    def from_dict(cls, data):
        """Creates an instance from a dictionary.

        Args:
            data (dict): A dictionary representation of the geometry.

        Returns:
            ForbesQbfsGeometry: An instance of the class.
        """
        cs = CoordinateSystem.from_dict(data["cs"])
        surface_config = ForbesSurfaceConfig(**data["surface_config"])
        solver_config = ForbesSolverConfig(**data.get("solver_config", {}))
        return cls(cs, surface_config, solver_config)

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature and of every Q-2D (freeform)
        coefficient, and updates ``surface_config`` to match. The conic
        constant and the normalization radius remain unchanged.
        """
        self._flip_terms("freeform_coeffs")

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Scales the radius, normalization radius, and freeform coefficients.
        The polynomial coefficients scale linearly with the sag when the
        normalization radius is also scaled.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        self.surface_config.radius = self.radius
        if self.surface_config.norm_radius is not None:
            self.surface_config.norm_radius *= scale_factor
        self.norm_radius = self.norm_radius * scale_factor

        for key in self.freeform_coeffs:
            self.freeform_coeffs[key] = self.freeform_coeffs[key] * scale_factor

        self.c = 1 / self.radius if self.radius != 0 else 0
        # As for the Q surface: the next evaluation prepares the scaled
        # coefficients, under torch.enable_grad() (see _ensure_coeffs). A
        # preparation here, inside torch.no_grad(), would keep the scaled
        # coefficients without a graph, also if they require grad later.
        self._coeffs_dirty = True

    def update_normalization(self, semi_aperture: float) -> None:
        if self.normalization_mode == "auto":
            self.norm_radius = be.array(semi_aperture * 1.25)
            self.surface_config.norm_radius = float(self.norm_radius)

    def __str__(self):
        return "ForbesQ2d"


class ForbesQbfsGeometry(ForbesQNormalSlopeGeometry):
    """Deprecated alias for ForbesQNormalSlopeGeometry.

    .. deprecated::
        Use :class:`ForbesQNormalSlopeGeometry` instead. The name \"Qbfs\" was
        historically used in the Forbes literature for the slope-orthogonal
        Q basis, but it misleadingly suggests a best-fit sphere reference.
        The implementation supports a general conic reference.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        surface_config: ForbesSurfaceConfig,
        solver_config: ForbesSolverConfig = None,
    ):
        warnings.warn(
            "ForbesQbfsGeometry is deprecated; use ForbesQNormalSlopeGeometry "
            "(slope-orthogonal Q basis with a general conic reference).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(coordinate_system, surface_config, solver_config)

    def __str__(self):
        # Keep legacy string for backward compatibility in displays
        return "ForbesQbfs"
