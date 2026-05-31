"""
Tools for working with Q (Forbes) polynomials.

code adapted in its majority from the prysm package - (https://github.com/brandondube/prysm)
Manuel Fragata Mendes, 2025

Copyright notice:
Copyright (c) 2017 Brandon Dube
"""

from __future__ import annotations

from collections import defaultdict
from functools import cache

from scipy import special

import optiland.backend as be


def kronecker(i: int, j: int) -> int:
    """The Kronecker delta function."""
    return 1 if i == j else 0


def _trim_trailing_zeros(coefs):
    """Drop trailing exact-zero entries from a coefficient sequence.

    Only *trailing* exact-zero entries are removed;
    interior zeros are preserved. Returns an empty list when all entries
    are zero or the input is empty/``None``.

    Args:
        coefs: A sequence of scalar coefficients.

    Returns:
        list: A list of coefficients with trailing zeros removed.
    """
    if coefs is None:
        return []
    try:
        n = len(coefs)
    except TypeError:
        # Scalar (or 0-d tensor) — wrap in a single-element list, trimming
        # if it is zero.
        try:
            return [] if bool(coefs == 0) else [coefs]
        except Exception:
            return [coefs]
    while n > 0:
        v = coefs[n - 1]
        try:
            is_zero = bool(v == 0)
        except Exception:
            is_zero = False
        if not is_zero:
            break
        n -= 1
    if n == 0:
        return []
    if isinstance(coefs, list | tuple):
        return list(coefs[:n])
    # NumPy array or Torch tensor — keep entries by reference so that any
    # autograd graph attached to surviving entries is preserved.
    return [coefs[i] for i in range(n)]


@cache
def gamma_func(n: int, m: int) -> float:
    """Recursive gamma function for Q2D polynomials."""
    if n == 1 and m == 2:
        return 3 / 8
    if n == 1 and m > 2:
        mm1 = m - 1
        numerator = 2 * mm1 + 1
        denominator = 2 * (mm1 - 1)
        return (numerator / denominator) * gamma_func(1, mm1)

    nm1 = n - 1
    num = (nm1 + 1) * (2 * m + 2 * nm1 - 1)
    den = (m + nm1 - 2) * (2 * nm1 + 1)
    return (num / den) * gamma_func(nm1, m)


# -----------------------------------------------------------------------------
# Forbes slope-orthogonal (Q^bfs) polynomial basis functions
# -----------------------------------------------------------------------------
# NOTE: The "qbfs" suffix in function names below is a historical identifier
# from Forbes' 2007 paper, where these polynomials were called Q^bfs for
# "best-fit sphere." However, the modern formulation (Forbes 2011) uses these
# same polynomial basis functions with a general conic reference surface
# (conic constant k may be nonzero). The "qbfs" naming is retained here for
# code stability and to match the original literature, but users should NOT
# infer that a spherical reference is required or used.
# -----------------------------------------------------------------------------


@cache
def g_qbfs(n_minus_1: int) -> float:
    """Recurrence coefficient g for Q-BFS polynomials."""
    if n_minus_1 == 0:
        return -1 / 2
    n_minus_2 = n_minus_1 - 1
    return -(1 + g_qbfs(n_minus_2) * h_qbfs(n_minus_2)) / f_qbfs(n_minus_1)


@cache
def h_qbfs(n_minus_2: int) -> float:
    """Recurrence coefficient h for Q-BFS polynomials."""
    n = n_minus_2 + 2
    return -n * (n - 1) / (2 * f_qbfs(n_minus_2))


@cache
def f_qbfs(n: int) -> float:
    """Recurrence coefficient f for Q-BFS polynomials."""
    if n == 0:
        return 2.0
    if n == 1:
        return 19**0.5 / 2

    term1 = float(n * (n + 1) + 3)
    term2 = g_qbfs(n - 1) ** 2
    term3 = h_qbfs(n - 2) ** 2

    return (term1 - term2 - term3) ** 0.5


def change_basis_qbfs_to_pn(cs: list[float], _no_trim: bool = False) -> be.array:
    """
    Changes the basis of Q-BFS coefficients to orthonormal Pn coefficients.

    Trailing exact-zero entries are removed before basis conversion;
    they would not contribute to the polynomial sum, and keeping them
    drives the Clenshaw recurrence to needlessly high order.

    Args:
        cs: Q-BFS coefficient sequence.
        _no_trim: When True, ``cs`` is assumed already trimmed (trailing
            zeros removed). This avoids the ``bool(v == 0)`` element checks,
            which force a device-to-host synchronization on CUDA tensors.
    """
    if not _no_trim:
        cs = _trim_trailing_zeros(cs)
    m = len(cs) - 1
    if m < 0:
        return be.array(cs)

    bs_list = [0.0] * (m + 1)

    f_m = f_qbfs(m)
    if not isinstance(f_m, (int | float)):
        cs = be.stack(cs)

    bs_list[m] = cs[m] / f_m
    if m == 0:
        return be.array(bs_list) if be.get_backend() != "torch" else be.stack(bs_list)

    g = g_qbfs(m - 1)
    f = f_qbfs(m - 1)
    bs_list[m - 1] = (cs[m - 1] - g * bs_list[m]) / f

    for i in range(m - 2, -1, -1):
        g = g_qbfs(i)
        h = h_qbfs(i)
        f = f_qbfs(i)
        bs_list[i] = (cs[i] - g * bs_list[i + 1] - h * bs_list[i + 2]) / f

    return be.array(bs_list) if be.get_backend() != "torch" else be.stack(bs_list)


def _initialize_alphas_q(cs, x, alphas, j=0):
    """Initializes the alpha array for Clenshaw's algorithm."""
    if alphas is not None:
        return alphas
    n_modes = max(len(cs), 2)
    shape = (n_modes, *be.shape(x)) if hasattr(x, "shape") else (n_modes,)
    if j != 0:
        shape = (j + 1, *shape)
    zeros = be.zeros(shape)
    if be.get_backend() == "torch":
        zeros.requires_grad = False
    return zeros


def _clenshaw_qbfs_recurrence(bs, usq, alphas):
    """Backend-agnostic Clenshaw recurrence calculation for Q-BFS."""
    m = len(bs) - 1
    if m < 0:
        return alphas

    prefix = 2 - 4 * usq
    alphas[m] = bs[m]
    if m > 0:
        alphas[m - 1] = bs[m - 1] + prefix * alphas[m]
    for i in range(m - 2, -1, -1):
        alphas[i] = bs[i] + prefix * alphas[i + 1] - alphas[i + 2]
    return alphas


def clenshaw_qbfs(
    cs: list[float], usq: be.array, alphas: be.array = None, _no_trim: bool = False
):
    """Computes the sum of Q-BFS polynomials using Clenshaw's algorithm.

    Trailing exact-zero coefficients are trimmed before evaluation unless
    ``_no_trim`` is set (the caller has already prepared a trimmed sequence).
    """
    if not _no_trim:
        cs = _trim_trailing_zeros(cs)
    bs = change_basis_qbfs_to_pn(cs, _no_trim=True)
    m = len(bs) - 1
    if m < 0:
        return be.zeros_like(usq) if hasattr(usq, "shape") else 0.0

    if be.get_backend() == "torch":
        s, _, _ = _clenshaw_qbfs_functional(bs, usq)
        if alphas is not None:
            alphas_res = _clenshaw_qbfs_recurrence(bs, usq, be.empty_like(alphas))
            alphas[...] = alphas_res
        return s

    alphas = _initialize_alphas_q(cs, usq, alphas)
    alphas = _clenshaw_qbfs_recurrence(bs, usq, alphas)
    return 2 * (alphas[0] + alphas[1]) if m > 0 else 2 * alphas[0]


def _clenshaw_qbfs_functional(bs, usq):
    """Pure-functional Clenshaw that returns (S, alpha0, alpha1)."""
    m = len(bs) - 1
    if m < 0:
        zeros = be.zeros_like(usq)
        return zeros, zeros, zeros

    prefix = 2 - 4 * usq
    b_curr = bs[m] + usq * 0
    b_next = be.zeros_like(b_curr)

    for n in range(m - 1, -1, -1):
        b_new = bs[n] + prefix * b_curr - b_next
        b_next, b_curr = b_curr, b_new

    alpha0, alpha1 = b_curr, b_next
    s = 2 * (alpha0 + alpha1) if m > 0 else 2 * alpha0
    return s, alpha0, alpha1


def clenshaw_qbfs_der(cs, usq, j=1, alphas=None, _no_trim: bool = False):
    """Computes derivatives of Q-BFS polynomials using Clenshaw's method.

    Trailing exact-zero coefficients are trimmed before evaluation unless
    ``_no_trim`` is set. When the derivative order ``j`` exceeds the trimmed
    polynomial degree the higher-order alpha tables are returned as zero.
    """
    if not _no_trim:
        cs = _trim_trailing_zeros(cs)
    if be.get_backend() == "torch":
        return _clenshaw_qbfs_der_functional(cs, usq, j)

    m = len(cs) - 1
    alphas = _initialize_alphas_q(cs, usq, alphas, j=j)
    if m < 0:
        return alphas

    clenshaw_qbfs(cs, usq, alphas=alphas[0])

    prefix = 2 - 4 * usq
    for jj in range(1, j + 1):
        if m - jj < 0:
            continue
        alphas[jj][m - jj] = -4 * jj * alphas[jj - 1][m - jj + 1]
        if m - jj - 1 >= 0:
            alphas[jj][m - jj - 1] = (
                prefix * alphas[jj][m - jj] - 4 * jj * alphas[jj - 1][m - jj]
            )
        for n in range(m - jj - 2, -1, -1):
            alphas[jj][n] = (
                prefix * alphas[jj][n + 1]
                - alphas[jj][n + 2]
                - 4 * jj * alphas[jj - 1][n + 1]
            )
    return alphas


def _clenshaw_qbfs_der_functional(cs, usq, j=1):
    """Pure-functional Clenshaw for Q-BFS derivatives (PyTorch backend)."""
    m = len(cs) - 1
    if m < 0:
        shape = (
            (j + 1, len(cs), *be.shape(usq))
            if hasattr(usq, "shape")
            else (j + 1, len(cs))
        )
        return be.zeros(shape)

    bs = change_basis_qbfs_to_pn(cs, _no_trim=True)
    prefix = 2 - 4 * usq

    # functional implementation of the base case (j=0)
    alphas_j0_list = [be.zeros_like(usq) for _ in range(m + 1)]
    if m >= 0:
        #  first scalar coefficient is broadcast to the full
        # tensor size
        alphas_j0_list[m] = bs[m] + be.zeros_like(usq)
    if m >= 1:
        alphas_j0_list[m - 1] = bs[m - 1] + prefix * alphas_j0_list[m]
    for i in range(m - 2, -1, -1):
        alphas_j0_list[i] = (
            bs[i] + prefix * alphas_j0_list[i + 1] - alphas_j0_list[i + 2]
        )

    all_alphas_tensors = [be.stack(alphas_j0_list)]
    prev_alphas_j_list = alphas_j0_list

    for jj in range(1, j + 1):
        alphas_jj_list = [be.zeros_like(usq) for _ in range(m + 1)]
        if m - jj >= 0:
            alphas_jj_list[m - jj] = -4 * jj * prev_alphas_j_list[m - jj + 1]
        if m - jj - 1 >= 0:
            alphas_jj_list[m - jj - 1] = (
                prefix * alphas_jj_list[m - jj] - 4 * jj * prev_alphas_j_list[m - jj]
            )
        for n in range(m - jj - 2, -1, -1):
            alphas_jj_list[n] = (
                prefix * alphas_jj_list[n + 1]
                - alphas_jj_list[n + 2]
                - 4 * jj * prev_alphas_j_list[n + 1]
            )
        all_alphas_tensors.append(be.stack(alphas_jj_list))
        prev_alphas_j_list = alphas_jj_list

    return be.stack(all_alphas_tensors)


def compute_z_qbfs(
    coefs: list[float], usq: be.array, _no_trim: bool = False
) -> be.array:
    """Sag-only Q-BFS polynomial sum (no derivative table built).

    Equivalent to the first return value of :func:`compute_z_zprime_qbfs`
    but skips the j=1 Clenshaw pass entirely. Use this from ``sag()``
    code paths where the derivative is not needed.

    Args:
        coefs: Q-BFS coefficient sequence (trailing zeros are trimmed).
        usq: Squared normalized radius ``u**2``.
        _no_trim: When True, ``coefs`` is assumed already trimmed.

    Returns:
        be.array: The raw Q-BFS polynomial sum at each ``usq`` sample.
    """
    if not _no_trim:
        coefs = _trim_trailing_zeros(coefs)
    if len(coefs) == 0:
        return be.zeros_like(usq) if hasattr(usq, "shape") else be.array(0.0)
    return clenshaw_qbfs(coefs, usq, _no_trim=True)


def compute_z_zprime_qbfs(
    coefs: list[float], u: be.array, usq: be.array, _no_trim: bool = False
) -> tuple[be.array, be.array]:
    """Computes the raw Q-BFS polynomial sum and its derivative w.r.t. u."""
    if not _no_trim:
        coefs = _trim_trailing_zeros(coefs)
    if len(coefs) == 0:
        zeros = be.zeros_like(u)
        return zeros, zeros

    alphas = clenshaw_qbfs_der(coefs, usq, j=1, _no_trim=True)

    if len(coefs) > 1:
        s = 2 * (alphas[0, 0] + alphas[0, 1])
        ds_dusq = 2 * (alphas[1, 0] + alphas[1, 1])
    else:
        s = 2 * alphas[0, 0]
        ds_dusq = 2 * alphas[1, 0]

    ds_du = ds_dusq * 2 * u
    return s, ds_du


# q2d polynomials logic


@cache
def _g_q2d_raw(n: int, m: int) -> float:
    """Raw G coefficient for Q2D polynomials."""
    if n == 0:
        num = special.factorial2(2 * m - 1)
        den = 2 ** (m + 1) * special.factorial(m - 1)
        return num / den
    if n > 0 and m == 1:
        t1num = (2 * n**2 - 1) * (n**2 - 1)
        t1den = 8 * (4 * n**2 - 1)
        term1 = -t1num / t1den
        term2 = 1 / 24 * kronecker(n, 1)
        return term1 - term2

    nt1 = 2 * n * (m + n - 1) - m
    nt2 = (n + 1) * (2 * m + 2 * n - 1)
    num = nt1 * nt2
    dt1 = (m + 2 * n - 2) * (m + 2 * n - 1)
    dt2 = (m + 2 * n) * (2 * n + 1)
    den = dt1 * dt2
    term1 = -num / den
    return term1 * gamma_func(n, m)


@cache
def _f_q2d_raw(n: int, m: int) -> float:
    """Raw F coefficient for Q2D polynomials."""
    if n == 0 and m == 1:
        return 0.25
    if n == 0:
        num = m**2 * special.factorial2(2 * m - 3)
        den = 2 ** (m + 1) * special.factorial(m - 1)
        return num / den
    if n > 0 and m == 1:
        t1num = 4 * (n - 1) ** 2 * n**2 + 1
        t1den = 8 * (2 * n - 1) ** 2
        term1 = t1num / t1den
        term2 = 11 / 32 * kronecker(n, 1)
        return term1 + term2

    chi = m + n - 2
    nt1 = 2 * n * chi * (3 - 5 * m + 4 * n * chi)
    nt2 = m**2 * (3 - m + 4 * n * chi)
    num = nt1 + nt2
    dt1 = (m + 2 * n - 3) * (m + 2 * n - 2)
    dt2 = (m + 2 * n - 1) * (2 * n - 1)
    den = dt1 * dt2
    term1 = num / den
    return term1 * gamma_func(n, m)


@cache
def g_q2d(n: int, m: int) -> float:
    """Recurrence coefficient g for Q2D polynomials."""
    return _g_q2d_raw(n, m) / f_q2d(n, m)


@cache
def f_q2d(n: int, m: int) -> float:
    """Recurrence coefficient f for Q2D polynomials."""
    if n == 0:
        return _f_q2d_raw(n=0, m=m) ** 0.5

    return (_f_q2d_raw(n, m) - g_q2d(n - 1, m) ** 2) ** 0.5


def change_basis_q2d_to_pnm(
    cns: list[float], m: int, _no_trim: bool = False
) -> be.array:
    """
    Changes the basis of Q2D coefficients to orthonormal Pnm coefficients.
    """
    if not _no_trim:
        cns = _trim_trailing_zeros(cns)
    m = abs(m)
    n_max = len(cns) - 1
    if n_max < 0:
        return be.array(cns)

    ds_list = [be.array(0.0)] * (n_max + 1)
    ds_list[n_max] = cns[n_max] / f_q2d(n_max, m)

    for n in range(n_max - 1, -1, -1):
        ds_list[n] = (cns[n] - g_q2d(n, m) * ds_list[n + 1]) / f_q2d(n, m)

    return be.stack(ds_list)


_ABC_Q2D_SPECIAL_CASES = {
    (1, 0): (2, -1, 0),
    (1, 1): (-4 / 3, -8 / 3, -11 / 3),
    (1, 2): (9 / 5, -24 / 5, 0),
    (2, 0): (3, -2, 0),
    (3, 0): (5, -4, 0),
}


@cache
def abc_q2d(n: int, m: int) -> tuple[float, float, float]:
    """Recurrence coefficients A, B, C for Q2D Clenshaw algorithm."""
    d = (4 * n**2 - 1) * (m + n - 2) * (m + 2 * n - 3)
    if d == 0:
        d = 1e-99
    term1 = (2 * n - 1) * (m + 2 * n - 2)
    term2 = 4 * n * (m + n - 2) + (m - 3) * (2 * m - 1)
    a = (term1 * term2) / d
    num_b = -2 * (2 * n - 1) * (m + 2 * n - 3) * (m + 2 * n - 2) * (m + 2 * n - 1)
    b = num_b / d
    num_c = n * (2 * n - 3) * (m + 2 * n - 1) * (2 * m + 2 * n - 3)
    c = num_c / d
    return a, b, c


def abc_q2d_clenshaw(n: int, m: int) -> tuple[float, float, float]:
    """Provides A, B, C coefficients for Clenshaw, handling special cases."""
    return _ABC_Q2D_SPECIAL_CASES.get((m, n), abc_q2d(n, m))


def q2d_sum_from_alphas(alphas: be.array, m: int, num_coeffs: int) -> be.array:
    """
    Computes the final sum from the alpha coefficients returned by Clenshaw's
    method, applying the special summation rule for m=1.

    The m==1 correction reads ``alphas[3]``; the read is also guarded by
    the actual alpha-table length, so a caller passing a stale
    ``num_coeffs`` does not over-index.
    """
    s = 0.5 * alphas[0]
    # special case for m=1, as in Forbes' papers
    if m == 1 and num_coeffs - 1 > 2 and be.shape(alphas)[0] > 3:
        s -= 2 / 5 * alphas[3]
    return s


def _get_s_and_s_prime(alphas, m, num_coeffs):
    """Helper to compute S and S' from alpha derivatives for Q2D."""
    s = q2d_sum_from_alphas(alphas[0], m, num_coeffs)
    s_prime = q2d_sum_from_alphas(alphas[1], m, num_coeffs)
    return s, s_prime


def _compute_m_gt0_components(ams, bms, u, t, usq, _no_trim: bool = False):
    """Computes the sum and derivatives for all m>0 components.

    Per-m terms are accumulated incrementally rather than collected into
    lists and ``be.stack``-ed, which avoids allocating a ``(n_modes, *shape)``
    intermediate per accumulator (a measurable kernel-launch/allocation cost
    on CUDA). The running ``a + b`` adds are out-of-place, so the PyTorch
    autograd graph is preserved.
    """
    poly_sum = dr_sum = dt_sum = None

    for m_idx, (a_coef, b_coef) in enumerate(zip(ams, bms, strict=False)):
        m = m_idx + 1
        # Trim trailing zeros independently per family so the m==1 special
        # case in q2d_sum_from_alphas reads a correctly sized alpha table
        # and does not over index when the user supplied vector ends in
        # zeros. Skipped when the caller passes already-prepared families.
        if not _no_trim:
            a_coef = _trim_trailing_zeros(a_coef)
            b_coef = _trim_trailing_zeros(b_coef)

        s_a, s_b, s_prime_a, s_prime_b = 0, 0, 0, 0
        if a_coef:
            alphas_a = clenshaw_q2d_der(a_coef, m, usq, j=1, _no_trim=True)
            s_a, s_prime_a = _get_s_and_s_prime(alphas_a, m, len(a_coef))
        if b_coef:
            alphas_b = clenshaw_q2d_der(b_coef, m, usq, j=1, _no_trim=True)
            s_b, s_prime_b = _get_s_and_s_prime(alphas_b, m, len(b_coef))

        um = u**m
        cost = be.cos(m * t)
        sint = be.sin(m * t)

        poly_term = um * (cost * s_a + sint * s_b)
        umm1 = u ** (m - 1) if m > 0 else be.ones_like(u)
        two_usq = 2 * usq

        aterm = cost * (two_usq * s_prime_a + m * s_a)
        bterm = sint * (two_usq * s_prime_b + m * s_b)
        dr_term = umm1 * (aterm + bterm)
        dt_term = m * um * (-s_a * sint + s_b * cost)

        poly_sum = poly_term if poly_sum is None else poly_sum + poly_term
        dr_sum = dr_term if dr_sum is None else dr_sum + dr_term
        dt_sum = dt_term if dt_sum is None else dt_sum + dt_term

    zeros = be.zeros_like(u)
    return (
        poly_sum if poly_sum is not None else zeros,
        dr_sum if dr_sum is not None else zeros,
        dt_sum if dt_sum is not None else zeros,
    )


def _harmonic_powers(X, Y, m_max):
    """Compute Re/Im parts of (X + iY)**k for k = 0 .. m_max.

    Iteratively applies the complex-multiplication recurrence

        H_c[k+1] = X * H_c[k] - Y * H_s[k]
        H_s[k+1] = X * H_s[k] + Y * H_c[k]

    Backend-agnostic and singularity-free at the origin (no division
    by ``r``); autograd-safe (purely functional list construction).

    Args:
        X: Normalized x-coordinate (be.array).
        Y: Normalized y-coordinate (be.array).
        m_max: Highest azimuthal order needed (inclusive).

    Returns:
        tuple[list[be.array], list[be.array]]: ``(H_c, H_s)`` each of
            length ``m_max + 1``.
    """
    ones = be.ones_like(X) if hasattr(X, "shape") else be.array(1.0)
    zeros = be.zeros_like(X) if hasattr(X, "shape") else be.array(0.0)
    H_c = [ones]
    H_s = [zeros]
    for _ in range(m_max):
        c_prev, s_prev = H_c[-1], H_s[-1]
        H_c.append(X * c_prev - Y * s_prev)
        H_s.append(X * s_prev + Y * c_prev)
    return H_c, H_s


def _q2d_cartesian_eval(X, Y, cm0, ams, bms, _no_trim: bool = False):
    """Evaluate the Q2D polynomial sum P(X, Y) and its Cartesian derivatives.

    P is the dimensionless polynomial part of the Forbes Q2D departure,

        P(X, Y) = u**2 * (1 - u**2) * S_cm0(u**2)
                + sum_{m>=1} [ Re((X + iY)**m) * S_a_m(u**2)
                             + Im((X + iY)**m) * S_b_m(u**2) ],

    with ``u**2 = X**2 + Y**2``. Derivatives are computed in normalized
    Cartesian coordinates via harmonic powers, so the result is regular
    at ``X = Y = 0`` (no polar ``1/r`` artifact). The caller is
    responsible for applying the conic-correction factor, the base-sag
    derivative, and the chain rule from ``X = x / R_n`` to physical
    coordinates.

    Args:
        X: Normalized x-coordinate (be.array).
        Y: Normalized y-coordinate (be.array).
        cm0: m=0 (Qbfs-style) coefficient sequence; trimmed internally.
        ams: Per-m cosine coefficient families (index ``i`` is m == i+1).
        bms: Per-m sine coefficient families, same layout as ``ams``.

    Returns:
        tuple: ``(P, dP_dX, dP_dY)`` as be.arrays, broadcast to
            ``X`` / ``Y`` shape.
    """
    usq = X * X + Y * Y

    # m == 0 envelope: P_m0 = u^2 (1 - u^2) * S_cm0(u^2)
    if not _no_trim:
        cm0 = _trim_trailing_zeros(cm0)
    if cm0:
        # Reuse derivative Clenshaw to get S and dS/du^2 in one sweep.
        alphas_m0 = clenshaw_qbfs_der(cm0, usq, j=1, _no_trim=True)
        if len(cm0) > 1:
            s_cm0 = 2 * (alphas_m0[0, 0] + alphas_m0[0, 1])
            dsdu2_cm0 = 2 * (alphas_m0[1, 0] + alphas_m0[1, 1])
        else:
            s_cm0 = 2 * alphas_m0[0, 0]
            dsdu2_cm0 = 2 * alphas_m0[1, 0]
        env = usq * (1 - usq)
        P_m0 = env * s_cm0
        # d/dX [ u^2(1-u^2) * S ] = 2X * [ (1 - 2u^2) S + u^2(1-u^2) dS/du^2 ]
        radial_chain = (1 - 2 * usq) * s_cm0 + env * dsdu2_cm0
        dP_m0_dX = 2 * X * radial_chain
        dP_m0_dY = 2 * Y * radial_chain
    else:
        zeros = be.zeros_like(usq)
        P_m0 = zeros
        dP_m0_dX = zeros
        dP_m0_dY = zeros

    # m >= 1: harmonic powers + per-m radial Clenshaw.
    m_max = max(len(ams), len(bms))
    if m_max == 0:
        return P_m0, dP_m0_dX, dP_m0_dY

    H_c, H_s = _harmonic_powers(X, Y, m_max)

    # Accumulate the m>0 contributions incrementally to avoid building three
    # term lists and the corresponding ``be.stack`` / ``be.sum`` intermediates
    # (heavy allocation on CUDA float32 for dense freeforms). Out-of-place adds
    # keep the autograd graph intact.
    P_mgt0 = dPx_mgt0 = dPy_mgt0 = None
    for m_idx in range(m_max):
        m = m_idx + 1
        a_coef = ams[m_idx] if m_idx < len(ams) else []
        b_coef = bms[m_idx] if m_idx < len(bms) else []
        if not _no_trim:
            a_coef = _trim_trailing_zeros(a_coef)
            b_coef = _trim_trailing_zeros(b_coef)
        if not a_coef and not b_coef:
            continue

        s_a = s_b = dsdu2_a = dsdu2_b = 0.0
        if a_coef:
            alphas_a = clenshaw_q2d_der(a_coef, m, usq, j=1, _no_trim=True)
            s_a = q2d_sum_from_alphas(alphas_a[0], m, len(a_coef))
            dsdu2_a = q2d_sum_from_alphas(alphas_a[1], m, len(a_coef))
        if b_coef:
            alphas_b = clenshaw_q2d_der(b_coef, m, usq, j=1, _no_trim=True)
            s_b = q2d_sum_from_alphas(alphas_b[0], m, len(b_coef))
            dsdu2_b = q2d_sum_from_alphas(alphas_b[1], m, len(b_coef))

        Hc_m = H_c[m]
        Hs_m = H_s[m]
        Hc_mm1 = H_c[m - 1]
        Hs_mm1 = H_s[m - 1]

        P_term = Hc_m * s_a + Hs_m * s_b
        # d/dX [ H_c[m] S_a + H_s[m] S_b ]
        #   = m*H_c[m-1]*S_a + H_c[m]*2X*dS_a + m*H_s[m-1]*S_b + H_s[m]*2X*dS_b
        dPx_term = m * (Hc_mm1 * s_a + Hs_mm1 * s_b) + 2 * X * (
            Hc_m * dsdu2_a + Hs_m * dsdu2_b
        )
        # d/dY [ H_c[m] S_a + H_s[m] S_b ]
        #   = -m*H_s[m-1]*S_a + H_c[m]*2Y*dS_a + m*H_c[m-1]*S_b + H_s[m]*2Y*dS_b
        dPy_term = m * (-Hs_mm1 * s_a + Hc_mm1 * s_b) + 2 * Y * (
            Hc_m * dsdu2_a + Hs_m * dsdu2_b
        )

        P_mgt0 = P_term if P_mgt0 is None else P_mgt0 + P_term
        dPx_mgt0 = dPx_term if dPx_mgt0 is None else dPx_mgt0 + dPx_term
        dPy_mgt0 = dPy_term if dPy_mgt0 is None else dPy_mgt0 + dPy_term

    if P_mgt0 is None:
        P_mgt0 = be.zeros_like(usq)
        dPx_mgt0 = be.zeros_like(usq)
        dPy_mgt0 = be.zeros_like(usq)

    return P_m0 + P_mgt0, dP_m0_dX + dPx_mgt0, dP_m0_dY + dPy_mgt0


def _compute_m_gt0_sag_only(ams, bms, u, t, usq, _no_trim: bool = False):
    """Sag-only counterpart of :func:`_compute_m_gt0_components`.

    Skips the derivative Clenshaw pass and the radial / azimuthal
    derivative accumulators; only the m>0 polynomial sum is returned.
    Terms are accumulated incrementally (see
    :func:`_compute_m_gt0_components`).
    """
    poly_sum = None
    for m_idx, (a_coef, b_coef) in enumerate(zip(ams, bms, strict=False)):
        m = m_idx + 1
        if not _no_trim:
            a_coef = _trim_trailing_zeros(a_coef)
            b_coef = _trim_trailing_zeros(b_coef)

        s_a, s_b = 0, 0
        if a_coef:
            alphas_a = clenshaw_q2d(a_coef, m, usq, _no_trim=True)
            s_a = q2d_sum_from_alphas(alphas_a, m, len(a_coef))
        if b_coef:
            alphas_b = clenshaw_q2d(b_coef, m, usq, _no_trim=True)
            s_b = q2d_sum_from_alphas(alphas_b, m, len(b_coef))

        um = u**m
        cost = be.cos(m * t)
        sint = be.sin(m * t)
        poly_term = um * (cost * s_a + sint * s_b)
        poly_sum = poly_term if poly_sum is None else poly_sum + poly_term

    return poly_sum if poly_sum is not None else be.zeros_like(u)


def compute_z_q2d(cm0, ams, bms, u, t, _no_trim: bool = False):
    """Sag-only Q2D polynomial sum (no derivative table built).

    Returns the pair ``(poly_sum_m0, poly_sum_m_gt0)`` — the same first
    and third entries as :func:`compute_z_zprime_q2d` would return, but
    without the j=1 Clenshaw pass over each per-m family. Use this from
    ``sag()`` code paths where the derivative is not needed.

    Args:
        cm0: m==0 Qbfs-style coefficient sequence.
        ams: Per-m cosine coefficient families (index ``i`` is m == i+1).
        bms: Per-m sine coefficient families, same layout as ``ams``.
        u: Normalized radius.
        t: Azimuth in radians.

    Returns:
        tuple: ``(poly_sum_m0, poly_sum_m_gt0)``.
    """
    usq = u * u
    zeros = be.zeros_like(u)

    if not _no_trim:
        cm0 = _trim_trailing_zeros(cm0)
    poly_sum_m0 = zeros if not cm0 else compute_z_qbfs(cm0, usq, _no_trim=True)
    poly_sum_m_gt0 = _compute_m_gt0_sag_only(ams, bms, u, t, usq, _no_trim=_no_trim)
    return poly_sum_m0, poly_sum_m_gt0


def compute_z_zprime_q2d(cm0, ams, bms, u, t, _no_trim: bool = False):
    """Computes the polynomial sum components for a Q2D surface."""
    usq = u * u
    zeros = be.zeros_like(u)

    if not _no_trim:
        cm0 = _trim_trailing_zeros(cm0)
    poly_sum_m0, d_poly_sum_m0_du = zeros, zeros
    if cm0:
        poly_sum_m0, d_poly_sum_m0_du = compute_z_zprime_qbfs(
            cm0, u, usq, _no_trim=True
        )

    poly_sum_m_gt0, dr_m_gt0, dt_m_gt0 = _compute_m_gt0_components(
        ams, bms, u, t, usq, _no_trim=_no_trim
    )

    return poly_sum_m0, d_poly_sum_m0_du, poly_sum_m_gt0, dr_m_gt0, dt_m_gt0


def q2d_nm_coeffs_to_ams_bms(nms: list[tuple[int, int]], coefs: list[float]):
    """Converts a list of (n, m) indexed coefficients to grouped a_m and b_m lists."""
    cms = []
    ac = defaultdict(list)
    bc = defaultdict(list)

    for (n, m), c in zip(nms, coefs, strict=False):
        if m == 0:
            if n >= len(cms):
                cms.extend([0.0] * (n - len(cms) + 1))
            cms[n] = c
            continue

        target_dict = ac if m > 0 else bc
        m_abs = abs(m)
        if n >= len(target_dict[m_abs]):
            target_dict[m_abs].extend([0.0] * (n - len(target_dict[m_abs]) + 1))
        target_dict[m_abs][n] = c

    max_m = 0
    if ac:
        max_m = max(max_m, max(ac.keys()))
    if bc:
        max_m = max(max_m, max(bc.keys()))

    ams_ret = [ac.get(i, []) for i in range(1, max_m + 1)]
    bms_ret = [bc.get(i, []) for i in range(1, max_m + 1)]

    return cms, ams_ret, bms_ret


def clenshaw_q2d(cns, m, usq, alphas=None, _no_trim: bool = False):
    """Evaluates the Q2D Clenshaw alpha table for azimuthal order ``m``."""
    if not _no_trim:
        cns = _trim_trailing_zeros(cns)
    if be.get_backend() == "torch":
        ds = change_basis_q2d_to_pnm(cns, m, _no_trim=True)
        all_alphas_list = _clenshaw_q2d_functional(ds, m, usq)

        if not all_alphas_list:
            return _initialize_alphas_q(cns, usq, alphas)

        result_tensor = be.stack(all_alphas_list)
        if alphas is not None:
            alphas[...] = result_tensor
            return alphas
        return result_tensor

    ds = change_basis_q2d_to_pnm(cns, m, _no_trim=True)
    alphas = _initialize_alphas_q(ds, usq, alphas)
    n_max = len(ds) - 1
    if n_max < 0:
        return alphas

    alphas[n_max] = ds[n_max]
    if n_max > 0:
        a, b, _ = abc_q2d_clenshaw(n_max - 1, m)
        alphas[n_max - 1] = ds[n_max - 1] + (a + b * usq) * alphas[n_max]

    for n in range(n_max - 2, -1, -1):
        a, b, _ = abc_q2d_clenshaw(n, m)
        _, _, c = abc_q2d_clenshaw(n + 1, m)
        alphas[n] = ds[n] + (a + b * usq) * alphas[n + 1] - c * alphas[n + 2]
    return alphas


def _clenshaw_q2d_functional(ds, m, usq):
    """Pure-functional Clenshaw for Q2D polynomials."""
    n_max = len(ds) - 1
    if n_max < 0:
        return []

    all_alphas = [be.zeros_like(usq) for _ in range(n_max + 1)]
    if n_max >= 0:
        all_alphas[n_max] = ds[n_max] + usq * 0
    if n_max >= 1:
        a, b, _ = abc_q2d_clenshaw(n_max - 1, m)
        all_alphas[n_max - 1] = ds[n_max - 1] + (a + b * usq) * all_alphas[n_max]
    for n in range(n_max - 2, -1, -1):
        a, b, _ = abc_q2d_clenshaw(n, m)
        _, _, c = abc_q2d_clenshaw(n + 1, m)
        all_alphas[n] = (
            ds[n] + (a + b * usq) * all_alphas[n + 1] - c * all_alphas[n + 2]
        )
    return all_alphas


def clenshaw_q2d_der(cns, m, usq, j=1, alphas=None, _no_trim: bool = False):
    """Computes derivatives of Q-2D polynomials using Clenshaw's method."""
    if not _no_trim:
        cns = _trim_trailing_zeros(cns)
    if be.get_backend() == "torch":
        return _clenshaw_q2d_der_functional(cns, m, usq, j)

    n_max = len(cns) - 1
    alphas = _initialize_alphas_q(cns, usq, alphas, j=j)
    if n_max < 0:
        return alphas

    clenshaw_q2d(cns, m, usq, alphas[0])
    for jj in range(1, j + 1):
        if n_max - jj < 0:
            continue
        _, b, _ = abc_q2d_clenshaw(n_max - jj, m)
        alphas[jj][n_max - jj] = jj * b * alphas[jj - 1][n_max - jj + 1]
        for n in range(n_max - jj - 1, -1, -1):
            a, b, _ = abc_q2d_clenshaw(n, m)
            _, _, c = abc_q2d_clenshaw(n + 1, m)
            alphas[jj][n] = (
                jj * b * alphas[jj - 1][n + 1]
                + (a + b * usq) * alphas[jj][n + 1]
                - c * alphas[jj][n + 2]
            )
    return alphas


def _clenshaw_q2d_der_functional(cns, m, usq, j=1):
    """Pure-functional Clenshaw for Q-2D derivatives (PyTorch backend)."""
    n_max = len(cns) - 1
    if n_max < 0:
        shape = (
            (j + 1, len(cns), *be.shape(usq))
            if hasattr(usq, "shape")
            else (j + 1, len(cns))
        )
        return be.zeros(shape)

    ds = change_basis_q2d_to_pnm(cns, m, _no_trim=True)
    alphas_j0_list = _clenshaw_q2d_functional(ds, m, usq)
    all_alphas_tensors = [be.stack(alphas_j0_list)]
    prev_alphas_j_list = alphas_j0_list

    for jj in range(1, j + 1):
        alphas_jj_list = [be.zeros_like(usq) for _ in range(n_max + 1)]
        if n_max - jj >= 0:
            _, b, _ = abc_q2d_clenshaw(n_max - jj, m)
            alphas_jj_list[n_max - jj] = jj * b * prev_alphas_j_list[n_max - jj + 1]
            for n in range(n_max - jj - 1, -1, -1):
                a, b, _ = abc_q2d_clenshaw(n, m)
                _, _, c = abc_q2d_clenshaw(n + 1, m)
                alphas_jj_list[n] = (
                    jj * b * prev_alphas_j_list[n + 1]
                    + (a + b * usq) * alphas_jj_list[n + 1]
                    - c * alphas_jj_list[n + 2]
                )
        all_alphas_tensors.append(be.stack(alphas_jj_list))
        prev_alphas_j_list = alphas_jj_list
    return be.stack(all_alphas_tensors)
