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
    if isinstance(coefs, (list, tuple)):
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


def change_basis_qbfs_to_pn(cs: list[float]) -> be.array:
    """
    Changes the basis of Q-BFS coefficients to orthonormal Pn coefficients.

    Trailing exact-zero entries are removed before basis conversion;
    they would not contribute to the polynomial sum, and keeping them
    drives the Clenshaw recurrence to needlessly high order.
    """
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


def clenshaw_qbfs(cs: list[float], usq: be.array, alphas: be.array = None):
    """Computes the sum of Q-BFS polynomials using Clenshaw's algorithm.

    Trailing exact-zero coefficients are trimmed before evaluation.
    """
    cs = _trim_trailing_zeros(cs)
    bs = change_basis_qbfs_to_pn(cs)
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


def clenshaw_qbfs_der(cs, usq, j=1, alphas=None):
    """Computes derivatives of Q-BFS polynomials using Clenshaw's method.

    Trailing exact-zero coefficients are trimmed before evaluation. When
    the derivative order ``j`` exceeds the trimmed polynomial degree the
    higher-order alpha tables are returned as zero.
    """
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

    bs = change_basis_qbfs_to_pn(cs)
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


def compute_z_zprime_qbfs(
    coefs: list[float], u: be.array, usq: be.array
) -> tuple[be.array, be.array]:
    """Computes the raw Q-BFS polynomial sum and its derivative w.r.t. u."""
    coefs = _trim_trailing_zeros(coefs)
    if len(coefs) == 0:
        zeros = be.zeros_like(u)
        return zeros, zeros

    alphas = clenshaw_qbfs_der(coefs, usq, j=1)

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


def change_basis_q2d_to_pnm(cns: list[float], m: int) -> be.array:
    """
    Changes the basis of Q2D coefficients to orthonormal Pnm coefficients.
    """
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


def _compute_m_gt0_components(ams, bms, u, t, usq):
    """Computes the sum and derivatives for all m>0 components."""
    poly_sum_terms = []
    dr_terms = []
    dt_terms = []

    for m_idx, (a_coef, b_coef) in enumerate(zip(ams, bms, strict=False)):
        m = m_idx + 1
        # Trim trailing zeros independently per family so the m==1 special
        # case in q2d_sum_from_alphas reads a correctly sized alpha table
        # and does not over index when the user supplied vector ends in
        # zeros.
        a_coef = _trim_trailing_zeros(a_coef)
        b_coef = _trim_trailing_zeros(b_coef)

        s_a, s_b, s_prime_a, s_prime_b = 0, 0, 0, 0
        if a_coef:
            alphas_a = clenshaw_q2d_der(a_coef, m, usq, j=1)
            s_a, s_prime_a = _get_s_and_s_prime(alphas_a, m, len(a_coef))
        if b_coef:
            alphas_b = clenshaw_q2d_der(b_coef, m, usq, j=1)
            s_b, s_prime_b = _get_s_and_s_prime(alphas_b, m, len(b_coef))

        um = u**m
        cost = be.cos(m * t)
        sint = be.sin(m * t)

        poly_sum_terms.append(um * (cost * s_a + sint * s_b))
        umm1 = u ** (m - 1) if m > 0 else be.ones_like(u)
        two_usq = 2 * usq

        aterm = cost * (two_usq * s_prime_a + m * s_a)
        bterm = sint * (two_usq * s_prime_b + m * s_b)
        dr_terms.append(umm1 * (aterm + bterm))
        dt_terms.append(m * um * (-s_a * sint + s_b * cost))

    zeros = be.zeros_like(u)
    poly_sum_m_gt0 = (
        be.sum(be.stack(poly_sum_terms), axis=0) if poly_sum_terms else zeros
    )
    dr_m_gt0 = be.sum(be.stack(dr_terms), axis=0) if dr_terms else zeros
    dt_m_gt0 = be.sum(be.stack(dt_terms), axis=0) if dt_terms else zeros

    return poly_sum_m_gt0, dr_m_gt0, dt_m_gt0


def compute_z_zprime_q2d(cm0, ams, bms, u, t):
    """Computes the polynomial sum components for a Q2D surface."""
    usq = u * u
    zeros = be.zeros_like(u)

    cm0 = _trim_trailing_zeros(cm0)
    poly_sum_m0, d_poly_sum_m0_du = zeros, zeros
    if cm0:
        poly_sum_m0, d_poly_sum_m0_du = compute_z_zprime_qbfs(cm0, u, usq)

    poly_sum_m_gt0, dr_m_gt0, dt_m_gt0 = _compute_m_gt0_components(ams, bms, u, t, usq)

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


def clenshaw_q2d(cns, m, usq, alphas=None):
    """Evaluates the Q2D Clenshaw alpha table for azimuthal order ``m``."""
    cns = _trim_trailing_zeros(cns)
    if be.get_backend() == "torch":
        ds = change_basis_q2d_to_pnm(cns, m)
        all_alphas_list = _clenshaw_q2d_functional(ds, m, usq)

        if not all_alphas_list:
            return _initialize_alphas_q(cns, usq, alphas)

        result_tensor = be.stack(all_alphas_list)
        if alphas is not None:
            alphas[...] = result_tensor
            return alphas
        return result_tensor

    ds = change_basis_q2d_to_pnm(cns, m)
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


def clenshaw_q2d_der(cns, m, usq, j=1, alphas=None):
    """Computes derivatives of Q-2D polynomials using Clenshaw's method."""
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

    ds = change_basis_q2d_to_pnm(cns, m)
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
