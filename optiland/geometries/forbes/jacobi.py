"""
High performance / recursive jacobi polynomial calculation.

code adapted in its majority from the prysm package - (https://github.com/brandondube/prysm)
Manuel Fragata Mendes, 2025

Copyright notice:
Copyright (c) 2017 Brandon Dube
"""

from __future__ import annotations

from functools import lru_cache

import optiland.backend as be


def weight(alpha, beta, x):
    """The weight function of the jacobi polynomials for a given alpha, beta value."""
    return (1 - x) ** alpha * (1 + x) ** beta


@lru_cache(512)
def recurrence_abc(n, alpha, beta):
    """See A&S online - https://dlmf.nist.gov/18.9 .

    Pn = (an-1 x + bn-1) Pn-1 - cn-1 * Pn-2

    This function makes a, b, c for the given n,
    i.e. to get a(n-1), do recurrence_abc(n-1)

    """
    aplusb = alpha + beta
    if n == 0 and (aplusb == 0 or aplusb == -1):
        A = 1 / 2 * (alpha + beta) + 1
        B = 1 / 2 * (alpha - beta)
        C = 1
    else:
        Anum = (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2)
        Aden = 2 * (n + 1) * (n + alpha + beta + 1)
        A = Anum / Aden

        Bnum = (alpha**2 - beta**2) * (2 * n + alpha + beta + 1)
        Bden = 2 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta)
        B = Bnum / Bden

        Cnum = (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2)
        Cden = (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta)
        C = Cnum / Cden

    return A, B, C


def jacobi(n, alpha, beta, x):
    """Jacobi polynomial of order n with weight parameters alpha and beta."""
    if n == 0:
        return be.ones_like(x)
    elif n == 1:
        term1 = alpha + 1
        term2 = alpha + beta + 2
        term3 = (x - 1) / 2
        return term1 + term2 * term3

    Pnm1 = alpha + 1 + (alpha + beta + 2) * ((x - 1) / 2)
    A, B, C = recurrence_abc(1, alpha, beta)
    Pn = (A * x + B) * Pnm1 - C
    if n == 2:
        return Pn

    for i in range(3, n + 1):
        Pnm2, Pnm1 = Pnm1, Pn
        A, B, C = recurrence_abc(i - 1, alpha, beta)
        Pn = (A * x + B) * Pnm1 - C * Pnm2

    return Pn


def jacobi_sum_clenshaw(s, alpha, beta, x, alphas=None):
    """Compute a weighted sum of Jacobi polynomials using Clenshaw's method."""
    alphas = _initialize_alphas(s, x, alphas)
    M = len(s) - 1
    if M < 0:
        return alphas
    alphas[M] = s[M]
    if M == 0:
        return alphas

    a, b, c = recurrence_abc(M - 1, alpha, beta)
    alphas[M - 1] = s[M - 1] + (a * x + b) * s[M]
    for n in range(M - 2, -1, -1):
        a, b, _ = recurrence_abc(n, alpha, beta)
        _, _, c = recurrence_abc(n + 1, alpha, beta)
        alphas[n] = s[n] + (a * x + b) * alphas[n + 1] - c * alphas[n + 2]

    return alphas


def jacobi_sum_clenshaw_der(s, alpha, beta, x, j=1, alphas=None):
    """Compute a weighted sum of partial derivatives w.r.t. x of Jacobi polynomials."""
    alphas = _initialize_alphas(s, x, None, j=j)
    M = len(s) - 1
    if M < 0:
        return alphas

    jacobi_sum_clenshaw(s, alpha, beta, x, alphas=alphas[0])
    for jj in range(1, j + 1):
        if M - jj < 0:
            continue
        a, *_ = recurrence_abc(M - jj, alpha, beta)
        alphas[jj][M - jj] = j * a * alphas[jj - 1][M - jj + 1]
        for n in range(M - jj - 1, -1, -1):
            a, b, _ = recurrence_abc(n, alpha, beta)
            _, _, c = recurrence_abc(n + 1, alpha, beta)
            alphas[jj][n] = (
                jj * a * alphas[jj - 1][n + 1]
                + (a * x + b) * alphas[jj][n + 1]
                - c * alphas[jj][n + 2]
            )

    return alphas


def _initialize_alphas(s, x, alphas, j=0):
    if alphas is None:
        shape = (len(s), *be.shape(x)) if hasattr(x, "shape") else (len(s),)
        if j != 0:
            shape = (j + 1, *shape)
        if be.get_backend() == "torch":
            import torch

            prec = be.get_precision()
            dtype = torch.float32 if prec == 32 else torch.float64
            # Ensure alphas doesn't require grad to allow in-place ops
            alphas = torch.zeros(
                shape,
                device=be.get_device(),
                dtype=dtype,
                requires_grad=False,
            )
        else:
            alphas = be.zeros(shape)
    return alphas
