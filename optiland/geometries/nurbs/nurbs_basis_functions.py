"""NURBS basis function evaluation routines."""

from __future__ import annotations

import numba as nb
import numpy as np

from optiland import backend as be

if be.get_backend() == "torch":

    def jit(nopython=True, cache=True):
        """No-op stand-in for numba.jit when running on the torch backend."""

        def decorator(func):
            return func

        return decorator

else:
    jit = nb.jit


@jit(nopython=True, cache=True)
def compute_basis_polynomials(n, p, U, u, return_degree=None):
    """Evaluates the n-th B-Spline basis polynomials of degree ´p´.

    The basis polynomials are computed from their definition by implementing
    equation 2.5 directly from The NURBS Book.

    Args:
        n: The highest index of the basis polynomials (n+1 basis
            polynomials).
        p: The degree of the basis polynomials.
        U: A knot vector of the basis polynomials with shape (r+1=n+p+2,).
            Set the multiplicity of the first and last entries equal to ´p+1´
            to obtain a clamped spline.
        u: The parameter used to evaluate the basis polynomials.
        return_degree: The degree of the returned basis polynomials.

    Returns:
        An array containing the basis polynomials of order ´p´ evaluated at
        ´u´.
    """
    u = np.asarray(u * 1.0)
    Nu = u.size

    m = n + p + 1

    N = np.zeros((p + 1, m, Nu), dtype=u.dtype)

    for i in range(m):
        N[0, i, :] = (
            0.0
            + 1.0 * (np.real(u) >= U[i]) * (np.real(u) < U[i + 1])
            + 1.00 * (np.logical_and(np.real(u) == U[-1], i == n))
        )

    for k in range(1, p + 1):
        m = m - 1
        for i in range(m):
            if (U[i + k] - U[i]) == 0:
                n1 = np.zeros((Nu,), dtype=u.dtype)
            else:
                n1 = (u - U[i]) / (U[i + k] - U[i]) * N[k - 1, i, :]

            if (U[i + k + 1] - U[i + 1]) == 0:
                n2 = np.zeros((Nu,), dtype=u.dtype)
            else:
                n2 = (U[i + k + 1] - u) / (U[i + k + 1] - U[i + 1]) * N[k - 1, i + 1, :]
            N[k, i, ...] = n1 + n2

    N = N[p, 0 : n + 1, :] if return_degree is None else N[return_degree, 0 : n + 1, :]

    return N


@jit(nopython=True, cache=True)
def compute_basis_polynomials_derivatives(n, p, U, u, derivative_order):
    """Evaluates the derivative of the n-th B-Spline basis polynomials.

    The basis polynomials derivatives are computed recursively by implementing
    equations 2.7 and 2.9 directly from The NURBS Book.

    Args:
        n: The highest index of the basis polynomials (n+1 basis
            polynomials).
        p: The degree of the original basis polynomials.
        U: A knot vector of the basis polynomials with shape (r+1=n+p+2,).
            Set the multiplicity of the first and last entries equal to ´p+1´
            to obtain a clamped spline.
        u: The parameter used to evaluate the basis polynomials.
        derivative_order: The order of the basis polynomial derivatives.

    Returns:
        An array containing the basis spline polynomials derivatives
        evaluated at ´u´.
    """
    if derivative_order > p:
        print("The derivative order is higher than the degree of the basis polynomials")

    u = np.asarray(u * 1.0)
    Nu = u.size

    if derivative_order >= 1:
        derivative_order -= 1
        N = compute_basis_polynomials_derivatives(n, p - 1, U, u, derivative_order)
        # The recursive logic for calculating derivatives requires n+2 basis
        # functions of degree p-1 to compute n+1 derivatives of degree p.
        # The following line pads the array of basis functions received from
        # the recursive call to prevent an out-of-bounds error.
        N = np.concatenate((N, np.zeros((1, Nu), dtype=u.dtype)), axis=0)
    elif derivative_order == 0:
        N = compute_basis_polynomials(n, p, U, u)
        return N
    else:
        print(
            "Oooopps, something went wrong in compute_basis_polynomials_derivatives()"
        )
        N = compute_basis_polynomials(n, p, U, u)
        return N

    N_ders = np.zeros((n + 1, Nu), dtype=u.dtype)

    for i in range(n + 1):
        if (U[i + p] - U[i]) == 0:
            n1 = np.zeros(Nu, dtype=u.dtype)
        else:
            n1 = p * N[i, :] / (U[i + p] - U[i])

        if (U[i + p + 1] - U[i + 1]) == 0:
            n2 = np.zeros(Nu, dtype=u.dtype)
        else:
            n2 = p * N[i + 1, :] / (U[i + p + 1] - U[i + 1])

        N_ders[i, :] = n1 - n2

    return N_ders


def basis_function(degree, knot_vector, span, knot):
    """Computes the non-vanishing basis functions for a single parameter.

    This is an implementation of Algorithm A2.2 from The NURBS Book by Piegl &
    Tiller. It uses recurrence to compute the basis functions, also known as
    the Cox - de Boor recursion formula.

    Args:
        degree: The degree, :math:`p`.
        knot_vector: The knot vector, :math:`U`.
        span: The knot span, :math:`i`.
        knot: The knot or parameter, :math:`u`.

    Returns:
        The basis functions.
    """
    left = [0.0 for _ in range(degree + 1)]
    right = [0.0 for _ in range(degree + 1)]
    N = [1.0 for _ in range(degree + 1)]

    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N


def basis_function_one(degree, knot_vector, span, knot):
    """Computes the value of a basis function for a single parameter.

    This is an implementation of Algorithm 2.4 from The NURBS Book by Piegl &
    Tiller.

    Args:
        degree: The degree, :math:`p`.
        knot_vector: The knot vector.
        span: The knot span, :math:`i`.
        knot: The knot or parameter, :math:`u`.

    Returns:
        The basis function, :math:`N_{i,p}`.
    """
    if (span == 0 and knot == knot_vector[0]) or (
        (span == len(knot_vector) - degree - 2)
        and knot == knot_vector[len(knot_vector) - 1]
    ):
        return 1.0

    if knot < knot_vector[span] or knot >= knot_vector[span + degree + 1]:
        return 0.0

    N = [0.0 for _ in range(degree + span + 1)]

    for j in range(0, degree + 1):
        if knot_vector[span + j] <= knot < knot_vector[span + j + 1]:
            N[j] = 1.0

    for k in range(1, degree + 1):
        saved = 0.0
        if N[0] != 0.0:
            saved = ((knot - knot_vector[span]) * N[0]) / (
                knot_vector[span + k] - knot_vector[span]
            )

        for j in range(0, degree - k + 1):
            Uleft = knot_vector[span + j + 1]
            Uright = knot_vector[span + j + k + 1]

            if N[j + 1] == 0.0:
                N[j] = saved
                saved = 0.0
            else:
                temp = N[j + 1] / (Uright - Uleft)
                N[j] = saved + (Uright - knot) * temp
                saved = (knot - Uleft) * temp

    return N[0]
