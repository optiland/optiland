"""Derivatives of the zero Forbes coefficients that an evaluation leaves out.

The Forbes geometries remove the trailing zero coefficients of each family
once per coefficient change (``_trim_trailing_zeros``). The Clenshaw sums of
``qpoly`` then evaluate only the coefficients that stay. The values do not
change, because a zero term adds nothing. But the derivative of a sum with
respect to a coefficient is the basis polynomial of that coefficient, also
when the coefficient is zero. Without this derivative, an optimizer cannot
move a coefficient that starts at zero.

A geometry collects the removed coefficients that can carry a derivative
(``zero_terms``). In each evaluation, ``ZeroTerms.mode`` selects one of three
paths:

- None: no derivative can be requested (NumPy, ``torch.no_grad``, tensors
  without requires_grad). The geometry evaluates the trimmed coefficients,
  as before.
- ``REVERSE``: only reverse mode can request a derivative. The geometry
  evaluates the trimmed coefficients, and ``ZeroTerms.attach`` gives the
  outputs of the Clenshaw sums to ``ZeroTermDerivative``. Its forward pass
  returns the outputs unchanged, so the values keep their bits. Its backward
  pass computes the derivative with respect to the removed coefficients. Only
  a backward pass pays for this derivative.
- ``KEEP``: forward mode or a ``torch.func`` transform can request a
  derivative. ``ZeroTermDerivative`` has no rules for these, so the geometry
  evaluates the coefficients with their trailing zeros.

The backward pass. A family sums S(x) = sum_n c_n Q_n(x), with x = u**2. The
Clenshaw sums first change the basis, d = D c (``change_basis_qbfs_to_pn``
and ``change_basis_q2d_to_pnm``), and then sum S(x) = sum_i d_i psi_i(x). The
functions psi_i follow from the recurrence of the sum (``_recurrence``). So
dS/dc_n = Q_n(x) = sum_i D[i, n] psi_i(x). For the weights w of one backward
pass, sum_x w(x) Q_n(x) = sum_i D[i, n] sum_x w(x) psi_i(x): one table of
psi_i, one product with the weights, and one small matrix product. The slope
dpsi_i/dx is a sum of the psi_j too (``_slope_matrix``), so the slope parts
use the same table. The weights come from the gradients of the outputs and
from the factors that the outputs apply to S and dS/dx, for example
u**m cos(m theta).

The positions enter the backward pass without their graph. Each removed
coefficient is zero, so the first derivatives with respect to the positions
do not change. A second derivative with respect to a removed coefficient
and a position is not included.
"""

from __future__ import annotations

import functools

import numpy as np

from optiland.geometries._derivatives import (
    can_carry_derivative,
    derivative_requested,
    reverse_mode_only,
)

from .qpoly import abc_q2d_clenshaw, f_q2d, f_qbfs, g_q2d, g_qbfs, h_qbfs

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

# The paths of ``ZeroTerms.mode``.
REVERSE = "reverse"
KEEP = "keep"


class _Family:
    """The removed coefficients of one family that can carry a derivative.

    Attributes:
        role (str): "m0" for the terms of order m = 0 (Q^bfs basis), "cos"
            or "sin" for the terms of order m >= 1 (Q^m basis).
        m (int): The azimuthal order.
        degrees (tuple[int, ...]): The degrees n of the coefficients.
        size (int): The number of basis functions psi_i that the derivative
            needs: the largest degree plus one.
        rows (torch.Tensor): Row k holds D[i, n] for n = degrees[k] and
            i < size (see ``_rows``).
        slope_rows (torch.Tensor): The rows for the slope dQ_n/dx.
    """

    __slots__ = ("degrees", "m", "role", "rows", "size", "slope_rows")

    def __init__(self, role, m, degrees, like):
        self.role = role
        self.m = m
        self.degrees = degrees
        self.size = degrees[-1] + 1
        self.rows = _rows(role == "m0", m, degrees, False, like.dtype, like.device)
        self.slope_rows = _rows(role == "m0", m, degrees, True, like.dtype, like.device)


def zero_terms(families, like):
    """Collect the removed coefficients that can carry a derivative.

    Args:
        families (Iterable[tuple]): One tuple for each family: the role
            ("m0", "cos" or "sin"), the order m, the coefficients before the
            trimming (tensors, numbers or None for an absent coefficient), and
            the number of coefficients that the trimming kept.
        like (torch.Tensor): A tensor with the dtype and the device of the
            derivatives.

    Returns:
        ZeroTerms or None: None if no removed coefficient can carry a
        derivative, for example on NumPy.
    """
    found, sources = [], []
    for role, m, coefficients, kept in families:
        degrees = tuple(
            n
            for n in range(kept, len(coefficients))
            if can_carry_derivative(coefficients[n])
        )
        if degrees:
            found.append(_Family(role, m, degrees, like))
            sources.extend(coefficients[n] for n in degrees)
    return ZeroTerms(tuple(found), tuple(sources)) if found else None


class ZeroTerms:
    """The removed coefficients of a Forbes geometry that can carry a derivative.

    Args:
        families (tuple[_Family, ...]): The families with such coefficients.
        sources (tuple[torch.Tensor, ...]): The coefficients, family by
            family, as the geometry got them (not the views of the stacked
            coefficients). So the derivative goes to them directly, and the
            gradients of the other coefficients keep their bits.
    """

    __slots__ = ("dtype", "families", "groups", "scalars", "sources")

    def __init__(self, families, sources):
        self.families = families
        self.sources = sources
        # If all coefficients are scalars of one dtype, the backward pass
        # splits its result into their gradients with one call.
        self.dtype = sources[0].dtype
        self.scalars = all(
            source.dim() == 0 and source.dtype == self.dtype for source in sources
        )
        # The families of one basis share one table of basis functions.
        groups = {}
        for index, family in enumerate(families):
            groups.setdefault((family.role == "m0", family.m), []).append(index)
        self.groups = tuple(
            (m0, m, max(families[i].size for i in members), tuple(members))
            for (m0, m), members in groups.items()
        )

    def mode(self):
        """Return how the removed terms get their derivative in this evaluation.

        Returns:
            str or None: ``REVERSE``, ``KEEP``, or None if no derivative with
            respect to a removed coefficient can be requested now.
        """
        if reverse_mode_only():
            if torch.is_grad_enabled() and any(s.requires_grad for s in self.sources):
                return REVERSE
            return None
        if any(derivative_requested(s) for s in self.sources):
            return KEEP
        return None

    def attach(self, weights, saved, *values):
        """Return the values with the derivative of the removed terms attached.

        Args:
            weights (Callable): Gives the variable x of the basis and the
                weights of each family from ``saved`` and the gradients of
                the values, for example ``q2d_sum_weights``.
            saved (tuple): The positions and parameters that ``weights``
                needs. The backward pass uses them without their graph.
                Give values that the graph keeps anyway: a value that only
                this function keeps lives until the graph is freed, and on
                Windows a later large allocation then costs page faults.
            *values (torch.Tensor): The outputs of one evaluation.

        Returns:
            tuple[torch.Tensor, ...]: The values, with the same bits.
        """
        return ZeroTermDerivative.apply(self, weights, saved, *self.sources, *values)


if torch is not None:

    class ZeroTermDerivative(torch.autograd.Function):
        """Return values unchanged; the backward pass adds the zero-term derivative.

        Inputs: the ``ZeroTerms``, the weights function and the saved values
        of ``ZeroTerms.attach``, the removed coefficients, and the values.
        The forward pass returns the values themselves (as views), so the
        bits stay and nothing is copied. The backward pass passes the
        gradients of the values through unchanged, and computes the
        derivative with respect to the removed coefficients (see the module
        docstring).
        """

        @staticmethod
        def forward(ctx, terms, weights, saved, *tensors):
            # The form with ctx in forward costs less per call than the form
            # with setup_context. Only the second works inside torch.func
            # transforms, but ZeroTerms.mode selects this function only
            # outside them. The saved values come before the values in the
            # graph, so keeping them here makes no reference cycle.
            ctx.terms, ctx.weights, ctx.saved = terms, weights, saved
            return tensors[len(terms.sources) :]

        @staticmethod
        def backward(ctx, *gradients):
            count = len(ctx.terms.sources)
            if any(ctx.needs_input_grad[3 : 3 + count]):
                found = _source_gradients(ctx, gradients)
            else:
                found = (None,) * count
            return (None, None, None, *found, *gradients)


def _source_gradients(ctx, gradients):
    """Return the gradients of the removed coefficients for one backward pass.

    Args:
        ctx: The context of ``ZeroTermDerivative``, with the ``ZeroTerms``,
            the weights function and the saved values.
        gradients (tuple[torch.Tensor, ...]): The gradients of the values.

    Returns:
        tuple[torch.Tensor, ...]: One gradient for each removed coefficient.
    """
    terms = ctx.terms
    shape = gradients[0].shape
    # Without their graph, the positions give constants: the products below
    # record nothing, also with create_graph=True, except the products with
    # the gradients.
    saved = tuple(v.detach() if hasattr(v, "detach") else v for v in ctx.saved)
    x, weights = ctx.weights(saved, gradients, terms.families)
    points = _flat(x, shape)
    totals = [None] * len(terms.families)
    for m0, m, size, members in terms.groups:
        table = None
        for i in members:
            if weights[i][0] is None and weights[i][1] is None:
                continue
            if table is None:
                table = _basis_table(m0, m, size, points)
            totals[i] = _family_gradient(terms.families[i], table, weights[i], shape)
    parts = [
        points.new_zeros(len(family.degrees)) if total is None else total
        for family, total in zip(terms.families, totals, strict=True)
    ]
    flat = torch.cat(parts) if len(parts) > 1 else parts[0]
    if terms.scalars and flat.dtype == terms.dtype:
        return flat.unbind()
    return tuple(
        flat[k].reshape(source.shape).to(source.dtype)
        for k, source in enumerate(terms.sources)
    )


def _flat(value, shape):
    """Return ``value`` broadcast to ``shape``, as a flat array."""
    if value.shape != shape:
        value = value.expand(shape)
    return value if value.dim() == 1 else value.reshape(-1)


def _family_gradient(family, table, weights, shape):
    """Return sum_x (w Q_n + w_slope dQ_n/dx) for the degrees n of a family.

    With Q_n = sum_i D[i, n] psi_i and dpsi_i/dx = sum_j A[i, j] psi_j, both
    parts need only sums of the weights times psi_j: ``family.rows`` holds
    D^T and ``family.slope_rows`` holds D^T A for the degrees of the family.
    One matrix-vector product per weight is the fastest sum over the points
    on a CPU and on a GPU (a matrix product with a long inner dimension is
    slow on a GPU).

    Args:
        family (_Family): The family.
        table (torch.Tensor): The (rows, points) table of ``_basis_table``.
        weights (tuple): The weights w and w_slope of the points, with the
            shape of the values, or None.
        shape (torch.Size): The shape of the values.

    Returns:
        torch.Tensor: One sum for each degree of the family.
    """
    if family.size < table.shape[0]:
        table = table[: family.size]
    total = None
    for matrix, weight in zip((family.rows, family.slope_rows), weights, strict=True):
        if weight is None:
            continue
        if matrix.dtype != table.dtype or matrix.device != table.device:
            matrix = matrix.to(dtype=table.dtype, device=table.device)
        # Out-of-place products with the gradients, so that a backward pass
        # with create_graph=True can differentiate them.
        part = matrix @ (table @ _flat(weight, shape))
        total = part if total is None else total + part
    return total


def _recurrence(m0, m, n):
    """Return (a, b, s, e) of psi_n = e + (a + b x) psi_{n-1} + s psi_{n-2}.

    The Clenshaw sums of ``qpoly`` give S(x) = sum_i d_i psi_i(x) for the
    coefficients d after the change of basis. Each sum has the form
    alpha_n = d_n + T[n, n + 1] alpha_{n+1} + T[n, n + 2] alpha_{n+2} and
    S = sum_n e_n alpha_n. So psi = (I - T)^-T e, which is the recurrence
    psi_n = e_n + T[n - 1, n] psi_{n-1} + T[n - 2, n] psi_{n-2}:

    - Q^bfs (m = 0): T[n, n + 1] = 2 - 4 x, T[n, n + 2] = -1 and
      S = 2 (alpha_0 + alpha_1). So psi_0 = 2 and e_1 = 2.
    - Q^m (m >= 1): T[n, n + 1] = A_n + B_n x, T[n, n + 2] = -C_{n+1}
      (``abc_q2d_clenshaw``) and S = alpha_0 / 2, minus 2 alpha_3 / 5 for
      m = 1. So psi_0 = 1/2 and e_3 = -2/5 for m = 1.

    Args:
        m0 (bool): True for the Q^bfs basis, False for the Q^m basis.
        m (int): The azimuthal order of the Q^m basis.
        n (int): The degree, n >= 1.

    Returns:
        tuple[float, float, float, float]: a, b, s and e.
    """
    if m0:
        return 2.0, -4.0, -1.0, 2.0 if n == 1 else 0.0
    a, b, c = abc_q2d_clenshaw(n - 1, m)
    return a, b, -c, -2 / 5 if (m == 1 and n == 3) else 0.0


def _first(m0):
    """Return psi_0, a constant: 2 for Q^bfs and 1/2 for Q^m."""
    return 2.0 if m0 else 0.5


@functools.cache
def _recurrences(m0, m, size):
    """Return ``_recurrence(m0, m, n)`` for n = 1 to size - 1."""
    return tuple(_recurrence(m0, m, n) for n in range(1, size))


def _basis_table(m0, m, size, x):
    """Return psi_i(x) for i < size as a (size, points) table.

    See ``_recurrence``. Only the backward pass of ``ZeroTermDerivative``
    calls this function, with x without a graph.

    Args:
        m0 (bool): True for the Q^bfs basis, False for the Q^m basis.
        m (int): The azimuthal order of the Q^m basis.
        size (int): The number of basis functions.
        x (torch.Tensor): The variable u**2 of the basis, a flat array.

    Returns:
        torch.Tensor: The table.
    """
    rows = [x.new_full(x.shape, _first(m0))]
    if m0 and size > 1:
        # a and b do not depend on n: t = 2 - 4 x serves all rows.
        t = 2.0 - 4.0 * x
        rows.append(t * 2.0 + 2.0)
        for _ in range(2, size):
            rows.append(torch.mul(t, rows[-1]).sub_(rows[-2]))
    elif not m0:
        for a, b, s, e in _recurrences(m0, m, size):
            row = torch.mul(rows[-1], a).addcmul_(x, rows[-1], value=b)
            if len(rows) >= 2:
                row.add_(rows[-2], alpha=s)
            if e:
                row += e
            rows.append(row)
    return torch.stack(rows)


@functools.cache
def _slope_matrix(m0, m, size):
    """Return A with dpsi_n/dx = sum_j A[n, j] psi_j for n < size.

    The derivative of the recurrence (``_recurrence``) is
    dpsi_n/dx = b psi_{n-1} + (a + b x) dpsi_{n-1}/dx + s dpsi_{n-2}/dx. In
    the basis psi, the product with x follows from the recurrence too:
    x psi_j = (psi_{j+1} - e_{j+1} - a psi_j - s psi_{j-1}) / b, with the
    coefficients of degree j + 1, and the constant 1 = psi_0 / psi_0(x).

    Args:
        m0 (bool): True for the Q^bfs basis, False for the Q^m basis.
        m (int): The azimuthal order of the Q^m basis.
        size (int): The number of basis functions.

    Returns:
        tuple[tuple[float, ...], ...]: The rows of A.
    """
    first = _first(m0)

    def times_x(row):
        product = [0.0] * size
        for j, value in enumerate(row):
            if value == 0.0:
                continue
            a, b, s, e = _recurrence(m0, m, j + 1)
            product[j + 1] += value / b
            product[0] -= value * e / (first * b)
            product[j] -= value * a / b
            if j >= 1:
                product[j - 1] -= value * s / b
        return product

    rows = [[0.0] * size]
    for n in range(1, size):
        a, b, s, _ = _recurrence(m0, m, n)
        shifted = times_x(rows[n - 1])
        row = [a * p + b * q for p, q in zip(rows[n - 1], shifted, strict=True)]
        row[n - 1] += b
        if n >= 2:
            row = [r + s * p for r, p in zip(row, rows[n - 2], strict=True)]
        rows.append(row)
    return tuple(tuple(row) for row in rows)


@functools.cache
def _q_in_basis(m0, m, n):
    """Return D[i, n] for i <= n: the coefficients of Q_n in the basis psi_i.

    This is the change of basis of ``change_basis_qbfs_to_pn`` or
    ``change_basis_q2d_to_pnm`` for the coefficients c = e_n. Q_n does not
    depend on the number of coefficients after n.

    Args:
        m0 (bool): True for the Q^bfs basis, False for the Q^m basis.
        m (int): The azimuthal order of the Q^m basis.
        n (int): The degree.

    Returns:
        tuple[float, ...]: The n + 1 coefficients.
    """
    d = [0.0] * (n + 1)
    if m0:
        d[n] = 1.0 / f_qbfs(n)
        for i in range(n - 1, -1, -1):
            later = g_qbfs(i) * d[i + 1]
            if i + 2 <= n:
                later += h_qbfs(i) * d[i + 2]
            d[i] = -later / f_qbfs(i)
    else:
        d[n] = 1.0 / f_q2d(n, m)
        for i in range(n - 1, -1, -1):
            d[i] = -g_q2d(i, m) * d[i + 1] / f_q2d(i, m)
    return tuple(float(v) for v in d)


@functools.cache
def _rows(m0, m, degrees, slopes, dtype, device):
    """Return the rows D^T (or D^T A with ``slopes``) for ``degrees``.

    Row k of D^T holds D[i, n] for n = degrees[k], so that
    sum_x w Q_n = sum_i D[i, n] sum_x w psi_i. Row k of D^T A gives
    sum_x w dQ_n/dx in the same way (see ``_slope_matrix``).

    Args:
        m0 (bool): True for the Q^bfs basis, False for the Q^m basis.
        m (int): The azimuthal order of the Q^m basis.
        degrees (tuple[int, ...]): The degrees n, in rising order.
        slopes (bool): Return D^T A instead of D^T.
        dtype (torch.dtype): The dtype of the matrix.
        device (torch.device): The device of the matrix.

    Returns:
        torch.Tensor: The (len(degrees), max(degrees) + 1) matrix.
    """
    size = degrees[-1] + 1
    rows = np.array(
        [list(_q_in_basis(m0, m, n)) + [0.0] * (size - n - 1) for n in degrees]
    )
    if slopes:
        rows = rows @ np.array(_slope_matrix(m0, m, size))
    return torch.tensor(rows, dtype=dtype, device=device)


def _harmonic_powers(X, Y, m_max):
    """Return the real and imaginary parts of (X + iY)**k for k <= m_max.

    The backward pass needs the same values as ``qpoly._harmonic_powers``,
    but without Optiland's grad mode (``be.ones_like`` can make a tensor
    that requires grad): these values are constants here. The powers 0 and
    1 are the numbers 1 and 0 and the arrays X and Y themselves.

    Args:
        X (torch.Tensor): The normalized x-coordinates.
        Y (torch.Tensor): The normalized y-coordinates.
        m_max (int): The largest power.

    Returns:
        tuple[list, list]: The real and the imaginary parts.
    """
    h_c, h_s = [1.0, X], [0.0, Y]
    for _ in range(1, m_max):
        c, s = h_c[-1], h_s[-1]
        h_c.append(torch.addcmul(X * c, Y, s, value=-1))
        h_s.append(torch.addcmul(X * s, Y, c))
    return h_c, h_s


# Weights of the outputs. Each function gets the saved positions, the
# gradients of the outputs and the families, and returns the variable x of
# the basis and one pair (w, w_slope) for each family: the derivative of the
# outputs with respect to a coefficient c_n of the family is
# w Q_n(x) + w_slope dQ_n/dx, summed over the outputs with the gradients.
# A pair (None, None) means that the outputs do not depend on the family.


def sum_weights(saved, gradients, families):
    """Weights for (S,) of ``clenshaw_qbfs``: one m = 0 family; saved (x,)."""
    (x,) = saved
    (gradient,) = gradients
    return x, [(gradient, None)]


def sum_and_slope_weights(saved, gradients, families):
    """Weights for (S, dS/du) of ``compute_z_zprime_qbfs`` with x = u**2; saved (u,).

    dS/du = 2 u dS/dx.
    """
    (u,) = saved
    gradient, gradient_u = gradients
    return u**2, [(gradient, gradient_u * (2 * u))]


def q2d_sum_weights(saved, gradients, families):
    """Weights for (S_0, P) of ``compute_z_q2d``; saved (u, y, x).

    P = sum_m u**m (cos(m theta) S_a + sin(m theta) S_b), with
    theta = atan2(y, x).
    """
    u, y, x = saved
    theta = torch.atan2(y, x)
    gradient_m0, gradient = gradients
    weights = []
    for family in families:
        if family.role == "m0":
            weights.append((gradient_m0, None))
            continue
        m = family.m
        angle = theta if m == 1 else m * theta
        h = torch.cos(angle) if family.role == "cos" else torch.sin(angle)
        weights.append((gradient * (u * h if m == 1 else u**m * h), None))
    return u * u, weights


def q2d_polar_weights(saved, gradients, families):
    """Weights for the five outputs of ``compute_z_zprime_q2d``; saved (u, y, x).

    The outputs are S_0, dS_0/du, P, dP/du and dP/dtheta, with
    P = sum_m u**m (cos(m theta) S_a + sin(m theta) S_b) and
    theta = atan2(y, x).
    """
    u, y, x = saved
    theta = torch.atan2(y, x)
    gradient_m0, gradient_m0_u, gradient, gradient_u, gradient_theta = gradients
    weights = []
    x = u * u
    two_u = 2 * u
    factors = [_polar_harmonic(family, u, theta) for family in families]
    for family, (value, radial, radial_slope, angular) in zip(
        families, factors, strict=True
    ):
        if family.role == "m0":
            weights.append((gradient_m0, gradient_m0_u * two_u))
            continue
        weights.append(
            (
                gradient * value + gradient_u * radial + gradient_theta * angular,
                gradient_u * radial_slope,
            )
        )
    return x, weights


def _polar_harmonic(family, u, theta):
    """Return the factors of S and dS/dx in the polar outputs of one family.

    For a family of order m with h = cos(m theta) ("cos") or sin(m theta)
    ("sin"), the outputs are P = u**m h S, dP/du = u**(m-1) h (2 x dS/dx +
    m S) and dP/dtheta = m u**m h' S, with h' = dh/dtheta / m.

    Args:
        family (_Family): The family.
        u (torch.Tensor): The normalized radius.
        theta (torch.Tensor): The azimuth.

    Returns:
        tuple: The factors of S in P, of S in dP/du, of dS/dx in dP/du, and
        of S in dP/dtheta. All are None for the m = 0 family.
    """
    if family.role == "m0":
        return None, None, None, None
    m = family.m
    h, h_turned = _harmonic(family, theta)
    power = u**m
    lower = u ** (m - 1) * h
    return power * h, m * lower, 2 * (u * u) * lower, m * power * h_turned


def _harmonic(family, theta):
    """Return h and dh/dtheta / m for h = cos(m theta) or sin(m theta).

    Args:
        family (_Family): A family of order m >= 1.
        theta (torch.Tensor): The azimuth.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: h and its turned function.
    """
    angle = family.m * theta
    if family.role == "cos":
        return torch.cos(angle), -torch.sin(angle)
    return torch.sin(angle), torch.cos(angle)


def q2d_cartesian_weights(saved, gradients, families):
    """Weights for (P, dP/dX, dP/dY) of ``_q2d_cartesian_eval``; saved (X, Y).

    P = x (1 - x) S_0 + sum_m (Re (X + iY)**m S_a + Im (X + iY)**m S_b),
    with x = X**2 + Y**2.
    """
    X, Y = saved
    gradient, gradient_x, gradient_y = gradients
    x = torch.addcmul(X * X, Y, Y)
    h_c, h_s = _harmonic_powers(X, Y, max(family.m for family in families))
    # d/dX of a function of x is 2 X d/dx, so the slope weights of all
    # families share this factor.
    radial = 2 * torch.addcmul(gradient_x * X, gradient_y, Y)
    weights = []
    for family in families:
        m = family.m
        if family.role == "m0":
            envelope = x - x * x
            weight = torch.addcmul(gradient * envelope, radial, 1 - 2 * x)
            weights.append((weight, radial * envelope))
            continue
        # cos: d/dX Re(X + iY)**m = m Re(X + iY)**(m-1) and
        # d/dY Re(X + iY)**m = -m Im(X + iY)**(m-1); sin: m Im and m Re.
        if family.role == "cos":
            h, lower_x, lower_y, sign = h_c[m], h_c[m - 1], h_s[m - 1], -1
        else:
            h, lower_x, lower_y, sign = h_s[m], h_s[m - 1], h_c[m - 1], 1
        if m == 1:
            # The lower powers are the numbers 1 and 0.
            weight = gradient * h + (gradient_x if sign < 0 else gradient_y)
        else:
            weight = torch.addcmul(gradient * h, gradient_x, lower_x, value=m)
            weight = torch.addcmul(weight, gradient_y, lower_y, value=sign * m)
        weights.append((weight, radial * h))
    return x, weights


def q2d_vertex_weights(saved, gradients, families):
    """Weights for the slopes (dz/dx, dz/dy) at the vertex of the polar path.

    At the vertex, dz/dx = S_a(0) / R and dz/dy = S_b(0) / R for the family
    m = 1 (``ForbesQ2dGeometry._surface_normal_analytical_vertex``); saved
    (R, x), with x a zero.
    """
    norm_radius, x = saved
    gradient_x, gradient_y = gradients
    weights = []
    for family in families:
        if family.m != 1:
            weights.append((None, None))
        elif family.role == "cos":
            weights.append((gradient_x / norm_radius, None))
        else:
            weights.append((gradient_y / norm_radius, None))
    return x, weights
