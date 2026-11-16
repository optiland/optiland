"""Chebyshev Geometry

The Chebyshev polynomial geometry represents a surface defined by a Chebyshev
polynomial in two dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
    sum(Cij * T_i(x / norm_x) * T_j(y / norm_y))

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Cij are the Chebyshev polynomial coefficients
- T_i(x) is the Chebyshev polynomial of the first kind of degree i
- norm_x and norm_y are normalization factors for the x and y coordinates

Chebyshev polynomials are derived in Cartesian coordinates,
which - unlike many other polynomial freeform surfaces used
to describe rotationally-symmetric systems - allows for
straightforward definition of anamorphic or non-rotationally
symmetric systems and non-elliptical apertures.

Kramer Harrison, 2024
"""

from __future__ import annotations

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries._derivatives import (
    derivative_requested,
    is_transformed,
    reverse_mode_only,
)
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

# Largest broadcast product (in elements) of ``_product_sum`` on a GPU:
# 2**24 values are 128 MiB in float64.
_BROADCAST_LIMIT = 2**24
# Points per part of a matrix product that ``_product_sum`` splits on a CPU.
_POINTS_PER_PART = 2048


class ChebyshevPolynomialGeometry(NewtonRaphsonGeometry):
    """Represents a Chebyshev polynomial geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
        sum(Cij * T_i(x / norm_x) * T_j(y / norm_y))

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Cij are the Chebyshev polynomial coefficients
    - T_i(x) is the Chebyshev polynomial of the first kind of degree i
    - norm_x and norm_y are normalization factors for the x and y coordinates

    The coefficients are defined in a 2D array where coefficients[i][j] is the
    coefficient for T_i(x) * T_j(y).

    Chebyshev polynomials are derived in Cartesian coordinates,
    which - unlike many other polynomial freeform surfaces used
    to describe rotationally-symmetric systems - allows for
    straightforward definition of anamorphic or non-rotationally
    symmetric systems and non-elliptical apertures.

    Args:
        coordinate_system (CoordinateSystem): The coordinate system of the geometry.
        radius (float): The radius of curvature of the base sphere.
        conic (float, optional): The conic constant of the base sphere.
            Defaults to 0.0.
        tol (float, optional): Tolerance for Newton-Raphson iteration.
            Defaults to 1e-10.
        max_iter (int, optional): Maximum iterations for Newton-Raphson.
            Defaults to 100.
        coefficients (list[list[float]] or be.ndarray, optional): A 2D array or
            list of lists representing the Chebyshev coefficients Cij.
            `coefficients[i][j]` is the coefficient for T_i(x) * T_j(y).
            Defaults to an empty list (no polynomial contribution).
        norm_x (float, optional): Normalization radius for the x-coordinate.
            If None, the radius scales automatically during paraxial updates.
            Defaults to None.
        norm_y (float, optional): Normalization radius for the y-coordinate.
            If None, the radius scales automatically during paraxial updates.
            Defaults to None.

    Attributes:
        c (be.ndarray): 2D array of Chebyshev coefficients.
        norm_x (be.ndarray): Normalization factor for x.
        norm_y (be.ndarray): Normalization factor for y.

    """

    def __init__(
        self,
        coordinate_system,
        radius,
        conic=0.0,
        tol=1e-10,
        max_iter=100,
        coefficients=None,
        norm_x=None,
        norm_y=None,
    ):
        if coefficients is None:
            coefficients = []
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.coefficients = be.atleast_2d(coefficients)

        self.normalization_mode = (
            "manual" if (norm_x is not None or norm_y is not None) else "auto"
        )
        self.norm_x = be.array(norm_x if norm_x is not None else 1.0)
        self.norm_y = be.array(norm_y if norm_y is not None else 1.0)

        self.is_symmetric = False

    def __str__(self):
        return "Chebyshev Polynomial"

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        self.norm_x = self.norm_x * scale_factor
        self.norm_y = self.norm_y * scale_factor
        self.coefficients = self.coefficients * scale_factor

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature and of every Chebyshev
        coefficient, so that the sag changes sign. The conic constant remains
        unchanged. The negated coefficients are computed first, so the
        geometry is left unchanged if they cannot be converted (for example
        when they were assigned as a ragged nested list).
        """
        flipped = -be.atleast_2d(self.coefficients)
        super().flip()
        self.coefficients = flipped

    def update_normalization(self, semi_aperture: float) -> None:
        if self.normalization_mode == "auto":
            self.norm_x = semi_aperture * 1.25
            self.norm_y = semi_aperture * 1.25

    def sag(self, x=0, y=0):
        """Calculates the sag of the Chebyshev polynomial surface at the given
        coordinates.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            be.ndarray or float: The sag value(s) at the given coordinates.

        """
        x_norm = x / self.norm_x
        y_norm = y / self.norm_y

        self._validate_inputs(x_norm, y_norm)

        r2 = x**2 + y**2
        z = r2 / (self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))

        if not derivative_requested(self.coefficients):
            for i, j in self._nonzero_terms():
                z = z + self.coefficients[i, j] * self._chebyshev(
                    i, x_norm
                ) * self._chebyshev(j, y_norm)
            return z

        # A derivative can be requested: the zero terms must carry it.
        terms, add_zero_terms = self._derivative_terms()
        if terms:
            tx = self._chebyshev_values(max(i for i, _ in terms), x_norm)
            ty = self._chebyshev_values(max(j for _, j in terms), y_norm)
            for i, j in terms:
                z = z + self.coefficients[i, j] * tx[i] * ty[j]
        if add_zero_terms:
            (z,) = self._with_zero_terms(x_norm, y_norm, z)
        return z

    def _surface_normal(self, x, y):
        """Calculates the surface normal of the Chebyshev polynomial surface at
        the given x and y position.

        Args:
            x (be.ndarray): The x-coordinate(s) at which to calculate the normal.
            y (be.ndarray): The y-coordinate(s) at which to calculate the normal.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The surface normal
            components (nx, ny, nz).

        """
        x_norm = x / self.norm_x
        y_norm = y / self.norm_y

        self._validate_inputs(x_norm, y_norm)

        r2 = x**2 + y**2
        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dzdx = x / denom
        dzdy = y / denom

        # Chain rule: d/dx T_i(x / norm_x) = T_i'(x / norm_x) / norm_x, so the
        # sag and normal derivatives describe the same surface for any
        # normalization.
        if not derivative_requested(self.coefficients):
            for i, j in self._nonzero_terms():
                dzdx = dzdx + (
                    self._chebyshev_derivative(i, x_norm)
                    / self.norm_x
                    * self.coefficients[i, j]
                    * self._chebyshev(j, y_norm)
                )
                dzdy = dzdy + (
                    self._chebyshev_derivative(j, y_norm)
                    / self.norm_y
                    * self.coefficients[i, j]
                    * self._chebyshev(i, x_norm)
                )
        else:
            # A derivative can be requested: the zero terms must carry it.
            terms, add_zero_terms = self._derivative_terms()
            if terms:
                n_x = max(i for i, _ in terms)
                n_y = max(j for _, j in terms)
                tx = self._chebyshev_values(n_x, x_norm)
                ty = self._chebyshev_values(n_y, y_norm)
                dtx = [
                    d / self.norm_x
                    for d in self._chebyshev_derivative_values(n_x, x_norm)
                ]
                dty = [
                    d / self.norm_y
                    for d in self._chebyshev_derivative_values(n_y, y_norm)
                ]
                for i, j in terms:
                    coefficient = self.coefficients[i, j]
                    dzdx = dzdx + (dtx[i] * coefficient * ty[j])
                    dzdy = dzdy + (dty[j] * coefficient * tx[i])
            if add_zero_terms:
                dzdx, dzdy = self._with_zero_terms(x_norm, y_norm, dzdx, dzdy)

        norm = be.sqrt(dzdx**2 + dzdy**2 + 1)
        nx = dzdx / norm
        ny = dzdy / norm
        nz = -1 / norm

        return nx, ny, nz

    def _nonzero_terms(self):
        """Return the degrees (i, j) of the non-zero coefficients, row by row.

        The zero pattern comes from one read of the coefficient values, so a
        GPU waits once per call and not once per index. ``tolist`` can also
        read a tensor inside a ``torch.func`` transform; NumPy cannot.

        Returns:
            list[tuple[int, int]]: The degrees, in the order of the array.
        """
        values = self.coefficients.tolist()
        return [
            (i, j)
            for i, row in enumerate(values)
            for j, value in enumerate(row)
            if value != 0
        ]

    def _derivative_terms(self):
        """Return the terms to add one by one when a derivative can be requested.

        The non-zero terms are added one by one, in the order and with the
        operations of the path without derivatives, so the values stay the
        same. ``_with_zero_terms`` then attaches the derivative of the zero
        terms. Inside a ``torch.func`` transform the coefficient values cannot
        be read, so every term is added one by one.

        Returns:
            tuple[list[tuple[int, int]], bool]: The degrees (i, j) to add one
            by one, and True if the zero terms must also be attached.
        """
        rows, columns = self.coefficients.shape
        if is_transformed(self.coefficients):
            return [(i, j) for i in range(rows) for j in range(columns)], False
        terms = self._nonzero_terms()
        return terms, len(terms) < rows * columns

    def _with_zero_terms(self, x_norm, y_norm, *values):
        """Return the values with the derivative of the zero terms attached.

        The terms of the zero coefficients add nothing, so the values keep
        their bits. Their derivative with respect to the coefficients is
        T_i(x) T_j(y) for the sag, and T_i'(x) T_j(y) / norm_x and
        T_i(x) T_j'(y) / norm_y for the two slopes. The positions enter
        without their graph: each of these coefficients is zero, so the first
        derivatives with respect to the positions are zero. A second
        derivative with respect to a zero coefficient and a position is not
        included.

        In reverse mode, ``_ZeroTermDerivative`` computes the derivative in
        the backward pass only, so the forward pass costs what the path
        without derivatives costs. It has no rules for forward mode and
        ``torch.func`` transforms, so there the zero terms are added as one
        sum, ``_zero_term_sum``, that is exactly zero.

        Args:
            x_norm (be.ndarray): The normalized x-coordinate(s).
            y_norm (be.ndarray): The normalized y-coordinate(s).
            *values (be.ndarray): The sag, or the slopes dz/dx and dz/dy.

        Returns:
            tuple[be.ndarray, ...]: The values, with the derivative attached.
        """
        x_norm, y_norm = x_norm.detach(), y_norm.detach()
        norms = None
        if len(values) == 2:
            norms = tuple(
                norm.detach() if hasattr(norm, "detach") else norm
                for norm in (self.norm_x, self.norm_y)
            )
        if reverse_mode_only():
            return _ZeroTermDerivative.apply(
                self.coefficients, x_norm, y_norm, norms, *values
            )

        rows, columns = self.coefficients.shape
        tx = self._chebyshev_values(rows - 1, x_norm)
        ty = self._chebyshev_values(columns - 1, y_norm)
        if norms is None:
            pairs = [(tx, ty)]
        else:
            dtx = self._chebyshev_derivative_values(rows - 1, x_norm)
            dty = self._chebyshev_derivative_values(columns - 1, y_norm)
            pairs = [
                ([d / norms[0] for d in dtx], ty),
                (tx, [d / norms[1] for d in dty]),
            ]
        zero = self.coefficients * (self.coefficients == 0)
        shape = be.shape(values[0])
        result = []
        for value, (a, b) in zip(values, pairs, strict=True):
            total = self._zero_term_sum(zero, a, b, shape)
            # total - total is +0.0, and value - (+0.0) keeps every bit of
            # the value (value + 0.0 turns -0.0 into +0.0). The expression
            # still carries the derivative of the sum.
            result.append(value - (total.detach() - total))
        return tuple(result)

    @staticmethod
    def _zero_term_sum(zero, a_values, b_values, shape):
        """Return sum_ij zero[i, j] a_i b_j at each point: zero, with a derivative.

        ``_with_zero_terms`` calls this in forward mode and inside
        ``torch.func`` transforms.

        Args:
            zero (be.ndarray): The coefficients, with the non-zero entries set
                to zero. The derivative with respect to a zero entry is one,
                and with respect to a non-zero entry zero, because the
                one-by-one terms carry that derivative.
            a_values (list[be.ndarray]): One array per row degree i.
            b_values (list[be.ndarray]): One array per column degree j.
            shape (tuple[int, ...]): The shape of the evaluation points.

        Returns:
            be.ndarray: The sum, with the given shape.
        """
        a, b = (
            be.reshape(
                be.stack([be.broadcast_to(v, shape) for v in values]),
                (len(values), -1),
            )
            for values in (a_values, b_values)
        )
        return be.reshape(be.sum(a * (zero @ b), axis=0), shape)

    @staticmethod
    def _chebyshev_values(n_max, x):
        """Return T_0(x) to T_n_max(x) from one pass of the recurrence.

        Each value has the same operations, and so the same bits, as
        ``_chebyshev(n, x)``.

        Args:
            n_max (int): The largest degree.
            x (be.ndarray): The normalized coordinate(s).

        Returns:
            list[be.ndarray]: The values T_n(x) for n = 0 to n_max.
        """
        t_prev = be.ones_like(x)
        values = [t_prev]
        if n_max == 0:
            return values
        t_curr = x
        values.append(t_curr)
        two_x = 2.0 * x  # the value that 2.0 * x * t_curr computes first
        for _ in range(n_max - 1):
            t_prev, t_curr = t_curr, two_x * t_curr - t_prev
            values.append(t_curr)
        return values

    @staticmethod
    def _chebyshev_derivative_values(n_max, x):
        """Return T_0'(x) to T_n_max'(x) from one pass of the recurrence.

        Each value has the same operations, and so the same bits, as
        ``_chebyshev_derivative(n, x)``.

        Args:
            n_max (int): The largest degree.
            x (be.ndarray): The normalized coordinate(s).

        Returns:
            list[be.ndarray]: The derivatives dT_n/dx for n = 0 to n_max.
        """
        values = [be.zeros_like(x)]
        if n_max == 0:
            return values
        u_prev = be.ones_like(x)
        values.append(u_prev)
        if n_max == 1:
            return values
        two_x = 2.0 * x  # the value that 2.0 * x * u_curr computes first
        u_curr = two_x
        values.append(2 * u_curr)
        for n in range(3, n_max + 1):
            u_prev, u_curr = u_curr, two_x * u_curr - u_prev
            values.append(n * u_curr)
        return values

    def _chebyshev(self, n, x):
        """Calculates the Chebyshev polynomial of the first kind of degree n at
        the given x value.

        Uses the recurrence ``T_0 = 1``, ``T_1 = x``,
        ``T_n = 2 x T_{n-1} - T_{n-2}``, which -- unlike the
        ``cos(n arccos(x))`` form -- is exact, differentiable and finite over
        the whole valid domain, including the endpoints ``x = +-1`` where the
        ``arccos`` form has an autograd singularity.

        Args:
            n (int): The degree of the Chebyshev polynomial.
            x (be.ndarray or float): The coordinate value(s) (normalized).

        Returns:
            be.ndarray or float: The Chebyshev polynomial T_n(x).

        """
        n = int(n)
        t_prev = be.ones_like(x)
        if n == 0:
            return t_prev
        t_curr = x
        for _ in range(n - 1):
            t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
        return t_curr

    def _chebyshev_derivative(self, n, x):
        """Derivative of the Chebyshev polynomial of the first kind, degree n.

        Uses ``T_n'(x) = n * U_{n-1}(x)`` with the second-kind recurrence
        ``U_0 = 1``, ``U_1 = 2x``, ``U_n = 2 x U_{n-1} - U_{n-2}``. The
        recurrence gives the exact finite endpoint values
        ``T_n'(+-1) = (+-1)^{n-1} n^2``, where the closed form containing
        ``sqrt(1 - x^2)`` produces an autograd ``0/0``.

        The returned value is the derivative with respect to the *normalized*
        coordinate; the caller applies the ``1/norm`` chain-rule factor.

        Args:
            n (int): The degree of the Chebyshev polynomial.
            x (be.ndarray or float): The coordinate value(s) (normalized).

        Returns:
            be.ndarray or float: ``dT_n/dx``. Returns 0 for n=0.

        """
        n = int(n)
        if n == 0:
            return be.zeros_like(x)
        u_prev = be.ones_like(x)
        if n == 1:
            return u_prev
        u_curr = 2.0 * x
        for _ in range(n - 2):
            u_prev, u_curr = u_curr, 2.0 * x * u_curr - u_prev
        return n * u_curr

    def _validate_inputs(self, x_norm, y_norm):
        """Validates the input coordinates for the Chebyshev polynomial surface.

        Args:
            x_norm (be.ndarray or float): The normalized x-coordinate(s).
            y_norm (be.ndarray or float): The normalized y-coordinate(s).

        """
        if be.any(be.abs(x_norm) > 1) or be.any(be.abs(y_norm) > 1):
            raise ValueError(
                "Chebyshev input coordinates must be normalized "
                "to [-1, 1]. Consider updating the normalization "
                "factors.",
            )

    def to_dict(self):
        """Converts the Chebyshev polynomial geometry to a dictionary.

        Returns:
            dict: A dictionary representation of the Chebyshev polynomial geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "coefficients": self.coefficients.tolist(),
                "norm_x": self.norm_x,
                "norm_y": self.norm_y,
            },
        )

        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a Chebyshev polynomial geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the Chebyshev
                polynomial geometry.

        Returns:
            ChebyshevPolynomialGeometry: An instance of
            ChebyshevPolynomialGeometry.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(
            cs,
            data["radius"],
            data.get("conic", 0.0),
            data.get("tol", 1e-10),
            data.get("max_iter", 100),
            data.get("coefficients", []),
            data.get("norm_x", 1),
            data.get("norm_y", 1),
        )


def _chebyshev_table(n_max, x, y, slopes):
    """Return Chebyshev values of x and y as one (n_max + 1, kinds, 2, points) table.

    Row n of kind 0 holds T_n(x) and T_n(y). With ``slopes``, row n of kind 1
    holds U_{n-1}(x) and U_{n-1}(y), with zero in row 0, so that n U_{n-1} is
    T_n'. Both kinds follow the recurrence row_n = 2 x row_{n-1} - row_{n-2};
    only the first two rows differ. The rows are written in place: x and y
    carry no graph, and only the backward pass of ``_ZeroTermDerivative``
    calls this.

    Args:
        n_max (int): The largest degree.
        x (torch.Tensor): The normalized x-coordinates, a flat array.
        y (torch.Tensor): The normalized y-coordinates, a flat array.
        slopes (bool): Add kind 1 for the slopes.

    Returns:
        torch.Tensor: The table.
    """
    table = x.new_empty((n_max + 1, 2 if slopes else 1, 2, x.numel()))
    table[0, 0] = 1.0  # T_0
    table[0, 1:] = 0.0  # T_0' = 0
    if n_max > 0:
        table[1, 0, 0] = x  # T_1
        table[1, 0, 1] = y
        table[1, 1:] = 1.0  # U_0
    two_xy = 2.0 * table[1, 0]
    for n in range(2, n_max + 1):
        torch.mul(two_xy, table[n - 1], out=table[n])
        table[n] -= table[n - 2]
    return table


def _product_sum(a, b):
    """Return the sum over the points of a[i] * b[j], as an (i, j) array.

    Args:
        a (torch.Tensor): A (rows, points) array.
        b (torch.Tensor): A (columns, points) array.

    Returns:
        torch.Tensor: The (rows, columns) array of sums.
    """
    rows, points = a.shape
    if a.device.type == "cpu":
        parts = points // _POINTS_PER_PART
        if parts < 8:
            return a @ b.T
        # A matrix product with a very long inner dimension runs on one
        # core. The products of the parts run in parallel: from 16k points
        # on, this is faster (on 12 cores, 3x faster for 100k points).
        n = parts * _POINTS_PER_PART
        a_parts = a[:, :n].reshape(rows, parts, -1).transpose(0, 1)
        b_parts = b[:, :n].reshape(b.shape[0], parts, -1).permute(1, 2, 0)
        return (a_parts @ b_parts).sum(0) + a[:, n:] @ b[:, n:].T
    if rows * b.shape[0] * points <= _BROADCAST_LIMIT:
        # On a GPU, the library runs a slow kernel for a matrix product with
        # a very long inner dimension (all points): 20 ms for 100k points.
        # A broadcast product and a sum use fast reductions.
        return (a[:, None, :] * b[None, :, :]).sum(-1)
    # For very many points, one product per row needs less memory.
    return torch.stack([(a[i] * b).sum(-1) for i in range(rows)])


if torch is not None:

    class _ZeroTermDerivative(torch.autograd.Function):
        """Copy values; the backward pass adds the derivative of the zero terms.

        See ``ChebyshevPolynomialGeometry._with_zero_terms``. The forward pass
        only copies the values, which keeps their bits, and saves the
        coefficients and the positions (without a graph). The backward pass
        computes the derivative with respect to the zero coefficients from
        them. The non-zero coefficients get no derivative here, because their
        terms carry it. Inputs: the coefficients, the normalized x and y, the
        normalization radii (None for the sag), and the values.
        """

        @staticmethod
        def forward(ctx, coefficients, x, y, norms, *values):
            # The form with ctx in forward costs 6 us per call, the form with
            # setup_context 14 us. Only the second works inside torch.func
            # transforms, but _with_zero_terms uses this function only
            # outside them.
            ctx.norms = norms
            ctx.save_for_backward(coefficients, x, y)
            return tuple(value.clone() for value in values)

        @staticmethod
        def backward(ctx, *gradients):
            coefficients, x, y = ctx.saved_tensors
            rows, columns = coefficients.shape
            shape = gradients[0].shape
            x = x.expand(shape).reshape(-1)
            y = y.expand(shape).reshape(-1)
            slopes = ctx.norms is not None
            table = _chebyshev_table(max(rows, columns) - 1, x, y, slopes)
            t_x, t_y = table[:rows, 0, 0], table[:columns, 0, 1]
            # Out-of-place products with the gradients, so that a backward
            # pass with create_graph=True can differentiate them.
            g = [gradient.reshape(1, -1) for gradient in gradients]
            if not slopes:  # the sag: T_i(x) T_j(y)
                derivative = _product_sum(t_x, t_y * g[0])
            else:  # the slopes: T_i'(x) T_j(y) / norm_x, T_i(x) T_j'(y) / norm_y
                u_x, u_y = table[:rows, 1, 0], table[:columns, 1, 1]
                norm_x, norm_y = ctx.norms
                scale_x = torch.arange(rows, dtype=x.dtype, device=x.device) / norm_x
                scale_y = torch.arange(columns, dtype=x.dtype, device=x.device) / norm_y
                derivative = (
                    _product_sum(u_x, t_y * g[0]) * scale_x[:, None]
                    + _product_sum(t_x, u_y * g[1]) * scale_y
                )
            derivative = torch.where(coefficients == 0, derivative, 0.0)
            return (derivative, None, None, None, *gradients)
