"""Polynomial XY Geometry

The Polynomial XY geometry represents a surface defined by a polynomial in two
dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Cij * x^i * y^j)

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Cij are the polynomial coefficients

The coefficients are defined in a 2D array where coefficients[i][j] is the
coefficient for x^i * y^j.

Historically, XY-polynomials were the first type of polynomials used
for low-order freeform surfaces (see https://doi.org/10.1364/AO.38.003572).
These polynomials remain common surface descriptors of freeform surfaces;
however their lack of orthogonality renders them less desirable than
their orthogonal counterparts for highly corrected imaging systems.

The polynomial uses nested Horner sums. Horner's method in y gives the row
values P_i(y) = sum_j Cij * y^j, and Horner's method in x then gives
sum_i x^i * P_i(y). The slopes come from the same row values:
dz/dx = sum_i i * x^(i - 1) * P_i(y), and dz/dy = sum_i x^i * P_i'(y), where
Horner's method gives P_i'(y) from the coefficients j * Cij. On the points,
only + - * / operate, with sequential sums, so a compiled kernel can repeat
the expressions.

Kramer Harrison, 2024
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

# With a 2-D tensor of coefficients, one Horner pass in y can give the values
# of all rows at once: each operation then acts on (rows x points) values.
# This needs fewer operations than one pass for each row, so it is faster for
# small bundles. For large bundles, its large arrays cost more time: on a CPU
# they get new memory pages from the operating system, and on a GPU they do
# not stay in the cache. Above these numbers of values, each row gets its own
# pass. In measurements (Ryzen 9 9900X, RTX 5070, Windows), the pass for all
# rows was faster up to 480k values on the CPU and up to 3M values on the GPU,
# and slower from 750k and 5M values. The two forms do the same operations on
# each value, so they give the same bits.
_ALL_ROWS_LIMIT_CPU = 2**19
_ALL_ROWS_LIMIT_GPU = 2**22


class _Coefficients(NamedTuple):
    """The polynomial coefficients of one evaluation.

    Attributes:
        matrix: The coefficients as a 2-D tensor, or None.
        rows: If ``matrix`` is None, the coefficients of each row.
        sparse: True if ``rows`` holds numbers. Then the rows have no
            trailing zero coefficients and no trailing rows of zeros, and
            the Horner sums leave out the zero coefficients. A number has no
            derivative, so this loses none. Tensors keep all their terms, so
            the derivative with respect to a zero coefficient Cij is
            x^i * y^j.
        num_rows: The number of rows.
        num_columns: The length of the longest row.
    """

    matrix: Any
    rows: Sequence[Any] | None
    sparse: bool
    num_rows: int
    num_columns: int


# The coefficients of a polynomial without terms.
_NO_TERMS = _Coefficients(None, (), True, 0, 0)


class _PolynomialTerms(NamedTuple):
    """The values at the points that the sag and the slopes share.

    Attributes:
        r2: x^2 + y^2.
        root: sqrt(1 - (1 + k) * r^2 / R^2) of the base conic.
        coefficients: The coefficients of this evaluation.
        rows: P_i(y) of each row i, or None for a row without terms.
    """

    r2: Any
    root: Any
    coefficients: _Coefficients
    rows: list[Any]


def _number_rows(rows: list[list[Any]]) -> _Coefficients:
    """Return rows of numbers without trailing zero coefficients and rows.

    The function removes the zeros from the lists in ``rows``, so these lists
    must be new lists.
    """
    width = 0
    for row in rows:
        end = len(row)
        while end and row[end - 1] == 0:
            end -= 1
        del row[end:]
        if end > width:
            width = end
    while rows and not rows[-1]:
        rows.pop()
    if not rows:
        return _NO_TERMS
    return _Coefficients(None, rows, True, len(rows), width)


def _prepare(coefficients: Any) -> _Coefficients:
    """Prepare the coefficients for one evaluation.

    The preparation reads no values from a device. A NumPy array gives its
    values as Python numbers, and a tensor stays a tensor.

    Args:
        coefficients: The coefficients of the geometry: a 2-D array or
            tensor, or a list of rows.

    Returns:
        _Coefficients: The coefficients for the Horner sums.
    """
    if isinstance(coefficients, np.ndarray):
        if coefficients.size == 0:
            return _NO_TERMS
        if coefficients.ndim == 2 and coefficients.dtype.kind in "biuf":
            return _number_rows(coefficients.tolist())
    elif torch is not None and isinstance(coefficients, torch.Tensor):
        if coefficients.numel() == 0:
            return _NO_TERMS
        if coefficients.dim() == 2:
            return _Coefficients(coefficients, None, False, *coefficients.shape)
    rows = [list(row) for row in coefficients]
    if all(isinstance(value, numbers.Real) for row in rows for value in row):
        return _number_rows(rows)
    width = max(map(len, rows), default=0)
    return _Coefficients(None, rows, False, len(rows), width)


def _horner(coefficients: Sequence[Any], t: Any, sparse: bool = False) -> Any:
    """Return the sum of coefficients[j] * t^j with Horner's method.

    Args:
        coefficients: The coefficients, from the power 0 up. At least one.
        t: The variable.
        sparse: If True, the coefficients are numbers, the last one is not
            zero, and the sum leaves out the addition of a zero coefficient.

    Returns:
        The sum. It is the last coefficient if there is only one.
    """
    value = coefficients[-1]
    for j in range(len(coefficients) - 2, -1, -1):
        coefficient = coefficients[j]
        skip = sparse and coefficient == 0
        value = t * value if skip else coefficient + t * value
    return value


def _all_rows_at_once(matrix: Any, y: Any) -> bool:
    """Return True if one Horner pass in y gives the values of all rows.

    ``y`` must be a tensor with the dtype and the device of ``matrix``, so
    that the result has the dtype of the pass for each row.
    """
    if (
        not isinstance(y, torch.Tensor)
        or y.dtype != matrix.dtype
        or y.device != matrix.device
    ):
        return False
    limit = _ALL_ROWS_LIMIT_CPU if y.device.type == "cpu" else _ALL_ROWS_LIMIT_GPU
    return matrix.shape[0] * y.numel() <= limit


def _row_values(coefficients: _Coefficients, y: Any, first: int = 0) -> list[Any]:
    """Return P_i(y) of each row i with Horner's method in y.

    Args:
        coefficients: The coefficients of this evaluation.
        y: The y-coordinates.
        first: The first row to evaluate. The rows before it give None.

    Returns:
        list: P_i(y) for each row, or None for a row without terms.
    """
    matrix = coefficients.matrix
    skipped = [None] * first
    if matrix is None:
        rows = coefficients.rows[first:]
        sparse = coefficients.sparse
        return skipped + [_horner(row, y, sparse) if row else None for row in rows]
    matrix = matrix[first:]
    num_rows, num_columns = matrix.shape
    if num_rows == 0 or num_columns == 0:
        return skipped + [None] * num_rows
    if _all_rows_at_once(matrix, y):
        # Each column broadcasts against y: (rows, 1, ..., 1).
        columns = matrix.T.reshape(num_columns, num_rows, *(1,) * y.dim())
        return skipped + list(_horner(columns.unbind(0), y).unbind(0))
    return skipped + [_horner(row.unbind(0), y) for row in matrix.unbind(0)]


def _derivative_values(coefficients: _Coefficients, y: Any) -> list[Any]:
    """Return P_i'(y) of each row i, from the coefficients j * Cij (j >= 1).

    Args:
        coefficients: The coefficients of this evaluation, with at least two
            columns.
        y: The y-coordinates.

    Returns:
        list: P_i'(y) for each row, or None for a row without terms in y.
    """
    matrix = coefficients.matrix
    num_rows, num_columns = coefficients.num_rows, coefficients.num_columns
    if matrix is None:
        rows = [[j * row[j] for j in range(1, len(row))] for row in coefficients.rows]
        matrix_rows = None
    else:
        powers = torch.arange(1, num_columns, dtype=matrix.dtype, device=matrix.device)
        rows, matrix_rows = None, matrix[:, 1:] * powers
    derivative = _Coefficients(
        matrix_rows, rows, coefficients.sparse, num_rows, num_columns - 1
    )
    return _row_values(derivative, y)


def _horner_in_x(values: list[Any], x: Any) -> Any:
    """Return the sum of values[i] * x^i with Horner's method.

    Args:
        values: The values from the power 0 up. None stands for a zero
            value, and the sum leaves out its addition.
        x: The x-coordinates.

    Returns:
        The sum, or None if all values are None.
    """
    total = None
    for value in reversed(values):
        if total is None:
            total = value
        elif value is None:
            total = x * total
        else:
            total = value + x * total
    return total


class PolynomialGeometry(NewtonRaphsonGeometry):
    """Represents a polynomial geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Cij * x^i * y^j)

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Cij are the polynomial coefficients

    The coefficients are defined in a 2D array where coefficients[i][j] is the
    coefficient for x^i * y^j.

    Historically, XY-polynomials were the first type of polynomials used
    for low-order freeform surfaces (see https://doi.org/10.1364/AO.38.003572).
    These polynomials remain common surface descriptors of freeform surfaces;
    however their lack of orthogonality renders them less desirable than
    their orthogonal counterparts for highly corrected imaging systems.

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
            list of lists representing the polynomial coefficients Cij.
            `coefficients[i][j]` is the coefficient for x^i * y^j.
            Defaults to an empty list (no polynomial contribution).

    Attributes:
        c (be.ndarray): 2D array of polynomial coefficients.

    """

    def __init__(
        self,
        coordinate_system,
        radius,
        conic=0.0,
        tol=1e-10,
        max_iter=100,
        coefficients=None,
    ):
        if coefficients is None:
            coefficients = []
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.coefficients = be.atleast_2d(coefficients)
        self.is_symmetric = False

        if len(self.coefficients) == 0:
            self.coefficients = be.zeros((1, 1))

    def __str__(self):
        return "Polynomial XY"

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        for i in range(len(self.coefficients)):
            for j in range(len(self.coefficients[i])):
                # C_ij' = C_ij * s^(1 - (i+j))
                self.coefficients[i][j] *= scale_factor ** (1 - (i + j))

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature and of every polynomial
        coefficient, so that the sag changes sign. The conic constant remains
        unchanged. The negated coefficients are computed first, so the
        geometry is left unchanged if they cannot be converted (for example
        when they were assigned as a ragged nested list).
        """
        flipped = -be.atleast_2d(self.coefficients)
        super().flip()
        self.coefficients = flipped

    def sag(self, x=0, y=0):
        """Calculates the sag of the polynomial surface at the given coordinates.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            be.ndarray or float: The sag value(s) at the given coordinates.

        """
        if not self._has_terms():
            # Without a term, the sag is the sag of the base conic.
            r2 = x**2 + y**2
            return r2 / (
                self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2))
            )
        return self._sag_and_terms(x, y)[0]

    def _terms(self, x: Any, y: Any, first_row: int = 0) -> _PolynomialTerms:
        """Return r**2, the conic root and the row values P_i(y) at (x, y).

        Each evaluation reads the coefficients again, so a change of a
        coefficient applies at the next call.

        Args:
            x: The x-coordinates.
            y: The y-coordinates.
            first_row: The first row i to evaluate P_i(y) for. The slopes do
                not use P_0(y), so the normal starts at row 1.

        Returns:
            _PolynomialTerms: The shared values.
        """
        r2 = x**2 + y**2
        root = be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        array = self.coefficients
        if type(array) is np.ndarray and array.size <= 1 and array.dtype.kind in "biuf":
            # No term or only C00, which only the sag uses: the values of
            # _prepare, without its work. item() gives the number of tolist().
            c00 = array.item() if array.size else 0
            if first_row or c00 == 0:
                return _PolynomialTerms(r2, root, _NO_TERMS, [])
            return _PolynomialTerms(
                r2, root, _Coefficients(None, [[c00]], True, 1, 1), [c00]
            )
        if torch is not None and isinstance(array, torch.Tensor) and array.numel() == 1:
            # Only C00, as a tensor: the sag adds it, also when it is zero, so
            # that C00 keeps its derivative. The slopes do not use it.
            if first_row:
                return _PolynomialTerms(r2, root, _NO_TERMS, [])
            c00 = array.reshape(())
            return _PolynomialTerms(
                r2, root, _Coefficients(None, [[c00]], False, 1, 1), [c00]
            )
        coefficients = _prepare(array)
        rows = []
        if coefficients.num_rows > first_row:
            rows = _row_values(coefficients, y, first_row)
        return _PolynomialTerms(r2, root, coefficients, rows)

    def _has_terms(self) -> bool:
        """Return False if the polynomial has no term.

        See :meth:`NewtonRaphsonGeometry._has_terms`. An empty array has no
        term (``coefficients=None`` gives one), and neither has a NumPy
        array whose only value is zero. The sag and the slopes are then
        those of the base conic, and the Newton loop calls ``sag`` and
        ``_surface_normal``. A tensor with values keeps every term, also a
        zero one, so that each coefficient keeps its derivative.

        Returns:
            bool: False for a polynomial without a term.
        """
        array = self.coefficients
        if type(array) is np.ndarray and array.size <= 1 and array.dtype.kind in "biuf":
            return array.size == 1 and array.item() != 0
        if torch is not None and isinstance(array, torch.Tensor):
            return array.numel() > 0
        return True

    def _sag_and_terms(self, x: Any, y: Any) -> tuple[Any, _PolynomialTerms]:
        """Return the sag at (x, y), and the terms that the slopes share.

        See :meth:`NewtonRaphsonGeometry._sag_and_terms`. The terms are r**2,
        the conic root and the row values P_i(y).
        """
        terms = self._terms(x, y)
        z = terms.r2 / (self.radius * (1 + terms.root))
        if terms.rows:
            polynomial = _horner_in_x(terms.rows, x)
            if polynomial is not None:
                z = z + polynomial
        return z, terms

    def _slopes_from_terms(
        self, x: Any, y: Any, terms: _PolynomialTerms
    ) -> tuple[Any, Any]:
        """Return d(sag)/dx and d(sag)/dy from the terms of ``_sag_and_terms``.

        See :meth:`NewtonRaphsonGeometry._slopes_from_terms`. The slope in x
        is the Horner sum in x of i * P_i(y). The slope in y is the Horner
        sum in x of P_i'(y), and each P_i'(y) is the Horner sum in y of the
        coefficients j * Cij.
        """
        denom = self.radius * terms.root
        dzdx = x / denom
        dzdy = y / denom
        coefficients = terms.coefficients
        if coefficients.num_rows > 1:  # terms in x
            weighted = [
                value if i == 1 or value is None else i * value
                for i, value in enumerate(terms.rows[1:], start=1)
            ]
            slope_x = _horner_in_x(weighted, x)
            if slope_x is not None:
                dzdx = dzdx + slope_x
        if coefficients.num_columns > 1:  # terms in y
            slope_y = _horner_in_x(_derivative_values(coefficients, y), x)
            if slope_y is not None:
                dzdy = dzdy + slope_y
        return dzdx, dzdy

    def _surface_normal(self, x, y):
        """Calculates the surface normal of the polynomial surface at the given x
        and y position.

        Args:
            x (be.ndarray): The x-coordinate(s) at which to calculate the normal.
            y (be.ndarray): The y-coordinate(s) at which to calculate the normal.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The surface normal
            components (nx, ny, nz).

        """
        if self._has_terms():
            terms = self._terms(x, y, first_row=1)
            dzdx, dzdy = self._slopes_from_terms(x, y, terms)
        else:
            # Without a term, the slopes are the slopes of the base conic.
            r2 = x**2 + y**2
            denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
            dzdx = x / denom
            dzdy = y / denom

        norm = be.sqrt(dzdx**2 + dzdy**2 + 1)
        nx = dzdx / norm
        ny = dzdy / norm
        nz = -1 / norm

        return nx, ny, nz

    def to_dict(self):
        """Converts the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict["coefficients"] = self.coefficients.tolist()
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a PolynomialGeometry from a dictionary.

        Args:
            data (dict): The dictionary containing the geometry data.

        Returns:
            PolynomialGeometry: An instance of PolynomialGeometry.

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
        )
