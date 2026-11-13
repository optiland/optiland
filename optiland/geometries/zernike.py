"""Zernike Geometry

The Zernike polynomial geometry represents a surface defined by a Zernike
polynomial in two dimensions. The surface is defined as:

z(x,y) = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
    sum_i [c[i] * Z_i(rho, phi)]

where:
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- c[i] is the coefficient for the i-th Zernike polynomial
- Z_i(...) is the i-th Zernike polynomial in polar coordinates
- rho = sqrt(x^2 + y^2) / normalization, phi = atan2(y, x)

Zernike polynomials are a set of orthogonal functions defined over the unit
disk, widely used in freeform optical surface design. They efficiently
describe wavefront aberrations and complex surface deformations by decomposing
them into radial and azimuthal components. Their orthogonality ensures minimal
cross-coupling between terms, making them ideal for optimizing optical systems.
In freeform optics, they enable precise control of surface shape,
improving performance beyond traditional spherical and aspheric designs.

The sag and the slopes come from tables. With X = x / normalization,
Y = y / normalization and t = 2 (X^2 + Y^2) - 1, a term with m >= 0 is
c_i N_i Re((X + iY)^m) Q_i(t), and a term with m < 0 is
c_i N_i Im((X + iY)^|m|) Q_i(t). N_i is the normalization constant and Q_i is
the radial polynomial divided by rho^|m|, as a polynomial in t. The
coefficients of Q_i in t are small and exact binary fractions, so the sum
keeps more digits than a sum of powers of rho. The terms of one harmonic make
one polynomial in t. On the points, only + - * / operate, with sequential
sums, so that a compiled kernel can repeat the expressions.

drpaprika, 2025
"""

from __future__ import annotations

import functools
import itertools
import math
from fractions import Fraction
from typing import TYPE_CHECKING, Any, NamedTuple

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from optiland.geometries.standard import _is_radius_infinite
from optiland.zernike import ZernikeFringe, ZernikeNoll, ZernikeStandard

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optiland._types import ZernikeType
    from optiland.zernike.base import BaseZernike


__all__ = [
    "ZernikePolynomialGeometry",
]

_ZERNIKE_TYPES: dict[ZernikeType, type[BaseZernike]] = {
    "standard": ZernikeStandard,
    "noll": ZernikeNoll,
    "fringe": ZernikeFringe,
}


class _Harmonic(NamedTuple):
    """The polynomial Q(t) of the terms of one harmonic: m and cosine or sine.

    ``offset`` and ``slope_offset`` are the indices of the t^0 coefficients of
    Q and of dQ/du (u = rho^2) in the combined coefficients.
    """

    m: int
    sine: bool
    offset: int
    degree: int
    slope_offset: int


class _ZernikeLayout(NamedTuple):
    """The tables of one Zernike type and number of terms.

    ``orders`` holds (m, index of the cosine polynomial, index of the sine
    polynomial) for each m, with None for a missing polynomial. ``norms`` is
    None if all normalization constants are 1. ``table`` has one row for each
    term. The tables do not depend on the coefficient values.
    """

    harmonics: tuple[_Harmonic, ...]
    orders: tuple[tuple[int, int | None, int | None], ...]
    max_m: int
    norms: tuple[float, ...] | None
    table: tuple[tuple[float, ...], ...]


@functools.cache
def _zernike_layout(zernike_type: type[BaseZernike], num_terms: int) -> _ZernikeLayout:
    """Build the tables of a Zernike type from exact rational numbers.

    Args:
        zernike_type: The Zernike class. It gives the index order, the
            normalization and the radial coefficients.
        num_terms: The number of terms.

    Returns:
        _ZernikeLayout: The tables.
    """
    terms = [(int(n), int(m)) for n, m in zernike_type._generate_indices(num_terms)]
    degrees: dict[tuple[int, bool], int] = {}
    for n, m in terms:
        key = (abs(m), m < 0)
        degrees[key] = max(degrees.get(key, 0), (n - abs(m)) // 2)
    # The coefficients of all polynomials first, then of their derivatives.
    keys = sorted(degrees)
    offsets = list(itertools.accumulate((degrees[k] + 1 for k in keys), initial=0))
    slopes = list(itertools.accumulate((degrees[k] for k in keys), initial=offsets[-1]))
    harmonics = {
        key: _Harmonic(*key, offsets[i], degrees[key], slopes[i])
        for i, key in enumerate(keys)
    }
    orders: dict[int, list[int | None]] = {}
    for index, (m, sine) in enumerate(keys):
        orders.setdefault(m, [None, None])[sine] = index

    table = []
    for n, m in terms:
        harmonic = harmonics[abs(m), m < 0]
        row = [0.0] * slopes[-1]
        # Q(u) = sum_j b_j u^j with u = (1 + t) / 2, so the coefficient of t^l
        # is sum_j b_j C(j, l) / 2^j: a binary fraction that a float holds.
        b = zernike_type._radial_coefficients(n, m)
        for power in range(len(b)):
            a = sum(
                Fraction(b[j] * math.comb(j, power), 2**j) for j in range(power, len(b))
            )
            row[harmonic.offset + power] = float(a)
            if power > 0:  # dQ/du = 2 dQ/dt
                row[harmonic.slope_offset + power - 1] = float(2 * power * a)
        table.append(tuple(row))

    norms = tuple(math.sqrt(zernike_type._norm_squared(n, m)) for n, m in terms)
    return _ZernikeLayout(
        harmonics=tuple(harmonics.values()),
        orders=tuple((m, *orders[m]) for m in sorted(orders)),
        max_m=max(orders, default=0),
        norms=None if all(norm == 1.0 for norm in norms) else norms,
        table=tuple(table),
    )


# The tables as backend arrays, for each Zernike type, number of terms,
# backend, device, precision and inference mode. They do not depend on the
# coefficient values, so they stay valid when the coefficients change.
_BACKEND_TABLES: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def _backend_tables(zernike_type: type[BaseZernike], num_terms: int) -> tuple[Any, Any]:
    """Return the normalization constants (or None) and the table as arrays.

    The arrays do not require gradients. They are made once for each backend
    context.
    """
    backend = be.get_backend()
    key = (zernike_type, num_terms, backend, be.get_precision())
    if backend == "torch":
        key += (be.get_device(), torch.is_inference_mode_enabled())
    tables = _BACKEND_TABLES.get(key)
    if tables is None:
        layout = _zernike_layout(zernike_type, num_terms)
        norms = None if layout.norms is None else be.asarray(layout.norms)
        tables = _BACKEND_TABLES[key] = (norms, be.asarray(layout.table))
    return tables


class _ZernikePointTables(NamedTuple):
    """The values at the points that the sag and the slopes share.

    ``r2`` and ``root`` are x^2 + y^2 and sqrt(1 - (1 + k) r^2 / R^2) of the
    base conic, or None for an infinite radius. ``x_norm`` and ``y_norm`` are
    X and Y, and ``t`` is 2 (X^2 + Y^2) - 1. ``cosines[m]`` and ``sines[m]``
    are Re and Im of (X + iY)^m, from m = 1. ``radial`` holds Q(t) of each
    harmonic, with None for m = 0, because only the sag uses that value.
    ``coefficients`` are the combined coefficients of this call.
    """

    x: Any
    y: Any
    r2: Any
    root: Any
    x_norm: Any
    y_norm: Any
    t: Any
    cosines: list[Any]
    sines: list[Any]
    radial: list[Any]
    coefficients: list[Any]
    layout: _ZernikeLayout


def _horner(coefficients: list[Any], offset: int, degree: int, t: Any) -> Any:
    """Evaluate a polynomial in t with Horner's method (a scalar for degree 0)."""
    value = coefficients[offset + degree]
    for j in range(offset + degree - 1, offset - 1, -1):
        value = coefficients[j] + t * value
    return value


def _accumulate(total: Any, term: Any) -> Any:
    """Add ``term`` to ``total``. None means that there is no value."""
    if term is None:
        return total
    return term if total is None else total + term


class ZernikePolynomialGeometry(NewtonRaphsonGeometry):
    """Represents a Zernike polynomial geometry defined as:

    z(x,y) = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
        sum_i [c[i] * Z_i(rho, phi)]

    where:
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - c[i] is the coefficient for the i-th Zernike polynomial
    - Z_i(...) is the i-th Zernike polynomial in polar coordinates
    - rho = sqrt(x^2 + y^2) / normalization, phi = atan2(y, x)

    The coefficients are defined in a 1D array where coefficients[i] is the
    coefficient for Z_i.

    Args:
        coordinate_system (str): The coordinate system used for the geometry.
        radius (float): The radius of curvature of the geometry.
        conic (float, optional): The conic constant of the geometry.
            Defaults to 0.0.
        tol (float, optional): The tolerance value used in calculations.
            Defaults to 1e-10.
        max_iter (int, optional): The maximum number of iterations used in
            calculations. Defaults to 100.
        coefficients (list or be.ndarray, optional): The coefficients of the
            Zernike polynomial surface. Defaults to an empty list, indicating
            no Zernike polynomial coefficients are used.
        zernike_type (str, optional): The type of Zernike polynomial to use.
            Defaults to "standard". Options are "standard", "noll", or "fringe".
        norm_radius (float, optional): The normalization radius for the
            Zernike polynomial coordinates. If None, the radius scales
            automatically during paraxial updates. Defaults to None.
    """

    def __init__(
        self,
        coordinate_system: str,
        radius: float,
        conic: float = 0.0,
        tol: float = 1e-10,
        max_iter: int = 100,
        coefficients: NDArray | None = None,
        zernike_type: ZernikeType = "standard",
        norm_radius: float | None = None,
    ):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)

        if zernike_type not in _ZERNIKE_TYPES:
            raise ValueError(
                "Zernike type must be one of 'standard', 'noll', or 'fringe', got "
                f"{zernike_type}",
            )
        if norm_radius is not None and norm_radius <= 0:
            raise ValueError(
                f"Normalization radius must be positive, got {norm_radius}"
            )

        coefficients = be.atleast_1d(coefficients if coefficients is not None else [])

        self.zernike = _ZERNIKE_TYPES[zernike_type](coeffs=coefficients)
        self.zernike_type: ZernikeType = zernike_type
        if norm_radius is not None:
            self.norm_radius = norm_radius
            self.normalization_mode = "manual"
        else:
            self.norm_radius = 1.0
            self.normalization_mode = "auto"
        self.is_symmetric = False

    @property
    def coefficients(self) -> NDArray:
        """Get the coefficients of the Zernike polynomial surface."""
        return self.zernike.coeffs

    @coefficients.setter
    def coefficients(self, value: NDArray) -> None:
        """Set the coefficients of the Zernike polynomial surface."""
        self.zernike = _ZERNIKE_TYPES[self.zernike_type](coeffs=be.atleast_1d(value))

    def __str__(self) -> str:
        return "Zernike Polynomial"

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        self.norm_radius = self.norm_radius * scale_factor
        self.coefficients = self.coefficients * scale_factor

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature and of every Zernike
        coefficient, so that the sag changes sign. The conic constant remains
        unchanged.
        """
        super().flip()
        self.coefficients = -self.coefficients

    def update_normalization(self, semi_aperture: float) -> None:
        if self.normalization_mode == "auto":
            self.norm_radius = semi_aperture * 1.25

    def sag(self, x: NDArray, y: NDArray) -> NDArray:  # type: ignore
        """Calculate the sag of the Zernike polynomial surface at the given
        coordinates.

        Args:
            x (float, be.ndarray): The Cartesian x-coordinate(s).
            y (float, be.ndarray): The Cartesian y-coordinate(s).

        Returns:
            be.ndarray: The sag value at the given Cartesian coordinates.
        """
        return self._sag_from_tables(self._point_tables(x, y, validate=True))

    def _surface_normal(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[float, float, float]:
        """Calculate the surface normal of the full surface (conic + Zernike)
        in Cartesian coordinates at (x, y).

        Args:
            x (float or be.ndarray): x-coordinate(s).
            y (float or be.ndarray): y-coordinate(s).

        Returns:
            (nx, ny, nz): Normal vector components in Cartesian coords.

        """
        nx, ny = self._slopes_from_tables(self._point_tables(x, y))
        norm = be.sqrt(nx**2 + ny**2 + 1)
        nx = nx / norm
        ny = ny / norm
        nz = -be.ones_like(x) / norm

        return (nx, ny, nz)

    def _point_tables(
        self, x: Any, y: Any, validate: bool = False
    ) -> _ZernikePointTables:
        """Calculate the values at the points that the sag and the slopes share.

        The polynomial coefficients come from the coefficient values of this
        call, so a change of a coefficient applies at the next call.

        Args:
            x: The x-coordinates.
            y: The y-coordinates.
            validate: If True, check that the points are in the normalization
                range, as ``sag`` does.

        Returns:
            _ZernikePointTables: The shared values.

        Raises:
            ValueError: If ``validate`` is True and a normalized coordinate is
                outside [-1, 1].
        """
        x_norm = x / self.norm_radius
        y_norm = y / self.norm_radius
        if validate:
            self._validate_inputs(x_norm, y_norm)

        # The base conic: r^2 and the square root of its sag and its slopes.
        r2 = root = None
        if not _is_radius_infinite(self.radius):
            r2 = x**2 + y**2
            root = be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)

        layout = _zernike_layout(type(self.zernike), len(self.zernike.indices))
        if not layout.harmonics:
            return _ZernikePointTables(
                x, y, r2, root, x_norm, y_norm, None, [], [], [], [], layout
            )

        norms, table = _backend_tables(type(self.zernike), len(self.zernike.indices))
        coeffs = self.zernike.coeffs
        if norms is not None:
            coeffs = coeffs * norms
        # One small matrix product gives the coefficients of all polynomials.
        coeffs = list(be.matmul(coeffs, table))

        t = 2 * (x_norm * x_norm + y_norm * y_norm) - 1
        # Re and Im of (X + iY)^m = rho^m cos(m phi) + i rho^m sin(m phi).
        cosines = [None, x_norm]
        sines = [None, y_norm]
        for m in range(2, layout.max_m + 1):
            cosines.append(x_norm * cosines[m - 1] - y_norm * sines[m - 1])
            sines.append(x_norm * sines[m - 1] + y_norm * cosines[m - 1])
        radial = [
            None if h.m == 0 else _horner(coeffs, h.offset, h.degree, t)
            for h in layout.harmonics
        ]
        return _ZernikePointTables(
            x, y, r2, root, x_norm, y_norm, t, cosines, sines, radial, coeffs, layout
        )

    def _sag_from_tables(self, tables: _ZernikePointTables) -> Any:
        """Calculate the sag from the shared values.

        Args:
            tables: The values of ``_point_tables``.

        Returns:
            The sag at the points.
        """
        if tables.root is None:
            z = be.zeros_like(tables.x)
        else:
            z = tables.r2 / (self.radius * (1 + tables.root))

        zernike = None
        for harmonic, value in zip(tables.layout.harmonics, tables.radial, strict=True):
            if harmonic.m == 0:
                term = _horner(
                    tables.coefficients, harmonic.offset, harmonic.degree, tables.t
                )
            else:
                terms = tables.sines if harmonic.sine else tables.cosines
                term = terms[harmonic.m] * value
            zernike = _accumulate(zernike, term)

        # Without terms, the Zernike part is the sum of no terms: 0.
        return z + (0 if zernike is None else zernike)

    def _slopes_from_tables(self, tables: _ZernikePointTables) -> tuple[Any, Any]:
        """Calculate the slopes dz/dx and dz/dy from the shared values.

        With H = (X + iY)^m, the polynomial Q of each harmonic and
        u = X^2 + Y^2, the product rule gives d(Re(H) Q)/dX =
        m Re((X + iY)^(m - 1)) Q + Re(H) (dQ/du) 2X, and the same form for Y
        and for Im(H). The center needs no special case.

        Args:
            tables: The values of ``_point_tables``.

        Returns:
            tuple: dz/dx and dz/dy at the points.
        """
        x, y = tables.x, tables.y
        if tables.root is None:
            dzdx = be.zeros_like(x)
            dzdy = be.zeros_like(y)
        else:
            denominator = self.radius * tables.root
            valid_denominator = be.abs(denominator) > 0
            safe_denominator = be.where(
                valid_denominator, denominator, be.ones_like(denominator)
            )
            dzdx = be.where(valid_denominator, x / safe_denominator, be.zeros_like(x))
            dzdy = be.where(valid_denominator, y / safe_denominator, be.zeros_like(y))

        layout = tables.layout
        # Add the derivatives dQ/du, each times its harmonic, in one sum.
        derivative_sum = None
        for harmonic in layout.harmonics:
            if harmonic.degree == 0:
                continue
            term = _horner(
                tables.coefficients,
                harmonic.slope_offset,
                harmonic.degree - 1,
                tables.t,
            )
            if harmonic.m > 0:
                terms = tables.sines if harmonic.sine else tables.cosines
                term = terms[harmonic.m] * term
            derivative_sum = _accumulate(derivative_sum, term)

        # The derivatives of the harmonics: d Re((X + iY)^m)/dX =
        # m Re((X + iY)^(m - 1)), d Re/dY = -m Im(...), d Im/dX = m Im(...),
        # d Im/dY = m Re(...). (X + iY)^0 = 1.
        slope_x = slope_y = None
        for m, cosine_index, sine_index in layout.orders:
            if m == 0:
                continue
            p_cos = None if cosine_index is None else tables.radial[cosine_index]
            p_sin = None if sine_index is None else tables.radial[sine_index]
            if m == 1:
                term_x, term_y = p_cos, p_sin
            else:
                cosine = tables.cosines[m - 1]
                sine = tables.sines[m - 1]
                if p_sin is None:
                    term_x, term_y = cosine * p_cos, -(sine * p_cos)
                elif p_cos is None:
                    term_x, term_y = sine * p_sin, cosine * p_sin
                else:
                    term_x = cosine * p_cos + sine * p_sin
                    term_y = cosine * p_sin - sine * p_cos
                term_x, term_y = m * term_x, m * term_y
            slope_x = _accumulate(slope_x, term_x)
            slope_y = _accumulate(slope_y, term_y)

        # du/dX = 2 X and du/dY = 2 Y.
        if derivative_sum is not None:
            slope_x = _accumulate(slope_x, 2 * tables.x_norm * derivative_sum)
            slope_y = _accumulate(slope_y, 2 * tables.y_norm * derivative_sum)

        # d(X)/dx = d(Y)/dy = 1 / normalization.
        if slope_x is not None:
            dzdx = dzdx + slope_x / self.norm_radius
        if slope_y is not None:
            dzdy = dzdy + slope_y / self.norm_radius
        return dzdx, dzdy

    def _validate_inputs(self, x_norm: float, y_norm: float) -> None:
        """Validate the input coordinates for the Zernike polynomial surface.

        Args:
            x_norm (be.ndarray): The normalized x values.

        """
        if be.any(be.abs(x_norm) > 1) or be.any(be.abs(y_norm) > 1):
            raise ValueError(
                "Zernike coordinates must be normalized "
                "to [-1, 1]. Consider updating the normalization "
                "radius to 1.1x the surface aperture.",
            )

    def to_dict(self) -> dict:
        """Convert the Zernike polynomial geometry to a dictionary.

        Returns:
            dict: The Zernike polynomial geometry as a dictionary.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "coefficients": list(self.zernike.coeffs),
                "zernike_type": self.zernike_type,
                "norm_radius": self.norm_radius,
            },
        )

        return geometry_dict

    @classmethod
    def from_dict(cls, data: dict) -> ZernikePolynomialGeometry:
        """Create a Zernike polynomial geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the Zernike
                polynomial geometry.

        Returns:
            ZernikePolynomialGeometry: The Zernike polynomial geometry.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(
            coordinate_system=cs,
            radius=data["radius"],
            conic=data.get("conic", 0.0),
            tol=data.get("tol", 1e-10),
            max_iter=data.get("max_iter", 100),
            coefficients=be.atleast_1d(data.get("coefficients", [])),
            zernike_type=data.get("zernike_type", "standard"),
            norm_radius=data.get("norm_radius", 1),
        )
