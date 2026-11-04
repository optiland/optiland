"""Model glass parameterized by Nd, Vd, dPgF, and optional IR partial dispersion.

This module defines a Buchdahl model glass based on the refractive index at the
Fraunhofer d-line, the Abbe number, and the deviation of the relative partial
dispersion. The default visible route uses these three descriptors only.

An optional IR route is enabled when both ``P_ref`` and ``lambda_ref`` are
provided. In that case, ``P_ref`` is the relative partial dispersion at the
reference wavelength:

    P_ref = (n(lambda_ref) - nd) / ((nd - 1) / vd)

Both routes use a 6th-order Buchdahl polynomial. The visible route predicts
nu3..nu6 from a 20-dimensional regression and solves nu1..nu2 from the F-C and
g-F anchors. The IR route predicts nu4..nu6 from a 35-dimensional regression
and solves nu1..nu3 from the F-C, g-F, and reference-wavelength anchors.

The extinction coefficient is ignored and is always set to zero.

by Ziyi Xiong 2026/11
"""

from __future__ import annotations

from functools import cache
from importlib import resources

import numpy as np

import optiland.backend as be
from optiland.materials.base import BaseMaterial
from optiland.materials.buchdahl import buchdahl_coordinate, evaluate_buchdahl


@cache
def _load_regression_matrix(path: str) -> np.ndarray:
    """Read a regression matrix once; the packaged data never changes.

    Loading per coefficient refresh would hit the disk on every evaluation for
    models whose parameters carry gradients, which are deliberately uncached.
    """
    # These files hold plain floats only; refusing pickled object arrays keeps a
    # tampered data file from executing code during deserialization.
    return np.load(path, allow_pickle=False)


class ModelMaterial(BaseMaterial):
    """6th-order Buchdahl model glass.

    The refractive index is evaluated as:

        n(lambda) = nd + sum(nu_k * omega**k), k = 1..6

    where:

        omega = (lambda - lambda_d) / (1 + alpha * (lambda - lambda_d))

    If no IR anchor is supplied, the model uses the visible regression matrix
    ``Regression_Matrix_Visible_Routing.npy``. That matrix predicts nu3..nu6 from the
    descriptors ``nd``, ``vd``, and ``dPgF``. The lower coefficients nu1 and nu2 are
    then solved exactly from the F-C and g-F partial-dispersion equations.

    If both ``P_ref`` and ``lambda_ref`` are supplied, the model uses the IR
    partial-dispersion regression matrix ``Regression_Matrix_IR_Routing.npy``. That
    matrix predicts nu4..nu6 from ``nd``, ``vd``, ``dPgF``, and ``P_ref``. The lower
    coefficients nu1..nu3 are then solved exactly from the F-C, g-F, and
    reference-wavelength equations.

    Args:
        nd (float): Refractive index at the d-line, 0.5875618 um.
        vd (float): Abbe number at the d-line.
        dPgF (float): Deviation of relative partial dispersion.
        P_ref (float | None): Optional relative partial dispersion at
            ``lambda_ref``.
        lambda_ref (float | None): Optional reference wavelength in microns.
    """

    MIN_WTH = 0.365  # Minimum wavelength in microns for this model
    MAX_WTH = 2.30  # Maximum wavelength in microns for this model
    MAX_INDEX = 3.0  # Above this polynomial is extrapolating
    MIN_INDEX = 1.0  # Below this the polynomial is unphysical
    D_LINE = 0.5875618  # Fraunhofer d-line wavelength in microns
    MAX_VISIBLE_WTH = 0.78  # Maximum wavelength in microns for visible route
    IR_REF_WTH = 1.55  # Reference wavelength in microns for infrared route

    REGRESSION_MATRIX_VISIBLE = resources.files("optiland.database").joinpath(
        "Regression_Matrix_Visible_Routing.npy"
    )

    REGRESSION_MATRIX_IR = resources.files("optiland.database").joinpath(
        "Regression_Matrix_IR_Routing.npy"
    )

    def __init__(
        self,
        nd: float,
        vd: float,
        dPgF: float,
        P_ref: float | None = None,
        lambda_ref: float | None = None,
    ) -> None:
        super().__init__()
        self.nd = nd
        self.vd = vd
        self.dPgf = dPgF
        self.p_ref = P_ref
        self.lambda_ref = lambda_ref

        if not self.MIN_INDEX < self.nd < self.MAX_INDEX:
            raise ValueError(
                "Refractive index is out of typical range for optical glasses: "
                f"{self.nd}"
            )

        if self.vd > 100 or self.vd < 10:
            raise ValueError(
                f"Abbe number is out of typical range for optical glasses: {self.vd}"
            )

        if self.dPgf > 1.0 or self.dPgf < -1.0:
            raise ValueError(
                f"dPgF is out of typical range for optical glasses: {self.dPgf}"
            )

        if (P_ref is None) != (lambda_ref is None):
            raise ValueError(
                "Both P_ref and lambda_ref must be provided together for the IR route."
            )

        self._p = None
        self._p_key = None

    @property
    def _is_ir(self) -> bool:
        """True when both IR anchors are set, i.e. the IR route applies.

        Derived from the live attributes rather than cached at construction, so
        assigning ``p_ref``/``lambda_ref`` later cannot leave a stale route in
        place and silently drop the IR anchor.
        """
        return self.p_ref is not None and self.lambda_ref is not None

    def _cache_state(self) -> tuple | None:
        """Track the model parameters; ``_p`` is refreshed from them on demand."""
        return self._state_key(
            (self.nd, self.vd, self.dPgf, self.p_ref, self.lambda_ref)
        )

    def _coefficients(self) -> be.ndarray:
        """Buchdahl coefficients derived from the live parameters.

        A stored vector cannot be reused: mutating ``nd``/``vd``/``dPgF`` would
        keep evaluating the old coefficients, and a vector carrying an autograd
        graph cannot be reused after ``backward()``. The refresh is keyed on the
        parameter state, so the ray-tracing path pays only a comparison.
        """
        state = self._cache_state()
        # The vector is built in the active backend, so its dtype and device are
        # part of the identity: switching backend must not reuse the old one.
        backend = be.get_backend()
        context = (
            backend,
            be.get_precision(),
            be.get_device() if backend == "torch" else "cpu",
        )
        # state is None means a parameter requires gradients: rebuild every call.
        key = None if state is None else (context, state)
        if key is None or key != self._p_key:
            self._p = self._get_coefficients()
            self._p_key = key
        return self._p

    def spectral_range(self, property_name: str = "n") -> tuple[float, float] | None:
        """Report the model's fitted interval for the refractive index."""
        super().spectral_range(property_name)
        return (self.MIN_WTH, self.MAX_WTH) if property_name == "n" else None

    def _calculate_n(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Returns the refractive index of the material.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float or be.ndarray: The refractive index of the material at the given
            wavelength(s).

        """
        wth = be.array(wavelength)

        # Membership test rather than two comparisons: NaN compares False against
        # both bounds and would otherwise reach the polynomial unrejected.
        if be.any(~((wth >= self.MIN_WTH) & (wth <= self.MAX_WTH))):
            raise ValueError("Wavelength out of range for this model.")

        # Same normalization as the coefficient solve, so the anchor term and the
        # fitted coefficients share one precision and device.
        n = evaluate_buchdahl(
            self._as_backend_array(self.nd), self._coefficients(), self._get_omega(wth)
        )

        # Unphysical index, likely due to extrapolation or bad inputs. NaN is
        # rejected here too, since it fails the membership test.
        if be.any(~((n >= self.MIN_INDEX) & (n <= self.MAX_INDEX))):
            wl = be.to_numpy(wth).ravel()
            where = (
                f"{wl[0]:.4f}" if wl.size == 1 else f"[{wl.min():.4f}, {wl.max():.4f}]"
            )
            raise ValueError(
                "Calculated refractive index is unphysical. Check inputs: "
                f"nd={self.nd}, vd={self.vd}, dPgF={self.dPgf}, "
                f"wavelength: {where} um, "
                f"(limit: {self.MIN_INDEX} <= n <= {self.MAX_INDEX})"
            )

        return n

    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Returns the extinction coefficient of the material, always 0 for this model.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float or be.ndarray: The extinction coefficient of the material, which
            is always 0 for this model. Returns a scalar 0 if wavelength is scalar,
            otherwise an array of zeros.

        """
        # Multiplying by 1.0 keeps the documented float result for integer input.
        return be.zeros_like(be.array(wavelength) * 1.0)

    def _get_omega(self, wavelength: float | be.ndarray) -> be.ndarray:
        """Return the nonlinear chromatic coordinate omega.

        Args:
            wavelength (float | be.ndarray): Wavelength(s) in microns.

        Returns:
            float | be.ndarray: The Buchdahl chromatic coordinate.
        """
        # alpha is experimentally determined per route: see
        # "docs/references/IR_Routing.ipynb" and
        # "docs/references/Visible_Routing.ipynb".
        alpha = 1.11016949153 if self._is_ir else 1.49152542373

        return buchdahl_coordinate(wavelength, self.D_LINE, alpha)

    def _get_lower_order_coeffs(self, higher_orders: be.ndarray) -> be.ndarray:
        """
        Solve lower-order Buchdahl coefficients from the exact anchors.

        In the visible route, ``higher_orders`` contains nu3..nu6 and this method
        solves nu1..nu2. In the IR route, ``higher_orders`` contains nu4..nu6 and
        this method solves nu1..nu3. For more details, see the
        "docs/references/IR_Routing.ipynb" and
        "docs/references/Visible_Routing.ipynb" notebooks.

        Args:
            higher_orders (be.ndarray): Regression-predicted coefficients, ordered
                from the lowest solved power upward (nu3.. or nu4..).

        Returns:
            be.ndarray: The solved lower-order coefficients.
        """
        # Anchors are constants, so keep them as plain floats; the matrix built
        # from them is fixed data rather than a trainable parameter.
        OMEGA_F = float(self._get_omega(0.4861327))  # Fraunhofer F line in microns
        OMEGA_C = float(self._get_omega(0.6562725))  # Fraunhofer C line in microns
        OMEGA_g = float(self._get_omega(0.4358343))  # Fraunhofer g line in microns

        # Parameters are normalized to the active backend's precision and device
        # so they can be combined with the backend-created matrices below.
        nd = self._as_backend_array(self.nd)
        vd = self._as_backend_array(self.vd)
        dn_FC = (nd - 1.0) / vd
        PgF = 0.6438 - 0.001682 * vd + self._as_backend_array(self.dPgf)
        dn_gF = PgF * dn_FC
        if self._is_ir:
            OMEGA_ref = float(self._get_omega(self.lambda_ref))
            dn_ref = self._as_backend_array(self.p_ref) * dn_FC

            M = be.array(
                [
                    [
                        OMEGA_F - OMEGA_C,
                        OMEGA_F**2 - OMEGA_C**2,
                        OMEGA_F**3 - OMEGA_C**3,
                    ],
                    [
                        OMEGA_g - OMEGA_F,
                        OMEGA_g**2 - OMEGA_F**2,
                        OMEGA_g**3 - OMEGA_F**3,
                    ],
                    [OMEGA_ref, OMEGA_ref**2, OMEGA_ref**3],
                ]
            )

            core_FC = 0.0
            core_gF = 0.0
            core_ref = 0.0
            for power, coeff in enumerate(higher_orders, start=4):
                core_FC += coeff * (OMEGA_F**power - OMEGA_C**power)
                core_gF += coeff * (OMEGA_g**power - OMEGA_F**power)
                core_ref += coeff * (OMEGA_ref**power)

            rhs = be.array([dn_FC - core_FC, dn_gF - core_gF, dn_ref - core_ref])
            return be.linalg.solve(M, rhs)
        else:
            M = be.array(
                [
                    [OMEGA_F - OMEGA_C, OMEGA_F**2 - OMEGA_C**2],
                    [OMEGA_g - OMEGA_F, OMEGA_g**2 - OMEGA_F**2],
                ]
            )

            core_FC = 0.0
            core_gF = 0.0
            for power, coeff in enumerate(higher_orders, start=3):
                core_FC += coeff * (OMEGA_F**power - OMEGA_C**power)
                core_gF += coeff * (OMEGA_g**power - OMEGA_F**power)

            rhs = be.array([dn_FC - core_FC, dn_gF - core_gF])
            return be.linalg.solve(M, rhs)

    def _get_higher_order_coeffs(self) -> be.ndarray:
        """Return regression-predicted higher-order Buchdahl coefficients.

        The visible route returns nu3..nu6. The IR route returns nu4..nu6.

        Returns:
            be.ndarray: A 1D array of regression-predicted coefficients.
        """
        path = (
            self.REGRESSION_MATRIX_IR if self._is_ir else self.REGRESSION_MATRIX_VISIBLE
        )
        reg_mat = be.asarray(_load_regression_matrix(str(path)))
        return be.matmul(self._get_feature_vector(), reg_mat)

    def _get_coefficients(self) -> be.ndarray:
        """
        Return the full Buchdahl coefficient vector [nu1, ..., nu6].

        Returns:
            be.ndarray: A 1D array of Buchdahl coefficients.
        """
        higher_orders = self._get_higher_order_coeffs()
        low_orders = self._get_lower_order_coeffs(higher_orders)

        return be.concatenate((low_orders, higher_orders))

    def _get_feature_vector(self) -> be.ndarray:
        """Return the regression feature vector for the active route.

        The visible route uses a fixed 20-dimensional feature vector in the same
        order as the visible regression matrix. The IR route uses the 35-dimensional
        total-degree <= 3 polynomial basis of nd, vd, dPgF, and P_ref.

        Returns:
            be.ndarray: A 1D array representing the feature vector.
        """
        nd = self._as_backend_array(self.nd)
        vd = self._as_backend_array(self.vd)
        dPgF = self._as_backend_array(self.dPgf)

        if not self._is_ir:
            # Term order must match the columns of the visible regression matrix.
            return be.array(
                [
                    # Constant and linear terms
                    1.0,
                    nd,
                    vd,
                    dPgF,
                    # Quadratic terms
                    nd**2,
                    vd**2,
                    dPgF**2,
                    nd * vd,
                    nd * dPgF,
                    vd * dPgF,
                    # Cubic terms
                    nd**3,
                    vd**3,
                    dPgF**3,
                    nd**2 * vd,
                    nd**2 * dPgF,
                    vd**2 * nd,
                    vd**2 * dPgF,
                    dPgF**2 * nd,
                    dPgF**2 * vd,
                    nd * vd * dPgF,
                ]
            )

        # IR route: total-degree <= 3 basis of nd, vd, dPgF and P_ref (35 terms),
        # in the same order as the columns of the IR regression matrix.
        x = [nd, vd, dPgF, self._as_backend_array(self.p_ref)]
        count = len(x)
        feats = [1.0, *x]
        feats.extend([x[i] * x[j] for i in range(count) for j in range(i, count)])
        feats.extend(
            [
                x[i] * x[j] * x[k]
                for i in range(count)
                for j in range(i, count)
                for k in range(j, count)
            ]
        )
        return be.array(feats)

    def partial_dispersion_ref(self, wavelength):
        """
        Returns the reference partial dispersion P_ref at a given wavelength.

        P_ref = (n(lambda) - nd) / ((nd - 1) / vd)

        Args:
            wavelength (float): Wavelength in microns, within the model range.

        Returns:
            float | be.ndarray: Reference partial dispersion at ``wavelength``.
        """
        n_ref = self._calculate_n(wavelength)

        return (n_ref - self.nd) / ((self.nd - 1.0) / self.vd)

    def to_dict(self):
        """Returns a dictionary representation of the material.

        Returns:
            dict: The dictionary representation of the material.

        """
        material_dict = super().to_dict()
        material_dict.update({"nd": float(self.nd), "vd": float(self.vd)})
        material_dict.update({"dPgF": float(self.dPgf)})
        material_dict.update(
            {"P_ref": float(self.p_ref) if self.p_ref is not None else None}
        )
        material_dict.update(
            {
                "lambda_ref": float(self.lambda_ref)
                if self.lambda_ref is not None
                else None
            }
        )
        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            ModelMaterial: The material object.

        """
        required_keys = ("nd", "vd", "dPgF")
        missing = [key for key in required_keys if data.get(key) is None]
        if missing:
            raise ValueError(f"Missing null required key(s): {', '.join(missing)}")

        return cls(
            nd=data["nd"],
            vd=data["vd"],
            dPgF=data["dPgF"],
            P_ref=data.get("P_ref"),
            lambda_ref=data.get("lambda_ref"),
        )
