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

by Ziyi Xiong 2026/4
"""

from __future__ import annotations

from importlib import resources

import optiland.backend as be
from optiland.materials.base import BaseMaterial


class ModelMaterial(BaseMaterial):
    r"""6th-order Buchdahl model glass.

    The refractive index is evaluated as:

        n(lambda) = nd + sum(nu_k * omega**k), k=1..6

    where:

        omega = (lambda - lambda_d) / (1 + alpha * (lambda - lambda_d))

    If no IR anchor is supplied, the model uses the visible regression matrix
    ``Regression_Matrix_Visible_Routing.npy``. That matrix predicts nu3..nu6
    from the descriptors ``nd``, ``vd``, and ``dPgF``. The lower coefficients
    nu1 and nu2 are then solved exactly from the F-C and g-F
    partial-dispersion equations.

    If both ``P_ref`` and ``lambda_ref`` are supplied, the model uses the IR
    partial-dispersion regression matrix
    ``Regression_Matrix_IR_Routing.npy``. That matrix predicts nu4..nu6 from
    ``nd``, ``vd``, ``dPgF``, and ``P_ref``. The lower coefficients nu1..nu3
    are then solved exactly from the F-C, g-F, and reference-wavelength
    equations.

    The reference partial dispersion P_ref is defined as::

        P_ref = (n(lambda_ref) - nd) / ((nd - 1) / vd)

    Args:
        nd (float): Refractive index at the d-line, 0.5875618 um.
        vd (float): Abbe number at the d-line.
        dPgF (float): Deviation of relative partial dispersion.
        P_ref (float | None): Optional reference partial dispersion at
            ``lambda_ref``.
        lambda_ref (float | None): Optional reference wavelength in microns.
    """

    MIN_WTH = 0.365  # Minimum wavelength in microns for this model
    MAX_WTH = 2.30  # Maximum wavelength in microns for this model

    REGRESSION_MATRIX_VISIBLE = resources.files("optiland.database").joinpath(
        "Regression_Matrix_Visible_Routing.npy",
    )

    REGRESSION_MATRIX = resources.files("optiland.database").joinpath(
        "Regression_Matrix_IR_Routing.npy",
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

        if self.vd > 100 or self.vd < 10:
            raise ValueError(
                f"Abbe number is out of typical range for optical glasses: {self.vd}"
            )

        if self.dPgf > 1.0 or self.dPgf < -1.0:
            raise ValueError(
                f"dPgF is out of typical range for optical glasses: {self.dPgf}"
            )

        self._p = self._get_coefficients()

    @staticmethod
    def buchdahl_eval(nd, coeffs, omega):
        """Evaluate the 6th-order Buchdahl polynomial.

        Args:
            nd (float): The refractive index at the d-line.
            coeffs (be.ndarray): Buchdahl coefficients [nu1, ..., nu6].
            omega (float or be.ndarray): Nonlinear chromatic coordinate(s).

        Returns:
            be.ndarray: The refractive index evaluated at the given omega.
        """
        omega = be.array(omega)
        n = be.full_like(omega, nd)
        for k, coeff in enumerate(coeffs, start=1):
            n = n + coeff * omega**k

        return n

    def _calculate_n(self, wavelength, **kwargs):
        """Returns the refractive index of the material.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            be.ndarray: The refractive index of the material at the given
            wavelength(s).

        """
        wth = be.array(wavelength)

        if be.any(wth < self.MIN_WTH) or be.any(wth > self.MAX_WTH):
            raise ValueError("Wavelength out of range for this model.")

        omegas = self._get_omega(wth)

        n = self.buchdahl_eval(self.nd, self._p, omegas)

        # Unphysically high index, likely due to extrapolation or bad inputs
        if be.any(n > 3.0):
            raise ValueError(
                "Calculated refractive index is unphysical. Check inputs."
                f"Nd: {self.nd}, Vd: {self.vd}, dPgF: {self.dPgf}",
                f"Wavelength: {wavelength}",
            )

        return n

    def _calculate_k(self, wavelength, **kwargs):
        """Returns the extinction coefficient of the material.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float or be.ndarray: The extinction coefficient of the material, which
            is always 0 for this model. Returns a scalar 0 if wavelength is scalar,
            otherwise an array of zeros.

        """
        return be.zeros_like(be.array(wavelength))

    def _get_omega(self, wavelength):
        """
        Returns the nonlinear chromatic coordinate omega for the given wavelengths.
        """
        if self.lambda_ref is not None and self.p_ref is not None:
            # IR-routing optimized runtime-regression alpha;
            # see docs/Buchdahl_model_material_details/IR_Routing.ipynb
            # section "Fit And Regression Matrix"
            ALPHA = 1.11016949153
        else:
            # Visible-routing optimized runtime-regression alpha;
            # see: docs/Buchdahl_model_material_details/Visible_Routing.ipynb
            # section "6th-Order Regression-Level Alpha Sweep"
            ALPHA = 1.49152542373

        D_LINE = 0.5875618  # Fraunhofer d-line wavelength in microns

        # Convert wavelength to omega using the second equation in the docstring.
        d_wl = wavelength - D_LINE
        omega = d_wl / (1 + ALPHA * d_wl)
        return omega

    def _get_lower_order_coeffs(self, higher_orders):
        """
        Solve lower-order Buchdahl coefficients from the exact anchors.

        In the visible route, ``higher_orders`` contains nu3..nu6 and this method
        solves nu1..nu2.

        In the IR route, ``higher_orders`` contains nu4..nu6 and
        this method solves nu1..nu3.

        Returns:
            be.ndarray: The solved lower-order coefficients.
        """
        OMEGA_F = self._get_omega(0.4861327)  # Fraunhofer F-line wavelength in microns
        OMEGA_C = self._get_omega(0.6562725)  # Fraunhofer C-line wavelength in microns
        OMEGA_g = self._get_omega(0.4358343)  # Fraunhofer g-line wavelength in microns

        # dn_FC = n_F - n_C = (nd - 1) / vd, definition of Abbe number
        dn_FC = (self.nd - 1.0) / self.vd

        # Pgf is the partial dispersion, and dPgF is its deviation from the normal line.
        # Please refer to: https://optics.ansys.com/hc/en-us/articles/42661781831571-How-to-use-the-Model-Glass
        # normal line: Pgf_normal = 0.6438 - 0.001682 * vd is given by Schott:
        # https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/10/tie-29_refractive_index_v2_us.pdf
        PgF = 0.6438 - 0.001682 * self.vd + self.dPgf

        # Pgf is also represented by: PgF = (n_g - n_F) / (n_F - n_C)
        dn_gF = PgF * dn_FC

        # IR routing:
        if self.lambda_ref is not None and self.p_ref is not None:
            OMEGA_ref = self._get_omega(self.lambda_ref)

            # dn_ref = n(lambda_ref) - nd = P_ref * ((nd - 1) / vd),
            # see definition of P_ref in the docstring
            dn_ref = self.p_ref * dn_FC

            # Matrix for solving nu1..nu3 from the three exact anchors and the predicted
            # higher-order coefficients. Please refer to the section "Theory" in the
            # notebook: Optiland\docs\Buchdahl_model_material_details\IR_Routing.ipynb.
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
        # Visible routing:
        else:
            # Similar way as the IR routing to solve nu1..nu2 from the two exact anchors
            # and the predicted higher-order coefficients. Please refer to the section
            # "Theory" in the notebook:
            # Optiland\docs\Buchdahl_model_material_details\Visible_Routing.ipynb.
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

    def _get_higher_order_coeffs(self):
        """
        Return regression-predicted higher-order Buchdahl coefficients.

        The visible route returns nu3..nu6. The IR route returns nu4..nu6.

        Returns:
            be.ndarray: A 1D array of regression-predicted coefficients.
        """
        feature = self._get_feature_vector()

        # IR routing's nu4..nu6 are predicted from the 35-dimensional regression matrix:
        # [nu4, nu5, nu6] = feature @ Regression_Matrix_IR_Routing.npy
        if self.lambda_ref is not None and self.p_ref is not None:
            reg_mat = be.load(str(self.REGRESSION_MATRIX))
            return be.matmul(feature, reg_mat)
        # Visible routing's nu3..nu6 are predicted from the 20-dimensional matrix:
        # [nu3, nu4, nu5, nu6] = feature @ Regression_Matrix_Visible_Routing.npy
        else:
            reg_mat = be.load(str(self.REGRESSION_MATRIX_VISIBLE))
            return be.matmul(feature, reg_mat)

    def _get_coefficients(self):
        """
        Return the full Buchdahl coefficient vector [nu1, ..., nu6].

        Returns:
            be.ndarray: A 1D array of Buchdahl coefficients.
        """
        higher_orders = self._get_higher_order_coeffs()
        low_orders = self._get_lower_order_coeffs(higher_orders)

        return be.concatenate((low_orders, higher_orders))

    def _get_feature_vector(self):
        """
        Return the regression feature vector.

        The visible route uses a fixed 20-dimensional feature vector in the same
        order as the visible regression matrix. The IR route uses the 35-dimensional
        total-degree <= 3 polynomial basis of nd, vd, dPgF, and P_ref.

        Returns:
            be.ndarray: A 1D array representing the feature vector.
        """

        nd, vd, dPgF = self.nd, self.vd, self.dPgf

        # IR routing's feature vector is the total-degree <= 3 polynomial basis of nd,
        # vd, dPgF, and P_ref.
        if self.lambda_ref is not None and self.p_ref is not None:
            x = be.array([nd, vd, dPgF, self.p_ref])
            feats = [be.array(1.0), x[0], x[1], x[2], x[3]]
            for i in range(4):
                for j in range(i, 4):
                    feats.append(x[i] * x[j])
            for i in range(4):
                for j in range(i, 4):
                    for k in range(j, 4):
                        feats.append(x[i] * x[j] * x[k])
            return be.stack(feats)
        # Visible routing's feature vector is a fixed 20-elements vector
        else:
            feats = be.array(
                [
                    1.0,
                    nd,
                    vd,
                    dPgF,
                    nd**2,
                    vd**2,
                    dPgF**2,
                    nd * vd,
                    nd * dPgF,
                    vd * dPgF,
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
            return feats

    def P_ref(self, wavelength):
        """
        Returns the reference partial dispersion P_ref at a given wavelength.

        Args:
            wavelength (float): The wavelength at which to evaluate P_ref.

        Returns:
            float: The reference partial dispersion at the given wavelength.
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
        required_keys = ["nd", "vd", "dPgF"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        return cls(
            data["nd"],
            data["vd"],
            data.get("dPgF"),
            data.get("P_ref"),
            data.get("lambda_ref"),
        )
