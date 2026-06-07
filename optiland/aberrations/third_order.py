"""Third-Order Aberrations

Computes third-order (Seidel) wavefront aberration terms.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland._types import BEArray
    from optiland.optic.context import OpticDataContext


class ThirdOrderAberrations:
    """Third-order (Seidel) wavefront aberration coefficients.

    Args:
        optic: Any object satisfying the OpticDataContext Protocol.
    """

    def __init__(self, optic: OpticDataContext) -> None:
        self._optic = optic

    def third_order(self) -> tuple[BEArray, ...]:
        """Compute all third-order aberrations and first-order color terms.

        Returns:
            A tuple of arrays: TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC,
            TAchC, LchC, TchC, S.
        """
        self._precalculations()
        TSC = self._compute_over_surfaces(self._TSC_term)
        CC = self._compute_over_surfaces(self._CC_term)
        TAC = self._compute_over_surfaces(self._TAC_term)
        TPC = self._compute_over_surfaces(self._TPC_term)
        DC = self._compute_over_surfaces(self._DC_term)
        TAchC = self._compute_over_surfaces(self._TAchC_term)
        TchC = self._compute_over_surfaces(self._TchC_term)

        SC = -TSC / self._ua[-1]
        AC = -TAC / self._ua[-1]
        PC = -TPC / self._ua[-1]
        LchC = -TAchC / self._ua[-1]

        S = self._sum_seidels(TSC, CC, TAC, TPC, DC)
        TCC = CC * 3

        return (
            TSC.flatten(),
            SC.flatten(),
            CC.flatten(),
            TCC.flatten(),
            TAC.flatten(),
            AC.flatten(),
            TPC.flatten(),
            PC.flatten(),
            DC.flatten(),
            TAchC.flatten(),
            LchC.flatten(),
            TchC.flatten(),
            S,
        )

    def TSC(self) -> BEArray:
        """Compute third-order transverse spherical aberration."""
        self._precalculations()
        return self._compute_over_surfaces(self._TSC_term).flatten()

    def SC(self) -> BEArray:
        """Compute third-order longitudinal spherical aberration."""
        self._precalculations()
        TSC = self._compute_over_surfaces(self._TSC_term)
        return (-TSC / self._ua[-1]).flatten()

    def CC(self) -> BEArray:
        """Compute third-order sagittal coma."""
        self._precalculations()
        return self._compute_over_surfaces(self._CC_term).flatten()

    def TCC(self) -> BEArray:
        """Compute third-order tangential coma."""
        return (self.CC() * 3).flatten()

    def TAC(self) -> BEArray:
        """Compute third-order transverse astigmatism."""
        self._precalculations()
        return self._compute_over_surfaces(self._TAC_term).flatten()

    def AC(self) -> BEArray:
        """Compute third-order longitudinal astigmatism."""
        self._precalculations()
        TAC = self._compute_over_surfaces(self._TAC_term)
        return (-TAC / self._ua[-1]).flatten()

    def TPC(self) -> BEArray:
        """Compute third-order transverse Petzval sum."""
        self._precalculations()
        return self._compute_over_surfaces(self._TPC_term).flatten()

    def PC(self) -> BEArray:
        """Compute third-order longitudinal Petzval sum."""
        self._precalculations()
        TPC = self._compute_over_surfaces(self._TPC_term)
        return (-TPC / self._ua[-1]).flatten()

    def DC(self) -> BEArray:
        """Compute third-order distortion."""
        self._precalculations()
        return self._compute_over_surfaces(self._DC_term).flatten()

    def _signed_refractive_indices(self, wavelength: float) -> BEArray:
        n_raw = self._optic.surfaces.n(wavelength)
        n_signed = []
        sign = 1.0
        for k, surf in enumerate(self._optic.surfaces):
            if getattr(surf.interaction_model, "is_reflective", False):
                sign = -sign
            n_signed.append(sign * be.abs(n_raw[k]))
        return be.array(n_signed)

    def _get_conic_term(self, k: int, p_ya: int, p_yb: int) -> float:
        dn = self._n[k] - self._n[k - 1]
        S_conic = (
            dn
            * self._K[k]
            * (self._C[k] ** 3)
            * (self._ya[k] ** p_ya)
            * (self._yb[k] ** p_yb)
        )
        return S_conic / (2 * self._n[-1] * self._ua[-1])

    def _compute_over_surfaces(self, term_func: Callable) -> BEArray:
        terms = [term_func(k) for k in range(1, self._N - 1)]
        return be.array(terms)

    def _precalculations(self) -> None:
        self._inv: float = self._optic.paraxial.invariant()
        self._on_axis = be.isclose(self._inv, be.array(0.0))
        self._n = self._signed_refractive_indices(self._optic.primary_wavelength)
        n_F = self._signed_refractive_indices(0.4861)
        n_C = self._signed_refractive_indices(0.6563)
        self._dn = n_F - n_C

        self._N: int = self._optic.surfaces.num_surfaces
        self._C = 1 / self._optic.surfaces.radii
        self._ya, self._ua = self._optic.paraxial.marginal_ray()
        self._yb, self._ub = self._optic.paraxial.chief_ray()
        self._hp = self._inv / (self._n[-1] * self._ua[-1])
        self._K = self._optic.surfaces.conic

        i_list, ip_list, B_list, Bp_list = [], [], [], []
        for k in range(1, self._N - 1):
            i_val = (self._C[k] * self._ya[k] + self._ua[k - 1])[0]
            ip_val = (self._C[k] * self._yb[k] + self._ub[k - 1])[0]
            i_list.append(i_val)
            ip_list.append(ip_val)

            if self._on_axis:
                B_list.append(0)
                Bp_list.append(0)
            else:
                denom = 2 * self._n[k] * self._inv
                B_val = (
                    self._n[k - 1]
                    * (self._n[k] - self._n[k - 1])
                    * self._ya[k]
                    * (self._ua[k] + i_val)
                    / denom
                )[0]
                Bp_val = (
                    self._n[k - 1]
                    * (self._n[k] - self._n[k - 1])
                    * self._yb[k]
                    * (self._ub[k] + ip_val)
                    / denom
                )[0]
                B_list.append(B_val)
                Bp_list.append(Bp_val)

        self._i = be.array(i_list)
        self._ip = be.array(ip_list)
        self._B = be.array(B_list)
        self._Bp = be.array(Bp_list)

    def _TSC_on_axis_term(self, k: int) -> float:
        i_val = self._C[k] * self._ya[k] + self._ua[k - 1]
        term = (
            self._n[k - 1]
            * (self._n[k] - self._n[k - 1])
            * self._ya[k]
            * (self._ua[k] + i_val)
            * i_val**2
        )
        spherical = term / (2 * self._n[k] * self._n[-1] * self._ua[-1])
        return spherical + self._get_conic_term(k, p_ya=4, p_yb=0)

    def _TSC_term(self, k: int) -> float:
        if self._on_axis:
            return self._TSC_on_axis_term(k)
        spherical = self._B[k - 1] * self._i[k - 1] ** 2 * self._hp
        return spherical + self._get_conic_term(k, p_ya=4, p_yb=0)

    def _CC_term(self, k: int) -> float:
        spherical = self._B[k - 1] * self._i[k - 1] * self._ip[k - 1] * self._hp
        return spherical + self._get_conic_term(k, p_ya=3, p_yb=1)

    def _TAC_term(self, k: int) -> float:
        spherical = self._B[k - 1] * self._ip[k - 1] ** 2 * self._hp
        return spherical + self._get_conic_term(k, p_ya=2, p_yb=2)

    def _TPC_term(self, k: int) -> BEArray:
        return (
            (self._n[k] - self._n[k - 1])
            * self._C[k]
            * self._hp
            * self._inv
            / (2 * self._n[k] * self._n[k - 1])
        )

    def _DC_term(self, k: int) -> BEArray:
        spherical = self._hp * (
            self._Bp[k - 1] * self._i[k - 1] * self._ip[k - 1]
            + 0.5 * (self._ub[k] ** 2 - self._ub[k - 1] ** 2)
        )
        return spherical + self._get_conic_term(k, p_ya=1, p_yb=3)

    def _TAchC_term(self, k: int) -> BEArray:
        return (
            -self._ya[k - 1]
            * self._i[k - 1]
            / (self._n[-1] * self._ua[-1])
            * (self._dn[k - 1] - self._n[k - 1] / self._n[k] * self._dn[k])
        )

    def _TchC_term(self, k: int) -> BEArray:
        return (
            -self._ya[k - 1]
            * self._ip[k - 1]
            / (self._n[-1] * self._ua[-1])
            * (self._dn[k - 1] - self._n[k - 1] / self._n[k] * self._dn[k])
        )

    def _sum_seidels(
        self,
        TSC: BEArray,
        CC: BEArray,
        TAC: BEArray,
        TPC: BEArray,
        DC: BEArray,
    ) -> BEArray:
        factor = self._n[-1] * self._ua[-1] * 2
        return be.array(
            [
                -be.sum(TSC) * factor,
                -be.sum(CC) * factor,
                -be.sum(TAC) * factor,
                -be.sum(TPC) * factor,
                -be.sum(DC) * factor,
            ]
        )
