"""
Aberrations Module

This module computes first- and third-order aberrations of optical systems.
The aberration calculations are based on the algorithms outlined in
Modern Optical Engineering by Warren Smith (Chapter 6.3).

Kramer Harrison, 2023
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland._types import BEArray
    from optiland.optic import Optic


class Aberrations:
    """Class for computation of optical aberrations.

    This class provides methods to compute first- and third-order aberrations
    of a general optical system, using an instance of an optic.Optic object.

    Args:
        optic (Optic): An instance of the optical system to be analyzed.
    """

    def __init__(self, optic: Optic):
        self.optic = optic

    def third_order(self) -> tuple[BEArray, ...]:
        """
        Compute all third-order aberrations and first-order color terms.

        Returns:
            A tuple containing the following `be.ndarray` instances:
                - TSC: Third-order transverse spherical aberration.
                - SC: Third-order longitudinal spherical aberration.
                - CC: Third-order sagittal coma.
                - TCC: Third-order tangential coma (3×CC).
                - TAC: Third-order transverse astigmatism.
                - AC: Third-order longitudinal astigmatism.
                - TPC: Third-order transverse Petzval sum.
                - PC: Third-order longitudinal Petzval sum.
                - DC: Third-order distortion.
                - TAchC: First-order transverse axial color.
                - LchC: First-order longitudinal axial color.
                - TchC: First-order lateral color.
                - S: Seidel aberration coefficients.
        """
        self._precalculations()

        # Compute aberration terms over surfaces 1 to N-2
        TSC = self._compute_over_surfaces(self._TSC_term)
        CC = self._compute_over_surfaces(self._CC_term)
        TAC = self._compute_over_surfaces(self._TAC_term)
        TPC = self._compute_over_surfaces(self._TPC_term)
        DC = self._compute_over_surfaces(self._DC_term)
        TAchC = self._compute_over_surfaces(self._TAchC_term)
        TchC = self._compute_over_surfaces(self._TchC_term)

        # Compute derived terms using the computed aberration values
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

    def seidels(self) -> BEArray:
        """
        Compute the Seidel aberration coefficients.

        Returns:
            Array of Seidel aberration coefficients.
        """
        self._precalculations()
        TSC = self._compute_over_surfaces(self._TSC_term)
        CC = self._compute_over_surfaces(self._CC_term)
        TAC = self._compute_over_surfaces(self._TAC_term)
        TPC = self._compute_over_surfaces(self._TPC_term)
        DC = self._compute_over_surfaces(self._DC_term)
        S = self._sum_seidels(TSC, CC, TAC, TPC, DC)
        return S.squeeze()

    def TSC(self) -> BEArray:
        """
        Compute third-order transverse spherical aberration.

        Returns:
            Third-order transverse spherical aberration.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TSC_term).flatten()

    def SC(self) -> BEArray:
        """
        Compute third-order longitudinal spherical aberration.

        Returns:
            Third-order longitudinal spherical aberration.
        """
        self._precalculations()
        TSC = self._compute_over_surfaces(self._TSC_term)
        SC = -TSC / self._ua[-1]
        return SC.flatten()

    def CC(self) -> BEArray:
        """
        Compute third-order sagittal coma.

        Returns:
            Third-order sagittal coma.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._CC_term).flatten()

    def TCC(self) -> BEArray:
        """
        Compute third-order tangential coma.

        Returns:
            Third-order tangential coma.
        """
        return (self.CC() * 3).flatten()

    def TAC(self) -> BEArray:
        """
        Compute third-order transverse astigmatism.

        Returns:
            Third-order transverse astigmatism.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TAC_term).flatten()

    def AC(self) -> BEArray:
        """
        Compute third-order longitudinal astigmatism.

        Returns:
            Third-order longitudinal astigmatism.
        """
        self._precalculations()
        TAC = self._compute_over_surfaces(self._TAC_term)
        AC = -TAC / self._ua[-1]
        return AC.flatten()

    def TPC(self) -> BEArray:
        """
        Compute third-order transverse Petzval sum.

        Returns:
            Third-order transverse Petzval sum.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TPC_term).flatten()

    def PC(self) -> BEArray:
        """
        Compute third-order longitudinal Petzval sum.

        Returns:
            Third-order longitudinal Petzval sum.
        """
        self._precalculations()
        TPC = self._compute_over_surfaces(self._TPC_term)
        PC = -TPC / self._ua[-1]
        return PC.flatten()

    def DC(self) -> BEArray:
        """
        Compute third-order distortion.

        Returns:
            Third-order distortion.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._DC_term).flatten()

    def TAchC(self) -> BEArray:
        """
        Compute first-order transverse axial color.

        Returns:
            First-order transverse axial color.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TAchC_term).flatten()

    def LchC(self) -> BEArray:
        """
        Compute first-order longitudinal axial color.

        Returns:
            First-order longitudinal axial color.
        """
        self._precalculations()
        TAchC = self._compute_over_surfaces(self._TAchC_term)
        LchC = -TAchC / self._ua[-1]
        return LchC.flatten()

    def TchC(self) -> BEArray:
        """
        Compute first-order lateral color.

        Returns:
            First-order lateral color.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TchC_term).flatten()

    def _signed_refractive_indices(self, wavelength: float) -> BEArray:
        """Return signed refractive indices following the Welford convention.

        For Seidel third-order aberration formulas to be valid in systems
        containing mirrors, the post-mirror medium is treated as having a
        sign-flipped refractive index. Each reflection toggles the sign for
        all subsequent surfaces — so a two-mirror system ends with the
        original sign, a three-mirror system with the flipped sign, etc.

        ``surfaces.n()`` itself returns magnitudes only (materials don't
        carry direction), which is why the Seidel formulas previously
        evaluated to zero on every reflective surface.

        Args:
            wavelength: Wavelength at which to evaluate the indices.

        Returns:
            Array of signed refractive indices, one per surface.
        """
        n_raw = self.optic.surfaces.n(wavelength)
        n_signed = []
        sign = 1.0
        for k, surf in enumerate(self.optic.surfaces):
            if getattr(surf.interaction_model, "is_reflective", False):
                sign = -sign
            # ``be.abs`` because the underlying material may already report a
            # signed n on some backends; we standardize on magnitude × sign.
            n_signed.append(sign * be.abs(n_raw[k]))
        return be.array(n_signed)

    def _get_conic_term(self, k: int, p_ya: int, p_yb: int) -> float:
        """
        Compute the transverse conic contribution for surface k.
        """
        dn = self._n[k] - self._n[k - 1]

        # Conic Seidel sum: S_conic = dn * K * c^3 * ya^p_ya * yb^p_yb
        S_conic = (
            dn
            * self._K[k]
            * (self._C[k] ** 3)
            * (self._ya[k] ** p_ya)
            * (self._yb[k] ** p_yb)
        )

        # Convert Wavefront Seidel Sum to Transverse Aberration contribution
        return S_conic / (2 * self._n[-1] * self._ua[-1])

    def _compute_over_surfaces(self, term_func: Callable) -> BEArray:
        """
        Compute a given aberration term over all relevant surfaces.

        Args:
            term_func (callable): Function that computes a term for a given
                surface index.

        Returns:
            Array of computed term values over surfaces 1 to N-2.
        """
        terms = [term_func(k) for k in range(1, self._N - 1)]
        return be.array(terms)

    def _precalculations(self):
        """
        Perform all necessary precalculations for aberration computations.

        This method computes and stores common parameters required by the
        various aberration term methods.
        """
        self._inv: float = self.optic.paraxial.invariant()  # Lagrange invariant
        self._on_axis = be.isclose(self._inv, be.array(0.0))
        self._n = self._signed_refractive_indices(self.optic.primary_wavelength)
        n_F = self._signed_refractive_indices(0.4861)
        n_C = self._signed_refractive_indices(0.6563)
        self._dn = n_F - n_C

        self._N: int = self.optic.surfaces.num_surfaces
        self._C = 1 / self.optic.surfaces.radii
        self._ya, self._ua = self.optic.paraxial.marginal_ray()
        self._yb, self._ub = self.optic.paraxial.chief_ray()
        self._hp = self._inv / (self._n[-1] * self._ua[-1])

        self._K = self.optic.surfaces.conic

        i_list = []
        ip_list = []
        B_list = []
        Bp_list = []

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
        """
        Compute third-order transverse spherical aberration term for surface k
        when the system is on-axis.

        Args:
            k: Surface index.

        Returns:
            Computed transverse spherical aberration term.
        """
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
        """
        Compute third-order transverse spherical aberration term for surface k.

        Args:
            k: Surface index.

        Returns:
            Computed transverse spherical aberration term.
        """
        if self._on_axis:
            return self._TSC_on_axis_term(k)
        spherical = self._B[k - 1] * self._i[k - 1] ** 2 * self._hp
        return spherical + self._get_conic_term(k, p_ya=4, p_yb=0)

    def _CC_term(self, k: int) -> float:
        """
        Compute third-order sagittal coma term for surface k.

        Args:
            k: Surface index.

        Returns:
            Computed sagittal coma term.
        """
        spherical = self._B[k - 1] * self._i[k - 1] * self._ip[k - 1] * self._hp
        return spherical + self._get_conic_term(k, p_ya=3, p_yb=1)

    def _TAC_term(self, k: int) -> float:
        """
        Compute third-order transverse astigmatism term for surface k.

        Args:
            k: Surface index.

        Returns:
            Computed transverse astigmatism term.
        """
        spherical = self._B[k - 1] * self._ip[k - 1] ** 2 * self._hp
        return spherical + self._get_conic_term(k, p_ya=2, p_yb=2)

    def _TPC_term(self, k: int) -> BEArray:
        """
        Compute third-order transverse Petzval sum term for surface k.

        Args:
            k: Surface index.

        Returns:
            Computed transverse Petzval sum term.
        """
        return (
            (self._n[k] - self._n[k - 1])
            * self._C[k]
            * self._hp
            * self._inv
            / (2 * self._n[k] * self._n[k - 1])
        )

    def _DC_term(self, k: int) -> BEArray:
        """
        Compute third-order distortion term for surface k.

        Args:
            k: Surface index.

        Returns:
            Computed distortion term.
        """
        spherical = self._hp * (
            self._Bp[k - 1] * self._i[k - 1] * self._ip[k - 1]
            + 0.5 * (self._ub[k] ** 2 - self._ub[k - 1] ** 2)
        )
        return spherical + self._get_conic_term(k, p_ya=1, p_yb=3)

    def _TAchC_term(self, k: int) -> BEArray:
        """
        Compute first-order transverse axial color term for surface k.

        Args:
            k: Surface index.

        Returns:
            Computed transverse axial color term.
        """
        return (
            -self._ya[k - 1]
            * self._i[k - 1]
            / (self._n[-1] * self._ua[-1])
            * (self._dn[k - 1] - self._n[k - 1] / self._n[k] * self._dn[k])
        )

    def _TchC_term(self, k: int) -> BEArray:
        """
        Compute first-order lateral color term for surface k.

        Args:
            k: Surface index.

        Returns:
            Computed lateral color term.
        """
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
        """
        Sum the Seidel aberration coefficients from the individual terms.

        Args:
            TSC: Transverse spherical aberration terms.
            CC: Sagittal coma terms.
            TAC: Transverse astigmatism terms.
            TPC: Transverse Petzval sum terms.
            DC: Distortion terms.

        Returns:
            Array of Seidel aberration coefficients.
        """
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
