"""Polarized Rays

This module contains the `PolarizedRays` class, which represents a class for
polarized rays in three-dimensional space. The class inherits from the
`RealRays` class.

Kramer Harrison, 2024
"""

from __future__ import annotations

import optiland.backend as be
from optiland.rays.polarization_state import PolarizationState
from optiland.rays.real_rays import RealRays


class PolarizedRays(RealRays):
    """Represents a class for polarized rays in three-dimensional space.

    Inherits from the `RealRays` class.

    Attributes:
        x (ndarray): The x-coordinates of the rays.
        y (ndarray): The y-coordinates of the rays.
        z (ndarray): The z-coordinates of the rays.
        L (ndarray): The x-components of the direction vectors of the rays.
        M (ndarray): The y-components of the direction vectors of the rays.
        N (ndarray): The z-components of the direction vectors of the rays.
        i (ndarray): The intensity of the rays.
        w (ndarray): The wavelength of the rays.
        opd (ndarray): The optical path length of the rays.
        p (be.ndarray): Array of polarization matrices of the rays.

    Methods:
        get_output_field(E: be.ndarray) -> be.ndarray:
            Compute the output electric field given the input electric field.
        update_intensity(state: PolarizationState):
            Update the ray intensity based on the polarization state.
        update(jones_matrix: be.ndarray = None):
            Update the polarization matrices after interaction with a surface.
        _get_3d_electric_field(state: PolarizationState) -> be.ndarray:
            Get the 3D electric fields given the polarization state and
            initial rays.

    """

    def __init__(self, x, y, z, L, M, N, intensity, wavelength):
        super().__init__(x, y, z, L, M, N, intensity, wavelength)

        self.p = be.tile(be.eye(3), (be.size(self.x), 1, 1))
        self._i0 = be.copy(intensity)
        self._L0 = be.copy(L)
        self._M0 = be.copy(M)
        self._N0 = be.copy(N)

    def get_output_field(self, E: be.ndarray) -> be.ndarray:
        """Compute the output electric field given the input electric field.

        Args:
            E (be.ndarray): The input electric field as a numpy array.

        Returns:
            be.ndarray: The computed output electric field as a numpy array.

        """
        return be.mult_p_E(self.p, E)

    def _compute_unscaled_exit_fields(
        self, state: PolarizationState | None
    ) -> list[be.ndarray]:
        """Compute the unscaled exit electric field(s) for the rays.

        Args:
            state (PolarizationState | None): The polarization state.

        Returns:
            list[be.ndarray]: A list of unscaled 3D electric field arrays.
        """
        if state is not None and state.is_polarized:
            E0 = self._get_3d_electric_field(state)
            E1 = self.get_output_field(E0)
            return [E1]
        else:
            state_x = PolarizationState(
                is_polarized=True,
                Ex=1.0,
                Ey=0.0,
                phase_x=0.0,
                phase_y=0.0,
            )
            E0_x = self._get_3d_electric_field(state_x)
            E1_x = self.get_output_field(E0_x)

            state_y = PolarizationState(
                is_polarized=True,
                Ex=0.0,
                Ey=1.0,
                phase_x=0.0,
                phase_y=0.0,
            )
            E0_y = self._get_3d_electric_field(state_y)
            E1_y = self.get_output_field(E0_y)

            return [E1_x, E1_y]

    def get_exit_fields(self, state: PolarizationState | None) -> list[be.ndarray]:
        """Compute the exit electric field(s) for the rays.

        Args:
            state (PolarizationState | None): The polarization state.

        Returns:
            list[be.ndarray]: A list of 3D electric field arrays. For polarized
            light, the list contains a single array. For unpolarized light, the
            list contains two orthogonal, incoherently superimposed arrays, each
            scaled down by 1/sqrt(2).
        """
        fields = self._compute_unscaled_exit_fields(state)
        scale_factor = be.unsqueeze_last(be.sqrt(self._i0 / len(fields)))
        return [E1 * scale_factor for E1 in fields]

    def update_intensity(self, state: PolarizationState):
        """Update ray intensity based on polarization state.

        Args:
            state (PolarizationState): The polarization state of the ray.

        """
        fields = self._compute_unscaled_exit_fields(state)
        intensity = be.zeros_like(self.i)
        for E1 in fields:
            intensity = intensity + be.sum(be.abs(E1) ** 2, axis=1)
        self.i = intensity * self._i0 / len(fields)

    @staticmethod
    def get_local_basis(
        k0: be.ndarray, k1: be.ndarray
    ) -> tuple[be.ndarray, be.ndarray, be.ndarray, be.ndarray]:
        """Get the local s, p0, p1 vectors and transforming matrices.

        Args:
            k0: (N, 3) array of pre-interaction ray directions.
            k1: (N, 3) array of post-interaction ray directions.

        Returns:
            tuple: (s, p0, p1, o_in, o_out) where s, p0, p1 are (N, 3) vectors
            and o_in, o_out are the projection matrices.
        """
        # find s-component
        s = be.cross(k0, k1)
        mag = be.linalg.norm(s, axis=1)

        # handle case when mag = 0 (i.e., k0 parallel to k1)
        mask = mag == 0
        if be.any(mask):
            x = be.broadcast_to(be.array([1.0, 0.0, 0.0]), k0[mask].shape)
            p_fallback = be.cross(k0[mask], x)

            p_norms = be.linalg.norm(p_fallback, axis=1)
            y = be.broadcast_to(be.array([0.0, 1.0, 0.0]), k0[mask].shape)
            p_fallback = be.where(
                be.unsqueeze_last(p_norms == 0), be.cross(k0[mask], y), p_fallback
            )

            s[mask] = be.cross(p_fallback, k0[mask])
            mag = be.linalg.norm(s, axis=1)

        s = s / be.unsqueeze_last(mag)

        # find p-component pre and post surface
        p0 = be.cross(k0, s)
        p1 = be.cross(k1, s)

        # othogonal transformation matrices
        o_in = be.stack((s, p0, k0), axis=1)
        o_out = be.stack((s, p1, k1), axis=2)

        return s, p0, p1, o_in, o_out

    def update(self, jones_matrix: be.ndarray = None):
        """Update polarization matrices after interaction with surface.

        Args:
            jones_matrix (be.ndarray, optional): Jones matrix representing the
                interaction with the surface. If not provided, the
                polarization matrix is computed assuming an identity matrix.

        """
        # merge k-vector components into matrix for speed
        k0 = be.stack([self.L0, self.M0, self.N0]).T
        k1 = be.stack([self.L, self.M, self.N]).T

        s, p0, p1, o_in, o_out = self.get_local_basis(k0, k1)

        # compute polarization matrix for surface
        if jones_matrix is None:
            p = be.matmul(o_out, o_in)
        else:
            p = be.batched_chain_matmul3(o_out, jones_matrix, o_in)

        # update polarization matrices of rays
        self.p = be.matmul(p, self.p)

    def _get_3d_electric_field(self, state: PolarizationState) -> be.ndarray:
        """Get 3D electric fields given polarization state and initial rays.

        Args:
            state (PolarizationState): The polarization state of the rays.

        Returns:
            be.ndarray: The 3D electric fields.

        """
        k = be.stack([self._L0, self._M0, self._N0]).T

        # TODO - efficiently handle case when k parallel to x-axis
        x = be.broadcast_to(be.array([1.0, 0.0, 0.0]), k.shape)
        p = be.cross(k, x)

        norms = be.linalg.norm(p, axis=1)
        if be.any(norms == 0):
            raise ValueError("k-vector parallel to x-axis is not currently supported.")

        p = p / be.unsqueeze_last(norms)

        s = be.cross(p, k)

        E = (
            state.Ex * be.exp(1j * state.phase_x) * s
            + state.Ey * be.exp(1j * state.phase_y) * p
        )

        return E
