"""Jones Module

The jones module contains classes for Jones matrices in optics. The module
defines the base class BaseJones, which is an abstract class that defines the
interface for Jones matrices. The module also contains classes for specific
Jones matrices, such as JonesFresnel, JonesPolarizerH, JonesPolarizerV,
JonesPolarizerL45, JonesPolarizerL135, JonesPolarizerRCP, JonesPolarizerLCP,
JonesLinearDiattenuator, JonesLinearRetarder, JonesQuarterWaveRetarder, and
JonesHalfWaveRetarder.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:  # pragma: no cover
    from optiland.rays import RealRays


class BaseJones(ABC):
    """Base class for Jones matrices.

    The class defines Jones matrices given ray properties. In the general case,
    the Jones matrix is an Nx3x3 array, where N is the number of rays. The
    array is padded to make it 3x3 to account for 3D ray calculations.
    """

    @abstractmethod
    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        return be.tile(be.eye(3), (be.size(rays.x), 1, 1))  # pragma: no cover


class JonesFresnel(BaseJones):
    """Class representing the Jones matrix for Fresnel calculations.

    Args:
        material_pre (Material): Material object representing the
            material before the surface.
        material_post (Material): Material object representing the
            material after the surface.

    """

    def __init__(self, material_pre, material_post):
        self.material_pre = material_pre
        self.material_post = material_post

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        # define local variables
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # precomputations for speed
        cos_theta_i = be.cos(aoi)
        n = n2 / n1
        radicand = be.to_complex(n**2 - be.sin(aoi) ** 2)
        root = be.sqrt(radicand)

        # compute fresnel coefficients & compute jones matrices
        jones_matrix = be.to_complex(be.zeros((be.size(rays.x), 3, 3)))
        if reflect:
            s = (cos_theta_i - root) / (cos_theta_i + root)
            p = (n**2 * cos_theta_i - root) / (n**2 * cos_theta_i + root)

            jones_matrix[:, 0, 0] = s
            jones_matrix[:, 1, 1] = -p
            jones_matrix[:, 2, 2] = -1
        else:
            s = 2 * cos_theta_i / (cos_theta_i + root)
            p = 2 * n * cos_theta_i / (n**2 * cos_theta_i + root)

            jones_matrix[:, 0, 0] = s
            jones_matrix[:, 1, 1] = p
            jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesLinearPolarizer(BaseJones):
    """Class representing a general linear polarizer in 3D space.

    Args:
        axis (tuple | list | be.ndarray): A 3D vector representing the transmission
            axis in global coordinates (e.g., [1, 0, 0] for horizontal).
    """

    def __init__(self, axis):
        self.axis = be.array(axis)
        self.axis = self.axis / be.linalg.norm(self.axis)

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected.
            aoi (be.ndarray, optional): Array representing the angle of incidence.

        Returns:
            be.ndarray: The calculated Jones matrix.
        """
        from optiland.rays.polarized_rays import PolarizedRays  # noqa: PLC0415

        k0 = be.stack([rays.L0, rays.M0, rays.N0]).T
        k1 = be.stack([rays.L, rays.M, rays.N]).T

        s, p0, p1, o_in, o_out = PolarizedRays.get_local_basis(k0, k1)

        # Broadcast axis to match rays
        axis_b = be.broadcast_to(self.axis, k0.shape)

        # Project transmission axis onto local incident and exit planes
        ts_in = be.sum(axis_b * s, axis=1)
        tp_in = be.sum(axis_b * p0, axis=1)
        norm_in = be.sqrt(ts_in**2 + tp_in**2)
        norm_in = be.where(norm_in == 0, be.ones_like(norm_in), norm_in)

        ts_out = be.sum(axis_b * s, axis=1)
        tp_out = be.sum(axis_b * p1, axis=1)
        norm_out = be.sqrt(ts_out**2 + tp_out**2)
        norm_out = be.where(norm_out == 0, be.ones_like(norm_out), norm_out)

        us_in = ts_in / norm_in
        up_in = tp_in / norm_in
        us_out = ts_out / norm_out
        up_out = tp_out / norm_out

        jones_matrix = be.to_complex(be.zeros((be.size(rays.x), 3, 3)))
        jones_matrix[:, 0, 0] = us_out * us_in
        jones_matrix[:, 0, 1] = us_out * up_in
        jones_matrix[:, 1, 0] = up_out * us_in
        jones_matrix[:, 1, 1] = up_out * up_in
        jones_matrix[:, 2, 2] = 1.0

        return jones_matrix


class JonesPolarizerH(JonesLinearPolarizer):
    """Class representing the Jones matrix for a horizontal polarizer."""

    def __init__(self):
        super().__init__([1, 0, 0])


class JonesPolarizerV(JonesLinearPolarizer):
    """Class representing the Jones matrix for a vertical polarizer."""

    def __init__(self):
        super().__init__([0, 1, 0])


class JonesPolarizerL45(JonesLinearPolarizer):
    """Class representing the Jones matrix for a linear polarizer at 45 degrees."""

    def __init__(self):
        # 45 deg in X-Y plane
        val = 1.0 / be.sqrt(be.array(2.0))
        super().__init__([val, val, 0])


class JonesPolarizerL135(JonesLinearPolarizer):
    """Class representing the Jones matrix for a linear polarizer at 135 degrees."""

    def __init__(self):
        val = 1.0 / be.sqrt(be.array(2.0))
        super().__init__([-val, val, 0])


class ConstantJones(BaseJones):
    """Base class for constant Jones matrices in the local ray frame.

    Args:
        j00 (complex): The (0, 0) element of the Jones matrix.
        j01 (complex): The (0, 1) element of the Jones matrix.
        j10 (complex): The (1, 0) element of the Jones matrix.
        j11 (complex): The (1, 1) element of the Jones matrix.
    """

    def __init__(self, j00: complex, j01: complex, j10: complex, j11: complex):
        self.j00 = j00
        self.j01 = j01
        self.j10 = j10
        self.j11 = j11

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays."""
        jones_matrix = be.to_complex(be.zeros((be.size(rays.x), 3, 3)))
        jones_matrix[:, 0, 0] = self.j00
        jones_matrix[:, 0, 1] = self.j01
        jones_matrix[:, 1, 0] = self.j10
        jones_matrix[:, 1, 1] = self.j11
        jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesPolarizerRCP(ConstantJones):
    """Class representing the Jones matrix for a right circular polarizer."""

    def __init__(self):
        super().__init__(0.5, 1j * 0.5, -1j * 0.5, 0.5)


class JonesPolarizerLCP(ConstantJones):
    """Class representing the Jones matrix for a left circular polarizer."""

    def __init__(self):
        super().__init__(0.5, -1j * 0.5, 1j * 0.5, 0.5)


class JonesLinearDiattenuator(BaseJones):
    """Represents a linear diattenuator in Jones calculus.

    Attributes:
        t_min (be.ndarray): Minimum amplitude transmission coefficient.
        t_max (be.ndarray): Maximum amplitude transmission coefficient.
        axis (be.ndarray): A 3D vector representing the fast transmission axis.

    Note:
        The intensity transmission is given by the square of the amplitude
        coefficients.
    """

    def __init__(self, t_min, t_max, axis=None, *, theta=None):
        self.t_min = be.array(t_min)
        self.t_max = be.array(t_max)

        if axis is not None and (
            isinstance(axis, int | float) or be.size(be.array(axis)) == 1
        ):
            theta = axis
            axis = None

        if axis is not None:
            self.axis = be.array(axis)
            self.axis = self.axis / be.linalg.norm(self.axis)
        elif theta is not None:
            self.axis = be.array([be.cos(theta), be.sin(theta), 0.0])
        else:
            self.axis = be.array([1.0, 0.0, 0.0])

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays."""
        from optiland.rays.polarized_rays import PolarizedRays  # noqa: PLC0415

        k0 = be.stack([rays.L0, rays.M0, rays.N0]).T
        k1 = be.stack([rays.L, rays.M, rays.N]).T
        s, p0, p1, o_in, o_out = PolarizedRays.get_local_basis(k0, k1)

        axis_b = be.broadcast_to(self.axis, k0.shape)
        ts_in = be.sum(axis_b * s, axis=1)
        tp_in = be.sum(axis_b * p0, axis=1)
        norm_in = be.sqrt(ts_in**2 + tp_in**2)
        norm_in = be.where(norm_in == 0, be.ones_like(norm_in), norm_in)

        us = ts_in / norm_in
        up = tp_in / norm_in

        j00 = self.t_max * us**2 + self.t_min * up**2
        j0x = (
            self.t_max * us * up - self.t_min * us * up
        )  # t_max*c*s - t_min*c*s = (t_max - t_min) * us * up
        j11 = self.t_max * up**2 + self.t_min * us**2

        jones_matrix = be.to_complex(be.zeros((be.size(rays.x), 3, 3)))
        jones_matrix[:, 0, 0] = j00
        jones_matrix[:, 0, 1] = j0x
        jones_matrix[:, 1, 0] = j0x
        jones_matrix[:, 1, 1] = j11
        jones_matrix[:, 2, 2] = 1.0

        return jones_matrix


class JonesLinearRetarder(BaseJones):
    """Represents a linear retarder in Jones calculus.

    Attributes:
        retardance (be.ndarray): Retardance of the retarder, or the absolute value
            of the phase difference between the two components of the electric
            field, in radians.
        axis (be.ndarray): A 3D vector representing the fast transmission axis.
    """

    def __init__(self, retardance, axis=None, *, theta=None):
        self.retardance = be.array(retardance)

        if axis is not None and (
            isinstance(axis, int | float) or be.size(be.array(axis)) == 1
        ):
            theta = axis
            axis = None

        if axis is not None:
            self.axis = be.array(axis)
            self.axis = self.axis / be.linalg.norm(self.axis)
        elif theta is not None:
            self.axis = be.array([be.cos(theta), be.sin(theta), 0.0])
        else:
            self.axis = be.array([1.0, 0.0, 0.0])

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays."""
        from optiland.rays.polarized_rays import PolarizedRays  # noqa: PLC0415

        d = self.retardance

        k0 = be.stack([rays.L0, rays.M0, rays.N0]).T
        k1 = be.stack([rays.L, rays.M, rays.N]).T
        s, p0, p1, o_in, o_out = PolarizedRays.get_local_basis(k0, k1)

        axis_b = be.broadcast_to(self.axis, k0.shape)
        ts_in = be.sum(axis_b * s, axis=1)
        tp_in = be.sum(axis_b * p0, axis=1)
        norm_in = be.sqrt(ts_in**2 + tp_in**2)
        norm_in = be.where(norm_in == 0, be.ones_like(norm_in), norm_in)

        us = ts_in / norm_in
        up = tp_in / norm_in

        j00 = be.exp(-1j * d / 2) * us**2 + be.exp(1j * d / 2) * up**2
        j0x = -2j * be.sin(d / 2) * us * up
        j11 = be.exp(1j * d / 2) * us**2 + be.exp(-1j * d / 2) * up**2

        jones_matrix = be.to_complex(be.zeros((be.size(rays.x), 3, 3)))
        jones_matrix[:, 0, 0] = j00
        jones_matrix[:, 0, 1] = j0x
        jones_matrix[:, 1, 0] = j0x
        jones_matrix[:, 1, 1] = j11
        jones_matrix[:, 2, 2] = 1.0

        return jones_matrix


class JonesQuarterWaveRetarder(JonesLinearRetarder):
    """Represents a quarter-wave retarder in Jones calculus."""

    def __init__(self, axis=None, *, theta=None):
        if axis is None and theta is None:
            theta = 0
        super().__init__(be.pi / 2, axis, theta=theta)


class JonesHalfWaveRetarder(JonesLinearRetarder):
    """Represents a half-wave retarder in Jones calculus."""

    def __init__(self, axis=None, *, theta=None):
        if axis is None and theta is None:
            theta = 0
        super().__init__(be.pi, axis, theta=theta)
