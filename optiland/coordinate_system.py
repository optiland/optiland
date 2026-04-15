"""Coordinate System Module

This module provides standard coordinate system transformation calculations.
The CoordinateSystem class represents a coordinate system in 3D space and
provides methods for localizing and globalizing rays. This class is used
to define the position and orientation of all optical surfaces in an optical
system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.spatial.transform import Rotation as R

import optiland.backend as be
from optiland.rays import RealRays

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from optiland._types import BEArray, ScalarOrArray


class CoordinateSystem:
    """Represents a coordinate system in 3D space.

    Attributes:
        x: The x-coordinate of the origin.
        y: The y-coordinate of the origin.
        z: The z-coordinate of the origin.
        rx: The rotation around the x-axis.
        ry: The rotation around the y-axis.
        rz: The rotation around the z-axis.
        reference_cs: The reference coordinate system.
    """

    def __init__(
        self,
        x: ScalarOrArray = 0,
        y: ScalarOrArray = 0,
        z: ScalarOrArray = 0,
        rx: ScalarOrArray = 0,
        ry: ScalarOrArray = 0,
        rz: ScalarOrArray = 0,
        reference_cs: CoordinateSystem | None = None,
    ):
        """Initialize a orthogonal coordinate system.

        Args:
            x: The x-coordinate of the origin.
            y: The y-coordinate of the origin.
            z: The z-coordinate of the origin.
            rx: The rotation around the x-axis.
            ry: The rotation around the y-axis.
            rz: The rotation around the z-axis.
            reference_cs: The reference coordinate system.
        """

        self.x = be.array(x)
        self.y = be.array(y)
        self.z = be.array(z)

        self.rx = be.array(rx)
        self.ry = be.array(ry)
        self.rz = be.array(rz)

        self.reference_cs = reference_cs

    def localize(self, rays: RealRays):
        """Localizes the rays in the coordinate system.

        Args:
            rays: The rays to be localized.

        """
        if self.reference_cs:
            self.reference_cs.localize(rays)

        rays.translate(-self.x, -self.y, -self.z)
        if self.rz:
            rays.rotate_z(-self.rz)
        if self.ry:
            rays.rotate_y(-self.ry)
        if self.rx:
            rays.rotate_x(-self.rx)

    def globalize(self, rays: RealRays):
        """Globalizes the rays from the coordinate system.

        Args:
            rays: The rays to be globalized.

        """
        if self.rx:
            rays.rotate_x(self.rx)
        if self.ry:
            rays.rotate_y(self.ry)
        if self.rz:
            rays.rotate_z(self.rz)
        rays.translate(self.x, self.y, self.z)

        if self.reference_cs:
            self.reference_cs.globalize(rays)

    @property
    def position_in_gcs(self) -> tuple[BEArray, BEArray, BEArray]:
        """Returns the position of the coordinate system in the global coordinate
            system.

        Returns:
            A tuple containing the x, y, and z coordinates of the position.
        """
        vector = RealRays(0, 0, 0, 0, 0, 1, 1, 1)
        self.globalize(vector)
        return vector.x, vector.y, vector.z

    def get_rotation_matrix(self) -> BEArray:
        """Get the rotation matrix of the coordinate system

        Returns:
            The rotation matrix of the coordinate system.
        """
        rx, ry, rz = self.rx, self.ry, self.rz

        Rx = be.array(
            [[1, 0, 0], [0, be.cos(rx), -be.sin(rx)], [0, be.sin(rx), be.cos(rx)]]
        )

        Ry = be.array(
            [[be.cos(ry), 0, be.sin(ry)], [0, 1, 0], [-be.sin(ry), 0, be.cos(ry)]]
        )

        Rz = be.array(
            [[be.cos(rz), -be.sin(rz), 0], [be.sin(rz), be.cos(rz), 0], [0, 0, 1]]
        )

        R = Rz @ Ry @ Rx

        return R

    def get_effective_transform(self) -> tuple[BEArray, BEArray]:
        """Get the effective translation and rotation matrix of the CS

        Returns:
            A tuple containing the effective translation and rotation matrix
        """
        translation = be.array([self.x.item(), self.y.item(), self.z.item()])
        if self.reference_cs is None:
            # No reference coordinate system, return the local transform
            return translation, self.get_rotation_matrix()

        # Get the effective transform of the reference coordinate system
        ref_translation, ref_rot_mat = self.reference_cs.get_effective_transform()

        # Combine translations
        eff_translation = ref_translation + ref_rot_mat @ translation

        # Combine rotations by multiplying the rotation matrices
        eff_rot_mat = ref_rot_mat @ self.get_rotation_matrix()

        return eff_translation, eff_rot_mat

    def get_effective_rotation_euler(self) -> NDArray[np.floating]:
        """Get the effective rotation in Euler angles.

        The Euler angles are returned in 'xyz' order.

        Returns:
            A NumPy array containing the effective rotation as Euler
                angles (rx, ry, rz). Note: This returns a NumPy array due to
                the use of SciPy for the conversion.
        """
        _, eff_rot_mat = self.get_effective_transform()
        # Convert the effective rotation matrix back to Euler angles
        # detach & convert to plain numpy so SciPy won’t try to call .numpy()
        # on a grad model
        matrix = be.to_numpy(eff_rot_mat)
        return R.from_matrix(matrix).as_euler("xyz")

    def to_dict(self) -> dict:
        """Convert the coordinate system to a dictionary.

        Returns:
            The dictionary representation of the coordinate system.
        """
        return {
            "x": float(self.x.item()),
            "y": float(self.y.item()),
            "z": float(self.z.item()),
            "rx": float(self.rx.item()),
            "ry": float(self.ry.item()),
            "rz": float(self.rz.item()),
            "reference_cs": self.reference_cs.to_dict() if self.reference_cs else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CoordinateSystem:
        """Create a coordinate system from a dictionary.

        Args:
            data: The dictionary representation of the coordinate
                system.

        Returns:
            The coordinate system instance defined by the provided dictionary.
        """
        reference_cs = (
            cls.from_dict(data["reference_cs"]) if data["reference_cs"] else None
        )

        return cls(
            data.get("x", 0),
            data.get("y", 0),
            data.get("z", 0),
            data.get("rx", 0),
            data.get("ry", 0),
            data.get("rz", 0),
            reference_cs,
        )
