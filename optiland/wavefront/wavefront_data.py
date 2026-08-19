"""
This module defines the WavefrontData class.

Kramer Harrison, 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from optiland._types import BEArrayT


@dataclass
class WavefrontData(Generic[BEArrayT]):
    """
    Data container for wavefront results at a given field and wavelength.

    Attributes:
        pupil_x (be.ndarray): x-coordinates of ray intersections at exit pupil.
        pupil_y (be.ndarray): y-coordinates of ray intersections at exit pupil.
        pupil_z (be.ndarray): z-coordinates of ray intersections at exit pupil.
        opd (be.ndarray): Optical path difference data, normalized to waves.
        intensity (be.ndarray): Ray intensities at the exit pupil.
        radius (be.ndarray): Radius of curvature of the exit pupil reference sphere.
        E_exits (list[be.ndarray] | None): A list of 3D electric field vectors at
            the exit pupil, representing incoherent polarization states.
    """

    pupil_x: BEArrayT
    pupil_y: BEArrayT
    pupil_z: BEArrayT
    opd: BEArrayT
    intensity: BEArrayT
    radius: float
    prt_matrix: BEArrayT | None = None
    E_exits: list[BEArrayT] | None = None
