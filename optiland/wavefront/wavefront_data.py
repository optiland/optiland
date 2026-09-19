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
        quadrature_weights (be.ndarray | None): Quadrature contributions associated
            with the samples. Generated data require ``assume_sample_order=True``
            to copy a distribution snapshot. Gaussian-quadrature values describe
            normalized distribution-coordinate unit-disk area. They are not
            physical-pupil Jacobian weights, are not multiplied by intensity,
            and are not changed when a ray is
            clipped. ``None`` means that no quadrature measure was supplied.
    """

    pupil_x: BEArrayT
    pupil_y: BEArrayT
    pupil_z: BEArrayT
    opd: BEArrayT
    intensity: BEArrayT
    radius: float
    prt_matrix: BEArrayT | None = None
    E_exits: list[BEArrayT] | None = None
    quadrature_weights: BEArrayT | None = None
