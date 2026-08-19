"""Scatter Module

The scatter module is used to model the scattering of rays based on a
Bidirectional Scattering Distribution Function (BSDF). The BSDF defines the
probability distribution of the scattered ray direction based on the incident
ray direction and the surface normal.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from optiland.rays import RealRays


@njit(fastmath=True, cache=True)
def get_point_lambertian():  # pragma: no cover
    """Generates a random point on the 2D unit disk.

    Returns:
        tuple: A tuple containing the x, y coordinates of the generated point.

    """
    r = np.random.rand()
    theta = np.random.uniform(0, 2 * np.pi)
    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)
    return x, y


@njit(fastmath=True, cache=True)
def get_point_gaussian(sigma):  # pragma: no cover
    """Generates a random point from a 2D Gaussian distribution using the
    Box-Muller transform.

    Returns:
        tuple: A tuple containing the x, y coordinates of the generated point.

    """
    u1, u2 = np.random.uniform(0, 1, 2)
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)
    x = sigma * z0
    y = sigma * z1
    return x, y


def func_wrapper(func, *args):  # pragma: no cover
    """Wrap a numba-jitted function with its args pre-bound and no arguments."""

    @njit(fastmath=True, cache=True)
    def wrapper():
        return func(*args)

    return wrapper


@njit(fastmath=True, cache=True)
def scatter(L, M, N, nx, ny, nz, get_point):  # pragma: no cover
    """Generate a scattered vector in the global coordinate system.

    Args:
        L (float): x-component of ray direction cosines.
        M (float): y-component of ray direction cosines.
        N (float): z-component of ray direction cosines.
        nx (float): x-component of the normal vectors.
        ny (float): y-component of the normal vectors.
        nz (float): z-component of the normal vectors.
        get_point (function): Function that generates a point on the unit disk.

    Returns:
        s (numpy.ndarray): Scattered vector in the global coordinate system.

    """
    while True:
        # Generate point on unit disk
        x, y = get_point()
        n = np.array((nx, ny, nz))
        r = np.array((L, M, N))

        # Arbitrary vector to use as a reference for the cross product
        arbitrary_vector = np.array((1, 0, 0)) if L < 0.999 else np.array((0, 1, 0))

        # First basis vector for the local coordinate system
        a = np.cross(n, arbitrary_vector)
        a /= np.linalg.norm(a)

        # Second basis vector for the local coordinate system
        b = np.cross(n, a)

        # Generate scattered vectors in local coordinate system
        s_loc_x = np.dot(r, a) + x
        s_loc_y = np.dot(r, b) + y
        radicand = 1 - s_loc_x**2 - s_loc_y**2

        # Check if the scattered vector is in the correct hemisphere
        if radicand < 0:
            continue
        s_loc_z = np.sqrt(radicand)

        # Transform scattered vectors to global coordinate system
        s = s_loc_x * a + s_loc_y * b + s_loc_z * n

        return s


@njit(parallel=True, fastmath=True, cache=True)
def scatter_parallel(L, M, N, nx, ny, nz, get_point):  # pragma: no cover
    """Perform scatter operation in parallel.

    Args:
        L (numpy.ndarray): Array of L values.
        M (numpy.ndarray): Array of M values.
        N (numpy.ndarray): Array of N values.
        nx (numpy.ndarray): Array of nx values.
        ny (numpy.ndarray): Array of ny values.
        nz (numpy.ndarray): Array of nz values.
        get_point (function): Function to get point.

    Returns:
        numpy.ndarray: Array of scattered vectors.

    """
    size = len(L)
    v = np.empty((size, 3), dtype=np.float64)
    for i in prange(size):
        v[i] = scatter(L[i], M[i], N[i], nx[i], ny[i], nz[i], get_point)
    return v


class BaseBSDF(ABC):  # noqa: B024
    """Abstract base class for Bidirectional Scattering Distribution Function
    (BSDF).

    Attributes:
        scattering_function: The scattering function associated with the BSDF.

    Methods:
        scatter(rays: RealRays, nx: be.ndarray, ny: be.ndarray, nz: be.ndarray):
            Scatters rays according to the BSDF.

    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseBSDF._registry[cls.__name__] = cls

    def scatter(self, rays: RealRays, nx: np.ndarray, ny: np.ndarray, nz: np.ndarray):
        """Scatter rays according to the BSDF.

        Args:
            rays (RealRays): The rays to be scattered.
            nx (np.ndarray): The x-component of the surface normal vector.
            ny (np.ndarray): The y-component of the surface normal vector.
            nz (np.ndarray): The z-component of the surface normal vector.

        Returns:
            RealRays: The updated rays after scattering is applied.

        """
        if np.isscalar(nx):
            nx = np.full_like(rays.L, nx)
        if np.isscalar(ny):
            ny = np.full_like(
                rays.M, ny
            )  # Corrected rays.L to rays.M for consistency if ny corresponds to M
        if np.isscalar(nz):
            nz = np.full_like(
                rays.N, nz
            )  # Corrected rays.L to rays.N for consistency if nz corresponds to N

        scattered_vec = scatter_parallel(
            rays.L,
            rays.M,
            rays.N,
            nx,
            ny,
            nz,
            self.scattering_function,
        )
        rays.L = scattered_vec[:, 0]
        rays.M = scattered_vec[:, 1]
        rays.N = scattered_vec[:, 2]
        return rays

    def to_dict(self):
        """Convert the BSDF to a dictionary.

        Returns:
            dict: A dictionary representation of the BSDF.

        """
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data):
        """Create a BSDF object from a dictionary."""
        bsdf_type = data["type"]
        return cls._registry[bsdf_type].from_dict(data)


class LambertianBSDF(BaseBSDF):
    """Lambertian Bidirectional Scattering Distribution Function (BSDF) class.

    This class represents a Lambertian BSDF, which is generally used to model
    diffuse scattering.
    """

    def __init__(self):
        self.scattering_function = get_point_lambertian

    def to_dict(self):
        """Convert the BSDF to a dictionary.

        Returns:
            dict: A dictionary representation of the BSDF.

        """
        return {
            "type": "LambertianBSDF",
        }

    @classmethod
    def from_dict(cls, data):
        """Create a LambertianBSDF object from a dictionary."""
        return cls()


class GaussianBSDF(BaseBSDF):
    """Gaussian Bidirectional Scattering Distribution Function (BSDF) class.

    This class represents a Gaussian BSDF, which models scattering based on a
    2D Gaussian distribution.
    """

    def __init__(self, sigma):
        self.sigma = sigma
        self.scattering_function = func_wrapper(get_point_gaussian, sigma)

    def to_dict(self):
        """Convert the BSDF to a dictionary.

        Returns:
            dict: A dictionary representation of the BSDF.

        """
        return {
            "type": "GaussianBSDF",
            "sigma": self.sigma,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a GaussianBSDF object from a dictionary."""
        return cls(data["sigma"])
