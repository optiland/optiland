"""Backend-specific strategies for Huygens-Fresnel summation.

This module provides different strategies for computing the Huygens-Fresnel
diffraction integral, optimized for different computation backends (NumPy with
Numba, and PyTorch). This allows the main `HuygensPSF` class to remain
backend-agnostic, delegating the computationally intensive summation to the
appropriate strategy.

Each strategy is encapsulated in a class that inherits from the abstract base
class `HuygensFresnelSummation`, ensuring a consistent interface.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange

import optiland.backend as be

try:
    import torch
except (ImportError, OSError):
    torch = None


class HuygensFresnelSummation(ABC):
    """Abstract base class for Huygens-Fresnel summation strategies."""

    @abstractmethod
    def compute(
        self,
        image_x,
        image_y,
        image_z,
        pupil_x,
        pupil_y,
        pupil_z,
        pupil_amp,
        pupil_opd,
        wavelength,
        Rp,
    ):
        """
        Compute the point spread function using a specific backend strategy.

        Args:
            image_x, image_y, image_z: 2D arrays of image surface coords.
            pupil_x, pupil_y, pupil_z: 1D arrays of pupil plane coords.
            pupil_amp: 1D array of pupil plane amplitudes.
            pupil_opd: 1D array of optical path difference in mm.
            wavelength: Wavelength of the light in mm.
            Rp: Radius of the exit pupil reference sphere in mm.

        Returns:
            2D array of the point spread function.
        """
        pass  # pragma: no cover


class NumbaSummation(HuygensFresnelSummation):
    """Huygens-Fresnel summation using Numba for NumPy arrays."""

    def compute(
        self,
        image_x,
        image_y,
        image_z,
        pupil_x,
        pupil_y,
        pupil_z,
        pupil_amp,
        pupil_opd,
        wavelength,
        Rp,
    ):
        """
        Compute the PSF using the Numba-jitted Huygens-Fresnel summation.
        """
        return self._huygens_fresnel_summation(
            be.to_numpy(image_x),
            be.to_numpy(image_y),
            be.to_numpy(image_z),
            be.to_numpy(pupil_x),
            be.to_numpy(pupil_y),
            be.to_numpy(pupil_z),
            be.to_numpy(pupil_amp),
            be.to_numpy(pupil_opd),
            wavelength,
            Rp,
        )

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _huygens_fresnel_summation(
        image_x,
        image_y,
        image_z,
        pupil_x,
        pupil_y,
        pupil_z,
        pupil_amp,
        pupil_opd,
        wavelength,
        Rp,
    ):  # pragma: no cover
        """
        Compute the point spread function using the Huygens–Fresnel diffraction integral

        Args:
            image_x, image_y, image_z (np.ndarray): 2D arrays of image surface coords.
            pupil_x, pupil_y, pupil_z (np.ndarray): 1D arrays of pupil plane coords.
            pupil_amp (np.ndarray): 1D array of pupil plane amplitudes.
            pupil_opd (np.ndarray): 1D array of optical path difference in mm.
            wavelength (float): Wavelength of the light in mm.
            Rp (float): Radius of the exit pupil reference sphere in mm.

        Returns:
            np.ndarray: 2D array of the point spread function.
        """
        k = 2.0 * np.pi / wavelength  # wavenumber
        Nx, Ny = image_x.shape
        field = np.zeros((Nx, Ny), dtype=np.complex128)

        # Loop over all image points
        for ix in prange(Nx):
            for iy in range(Ny):
                x = image_x[ix, iy]
                y = image_y[ix, iy]
                z = image_z[ix, iy]
                sum_val = 0.0 + 0.0j  # initialize field sum for image point (x, y)

                # Loop over all pupil points
                for j in range(pupil_x.shape[0]):
                    u = pupil_x[j]
                    v = pupil_y[j]
                    w = pupil_z[j]

                    # Compute distance R from the pupil point to the image point
                    dx = x - u
                    dy = y - v
                    dz = z - w
                    R = np.sqrt(dx * dx + dy * dy + dz * dz)

                    # Spherical propagation kernel
                    wave = np.exp(1j * k * R) / R

                    # Compute the unit normal at the pupil point
                    nux = u / Rp
                    nuy = v / Rp
                    nuz = w / Rp

                    # Compute the cosine of the angle between (P - Q) and pupil normal.
                    # P is the image point, Q is the pupil point
                    dot = dx * nux + dy * nuy + dz * nuz
                    cos_theta = dot / R
                    Q_obliq = 0.5 * (1.0 + cos_theta)  # obliquity factor

                    # Pupil function
                    pupil_phase = np.exp(-1j * k * pupil_opd[j])

                    # Add contribution to the field sum
                    sum_val += pupil_amp[j] * pupil_phase * wave * Q_obliq

                field[ix, iy] = sum_val

        # Compute psf as the squared magnitude of the field
        psf = np.abs(field) ** 2
        return psf


class TorchSummation(HuygensFresnelSummation):
    """Huygens-Fresnel summation using PyTorch for tensor operations."""

    def __init__(self, batch_size=1024):
        if torch is None:
            raise ImportError("PyTorch backend is not available.")
        self.batch_size = batch_size
        self.device = be.get_device()

    def compute(
        self,
        image_x,
        image_y,
        image_z,
        pupil_x,
        pupil_y,
        pupil_z,
        pupil_amp,
        pupil_opd,
        wavelength,
        Rp,
    ):
        """
        Compute the PSF using the PyTorch Huygens-Fresnel summation.
        """
        # Convert inputs to torch tensors and move to the correct device
        image_x = be.to_tensor(image_x, device=self.device)
        image_y = be.to_tensor(image_y, device=self.device)
        image_z = be.to_tensor(image_z, device=self.device)
        pupil_x = be.to_tensor(pupil_x, device=self.device)
        pupil_y = be.to_tensor(pupil_y, device=self.device)
        pupil_z = be.to_tensor(pupil_z, device=self.device)
        # pupil_amp may be complex (vectorial case); use complex dtype if needed
        pupil_amp_np = be.to_numpy(pupil_amp)
        if np.iscomplexobj(pupil_amp_np):
            pupil_amp = torch.tensor(
                pupil_amp_np, device=self.device, dtype=be.get_complex_precision()
            )
        else:
            pupil_amp = be.to_tensor(pupil_amp, device=self.device)
        pupil_opd = be.to_tensor(pupil_opd, device=self.device)

        k = 2.0 * torch.pi / wavelength

        # Reshape pupil data for vectorized operations
        pupil_x = pupil_x.reshape(1, -1)
        pupil_y = pupil_y.reshape(1, -1)
        pupil_z = pupil_z.reshape(1, -1)
        pupil_amp = pupil_amp.reshape(1, -1)
        pupil_opd = pupil_opd.reshape(1, -1)

        # Flatten image coordinates for batching
        image_x_flat = image_x.flatten()
        image_y_flat = image_y.flatten()
        image_z_flat = image_z.flatten()

        num_image_points = len(image_x_flat)
        field_flat = torch.zeros(
            num_image_points, dtype=be.get_complex_precision(), device=self.device
        )

        for i in range(0, num_image_points, self.batch_size):
            batch_end = min(i + self.batch_size, num_image_points)

            # Select batch of image points
            x = image_x_flat[i:batch_end].reshape(-1, 1)
            y = image_y_flat[i:batch_end].reshape(-1, 1)
            z = image_z_flat[i:batch_end].reshape(-1, 1)

            # Vectorized distance calculation
            dx = x - pupil_x
            dy = y - pupil_y
            dz = z - pupil_z
            R = torch.sqrt(dx**2 + dy**2 + dz**2)

            # Spherical propagation kernel
            wave = torch.exp(1j * k * R) / R

            # Unit normal at the pupil points
            nux = pupil_x / Rp
            nuy = pupil_y / Rp
            nuz = pupil_z / Rp

            # Obliquity factor
            dot = dx * nux + dy * nuy + dz * nuz
            cos_theta = dot / R
            Q_obliq = 0.5 * (1.0 + cos_theta)

            # Pupil function
            pupil_phase = torch.exp(-1j * k * pupil_opd)

            # Sum contributions for the batch
            sum_val = torch.sum(pupil_amp * pupil_phase * wave * Q_obliq, dim=1)
            field_flat[i:batch_end] = sum_val

        # Reshape field and compute PSF
        field = field_flat.reshape(image_x.shape)
        psf = torch.abs(field) ** 2

        return psf
