"""Image-plane sampling of pupil-FFT diffraction calculations.

Shared by the FFT PSF and the FFT MTF so that both use the same calibration
between pupil samples and image-plane coordinates.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.optic import Optic


def pupil_frequency_steps(
    optic: Optic,
    field: tuple[float, float],
    wavelength: float,
    num_rays: int,
) -> tuple[BEArray, BEArray]:
    """Image-plane spatial frequency of one pupil sample, per pupil axis.

    In the pupil FFT, image-plane position is Fourier conjugate to
    ``n * (L, M) / wavelength``, where ``(L, M)`` are the ray direction
    cosines in image space. A pair of ray directions separated by ``du``
    therefore gives an image-plane spatial frequency
    ``n * du_parallel / wavelength``, with ``du_parallel`` the separation
    projected onto the image surface's tangent plane at the chief ray.

    The two opposite marginal-ray differences (pupil Y and pupil X) are
    projected onto that plane; one pupil sample spans ``1 / (num_rays - 1)``
    of each diameter. Pupil Y gives the tangential step and pupil X the
    sagittal step. This retains anamorphism and the local slope of a curved
    image, and assumes an approximately affine pupil-to-direction-cosine
    mapping, as the pupil FFT itself does.

    The FFT MTF frequency axes are these steps directly. The FFT PSF pixel
    pitch along an axis is ``1 / (grid_size * step)`` along that axis.

    Args:
        optic (Optic): The optical system.
        field (tuple[float, float]): Normalized field coordinates (Hx, Hy).
        wavelength (float): The wavelength in micrometers.
        num_rays (int): The number of pupil samples across the diameter.

    Returns:
        tuple[BEArray, BEArray]: The tangential (pupil Y) and sagittal
        (pupil X) frequency steps, in cycles/mm.
    """
    projected, n_image = _projected_marginal_differences(optic, field, wavelength)
    bandwidths = be.sqrt(be.sum(projected**2, axis=-1))
    steps = n_image * bandwidths / ((num_rays - 1) * wavelength * 1e-3)
    return steps[0], steps[1]


def pupil_direction_matrix(
    optic: Optic,
    field: tuple[float, float],
    wavelength: float,
) -> BEArray:
    """Linear map from normalized pupil coordinates to image-space directions.

    Returns the 2x2 matrix ``A`` with ``n * (dL, dM) = A @ (Px, Py)``, where
    ``(dL, dM)`` is the change of a ray's direction cosines relative to the
    chief ray, projected onto the image surface's tangent plane at the chief
    ray and expressed along the image surface's local x and y axes, and ``n``
    is the image-space index. Each column is the secant over the full pupil
    diameter (half the difference between the marginal rays at +1 and -1),
    with its sign: the same affine approximation as `pupil_frequency_steps`,
    whose steps are the column norms of ``A`` divided by
    ``(num_rays - 1) / 2 * wavelength``.

    The scalar diffraction field at image-plane offset ``r = (x, y)`` from
    the chief ray is then ``sum_p g(p) exp(i k r . (A p))``, with
    ``k = 2 pi / wavelength``. For fields on the x or y axis of a rotationally
    symmetric system ``A`` is diagonal and the sum separates into one 1D
    transform per axis. For other fields the off-diagonal terms couple the
    axes.

    Args:
        optic (Optic): The optical system.
        field (tuple[float, float]): Normalized field coordinates (Hx, Hy).
        wavelength (float): The wavelength in micrometers.

    Returns:
        BEArray: The 2x2 matrix ``A``; rows are the image x and y directions,
        columns are pupil X and pupil Y.
    """
    projected, n_image = _projected_marginal_differences(optic, field, wavelength)
    half_y = projected[0] / 2  # pupil Y column
    half_x = projected[1] / 2  # pupil X column
    return n_image * be.stack(
        [be.stack([half_x[0], half_y[0]]), be.stack([half_x[1], half_y[1]])]
    )


def _projected_marginal_differences(
    optic: Optic,
    field: tuple[float, float],
    wavelength: float,
) -> tuple[BEArray, BEArray]:
    """Marginal-ray direction differences on the image tangent plane.

    Args:
        optic (Optic): The optical system.
        field (tuple[float, float]): Normalized field coordinates (Hx, Hy).
        wavelength (float): The wavelength in micrometers.

    Returns:
        tuple[BEArray, BEArray]: The differences ``d(+1) - d(-1)`` of the
        local direction-cosine vectors for pupil Y (row 0) and pupil X
        (row 1), projected onto the image surface's tangent plane at the chief
        ray (shape ``(2, 3)``), and the image-space refractive index.
    """
    rays = optic.trace_generic(
        *field,
        Px=be.array([0.0, 0.0, 0.0, 1.0, -1.0]),
        Py=be.array([0.0, 1.0, -1.0, 0.0, 0.0]),
        wavelength=wavelength,
    )
    geometry = optic.image_surface.geometry
    # trace_generic returns global coordinates; surface_normal expects local
    # coordinates. Localizing these fresh rays leaves the stored per-surface
    # trace snapshots in the global frame.
    geometry.cs.localize(rays)
    normal = be.stack(geometry.surface_normal(rays), axis=-1)[0]
    normal = normal / be.sqrt(be.sum(normal**2))
    directions = be.stack([rays.L, rays.M, rays.N], axis=-1)
    differences = be.stack(
        [directions[1] - directions[2], directions[3] - directions[4]]
    )
    perpendicular = be.sum(differences * normal, axis=-1)
    projected = differences - perpendicular[:, None] * normal
    n_image = be.abs(optic.image_surface.material_post.n(wavelength))
    return projected, n_image
