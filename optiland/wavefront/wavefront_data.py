"""
This module defines the WavefrontData class.

Kramer Harrison, 2024
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from dataclasses import dataclass
from typing import Generic

import numpy as np

import optiland.backend as be
from optiland._types import BEArrayT

from .evaluation import (
    WavefrontEvaluationResult,
    WavefrontRemoval,
    _evaluate_prepared,
    _prepare_wavefront_inputs,
)


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

    def evaluate(
        self,
        *,
        remove: WavefrontRemoval,
        weights: BEArrayT | Sequence[float] | None = None,
        use_quadrature: bool = False,
        rcond: float | None = None,
    ) -> WavefrontEvaluationResult[BEArrayT]:
        """Evaluate the cached OPD samples without tracing.

        Explicit ``weights`` are final effective weights and are not multiplied
        by intensity, apodization, or quadrature contributions. Alternatively,
        ``use_quadrature=True`` selects the stored quadrature snapshot. With
        neither option, samples receive equal weight.

        For ``remove="piston_tilt"``, the affine basis uses the stored ``pupil_x``
        and ``pupil_y`` coordinates. Quadrature metadata still describes the
        distribution's sampling measure, not a change of fitting coordinates.

        Native evaluation conservatively requires finite, strictly positive
        stored intensity on every positive-weight sample. An explicit zero
        weight can exclude an unusable sample, but the result then describes
        only that conditional support and is not evidence of trace completeness.
        This safety check runs before numerical evaluation. The standalone
        :func:`evaluate_wavefront` evaluator remains independent of intensity.
        Its first-order OPD gradient contract requires normal-range nonzero RMS,
        derivatives, and nonzero normalized weighted components. Subnormal forward
        values remain available, but gradients in that underflow regime and
        higher-order derivatives are not supported.

        Args:
            remove: Modes to remove: ``"none"``, ``"piston"``, or
                ``"piston_tilt"``.
            weights: Final effective evaluation weights. ``None`` selects equal
                samples unless ``use_quadrature`` is true.
            use_quadrature: Use the stored quadrature-weight snapshot.
            rcond: Relative affine-rank cutoff passed to
                :func:`evaluate_wavefront`. Requires a Python or NumPy real
                scalar; booleans, strings, arrays, and tensors are rejected.

        Returns:
            Weighted residual statistics and fit diagnostics from
            :func:`evaluate_wavefront`.

        Raises:
            ValueError: If weight selection is ambiguous, requested quadrature
                data are unavailable, stored intensity is not aligned, or a
                positive-weight sample has nonfinite or nonpositive intensity.
            TypeError: If weights or intensity are masked, or stored intensity
                does not use the active backend, a real numeric dtype, or the
                OPD device, or if rcond has an unsupported type.
        """
        if weights is not None and use_quadrature:
            raise ValueError("weights and use_quadrature=True are mutually exclusive.")
        if use_quadrature:
            if self.quadrature_weights is None:
                raise ValueError("No quadrature weights are available for this data.")
            effective_weights = self.quadrature_weights
        else:
            effective_weights = weights

        self._validate_intensity()
        prepared = _prepare_wavefront_inputs(
            self.opd,
            x=self.pupil_x,
            y=self.pupil_y,
            weights=effective_weights,
            remove=remove,
        )
        used_intensity = self.intensity[prepared.used_mask]
        if not be.all(be.isfinite(used_intensity)) or be.any(used_intensity <= 0.0):
            raise ValueError(
                "intensity must be finite and strictly positive on "
                "positive-weight support."
            )

        return _evaluate_prepared(
            prepared,
            rcond=rcond,
        )

    def _validate_intensity(self) -> None:
        """Validate native intensity representation before OPD evaluation."""
        if np.ma.isMaskedArray(self.intensity):
            raise TypeError("intensity must not be a NumPy MaskedArray.")
        intensity_is_torch = be.is_torch_tensor(self.intensity)
        active_intensity = (
            intensity_is_torch
            if be.get_backend() == "torch"
            else isinstance(self.intensity, be.ndarray) and not intensity_is_torch
        )
        if not active_intensity:
            raise TypeError(
                f"intensity must use the active {be.get_backend()} backend."
            )
        if self.intensity.ndim != 1 or self.intensity.shape != getattr(
            self.opd, "shape", None
        ):
            raise ValueError("intensity must be one-dimensional and aligned with opd.")
        if intensity_is_torch:
            import torch

            real_numeric = (
                not self.intensity.is_complex()
                and self.intensity.dtype != torch.bool
                and not self.intensity.is_quantized
            )
        else:
            real_numeric = np.issubdtype(
                self.intensity.dtype, np.number
            ) and not np.issubdtype(self.intensity.dtype, np.complexfloating)
        if not real_numeric:
            raise TypeError("intensity must have a real numeric, non-complex dtype.")
        if intensity_is_torch and (
            not be.is_torch_tensor(self.opd) or self.intensity.device != self.opd.device
        ):
            raise TypeError("intensity must be on the same device as opd.")
