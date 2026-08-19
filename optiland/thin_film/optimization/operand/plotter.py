"""Plotting utilities for thin-film optimization operands."""

from __future__ import annotations

import numpy as np


class ThinFilmOperandPlotter:
    """Dedicated plotter for thin-film optimization operands."""

    def __init__(self, operands):
        self.operands = list(operands)

    def _wavelength_range(self) -> tuple[float, float]:
        wl_values = []
        for operand in self.operands:
            wavelength_nm = getattr(operand, "wavelength_nm", None)
            if wavelength_nm is None:
                continue
            if isinstance(wavelength_nm, list | np.ndarray):
                wl_values.extend(wavelength_nm)
            else:
                wl_values.append(wavelength_nm)

        if not wl_values:
            return (400.0, 800.0)

        margin = (max(wl_values) - min(wl_values)) * 0.1
        return (min(wl_values) - margin, max(wl_values) + margin)

    def _angle_range(self) -> tuple[float, float]:
        angle_values = []
        for operand in self.operands:
            aoi_deg = getattr(operand, "aoi_deg", None)
            if aoi_deg is None:
                continue
            if isinstance(aoi_deg, list | np.ndarray):
                angle_values.extend(aoi_deg)
            else:
                angle_values.append(aoi_deg)

        if not angle_values:
            return (0.0, 80.0)

        margin = (max(angle_values) - min(angle_values)) * 0.1
        return (min(angle_values) - margin, max(angle_values) + margin)

    def plot(
        self,
        ax,
        plot_type: str = "wavelength",
        wavelength_range_nm: tuple[float, float] | None = None,
        angle_range_deg: tuple[float, float] | None = None,
        num_points: int = 100,
    ) -> None:
        if not self.operands:
            return

        if plot_type == "wavelength":
            wavelength_range_nm = wavelength_range_nm or self._wavelength_range()
            x_values = np.linspace(
                wavelength_range_nm[0], wavelength_range_nm[1], num_points
            )
            for operand in self.operands:
                operand.plot(
                    ax,
                    plot_type,
                    x_values,
                    wavelength_range_nm=wavelength_range_nm,
                )
        elif plot_type == "angle":
            angle_range_deg = angle_range_deg or self._angle_range()
            x_values = np.linspace(angle_range_deg[0], angle_range_deg[1], num_points)
            for operand in self.operands:
                operand.plot(
                    ax,
                    plot_type,
                    x_values,
                    angle_range_deg=angle_range_deg,
                )
        else:
            raise ValueError(
                f"Invalid plot_type '{plot_type}'. Must be 'wavelength' or 'angle'."
            )

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend()
