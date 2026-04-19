from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.interpolate import interp1d

from .thin_film import ThinFilmOperand

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.thin_film import ThinFilmStack

TargetType = Literal["equal", "below", "over"]


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class ThinFilmOperandRegistry:
    """Registry for thin-film operand metric functions."""

    def __init__(self):
        self._registry: dict[str, Callable[..., float]] = {}

    def register(
        self, name: str, func: Callable[..., float], overwrite: bool = False
    ) -> None:
        if name in self._registry and not overwrite:
            raise ValueError(f'Operand "{name}" is already registered.')
        self._registry[name] = func

    def get(self, name: str) -> Callable[..., float] | None:
        return self._registry.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._registry


thin_film_operand_registry = ThinFilmOperandRegistry()
for _name, _func in {
    "R": ThinFilmOperand.reflectance,
    "T": ThinFilmOperand.transmittance,
    "A": ThinFilmOperand.absorptance,
    "reflectance": ThinFilmOperand.reflectance,
    "transmittance": ThinFilmOperand.transmittance,
    "absorptance": ThinFilmOperand.absorptance,
    "reflection_phase": ThinFilmOperand.reflection_phase,
    "transmission_phase": ThinFilmOperand.transmission_phase,
    "reflection_gd": ThinFilmOperand.reflection_gd,
    "transmission_gd": ThinFilmOperand.transmission_gd,
    "reflection_gdd": ThinFilmOperand.reflection_gdd,
    "transmission_gdd": ThinFilmOperand.transmission_gdd,
    "field_phase": ThinFilmOperand.field_phase,
    "field_amplitude": ThinFilmOperand.field_amplitude,
}.items():
    thin_film_operand_registry.register(_name, _func)


@dataclass(slots=True)
class ThinFilmEvaluationContext:
    stack: ThinFilmStack


@dataclass
class OptimizationTarget:
    """Legacy spectral/angular target definition retained as public API."""

    property: str
    wavelength_nm: float | list[float]
    target_type: TargetType
    value: float | list[float]
    weight: float
    aoi_deg: float | list[float]
    polarization: str
    tolerance: float

    def interpolate_target_value(
        self,
        current_wl: float | None = None,
        current_aoi: float | None = None,
    ) -> float:
        if isinstance(self.value, int | float):
            return float(self.value)

        value_array = np.array(self.value)

        if isinstance(self.wavelength_nm, list | np.ndarray):
            if current_wl is None:
                raise ValueError(
                    "current_wl must be provided for wavelength interpolation"
                )
            wl_array = np.array(self.wavelength_nm)
            if len(value_array) != len(wl_array):
                raise ValueError("Value and wavelength arrays must have same length")

            interp_func = interp1d(
                wl_array,
                value_array,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            return float(interp_func(current_wl))

        if isinstance(self.aoi_deg, list | np.ndarray):
            if current_aoi is None:
                raise ValueError("current_aoi must be provided for AOI interpolation")
            aoi_array = np.array(self.aoi_deg)
            if len(value_array) != len(aoi_array):
                raise ValueError("Value and AOI arrays must have same length")

            interp_func = interp1d(
                aoi_array,
                value_array,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            return float(interp_func(current_aoi))

        return float(value_array[0])


class ThinFilmBaseOperand:
    """Base class for merit-function operands."""

    weight: float

    def delta(self, context: ThinFilmEvaluationContext) -> float:
        raise NotImplementedError

    def fun(self, context: ThinFilmEvaluationContext) -> float:
        return math.sqrt(float(self.weight)) * self.delta(context)

    def performance_data(self, context: ThinFilmEvaluationContext) -> dict[str, Any]:
        raise NotImplementedError

    def plot(self, ax, plot_type: str, x_values: np.ndarray, **kwargs) -> None:
        return None


class SpectralOptimizationOperand(OptimizationTarget, ThinFilmBaseOperand):
    """Concrete operand for R/T/A spectral and angular targets."""

    @property
    def display_name(self) -> str:
        return self.property

    def _metric_function(self) -> Callable[..., float]:
        metric_function = thin_film_operand_registry.get(self.property)
        if metric_function is None:
            raise ValueError(f"Unknown operand type: {self.property}")
        return metric_function

    def _sample_points(self) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []

        if isinstance(self.wavelength_nm, list | np.ndarray):
            aoi_deg = (
                float(self.aoi_deg)
                if not isinstance(self.aoi_deg, list | np.ndarray)
                else float(self.aoi_deg[0])
            )
            for wl in np.array(self.wavelength_nm):
                points.append(
                    (
                        float(wl),
                        aoi_deg,
                        self.interpolate_target_value(current_wl=float(wl)),
                    )
                )
            return points

        wavelength_nm = float(self.wavelength_nm)
        if isinstance(self.aoi_deg, list | np.ndarray):
            for aoi in np.array(self.aoi_deg):
                points.append(
                    (
                        wavelength_nm,
                        float(aoi),
                        self.interpolate_target_value(current_aoi=float(aoi)),
                    )
                )
            return points

        points.append(
            (
                wavelength_nm,
                float(self.aoi_deg),
                self.interpolate_target_value(),
            )
        )
        return points

    def _residual(self, current_value: float, target_value: float) -> float:
        if self.target_type == "equal":
            return current_value - target_value
        if self.target_type == "below":
            return max(0.0, current_value - target_value)
        if self.target_type == "over":
            return max(0.0, target_value - current_value)
        raise ValueError(f"Unknown target_type: {self.target_type}")

    def current_values(self, context: ThinFilmEvaluationContext) -> list[float]:
        metric_function = self._metric_function()
        values = []
        for wavelength_nm, aoi_deg, _target_value in self._sample_points():
            values.append(
                _to_float(
                    metric_function(
                        context.stack,
                        wavelength_nm,
                        aoi_deg,
                        self.polarization,
                    )
                )
            )
        return values

    def residuals(self, context: ThinFilmEvaluationContext) -> list[float]:
        residuals = []
        for current_value, (_wl, _aoi, target_value) in zip(
            self.current_values(context), self._sample_points(), strict=False
        ):
            residuals.append(self._residual(current_value, target_value))
        return residuals

    def delta(self, context: ThinFilmEvaluationContext) -> float:
        residuals = self.residuals(context)
        if not residuals:
            return 0.0
        return float(np.sqrt(np.mean(np.square(residuals))))

    def performance_data(self, context: ThinFilmEvaluationContext) -> dict[str, Any]:
        sample_points = self._sample_points()
        current_values = self.current_values(context)
        target_values = [target_value for _wl, _aoi, target_value in sample_points]

        if len(sample_points) == 1:
            wavelength_nm, aoi_deg, target_value = sample_points[0]
            current_value = current_values[0]
            return {
                "property": self.property,
                "wavelength_nm": wavelength_nm,
                "aoi_deg": aoi_deg,
                "target_type": self.target_type,
                "target_value": target_value,
                "current_value": current_value,
                "difference": current_value - target_value,
                "weight": self.weight,
            }

        wavelengths = [wavelength_nm for wavelength_nm, _aoi, _target in sample_points]
        angles = [aoi_deg for _wl, aoi_deg, _target in sample_points]
        wavelength_value: float | list[float]
        aoi_value: float | list[float]

        if isinstance(self.wavelength_nm, list | np.ndarray):
            wavelength_value = wavelengths
            aoi_value = angles[0]
        else:
            wavelength_value = wavelengths[0]
            aoi_value = angles

        return {
            "property": self.property,
            "wavelength_nm": wavelength_value,
            "aoi_deg": aoi_value,
            "target_type": self.target_type,
            "target_values": target_values,
            "current_values": current_values,
            "differences": [
                current - target
                for current, target in zip(current_values, target_values, strict=False)
            ],
            "weight": self.weight,
        }

    def plot(self, ax, plot_type: str, x_values: np.ndarray, **kwargs) -> None:
        color_map = {"R": "red", "T": "blue", "A": "green"}
        target_styles = {"equal": "-", "below": "--", "over": ":"}
        color = color_map.get(self.property, "black")
        style = target_styles.get(self.target_type, "-")

        if plot_type == "wavelength":
            wavelength_range_nm = kwargs.get("wavelength_range_nm")
            if isinstance(self.wavelength_nm, list | np.ndarray):
                wl_array = np.array(self.wavelength_nm)
                if isinstance(self.value, list | np.ndarray):
                    value_array = np.array(self.value)
                    interp_func = interp1d(
                        wl_array,
                        value_array,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    y_target = interp_func(x_values)
                else:
                    y_target = np.full_like(x_values, self.value)

                ax.plot(
                    x_values,
                    y_target,
                    linestyle=style,
                    color=color,
                    label=f"{self.property} {self.target_type}",
                )
                return

            if (
                wavelength_range_nm is not None
                and wavelength_range_nm[0]
                <= self.wavelength_nm
                <= wavelength_range_nm[1]
                and not isinstance(self.aoi_deg, list | np.ndarray)
            ):
                ax.axvline(
                    self.wavelength_nm,
                    color=color,
                    linestyle=style,
                    label=f"{self.property} @ {self.wavelength_nm}nm",
                )
            return

        if plot_type == "angle":
            angle_range_deg = kwargs.get("angle_range_deg")
            if isinstance(self.aoi_deg, list | np.ndarray):
                angle_array = np.array(self.aoi_deg)
                if isinstance(self.value, list | np.ndarray):
                    value_array = np.array(self.value)
                    interp_func = interp1d(
                        angle_array,
                        value_array,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    y_target = interp_func(x_values)
                else:
                    y_target = np.full_like(x_values, self.value)

                ax.plot(
                    x_values,
                    y_target,
                    linestyle=style,
                    color=color,
                    label=f"{self.property} {self.target_type}",
                )
                return

            if (
                angle_range_deg is not None
                and angle_range_deg[0] <= self.aoi_deg <= angle_range_deg[1]
                and not isinstance(self.wavelength_nm, list | np.ndarray)
            ):
                ax.axvline(
                    self.aoi_deg,
                    color=color,
                    linestyle=style,
                    label=f"{self.property} @ {self.aoi_deg}°",
                )


@dataclass
class ThinFilmCustomOperand(ThinFilmBaseOperand):
    """User-defined scalar operand registered in the thin-film registry."""

    operand_type: str
    weight: float = 1.0
    target: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    input_data: dict[str, Any] | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.target is not None and (
            self.min_val is not None or self.max_val is not None
        ):
            raise ValueError(
                "Custom operand cannot accept both equality and inequality targets"
            )

    @property
    def display_name(self) -> str:
        return self.label or self.operand_type

    def value(self, context: ThinFilmEvaluationContext) -> float:
        metric_function = thin_film_operand_registry.get(self.operand_type)
        if metric_function is None:
            raise ValueError(f"Unknown operand type: {self.operand_type}")

        input_data = dict(self.input_data or {})
        input_data.setdefault("stack", context.stack)
        return _to_float(metric_function(**input_data))

    def delta(self, context: ThinFilmEvaluationContext) -> float:
        current_value = self.value(context)
        if self.target is not None:
            return current_value - self.target

        lower_penalty = (
            max(0.0, self.min_val - current_value) if self.min_val is not None else 0.0
        )
        upper_penalty = (
            max(0.0, current_value - self.max_val) if self.max_val is not None else 0.0
        )
        return lower_penalty + upper_penalty

    def performance_data(self, context: ThinFilmEvaluationContext) -> dict[str, Any]:
        current_value = self.value(context)
        return {
            "property": self.display_name,
            "operand_type": self.operand_type,
            "target_value": self.target,
            "min_value": self.min_val,
            "max_value": self.max_val,
            "current_value": current_value,
            "difference": self.delta(context),
            "weight": self.weight,
        }

    def plot(self, ax, plot_type: str, x_values: np.ndarray, **kwargs) -> None:
        """Draw horizontal reference lines for min_val / max_val / target.

        Since a custom operand is a scalar metric without an intrinsic
        wavelength or angle axis, its constraint bounds are shown as horizontal
        lines so they can be read against whichever quantity is on the y-axis.
        """
        color = "darkorange"
        base = self.display_name
        if self.target is not None:
            ax.axhline(
                self.target,
                linestyle="-",
                color=color,
                label=f"{base} = {self.target:.3f}",
            )
        if self.min_val is not None:
            ax.axhline(
                self.min_val,
                linestyle="--",
                color=color,
                label=f"{base} \u2265 {self.min_val:.3f}",
            )
        if self.max_val is not None:
            ax.axhline(
                self.max_val,
                linestyle=":",
                color=color,
                label=f"{base} \u2264 {self.max_val:.3f}",
            )


class ThinFilmOperandManager:
    """Manages operand instances for the thin-film optimizer."""

    def __init__(self):
        self.operands: list[ThinFilmBaseOperand] = []

    def add(self, operand: ThinFilmBaseOperand) -> None:
        self.operands.append(operand)

    def clear(self) -> None:
        self.operands = []

    def __iter__(self):
        return iter(self.operands)

    def __len__(self) -> int:
        return len(self.operands)

    def __getitem__(self, index: int) -> ThinFilmBaseOperand:
        return self.operands[index]
