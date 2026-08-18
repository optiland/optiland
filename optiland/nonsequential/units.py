"""Photometric conversion layer.

The NSQ radiometric core is unchanged: everything inside the trace stays in
watts, W/mm^2, and micrometres. This module is a *read-only* conversion
layer on top of finished results (and, for sources, an input-side
lumens-to-watts helper) -- it never touches the trace loop.

Guardrail : converting a monochromatic result outside the visible
band, or a spectrum with negligible V(lambda) overlap, raises rather than
returning a near-zero photometric value that looks like a valid (if dim)
answer. Silently returning ~0 is the same defect class as D-2 (an accepted
configuration that quietly does nothing).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from optiland.nonsequential.results.far_field_pattern import FarFieldPattern
    from optiland.nonsequential.results.irradiance_map import IrradianceMap
    from optiland.nonsequential.results.spectral_result import SpectralResult
    from optiland.nonsequential.sources.base import Spectrum

Weighting = Literal["photopic", "scotopic"]

# CIE 1931 2-degree photopic luminous efficiency function V(lambda), and
# CIE 1951 scotopic V'(lambda), tabulated at 10 nm from 380-780 nm (standard
# reference values). optiland.nonsequential uses micrometres everywhere
# else, so wavelengths are stored in um to match.
_PHOTOPIC_WAVELENGTHS_UM = np.arange(0.380, 0.781, 0.010)
_PHOTOPIC_V = np.array(
    [
        0.0000,
        0.0001,
        0.0004,
        0.0012,
        0.0040,
        0.0116,
        0.0230,
        0.0380,
        0.0600,
        0.0910,
        0.1390,
        0.2080,
        0.3230,
        0.5030,
        0.7100,
        0.8620,
        0.9540,
        0.9950,
        0.9950,
        0.9520,
        0.8700,
        0.7570,
        0.6310,
        0.5030,
        0.3810,
        0.2650,
        0.1750,
        0.1070,
        0.0610,
        0.0320,
        0.0170,
        0.0082,
        0.0041,
        0.0021,
        0.0010,
        0.00052,
        0.00025,
        0.00012,
        0.00006,
        0.00003,
        0.000015,
    ]
)

_SCOTOPIC_WAVELENGTHS_UM = _PHOTOPIC_WAVELENGTHS_UM
_SCOTOPIC_V = np.array(
    [
        0.000589,
        0.002209,
        0.00929,
        0.03484,
        0.0966,
        0.1998,
        0.3281,
        0.4550,
        0.5670,
        0.6760,
        0.7930,
        0.9040,
        0.9820,
        0.9970,
        0.9350,
        0.8110,
        0.6500,
        0.4810,
        0.3288,
        0.2076,
        0.1212,
        0.0655,
        0.03315,
        0.01593,
        0.00737,
        0.003335,
        0.001497,
        0.000677,
        0.000313,
        0.000148,
        0.0000715,
        0.0000353,
        0.0000178,
        0.00000914,
        0.00000478,
        0.00000255,
        0.00000139,
        0.000000760,
        0.000000425,
        0.000000241,
        0.000000139,
    ]
)

# Luminous efficacy at the peak of each weighting function (CIE-defined for
# photopic; the commonly-cited value for scotopic).
KM_PHOTOPIC = 683.002  # lm/W at 555 nm
KM_SCOTOPIC = 1700.0  # lm/W at 507 nm

# Visible band the tables above cover. A wavelength (or an entire spectrum)
# outside this range has V(lambda) == 0 everywhere it's defined and 0
# (by extension, not tabulated) everywhere else -- the guardrail treats
# both the same way.
VISIBLE_BAND_UM = (0.380, 0.780)

# Below this fraction of the peak luminous efficacy, a spectrum's overlap
# with V(lambda) is treated as negligible (guardrail) rather than as
# a very dim but valid photometric answer.
_NEGLIGIBLE_EFFICACY_FRACTION = 1e-6


def _table(weighting: Weighting) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(wavelengths_um, V, Km)`` for a weighting function.

    Args:
        weighting: ``"photopic"`` or ``"scotopic"``.

    Returns:
        The wavelength grid, the V(lambda)/V'(lambda) table, and the peak
        luminous efficacy Km/K'm [lm/W].

    Raises:
        ValueError: If ``weighting`` is not recognised.
    """
    if weighting == "photopic":
        return _PHOTOPIC_WAVELENGTHS_UM, _PHOTOPIC_V, KM_PHOTOPIC
    if weighting == "scotopic":
        return _SCOTOPIC_WAVELENGTHS_UM, _SCOTOPIC_V, KM_SCOTOPIC
    raise ValueError(
        f"Unknown weighting {weighting!r}; expected 'photopic' or 'scotopic'."
    )


def v_lambda(wavelength_um: float | np.ndarray, weighting: Weighting = "photopic"):
    """Evaluate V(lambda) (or V'(lambda)) at one or more wavelengths.

    Args:
        wavelength_um: Wavelength(s) [um].
        weighting: ``"photopic"`` (default) or ``"scotopic"``.

    Returns:
        The luminous efficiency function value(s), 0 outside the tabulated
        visible band -- correct physics (the eye has no response there),
        not a missing-data placeholder. Callers that must distinguish
        "genuinely zero" from "out of band" should check
        :data:`VISIBLE_BAND_UM` themselves; :func:`to_photometric` and
        :func:`lumens_to_watts` do this via the negligible-overlap guardrail.
    """
    wl_grid, v_grid, _ = _table(weighting)
    return np.interp(wavelength_um, wl_grid, v_grid, left=0.0, right=0.0)


def luminous_efficacy_of_spectrum(
    wavelengths_um: np.ndarray,
    weights: np.ndarray,
    weighting: Weighting = "photopic",
) -> float:
    """Luminous efficacy [lm/W] of a normalised spectral power distribution.

    Args:
        wavelengths_um: Wavelength samples [um].
        weights: Relative spectral power at each wavelength (need not be
            normalised; only the shape matters).
        weighting: ``"photopic"`` or ``"scotopic"``.

    Returns:
        ``Km * sum(weights * V(wavelengths)) / sum(weights)`` -- the
        spectrum-averaged luminous efficacy, in [0, Km].
    """
    wl = np.asarray(wavelengths_um, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    _, _, km = _table(weighting)
    total_weight = w.sum()
    if total_weight <= 0:
        return 0.0
    return float(km * np.sum(w * v_lambda(wl, weighting)) / total_weight)


def _require_in_band(
    efficacy_or_v: float, km: float, weighting: Weighting, context: str
) -> None:
    """Raise if a photometric quantity's spectral overlap is negligible.

    Args:
        efficacy_or_v: Either a luminous efficacy [lm/W] or a bare V(lambda)
            value -- both are compared against ``km`` on the same
            fractional scale.
        km: The weighting's peak luminous efficacy [lm/W].
        weighting: ``"photopic"`` or ``"scotopic"``, for the error message.
        context: What was being converted, for the error message.

    Raises:
        ValueError: If ``efficacy_or_v`` is a negligible fraction of ``km``.
    """
    if efficacy_or_v < km * _NEGLIGIBLE_EFFICACY_FRACTION:
        lo, hi = VISIBLE_BAND_UM
        raise ValueError(
            f"Cannot convert {context} to a {weighting} photometric quantity: "
            f"its spectral content has negligible overlap with the "
            f"{weighting} V(lambda) curve (nonzero only within "
            f"{lo:g}-{hi:g} um). This would silently report ~0 rather than "
            "a meaningful value, so it raises instead (guardrail). "
            "Pass an explicit in-band wavelength_um, or verify the trace's "
            "source spectrum actually falls in the visible band."
        )


def lumens_to_watts(
    total_flux_lm: float, spectrum: Spectrum, weighting: Weighting = "photopic"
) -> float:
    """Convert a source's lumens to the radiometric watts NSQ traces in.

    Args:
        total_flux_lm: Source output in lumens.
        spectrum: The source's :class:`~optiland.nonsequential.sources.base.Spectrum`
            -- its shape (not absolute scale) determines the conversion.
        weighting: ``"photopic"`` (default) or ``"scotopic"``.

    Returns:
        Radiant flux [W] such that a source emitting this many watts, with
        this spectral shape, emits ``total_flux_lm`` lumens.

    Raises:
        ValueError: If the spectrum has negligible overlap with the chosen
            V(lambda) curve (guardrail) -- e.g. a monochromatic 1.5 um
            source has no photopic lumen equivalent worth reporting.
    """
    _, _, km = _table(weighting)
    efficacy = luminous_efficacy_of_spectrum(
        spectrum.wavelengths, spectrum.weights, weighting
    )
    _require_in_band(efficacy, km, weighting, "this source's spectrum")
    return total_flux_lm / efficacy


def _resolve_wavelength_weighting(
    result: object,
    weighting: Weighting,
    wavelength_um: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the (wavelengths, V) pair a result should be weighted by.

    Args:
        result: A detector result object.
        weighting: ``"photopic"`` or ``"scotopic"``.
        wavelength_um: Explicit monochromatic wavelength, required for a
            result type that carries no spectral breakdown of its own
            (``IrradianceMap``, ``FarFieldPattern``).

    Returns:
        ``(wavelengths_um, V_values)``, both length >= 1, describing the
        weighting to apply. For a ``SpectralResult`` this is its own
        wavelength bins; otherwise the single ``wavelength_um``.

    Raises:
        ValueError: If a spectral breakdown is required but neither the
            result nor ``wavelength_um`` provides one.
    """
    result_wavelengths = getattr(result, "wavelengths", None)
    if result_wavelengths is not None:
        wl = np.asarray(result_wavelengths, dtype=np.float64)
        return wl, v_lambda(wl, weighting)
    if wavelength_um is None:
        raise ValueError(
            f"{type(result).__name__} carries no per-wavelength breakdown "
            "(only SpectralResult does); pass wavelength_um= explicitly -- "
            "e.g. the monochromatic wavelength the source scene used."
        )
    wl = np.asarray([wavelength_um], dtype=np.float64)
    return wl, v_lambda(wl, weighting)


class PhotometricMap:
    """A 2D photometric map -- illuminance [lux] or luminance-equivalent.

    Mirrors :class:`~optiland.nonsequential.results.irradiance_map.IrradianceMap`'s
    shape so it can be plotted the same way, but the values and units are
    photometric, not radiometric.

    Attributes:
        data: Photometric map, shape (ny, nx). Lux for ``quantity=
            "illuminance"``.
        x_coords: Bin centre x-coordinates [mm], shape (nx,).
        y_coords: Bin centre y-coordinates [mm], shape (ny,).
        total: Total luminous flux [lm] over the whole map.
        quantity: The photometric quantity computed (``"illuminance"``).
        weighting: ``"photopic"`` or ``"scotopic"``.
    """

    def __init__(
        self,
        data: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        total: float,
        quantity: str,
        weighting: Weighting,
    ) -> None:
        """Initialize PhotometricMap."""
        self.data = data
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.total = float(total)
        self.quantity = quantity
        self.weighting = weighting

    def plot(self, ax=None, **kwargs):
        """Plot the photometric map.

        Args:
            ax: Optional Matplotlib Axes. If None, a new figure is created.
            **kwargs: Additional arguments passed to imshow.

        Returns:
            The Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        unit = "lux" if self.quantity == "illuminance" else self.quantity
        extent = [
            self.x_coords[0],
            self.x_coords[-1],
            self.y_coords[0],
            self.y_coords[-1],
        ]
        im = ax.imshow(
            self.data, origin="lower", extent=extent, aspect="equal", **kwargs
        )
        plt.colorbar(im, ax=ax, label=f"{self.quantity.capitalize()} [{unit}]")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{self.quantity.capitalize()} ({self.weighting}) map")
        return fig


class PhotometricScalar:
    """A single photometric quantity (e.g. total luminous flux).

    Attributes:
        value: The photometric value.
        quantity: ``"luminous_flux"`` (lumens).
        weighting: ``"photopic"`` or ``"scotopic"``.
    """

    def __init__(self, value: float, quantity: str, weighting: Weighting) -> None:
        """Initialize PhotometricScalar."""
        self.value = float(value)
        self.quantity = quantity
        self.weighting = weighting

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        unit = "lm" if self.quantity == "luminous_flux" else self.quantity
        return f"PhotometricScalar({self.value:.6g} {unit}, {self.weighting})"


def to_photometric(
    result: IrradianceMap | SpectralResult | FarFieldPattern,
    quantity: Literal["illuminance", "luminous_flux"] = "illuminance",
    weighting: Weighting = "photopic",
    wavelength_um: float | None = None,
) -> PhotometricMap | PhotometricScalar:
    """Convert a detector result to a photometric quantity.

    The radiometric result is unaffected; this returns a new object.

    Args:
        result: An ``IrradianceMap``, ``SpectralResult``, or
            ``FarFieldPattern`` (from ``SimulationResult.detectors[name]``).
        quantity: ``"illuminance"`` (lux, per-pixel map) or
            ``"luminous_flux"`` (total lumens, scalar).
        weighting: ``"photopic"`` (default, CIE 1931) or ``"scotopic"``
            (CIE 1951).
        wavelength_um: Required when ``result`` carries no per-wavelength
            breakdown of its own (i.e. anything but ``SpectralResult``) --
            the monochromatic wavelength to weight by.

    Returns:
        A :class:`PhotometricMap` for ``quantity="illuminance"``, or a
        :class:`PhotometricScalar` for ``quantity="luminous_flux"``.

    Raises:
        ValueError: If the result's spectral content -- explicit
            ``wavelength_um`` or the result's own wavelength bins -- has
            negligible overlap with the chosen V(lambda) curve (guardrail:
            this would otherwise silently return ~0).
        TypeError: If ``result`` has neither a pixel grid nor per
            -wavelength data (e.g. a ``RayDatabase``), or ``quantity`` is
            not recognised.
    """
    _, _, km = _table(weighting)
    wl, v_vals = _resolve_wavelength_weighting(result, weighting, wavelength_um)
    if v_vals.sum() <= 0.0:
        lo, hi = VISIBLE_BAND_UM
        wl_desc = f"{wl[0]:g}" if wl.size == 1 else f"{wl.min():g}-{wl.max():g}"
        raise ValueError(
            f"Cannot convert to a {weighting} photometric quantity: "
            f"wavelength(s) {wl_desc} um have negligible overlap with the "
            f"{weighting} V(lambda) curve (nonzero only within "
            f"{lo:g}-{hi:g} um). This would silently report ~0 rather than "
            "a meaningful value, so it raises instead (guardrail)."
        )

    if quantity == "illuminance":
        irradiance = getattr(result, "irradiance", None)
        if irradiance is None:
            raise TypeError(
                f"{type(result).__name__} has no 'irradiance' map; "
                "illuminance requires IrradianceMap or SpectralResult."
            )
        irradiance = np.asarray(irradiance, dtype=np.float64)
        irradiance_w_per_m2 = irradiance * 1.0e6  # W/mm^2 -> W/m^2
        if irradiance.ndim == 3:
            # SpectralResult: (ny, nx, n_lambda). Each bin's irradiance is
            # already the flux collected *within* that wavelength bin (not
            # a spectral density), so summing V(lambda_bin) * irradiance_bin
            # over bins is the correct discrete form of the lux integral --
            # no additional bin-width factor.
            weight = km * v_vals
            illuminance = np.tensordot(irradiance_w_per_m2, weight, axes=([2], [0]))
        else:
            illuminance = irradiance_w_per_m2 * (km * v_vals[0])
        pixel_area_mm2 = (
            (result.x_coords[1] - result.x_coords[0])
            * (result.y_coords[1] - result.y_coords[0])
            if len(result.x_coords) > 1 and len(result.y_coords) > 1
            else 1.0
        )
        total_lumens = float(illuminance.sum() * pixel_area_mm2 * 1e-6)
        return PhotometricMap(
            data=illuminance,
            x_coords=result.x_coords,
            y_coords=result.y_coords,
            total=total_lumens,
            quantity="illuminance",
            weighting=weighting,
        )

    if quantity == "luminous_flux":
        total_flux_w = getattr(result, "total_flux", None)
        if total_flux_w is None:
            raise TypeError(
                f"{type(result).__name__} has no 'total_flux'; cannot "
                "compute luminous_flux."
            )
        from optiland.backend.utils import to_numpy  # noqa: PLC0415

        total_flux_w = float(to_numpy(total_flux_w))
        irradiance = getattr(result, "irradiance", None)
        if wl.size > 1 and irradiance is not None:
            # SpectralResult: weight each bin by its actual share of the
            # collected flux (summed over pixels), not a flat average --
            # a spectrum concentrated near the V(lambda) peak should get
            # a higher effective efficacy than one spread evenly across
            # the same bins.
            per_bin_flux = np.asarray(irradiance, dtype=np.float64).sum(axis=(0, 1))
            total = per_bin_flux.sum()
            weight = (
                km * float(np.sum(per_bin_flux * v_vals) / total)
                if total > 0
                else km * v_vals.mean()
            )
        elif wl.size > 1:
            weight = km * v_vals.mean()
        else:
            weight = km * v_vals[0]
        return PhotometricScalar(
            value=total_flux_w * weight, quantity="luminous_flux", weighting=weighting
        )

    raise TypeError(
        f"Unknown quantity {quantity!r}; expected 'illuminance' or 'luminous_flux'."
    )
