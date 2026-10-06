"""HCIPy source -> Optiland scalar prescription -> Pyxel photon-rate file.

Run from the repository root with HCIPy installed. ESA Pyxel is optional; the
export uses its existing .npy loader and does not require it at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hcipy as hp
import numpy as np

import optiland.backend as be
from optiland.optic import Optic
from optiland.physical_apertures import RectangularAperture
from optiland.physical_optics import boundary_diagnostic
from optiland.physical_optics.interoperability import from_hcipy, to_hcipy
from optiland.physical_optics.radiometry import (
    field_to_irradiance,
    irradiance_to_photon_rate,
    normalized_psf,
)
from optiland.physical_optics.train import ScalarOpticalTrain


def main() -> None:
    """Generate calibrated monochromatic rate and conditional PSF arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    args = parser.parse_args()
    be.set_backend(args.backend)
    if args.backend == "torch":
        be.set_device("cpu")
        be.set_precision("float64")

    # HCIPy uses meters here. A waist field avoids ambiguity about off-waist
    # Gaussian phase conventions. We explicitly assign 1 nW captured power.
    # Binning by two produces odd detector dimensions, with a detector cell
    # centered on the optical axis: ready for native Pyxel PSF registration.
    ny, nx = 190, 254
    dx_m, dy_m = 6e-6, 8e-6
    wavelength_m, waist_m = 500e-9, 150e-6
    grid = hp.make_uniform_grid([nx, ny], [nx * dx_m, ny * dy_m])
    data = np.exp(-(grid.x**2 + grid.y**2) / waist_m**2)
    source = hp.Wavefront(hp.Field(data.astype(complex), grid), wavelength_m)
    source.total_power = 1e-9
    field = from_hcipy(source)  # Coordinates -> mm; amplitude -> sqrt(W/mm²).

    # Reuse Optiland's native ideal thin lens (f=50 mm in air), with an
    # explicit physical stop. The adapter uses its paraxial quadratic phase.
    optic = Optic()
    optic.surfaces.add(index=0, thickness=np.inf, material="air")
    optic.surfaces.add(
        index=1,
        z=0.0,
        surface_type="paraxial",
        f=50.0,
        material="air",
        aperture=RectangularAperture(-0.15, 0.20, -0.18, 0.18),
    )
    optic.surfaces.add(index=2, z=50.0, surface_type="plane", material="air")
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    diagnostic = boundary_diagnostic(output, edge_width=4, threshold=0.01, warn=True)

    # Calibration stays fixed: losses are NOT normalized away after the stop.
    irradiance = field_to_irradiance(output, irradiance_scale_w_per_mm2=1.0)
    rates = irradiance_to_photon_rate(
        irradiance,
        dx_mm=output.dx,
        dy_mm=output.dy,
        wavelength_nm=output.wavelength * 1e6,
        binning=(2, 2),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "photon_rate.npy", be.to_numpy(rates))
    np.save(args.output_dir / "conditional_psf.npy", be.to_numpy(normalized_psf(rates)))
    restored = to_hcipy(output)
    print(f"Input captured power: {float(be.to_numpy(field.power)):.12g} W")
    print(f"Output captured power: {float(be.to_numpy(output.power)):.12g} W")
    print(f"HCIPy exported power: {restored.total_power:.12g} W")
    print(f"Boundary fraction: {float(be.to_numpy(diagnostic.boundary_fraction)):.6g}")
    print(f"Photon rate: {float(be.to_numpy(be.sum(rates))):.12g} photons/s")
    print(f"Pyxel rows/columns: {rates.shape[0]}/{rates.shape[1]}")
    print(
        f"Pyxel vertical/horizontal pitch: {2 * output.dy * 1000:g}/"
        f"{2 * output.dx * 1000:g} um; wavelength: {output.wavelength * 1e6:g} nm"
    )
    print("Use load_image with time_scale=1.0 and convert_to_photons=False.")
    print(
        "Use the conditional PSF only as a normalized blur kernel, "
        "not as absolute illumination."
    )


if __name__ == "__main__":
    main()
