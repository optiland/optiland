"""Benchmark problems for the optimization v2 program (decision V26).

Each public function returns ``(optic, problem)`` with a fully populated
:class:`~optiland.optimization.OptimizationProblem`. All construction is
deterministic: any randomised perturbation uses a fixed seed.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


import numpy as np

import optiland.backend as be
from optiland.optic import Optic
from optiland.optimization import OptimizationProblem
from optiland.samples.objectives import CookeTriplet, DoubleGauss

SEED = 20261028
WAVELENGTHS = (0.4861, 0.5876, 0.6563)
SCALING_SIZES = (10, 25, 50, 100)
FIELDS_3 = ((0.0, 0.0), (0.0, 0.7), (0.0, 1.0))


def _add_spot_operands(
    problem: OptimizationProblem,
    optic: Optic,
    fields: tuple[tuple[float, float], ...],
    wavelengths: tuple[float, ...],
    num_rays: int = 5,
) -> None:
    """Add one RMS spot operand per field x wavelength."""
    for hx, hy in fields:
        for wl in wavelengths:
            problem.add_operand(
                operand_type="rms_spot_size",
                target=0.0,
                weight=1.0,
                input_data={
                    "optic": optic,
                    "surface_number": -1,
                    "Hx": hx,
                    "Hy": hy,
                    "num_rays": num_rays,
                    "wavelength": wl,
                    "distribution": "hexapolar",
                },
            )


def _add_efl_operand(problem: OptimizationProblem, optic: Optic) -> None:
    """Hold the effective focal length at its starting value."""
    problem.add_operand(
        operand_type="f2",
        target=float(be.to_numpy(optic.paraxial.f2())),
        weight=1.0,
        input_data={"optic": optic},
    )


def _is_curved(optic: Optic, surface_number: int) -> bool:
    """Whether the surface has a finite radius of curvature."""
    radius = float(be.to_numpy(optic.surfaces[surface_number].geometry.radius))
    return bool(np.isfinite(radius))


def cooke() -> tuple[Optic, OptimizationProblem]:
    """Cooke triplet: radii + thicknesses, RMS spot 3F x 3W + EFL."""
    optic = CookeTriplet()
    problem = OptimizationProblem()
    _add_spot_operands(problem, optic, FIELDS_3, WAVELENGTHS)
    _add_efl_operand(problem, optic)
    for i in range(1, 7):
        if _is_curved(optic, i):
            problem.add_variable(optic, "radius", surface_number=i)
        problem.add_variable(optic, "thickness", surface_number=i)
    return optic, problem


def double_gauss() -> tuple[Optic, OptimizationProblem]:
    """Double Gauss: radii + thicknesses, RMS spot 3F x 3W + EFL (Section 1)."""
    optic = DoubleGauss()
    problem = OptimizationProblem()
    _add_spot_operands(problem, optic, FIELDS_3, WAVELENGTHS)
    _add_efl_operand(problem, optic)
    for i in range(1, 12):
        if _is_curved(optic, i):
            problem.add_variable(optic, "radius", surface_number=i)
        problem.add_variable(optic, "thickness", surface_number=i)
    return optic, problem


_WIDE_TABLE = (
    (8.0107, 0.2500, "N-SK4"),
    (1.1856, 0.9613, "air"),
    (1.6747, 0.3578, ("SF1", "schott")),
    (-7.5157, 0.2136, "N-SK4"),
    (0.9411, 0.4146, "air"),
    (-1.7688, 0.3333, ("SF1", "schott")),
    (-1.5531, 0.3863, "N-SK4"),
    (-2.2281, 0.9842, "air"),
    (be.inf, 0.1000, "air"),  # stop
    (13.6803, 0.1400, "N-SK4"),
    (-3.4279, 0.0605, "air"),
    (20.0257, 0.3332, "N-SK4"),
    (-0.9258, 0.4374, ("SF1", "schott")),
    (-3.2233, 0.2679, "air"),
    (10.3847, 0.1400, ("SF1", "schott")),
    (2.4272, 0.2609, "N-SK4"),
    (-3.8828, 0.0150, "air"),
    (3.3650, 0.3167, "N-SK4"),
    (-14.7547, 1.5604, "air"),
)
_WIDE_ASPHERE_SURFACES = (2, 5, 8, 14, 19)
_WIDE_STOP = 9


def wide_asphere() -> tuple[Optic, OptimizationProblem]:
    """Wide-angle lens with even-asphere terms as variables, 5 fields.

    The 100 degree full-field wide-angle sample with aspheric terms on five
    surfaces. The edge fields are ray-failure prone.
    """
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    for i, (radius, thickness, material) in enumerate(_WIDE_TABLE, start=1):
        kwargs = {"index": i, "radius": radius, "thickness": thickness}
        if material != "air":
            kwargs["material"] = material
        if i == _WIDE_STOP:
            kwargs["is_stop"] = True
        if i in _WIDE_ASPHERE_SURFACES:
            kwargs["surface_type"] = "even_asphere"
            kwargs["coefficients"] = [0.0, 0.0, 0.0]
        optic.surfaces.add(**kwargs)
    optic.surfaces.add(index=len(_WIDE_TABLE) + 1, radius=be.inf, thickness=0.0)
    optic.set_aperture("imageFNO", 4.0)
    optic.fields.set_type("angle")
    for y in (0.0, 15.0, 30.0, 40.0, 50.0):
        optic.fields.add(y=y)
    for wl in WAVELENGTHS:
        optic.wavelengths.add(value=wl, is_primary=(wl == WAVELENGTHS[1]))
    optic.ray_tracer.set_aiming("iterative", cache=True)

    problem = OptimizationProblem()
    fields = tuple((0.0, h) for h in (0.0, 0.3, 0.6, 0.8, 1.0))
    _add_spot_operands(problem, optic, fields, WAVELENGTHS)
    _add_efl_operand(problem, optic)
    for i in _WIDE_ASPHERE_SURFACES:
        for k in range(3):
            problem.add_variable(
                optic, "asphere_coeff", surface_number=i, coeff_number=k
            )
        problem.add_variable(optic, "radius", surface_number=i)
    return optic, problem


def freeform() -> tuple[Optic, OptimizationProblem]:
    """Off-axis refractive system with a Zernike freeform and a tilted surface."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=40.0, thickness=6.0, material="N-BK7", is_stop=True, dy=3.0
    )
    optic.surfaces.add(
        index=2,
        surface_type="zernike",
        radius=-60.0,
        thickness=20.0,
        coefficients=[0.0] * 10,
        norm_radius=10.0,
        dy=3.0,
    )
    optic.surfaces.add(
        index=3, radius=35.0, thickness=5.0, material="N-SF11", rx=0.02, dy=1.0
    )
    optic.surfaces.add(index=4, radius=-80.0, thickness=60.0)
    optic.surfaces.add(index=5)
    optic.set_aperture("EPD", 10.0)
    optic.fields.set_type("angle")
    for y in (0.0, 3.0, 5.0):
        optic.fields.add(y=y)
    for wl in WAVELENGTHS:
        optic.wavelengths.add(value=wl, is_primary=(wl == WAVELENGTHS[1]))

    problem = OptimizationProblem()
    _add_spot_operands(problem, optic, FIELDS_3, WAVELENGTHS)
    _add_efl_operand(problem, optic)
    for i in (1, 3, 4):
        problem.add_variable(optic, "radius", surface_number=i)
    for i in (1, 2, 3, 4):
        problem.add_variable(optic, "thickness", surface_number=i)
    for idx in range(3, 10):
        problem.add_variable(optic, "zernike_coeff", surface_number=2, coeff_index=idx)
    problem.add_variable(optic, "tilt", surface_number=3, axis="x")
    problem.add_variable(optic, "decenter", surface_number=3, axis="y")
    return optic, problem


def scaling(n: int) -> tuple[Optic, OptimizationProblem]:
    """Synthetic weak-lens chain with exactly ``n`` radius variables.

    The chain holds ``ceil(n / 2)`` thin singlets. Every radius is a variable,
    except the last one when ``n`` is odd. Radii get a fixed-seed perturbation
    so the problem is not degenerate.

    Args:
        n: Number of optimization variables (a positive integer).

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(SEED)
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    index = 1
    for k in range((n + 1) // 2):
        r1 = 400.0 * (1.0 + 0.1 * rng.standard_normal())
        r2 = -400.0 * (1.0 + 0.1 * rng.standard_normal())
        optic.surfaces.add(
            index=index,
            radius=r1,
            thickness=2.0,
            material="N-BK7",
            is_stop=(k == 0),
        )
        optic.surfaces.add(index=index + 1, radius=r2, thickness=5.0)
        index += 2
    optic.surfaces.add(index=index)
    optic.set_aperture("EPD", 4.0)
    optic.fields.set_type("angle")
    for y in (0.0, 1.0, 2.0):
        optic.fields.add(y=y)
    optic.wavelengths.add(value=0.5876, is_primary=True)

    problem = OptimizationProblem()
    fields = tuple((0.0, h) for h in (0.0, 0.5, 1.0))
    _add_spot_operands(problem, optic, fields, (0.5876,), num_rays=3)
    _add_efl_operand(problem, optic)
    for i in range(1, n + 1):
        problem.add_variable(optic, "radius", surface_number=i)
    return optic, problem


PROBLEMS: dict[str, Callable[[], tuple[Optic, OptimizationProblem]]] = {
    "cooke": cooke,
    "double_gauss": double_gauss,
    "wide_asphere": wide_asphere,
    "freeform": freeform,
    **{f"scaling_{n}": (lambda n=n: scaling(n)) for n in SCALING_SIZES},
}

QUICK_PROBLEMS = ("cooke", "double_gauss", "scaling_10")
