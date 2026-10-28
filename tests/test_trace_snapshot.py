"""Trace snapshot / resume golden tests (opt v2 PR3.1).

A trace resumed at the surface returned by ``partial_trace_start`` must equal a
full trace at 1e-12 (decision V22) for every eligible variable type on every
benchmark optic. Each optic is first brought into the eligible configuration
(object-side aperture, upstream-only aiming) where it is not already, so every
variable type its surfaces support is exercised after the stop.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from benchmarks.optimization.problems import PROBLEMS
from optiland.geometries import EvenAsphere, ZernikePolynomialGeometry
from optiland.optimization.variable.variable import Variable
from optiland.raytrace.snapshot import TraceSnapshot

from .golden.snapshot import GOLDEN_PROBLEMS

pytestmark = pytest.mark.filterwarnings("ignore")

RTOL = 1e-12
RAY_FIELDS = ("x", "y", "z", "L", "M", "N", "opd", "i")
SURFACE_FIELDS = ("x", "y", "z", "L", "M", "N", "opd", "intensity")

# Two fields and three pupil points per field.
HY = (0.0, 0.0, 0.0, 0.7, 0.7, 0.7)
PX = (0.0, 0.5, -0.3, 0.0, 0.4, -0.2)
PY = (0.0, -0.4, 0.6, 0.0, 0.3, -0.5)


def _rays_args(optic):
    hx = be.zeros(len(HY))
    return hx, be.array(HY), be.array(PX), be.array(PY), optic.primary_wavelength


def _eligible_optic(name: str, aiming: str = "paraxial"):
    """Benchmark optic with an object-side aperture and upstream-only aiming."""
    optic, problem = PROBLEMS[name]()
    if optic.aperture.ap_type not in ("EPD", "float_by_stop_size", "objectNA"):
        optic.set_aperture("EPD", float(be.to_numpy(optic.paraxial.EPD())))
    optic.ray_tracer.set_aiming(aiming)  # also drops any cache wrapper
    return optic, problem


def _candidate_specs(optic) -> list[tuple[int, str, dict]]:
    """One (surface, type, kwargs) of every eligible type per surface after the stop."""
    wavelength = optic.primary_wavelength
    specs = []
    last = len(optic.surfaces.surfaces) - 1
    for s in range(optic.surfaces.stop_index + 1, last):
        geometry = optic.surfaces[s].geometry
        specs += [(s, "thickness", {}), (s, "index", {"wavelength": wavelength})]
        if hasattr(optic.surfaces[s].material_post, "name"):  # a catalog glass
            specs.append((s, "material", {"glass_selection": ["N-SF11", "N-BK7"]}))
        if hasattr(geometry, "radius"):
            specs += [(s, "radius", {}), (s, "reciprocal_radius", {}), (s, "conic", {})]
        if type(geometry) is EvenAsphere:
            specs.append((s, "asphere_coeff", {"coeff_number": 0}))
        if type(geometry) is ZernikePolynomialGeometry:
            specs.append((s, "zernike_coeff", {"coeff_index": 4}))
    return specs


def _perturb(variable: Variable) -> None:
    if variable.type == "material":
        current = variable.value
        variable.update("N-SF11" if current != "N-SF11" else "N-BK7")
    elif variable.type == "index":
        variable.update(variable.value + 0.01)
    else:
        raw = variable.variable.get_value()
        raw = float(be.to_numpy(raw))
        variable.variable.update_value(raw * 1.001 + 1e-4)
    variable.optic.updater.update()


def _state(rays, optic, start: int) -> dict[str, np.ndarray]:
    state = {f"rays.{a}": be.to_numpy(getattr(rays, a)).copy() for a in RAY_FIELDS}
    for k, surface in enumerate(optic.surfaces.surfaces[start:], start=start):
        for a in SURFACE_FIELDS:
            state[f"surface{k}.{a}"] = be.to_numpy(getattr(surface, a)).copy()
    return state


def _assert_states_close(actual: dict, expected: dict, label: str) -> None:
    assert actual.keys() == expected.keys()
    for key, e in expected.items():
        a = actual[key]
        scale = np.nanmax(np.abs(e), initial=0.0) if e.size else 0.0
        np.testing.assert_allclose(
            a,
            e,
            rtol=RTOL,
            atol=RTOL * max(scale, 1.0),
            equal_nan=True,
            err_msg=f"{label}: {key}",
        )


def _assert_resumes_exactly(optic, specs) -> set[str]:
    """Check every (surface, type, kwargs) variable resumes exactly.

    Returns:
        The variable types checked.
    """
    tracer = optic.ray_tracer
    args = _rays_args(optic)
    covered = set()
    for s, type_name, kwargs in specs:
        variable = Variable(optic, type_name, surface_number=s, **kwargs)
        assert tracer.partial_trace_start(variable) == s, str(variable)
        material = optic.surfaces[s].material_post
        raw = variable.variable.get_value()
        _, snapshot = tracer._trace(*args)

        _perturb(variable)
        full = _state(tracer.trace_generic(*args), optic, s)
        resumed = _state(tracer._trace_from(s, snapshot), optic, s)
        _assert_states_close(resumed, full, str(variable))
        covered.add(type_name)

        # Restore the material object itself: an index update replaces it.
        if type_name in ("index", "material"):
            optic.updater.set_material(material, s)
        else:
            variable.variable.update_value(raw)
        optic.updater.update()
    return covered


@pytest.mark.parametrize("name", GOLDEN_PROBLEMS)
def test_resumed_trace_equals_full_trace(set_test_backend, name):
    optic, _ = _eligible_optic(name)
    covered = _assert_resumes_exactly(optic, _candidate_specs(optic))
    assert {"radius", "reciprocal_radius", "conic", "thickness"} <= covered
    assert {"index", "material"} <= covered


def test_resumed_trace_with_iterative_aiming(set_test_backend):
    # Real aiming to the stop is upstream-only too. One surface after the stop
    # keeps the iterative aimer's cost down; it carries every variable type.
    optic, _ = _eligible_optic("wide_asphere", aiming="iterative")
    stop = optic.surfaces.stop_index
    asphere = next(
        s
        for s in range(stop + 1, len(optic.surfaces.surfaces))
        if type(optic.surfaces[s].geometry) is EvenAsphere
    )
    specs = [spec for spec in _candidate_specs(optic) if spec[0] == asphere]
    covered = _assert_resumes_exactly(optic, specs)
    assert "asphere_coeff" in covered


@pytest.mark.parametrize("name", GOLDEN_PROBLEMS)
def test_problem_variables_resume_exactly(set_test_backend, name):
    # The benchmark's own variables, on the optic as configured: any variable
    # judged eligible must resume exactly; ineligible ones are skipped.
    optic, problem = PROBLEMS[name]()
    tracer = optic.ray_tracer
    args = _rays_args(optic)
    for variable in problem.variables:
        start = tracer.partial_trace_start(variable)
        if start is None:
            continue
        initial = variable.value
        _, snapshot = tracer._trace(*args)
        _perturb(variable)
        full = _state(tracer.trace_generic(*args), optic, start)
        resumed = _state(tracer._trace_from(start, snapshot), optic, start)
        _assert_states_close(resumed, full, str(variable))
        variable.update(initial)
        optic.updater.update()


def test_snapshot_trace_matches_trace_generic(set_test_backend):
    optic, _ = PROBLEMS["cooke"]()
    tracer = optic.ray_tracer
    args = _rays_args(optic)
    expected = _state(tracer.trace_generic(*args), optic, 0)
    rays, snapshot = tracer._trace(*args)
    _assert_states_close(_state(rays, optic, 0), expected, "cooke")
    assert isinstance(snapshot, TraceSnapshot)
    assert len(snapshot) == len(optic.surfaces.surfaces)


def test_snapshot_state_is_before_each_surface(set_test_backend):
    optic, _ = PROBLEMS["cooke"]()
    rays, snapshot = optic.ray_tracer._trace(*_rays_args(optic))
    surfaces = optic.surfaces.surfaces
    for k in range(1, len(surfaces)):
        # The state before surface k is the state recorded after surface k-1.
        before = snapshot.states[k]
        for a, b in zip(RAY_FIELDS, SURFACE_FIELDS, strict=True):
            np.testing.assert_array_equal(
                be.to_numpy(getattr(before, a)),
                be.to_numpy(getattr(surfaces[k - 1], b)),
            )


def test_resume_at_zero_and_repeat_resumes(set_test_backend):
    optic, _ = PROBLEMS["cooke"]()
    tracer = optic.ray_tracer
    rays, snapshot = tracer._trace(*_rays_args(optic))
    expected = _state(rays, optic, 0)
    for _ in range(2):  # the snapshot is not consumed by a resume
        _assert_states_close(
            _state(tracer._trace_from(0, snapshot), optic, 0), expected, "cooke"
        )
        _assert_states_close(
            _state(tracer._trace_from(5, snapshot), optic, 5),
            {k: v for k, v in expected.items() if not k.startswith("surface")}
            | {
                k: v
                for k, v in expected.items()
                if k.startswith("surface") and int(k[7:].split(".")[0]) >= 5
            },
            "cooke",
        )


def test_trace_from_rejects_other_surfaces_and_bad_index():
    optic, _ = PROBLEMS["cooke"]()
    tracer = optic.ray_tracer
    _, snapshot = tracer._trace(*_rays_args(optic))
    with pytest.raises(IndexError):
        tracer._trace_from(len(snapshot), snapshot)
    with pytest.raises(IndexError):
        tracer._trace_from(-1, snapshot)
    optic.surfaces.remove(3)
    with pytest.raises(ValueError, match="different surface list"):
        tracer._trace_from(5, snapshot)
