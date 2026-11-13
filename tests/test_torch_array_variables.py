"""Variables that write one entry of a shared array work with Torch optimizers.

``polynomial_coeff``, ``chebyshev_coeff``, ``zernike_coeff`` and the NURBS
variables write one entry into a coefficient (or control-point) array that
all such variables of a surface share. A Torch optimizer writes the
parameters into the variables at each step. Before ``release_graph``, the
array kept the autograd graph of the last step: the second backward pass of
``polynomial_coeff`` and ``chebyshev_coeff`` then failed with "Trying to
backward through the graph a second time", and the Zernike graph grew with
each step. A coefficient outside the array was padded with ``np.pad``,
which turned a Torch tensor into a NumPy array.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.optimization import OptimizationProblem, TorchAdamOptimizer
from optiland.optimization.variable.nurbs import NurbsPointsVariable
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.surface_group import SurfaceGroup

from .nr_implicit_test_utils import backend_state

torch = pytest.importorskip("torch")

SHAPES = {
    "chebyshev_coeff": {
        "surface_type": "chebyshev",
        "coefficients": [[1e-4, 1e-3], [2e-4, 3e-4]],
        "norm_x": 12.0,
        "norm_y": 12.0,
    },
    "polynomial_coeff": {
        "surface_type": "polynomial",
        "coefficients": [[0.0, 0.0, 1e-4], [0.0, 2e-5, 0.0], [1e-4, 0.0, 0.0]],
    },
    "zernike_coeff": {
        "surface_type": "zernike",
        "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 1e-3, -1e-3, 0.0, 0.0],
        "norm_radius": 12.0,
    },
}
# Two variables per surface, so the test also checks that both keep their
# gradient within one step.
INDICES = {
    "chebyshev_coeff": [(1, 0), (0, 1)],
    "polynomial_coeff": [(1, 1), (0, 2)],
    "zernike_coeff": [5, 6],
}


def _singlet(shape):
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=30.0, thickness=5.0, is_stop=True, material="N-BK7"
    )
    optic.surfaces.add(index=2, radius=-45.0, thickness=45.0, **shape)
    optic.surfaces.add(index=3)
    optic.set_aperture("EPD", 10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.fields.add(y=4.0)
    optic.wavelengths.add(value=0.5876, is_primary=True)
    return optic


def _problem(variable_type, indices):
    optic = _singlet(SHAPES[variable_type])
    problem = OptimizationProblem()
    for hy in (0.0, 1.0):
        problem.add_operand(
            "rms_spot_size",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": optic,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": hy,
                "num_rays": 8,
                "wavelength": 0.5876,
                "distribution": "hexapolar",
            },
        )
    key = "coeff_index"
    for index in indices:
        problem.add_variable(optic, variable_type, surface_number=2, **{key: index})
    return optic, problem


def _graph_size(tensor):
    """Number of autograd nodes behind ``tensor``."""
    if getattr(tensor, "grad_fn", None) is None:
        return 0
    seen, stack = set(), [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        stack.extend(child for child, _ in node.next_functions)
    return len(seen)


@pytest.mark.parametrize("variable_type", list(SHAPES))
def test_adam_runs_several_steps_and_moves_each_variable(variable_type):
    """Five steps run, and each of the two variables of the surface moves."""
    with backend_state("torch", "float64"):
        be.grad_mode.enable()
        try:
            _, problem = _problem(variable_type, INDICES[variable_type])
            before = [float(be.to_numpy(v.value)) for v in problem.variables]
            optimizer = TorchAdamOptimizer(problem)
            optimizer.optimize(n_steps=5, lr=1e-5, disp=False)
            after = [float(be.to_numpy(v.value)) for v in problem.variables]
        finally:
            be.grad_mode.disable()
    for start, end in zip(before, after, strict=True):
        assert end != start


@pytest.mark.parametrize("variable_type", list(SHAPES))
def test_graph_does_not_grow_with_the_steps(variable_type):
    """The graph behind the array has the same size after step 1 and 10."""
    with backend_state("torch", "float64"):
        be.grad_mode.enable()
        try:
            optic, problem = _problem(variable_type, INDICES[variable_type])
            optimizer = TorchAdamOptimizer(problem)
            sizes = []
            for _ in range(10):
                optimizer.optimize(n_steps=1, lr=1e-5, disp=False)
                sizes.append(_graph_size(optic.surfaces[2].geometry.coefficients))
        finally:
            be.grad_mode.disable()
    assert sizes[-1] == sizes[0]


@pytest.mark.parametrize("variable_type", list(SHAPES))
def test_only_the_first_variable_of_a_step_copies_the_array(variable_type):
    """After one variable released the shared array, the others keep it.

    So each step copies a shared array once, and the setter of the Zernike
    coefficients builds one new Zernike object, not one per variable.
    """
    with backend_state("torch", "float64"):
        be.grad_mode.enable()
        try:
            optic, problem = _problem(variable_type, INDICES[variable_type])
            TorchAdamOptimizer(problem).optimize(n_steps=1, lr=1e-5, disp=False)
            geometry = optic.surfaces[2].geometry
            first, second = problem.variables
            assert geometry.coefficients.requires_grad
            first.release_graph()
            released = geometry.coefficients
            state = vars(geometry).get("zernike")
            assert not released.requires_grad
            second.release_graph()
            assert geometry.coefficients is released
            assert vars(geometry).get("zernike") is state
        finally:
            be.grad_mode.disable()


def test_index_outside_the_array_pads_a_tensor():
    """A polynomial coefficient outside the array pads it with zeros on Torch."""
    with backend_state("torch", "float64"):
        be.grad_mode.enable()
        try:
            optic, problem = _problem("polynomial_coeff", [(3, 4)])
            coefficients = optic.surfaces[2].geometry.coefficients
            assert isinstance(coefficients, torch.Tensor)
            assert tuple(coefficients.shape) == (4, 5)
            assert float(coefficients[3, 4]) == 0.0
            optimizer = TorchAdamOptimizer(problem)
            optimizer.optimize(n_steps=2, lr=1e-5, disp=False)
            assert isinstance(optic.surfaces[2].geometry.coefficients, torch.Tensor)
            assert optimizer.params[0].grad is not None
            assert float(optimizer.params[0].grad) != 0.0
        finally:
            be.grad_mode.disable()


def test_nurbs_points_survive_a_second_backward_pass():
    """Two update-backward cycles with release_graph in between both work."""
    with backend_state("torch", "float64"):
        geometry = NurbsGeometry(CoordinateSystem(), nurbs_norm_x=1, nurbs_norm_y=1)
        geometry.fit_surface()
        surface = Surface(
            previous_surface=None, geometry=geometry, material_post=IdealMaterial(n=1.0)
        )
        optic = Optic()
        optic.surfaces = SurfaceGroup([surface])
        variable = NurbsPointsVariable(optic, 0, (1, 1, 2), apply_scaling=False)
        param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        for _ in range(2):
            variable.release_graph()
            assert geometry.P.grad_fn is None
            variable.update_value(param * 2.0)
            (grad,) = torch.autograd.grad((geometry.P**2).sum(), param)
            assert float(grad) == pytest.approx(8.0 * 0.1)


def test_release_graph_keeps_numpy_values():
    """On NumPy, release_graph leaves the values and the array type unchanged."""
    with backend_state("numpy", "float64"):
        _, problem = _problem("polynomial_coeff", INDICES["polynomial_coeff"])
        geometry = problem.variables[0].variable.optic.surfaces[2].geometry
        before = np.array(geometry.coefficients, copy=True)
        problem.variables.release_graphs()
        assert isinstance(geometry.coefficients, np.ndarray)
        np.testing.assert_array_equal(geometry.coefficients, before)
