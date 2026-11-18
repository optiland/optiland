"""Forbes Q surface: the radial sums and their gradients in one backward pass.

In reverse mode, ``RadialSum`` gives the radial sums of a Q surface without
a graph and computes their gradients in one backward pass from a table of
the basis. A backward pass with ``create_graph`` evaluates the sums again
with autograd, and ``torch.func`` transforms take autograd. The gradients
are those of autograd, to rounding.
"""

from __future__ import annotations

import pytest

from .forbes_evaluation_test_utils import QBFS, X, Y, _geometry
from .nr_implicit_test_utils import backend_state


def _graph_has(tensor, name):
    """Return True if a node of the graph of ``tensor`` has ``name`` in its name."""
    seen, stack = set(), [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if name in type(node).__name__:
            return True
        stack.extend(next_node for next_node, _ in node.next_functions)
    return False


class TestRadialSum:
    @staticmethod
    def _loss(geometry, x, y):
        import torch

        weights = torch.linspace(0.3, 1.7, len(X), dtype=torch.float64)
        nx, ny, nz = geometry._surface_normal(x, y)
        return (weights * (geometry.sag(x, y) + 0.7 * nx - 1.3 * ny + nz)).sum()

    @staticmethod
    def _gradients(terms, whole_sum, monkeypatch, create_graph=False):
        """Gradients of the loss: with RadialSum, or with autograd only."""
        import torch

        from optiland.geometries.forbes import _whole_sum

        leaves = {
            key: torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for key, value in terms.items()
        }
        x = torch.tensor(X, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
        with monkeypatch.context() as patch:
            if not whole_sum:
                patch.setattr(_whole_sum, "records_reverse", lambda *args: False)
            loss = TestRadialSum._loss(_geometry(leaves), x, y)
        assert _graph_has(loss, "RadialSum") == whole_sum
        inputs = [*leaves.values(), x, y]
        return torch.autograd.grad(loss, inputs, create_graph=create_graph), inputs

    @pytest.mark.parametrize(
        "terms", [QBFS, {0: 2.0e-3}, {0: 1.0e-3, 1: 0.0, 2: -3.0e-4}], ids=str
    )
    def test_gradients_match_autograd(self, terms, monkeypatch):
        """Coefficients and positions: the gradients of autograd, to rounding."""
        torch = pytest.importorskip("torch")
        with backend_state("torch", "float64"):
            expected, _ = self._gradients(terms, False, monkeypatch)
            result, _ = self._gradients(terms, True, monkeypatch)
            for value, reference in zip(result, expected, strict=True):
                torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-18)

    def test_create_graph_gives_the_second_derivatives_of_autograd(self, monkeypatch):
        """A backward pass with create_graph evaluates the sums with autograd."""
        torch = pytest.importorskip("torch")
        with backend_state("torch", "float64"):
            results = []
            for whole_sum in (False, True):
                first, inputs = self._gradients(QBFS, whole_sum, monkeypatch, True)
                coefficient, x = first[1], first[-2]
                second = torch.autograd.grad(
                    coefficient + x.sum(), inputs, allow_unused=True
                )
                results.append(second)
            for value, reference in zip(*reversed(results), strict=True):
                torch.testing.assert_close(value, reference, rtol=1e-13, atol=1e-18)

    def test_transforms_take_autograd(self):
        """torch.func: no RadialSum; the Jacobian is that of reverse mode."""
        torch = pytest.importorskip("torch")
        with backend_state("torch", "float64"):
            keys = list(QBFS)
            x = torch.tensor(X, dtype=torch.float64)
            y = torch.tensor(Y, dtype=torch.float64)

            def sag(values):
                return _geometry(dict(zip(keys, values, strict=True))).sag(x, y)

            values = torch.tensor([QBFS[k] for k in keys], dtype=torch.float64)
            result = torch.func.jacrev(sag)(values)
            leaf = values.clone().requires_grad_()
            output = sag(list(leaf.unbind()))
            assert _graph_has(output, "RadialSum")
            rows = [
                torch.autograd.grad(output[i], leaf, retain_graph=True)[0]
                for i in range(len(X))
            ]
            torch.testing.assert_close(result, torch.stack(rows), rtol=1e-12, atol=0)
