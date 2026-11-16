"""One backward pass for the whole radial sum of a Q (slope-orthogonal) surface.

In reverse mode, autograd records each operation of the Clenshaw sums of a
Q surface: about three operations for each coefficient and each output,
for the sag and again for its slope. The backward pass then runs one node
for each of these operations.

``RadialSum`` evaluates the sums of the kept coefficients without a graph,
with the operations of the evaluation without a derivative, so the values
keep their bits. Its backward pass computes the gradients of the kept
coefficients and of u**2 from one table of the Clenshaw basis psi_i
(``_zero_terms._basis_table``):

- coefficient n: sum_x (w Q_n(x) + w_slope dQ_n/dx), with
  Q_n = sum_i D[i, n] psi_i and dQ_n/dx = sum_i (D^T A)[n, i] psi_i, as for
  the zero terms (see ``_zero_terms``);
- u**2: w dS/dx + w_slope d2S/dx2, with dS/dx = sum_j (A^T d)_j psi_j and
  d2S/dx2 = sum_j ((A A)^T d)_j psi_j for the Clenshaw coefficients d.

The gradients change in the last bits, because the sums run in another
order than the backward pass of autograd. A backward pass that records a
graph (``create_graph=True``) evaluates the sums again with autograd and
differentiates them, so higher derivatives are those of autograd.

The trimmed zero coefficients get their derivative from ``ZeroTerms``, as
before.
"""

from __future__ import annotations

import functools

import numpy as np

from optiland.geometries._derivatives import reverse_mode_only

from . import _zero_terms

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


def records_reverse(coefficients, usq) -> bool:
    """Return True if the evaluation records a graph for reverse mode only.

    Then ``RadialSum`` can give the derivatives: gradient recording is on, the
    coefficients or the positions require grad, and no ``torch.func``
    transform or forward-mode level is active (``RadialSum`` has no rules
    for them). The check reads no tensor values.

    Args:
        coefficients: The stacked coefficients of the surface.
        usq: The squared normalized radius u**2 of the evaluation.

    Returns:
        bool: False on NumPy and inside ``torch.no_grad``.
    """
    if torch is None or not torch.is_grad_enabled():
        return False
    if not (
        getattr(coefficients, "requires_grad", False)
        or getattr(usq, "requires_grad", False)
    ):
        return False
    return reverse_mode_only()


@functools.cache
def _slope_matrices(size, dtype, device):
    """Return A^T and (A A)^T of the Q-BFS basis of ``size`` terms."""
    a = np.array(_zero_terms._slope_matrix(True, 0, size))
    return (
        torch.tensor(a.T, dtype=dtype, device=device),
        torch.tensor((a @ a).T, dtype=dtype, device=device),
    )


class RadialJob:
    """The kept radial coefficients of a Q surface, for ``RadialSum``.

    A geometry builds two jobs (with and without the slope) at each
    preparation of its coefficients.

    Args:
        coefficients (list): The kept coefficients, at least one.
        basis (list): Their change of basis (``qpoly._family_basis``).
        total (int): The number of the stacked coefficients.
        slope (bool): The evaluation also gives dS/d(u**2).
        evaluate (Callable): ``evaluate(coefficients, basis, usq, slope)``
            gives the tuple (S,) or (S, dS/d(u**2)). With ``basis`` None, it
            computes the change of basis from the coefficients.
    """

    __slots__ = ("_matrices", "basis", "coefficients", "evaluate", "slope", "total")

    def __init__(self, coefficients, basis, total, slope, evaluate):
        self.coefficients = coefficients
        self.basis = basis
        self.total = total
        self.slope = slope
        self.evaluate = evaluate
        self._matrices = {}

    def matrices(self, dtype, device):
        """Return D^T, D^T A, A^T d and (A A)^T d for the kept degrees.

        Args:
            dtype (torch.dtype): The dtype of the matrices.
            device (torch.device): The device of the matrices.

        Returns:
            tuple: The rows for the coefficient gradients, and the
            coefficients of dS/dx and d2S/dx2 in the basis psi.
        """
        key = (dtype, device)
        found = self._matrices.get(key)
        if found is None:
            size = len(self.basis)
            degrees = tuple(range(size))
            rows = _zero_terms._rows(True, 0, degrees, False, dtype, device)
            slope_rows = _zero_terms._rows(True, 0, degrees, True, dtype, device)
            a_t, aa_t = _slope_matrices(size, dtype, device)
            d = torch.stack([value.detach() for value in self.basis])
            d = d.to(dtype=dtype, device=device)
            found = (rows, slope_rows, a_t @ d, aa_t @ d)
            self._matrices[key] = found
        return found


if torch is not None:

    class RadialSum(torch.autograd.Function):
        """The radial sums of ``RadialJob``, with one backward pass for all terms.

        Inputs: the job, the stacked coefficients, and u**2. The kept
        coefficients are the first ones. Outputs: S, and dS/d(u**2) for a
        job with the slope.
        """

        @staticmethod
        def forward(ctx, job, coefficients, usq):
            # The form with ctx in forward costs less per call than the form
            # with setup_context; records_reverse selects this function only
            # outside torch.func transforms.
            ctx.job = job
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(coefficients, usq)
            return job.evaluate(job.coefficients, job.basis, usq, job.slope)

        @staticmethod
        def backward(ctx, *gradients):
            coefficients, usq = ctx.saved_tensors
            needs = ctx.needs_input_grad[1:3]
            if torch.is_grad_enabled():
                found = _graph_gradients(ctx.job, coefficients, usq, gradients, needs)
            else:
                found = _table_gradients(ctx.job, coefficients, usq, gradients, needs)
            return (None, *found)


def _table_gradients(job, coefficients, usq, gradients, needs):
    """Return the gradients of the coefficients and of u**2 from the basis table.

    Args:
        job (RadialJob): The job of the evaluation.
        coefficients (torch.Tensor): The stacked coefficients.
        usq (torch.Tensor): u**2 of the evaluation.
        gradients (tuple): The gradients of S and dS/d(u**2), or None.
        needs (tuple[bool, bool]): Which inputs need a gradient.

    Returns:
        tuple: The gradients of the coefficients and of u**2, or None.
    """
    weight = gradients[0]
    weight_slope = gradients[1] if len(gradients) > 1 else None
    if weight is None and weight_slope is None:
        return None, None
    shape = usq.shape
    points = usq.detach().reshape(-1)
    flat = [None if g is None else g.reshape(-1) for g in (weight, weight_slope)]
    rows, slope_rows, first, second = job.matrices(points.dtype, points.device)
    table = _zero_terms._basis_table(True, 0, len(job.basis), points)
    gradient_c = gradient_u = None
    if needs[0]:
        kept = None
        for matrix, g in ((rows, flat[0]), (slope_rows, flat[1])):
            if g is not None:
                # Matrix-vector products: the fastest sums over the points
                # on a CPU and on a GPU.
                part = matrix @ (table @ g)
                kept = part if kept is None else kept + part
        gradient_c = coefficients.new_zeros(coefficients.shape)
        gradient_c[: len(job.basis)] = kept
    if needs[1]:
        for vector, g in ((first, flat[0]), (second, flat[1])):
            if g is not None:
                part = g * (vector @ table)
                gradient_u = part if gradient_u is None else gradient_u + part
        gradient_u = gradient_u.reshape(shape)
    return gradient_c, gradient_u


def _graph_gradients(job, coefficients, usq, gradients, needs):
    """Return the gradients with a graph, for a backward pass with create_graph.

    The sums are evaluated again with autograd, from the coefficients and
    u**2 with their graph, and differentiated with ``create_graph=True``.

    Args:
        job (RadialJob): The job of the evaluation.
        coefficients (torch.Tensor): The stacked coefficients.
        usq (torch.Tensor): u**2 of the evaluation.
        gradients (tuple): The gradients of S and dS/d(u**2), or None.
        needs (tuple[bool, bool]): Which inputs need a gradient.

    Returns:
        tuple: The gradients of the coefficients and of u**2, or None.
    """
    inputs = [
        value for value, need in zip((coefficients, usq), needs, strict=True) if need
    ]
    with torch.enable_grad():
        kept = list(coefficients.unbind())[: len(job.basis)]
        values = job.evaluate(kept, None, usq, job.slope)
    pairs = [(v, g) for v, g in zip(values, gradients, strict=True) if g is not None]
    if not inputs or not pairs:
        return None, None
    outputs, grad_outputs = zip(*pairs, strict=True)
    found = iter(
        torch.autograd.grad(
            outputs, inputs, grad_outputs, create_graph=True, allow_unused=True
        )
    )
    return tuple(next(found) if need else None for need in needs)
