"""Chunked vector-Jacobian product accumulation.

Differentiating a ray trace requires memory proportional to the number of
rays, because reverse-mode autograd retains the whole graph until the
backward pass runs. Many optical quantities are *reductions* over rays. Two
examples are a rendered image and an irradiance map. Each accumulates one
contribution per ray into a single total. Differentiating such a reduction
over enough rays produces an autograd graph that exceeds device memory.

The loss is computed from ``params`` in two steps::

    params --ray trace and reduce--> total --merit function--> loss

The gradient of the loss is computed with respect to ``params``. ``total`` is
the result of the reduction. The merit function is any scalar-valued function
of the total: for example a least-squares comparison against a target, or a
neural network computing a perceptual loss.

A *batch* is a subset of the rays. Summing the batch contributions in an
ordinary loop would not lower peak graph memory. Each batch's graph stays
reachable from the total, so autograd keeps them all until the backward pass
runs. ``chunked_vjp`` frees each batch's graph before building the next, so
only one exists at a time. Peak graph memory is then set by the largest graph
a single batch produces, and does not grow with the number of batches.

The total is the plain sum of the batch contributions, so every contribution
has the shape of the whole total. For example, when the total is a rendered
image, a batch's contribution is that whole image, dim and sparsely sampled.
A tile of the image, fully illuminated, would be the wrong shape. This holds
even when a batch's rays reach only part of the total: the contribution is
still the whole image, zero outside the region those rays cover.

``chunked_vjp`` passes each batch to ``batch_fn`` unchanged and never inspects
it. It places no restriction on what ``batch_fn`` computes. Any reduction over
rays can therefore be chunked, provided it is additive.

Every ray is traced twice: once in the forward pass to accumulate the total,
and once in the backward pass to differentiate it. Batch size does not change
how many times each ray is traced. Per-batch work runs once for each batch in
each pass, so ``setup_fn`` is called twice for every batch. Smaller batches
lower peak graph memory and raise the number of those calls.

How the computation is split
----------------------------
The graph is cut at the total. Writing ``L`` for the loss, the chain rule gives::

    dL/d(params) = dL/d(total) * d(total)/d(params)

The two factors can be evaluated in separate passes. dO calls this the
**separability property**, and uses it to split the work into three stages
(section II-D):

1. **Forward, no autograd.** :func:`chunked_vjp` accumulates the total over
   the batches with gradients disabled, so no graph is built.
2. **Forward and backward on the merit function.** The caller applies the
   merit function to the total and calls ``.backward()`` on the resulting
   scalar loss. Autograd propagates back through the merit function,
   including any network inside it, and arrives at the total carrying
   ``dL/d(total)``, a tensor of the same shape.
3. **Forward and backward on the reduction.** One batch at a time,
   :func:`chunked_vjp` re-evaluates ``batch_fn`` with gradients enabled,
   rebuilding the graph from ``params`` to that batch's contribution. That
   graph encodes ``d(contribution)/d(params)``, and its vector-Jacobian
   product with ``dL/d(total)`` is the batch's share of ``dL/d(params)``,
   which is accumulated into the parameter gradients. The graph is freed
   before the next batch is built.

Stage 1 runs no backward pass, and the total it returns is an ordinary
autograd tensor. Stage 2 is the caller's own code: they apply the merit
function to that total and call ``.backward()`` on the loss. Autograd reaches
:func:`chunked_vjp` during that call and runs stage 3. Stage 3 needs only
``dL/d(total)`` from stage 2, and never evaluates the merit function again.

Preconditions
-------------
Two conditions must hold for the gradients to be correct: the reduction must
be additive over rays, and ``batch_fn`` must be reproducible. ``chunked_vjp``
cannot check additivity, so a non-additive reduction fails silently.
Reproducibility could be checked by comparing what stage 1 and stage 3
compute, but this version does not compare them. It raises when a
contribution changes shape between the passes, and does not detect a
``batch_fn`` that returns different values.

.. warning::
   The reduction must be **additive** over rays. The total must be the
   plain sum of the batch contributions. This ensures stage 3 applies the same
   ``dL/d(total)`` to every batch.

   This precondition fails for ``max``. It also fails for any quantity
   normalized by something derived from all rays, such as a centroid reference
   or a count of surviving rays.

   :func:`chunked_vjp` cannot detect non-additive reductions. The module
   never inspects the batches, so it cannot regroup the rays to check if a
   different batching produces a different total. The contributions are just
   tensors; they do not record which operation produced them. Verifying
   additivity is the caller's responsibility.

.. warning::
   ``batch_fn`` must be **reproducible** even if it includes stochastic
   operations. Stage 3 re-evaluates ``batch_fn``. The resulting contribution
   must exactly match the contribution computed in stage 1. Otherwise, the
   gradient is computed for a different output than the one used in the merit
   function.

   Stochastic operations like ray generation or scattering require explicit
   state management. For example, explicitly setting the random seed before
   processing each batch guarantees that these operations yield the exact same
   values in both passes.

   The batch object must persist across passes. Do not pass a one-shot
   iterator. Instead, generate all the rays once, then pass slices into that
   array.

Relation to implicit differentiation
------------------------------------
Chunked accumulation composes with the implicit differentiation used by the
Newton-Raphson solvers (see :ref:`implicit_differentiation`). It does not
replace it. Implicit differentiation bounds how deep the graph is per ray.
Chunked accumulation bounds how wide the graph is across rays.

Reference
---------
Wang, Chen and Heidrich, "dO: A Differentiable Engine for Deep Lens Design of
Computational Imaging Systems", IEEE Transactions on Computational Imaging,
2022. Section II-D introduces the separability property and calls the
three-stage procedure adjoint back-propagation.

Ashish Verma, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import torch
except ImportError:  # pragma: no cover - only runs when torch is not installed
    torch = None

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# Pick the base class for _ChunkedVJP: torch.autograd.Function when torch is
# installed, and plain ``object`` when it is not. wrappers.py guards
# OpticalSystemModule with the same fallback.
#
# ml/__init__.py imports chunked_vjp, so importing optiland.ml runs this file,
# including on a numpy-only install. A class statement evaluates the base class
# named in its parentheses as soon as it runs. The class statement for
# _ChunkedVJP runs during that import, so the base class name has to resolve
# whether or not torch is installed. Naming torch.autograd.Function directly
# would raise AttributeError on a numpy-only install.
#
# With ``object`` as the base class, _ChunkedVJP has no apply(), but the
# import still succeeds because ``object`` is a Python builtin and does not
# depend on torch. chunked_vjp checks for torch before calling
# _ChunkedVJP.apply(), and raises a RuntimeError if torch is not installed.
# The missing apply() is therefore never reached.
_AutogradFunction = torch.autograd.Function if torch is not None else object

# Pick the decorator applied to backward(). This module computes first-order
# gradients only. Differentiating the gradient that backward() returns is
# unsupported, and fails with or without the decorator. once_differentiable
# only makes the error name the cause.
#
# A decorator is applied when the class body runs, which happens as the file
# is imported, so the name has to resolve whether or not torch is installed.
if torch is not None:
    _once_differentiable = torch.autograd.function.once_differentiable
else:  # pragma: no cover - only runs when torch is not installed

    def _once_differentiable(fn):
        return fn


def _check_contribution_shape(contribution, expected, expected_desc: str) -> None:
    """Raise if a batch's contribution shape differs from the expected shape.

    Both passes call this function to share the error message formatting. The
    forward pass compares the contribution against the total established by the
    first batch. The backward pass compares it against the incoming gradient.

    Both passes compare the contribution against a value produced by the same
    ``batch_fn``. Therefore, this check establishes consistency rather than
    correctness. If ``batch_fn`` consistently returns the wrong shape on every
    call, it will pass this check. For example, if the true total is shape
    (100, 100), but ``batch_fn`` consistently returns shape (50, 50) on every
    call, it will not raise an error.

    Args:
        contribution (torch.Tensor): What ``batch_fn`` returned for one batch.
        expected (torch.Tensor): The tensor whose shape it has to match.
        expected_desc (str): Names ``expected`` in the error message.

    Raises:
        ValueError: If the two shapes differ.
    """
    if contribution.shape != expected.shape:
        raise ValueError(
            f"batch_fn must return the same shape on every call. It returned "
            f"{tuple(contribution.shape)}, but {expected_desc} has shape "
            f"{tuple(expected.shape)}. The total is the sum of the batch "
            f"contributions, so every contribution must match the shape of the "
            f"total."
        )


class _ChunkedVJP(_AutogradFunction):
    """Autograd bridge for :func:`chunked_vjp`.

    This class is private. The public entry point is :func:`chunked_vjp`.
    That function validates the arguments and then calls
    ``_ChunkedVJP.apply()``.

    This class implements stage 1 and stage 3 from the module docstring.
    :meth:`forward` accumulates the total without building a graph.
    :meth:`backward` re-evaluates each batch to accumulate its vector-Jacobian
    product.

    This class does not implement stage 2. The caller evaluates the merit
    function and calls ``loss.backward()``. Autograd propagates the gradient
    through the merit function to compute the derivative of the loss with
    respect to the total. Autograd then passes this incoming gradient
    into our :meth:`backward` method as ``grad_output`` so that stage 3 can
    apply it to every batch.
    """

    @staticmethod
    def forward(
        ctx,
        batch_fn: Callable[[Any], torch.Tensor],
        setup_fn: Callable[[], None] | None,
        batches: Sequence[Any],
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate the total over the batches without building a graph.

        Args:
            ctx: Autograd context. Anything :meth:`backward` needs is stored
                on it here.
            batch_fn: Maps one batch to its contribution to the total.
            setup_fn: A callable taking no arguments, run before every batch.
                None when the caller supplied none.
            batches: The subsets of rays to reduce over. Each is passed to
                ``batch_fn`` unchanged.
            *params: The tensors to differentiate with respect to. Each is
                passed as a separate argument so that autograd records it.
                Gradients flow back only to what autograd records.

        Returns:
            torch.Tensor: The total, the sum of every batch's contribution.
        """
        # None until the first batch arrives. The total takes its shape and
        # dtype from whatever batch_fn returns, neither of which is known
        # before then, so there is nothing to preallocate.
        total = None

        # Accumulate the total without building a computational graph. We do
        # this by wrapping the loop in torch.no_grad(). If forward() is called
        # directly, gradients are enabled by default; explicitly disabling them
        # prevents the graph from being built. When Autograd calls forward()
        # via apply(), gradients are already disabled, making this a safe no-op.
        with torch.no_grad():
            for batch in batches:
                if setup_fn is not None:
                    setup_fn()
                contribution = batch_fn(batch)
                if total is None:
                    # Cloned because the following batches are added into it.
                    # batch_fn may return a tensor the caller still references,
                    # and accumulating into that would modify the caller's own
                    # tensor in place.
                    total = contribution.clone()
                else:
                    # Checked explicitly because the addition below would
                    # accept a contribution that broadcasts into the total's
                    # shape, such as (1, 4) into (3, 4).
                    _check_contribution_shape(
                        contribution, total, "the first batch's contribution"
                    )
                    # Added in place, so the whole reduction writes into the
                    # tensor the clone allocated instead of allocating one
                    # per batch.
                    total += contribution

        # backward re-evaluates batch_fn for every batch, so it needs batch_fn,
        # setup_fn and batches again. None of them is a tensor, so they go on
        # ctx as plain attributes.
        ctx.batch_fn = batch_fn
        ctx.setup_fn = setup_fn
        ctx.batches = batches
        # Save params for the backward pass while enforcing version checking.
        # We do this by passing them to ctx.save_for_backward() instead of
        # storing them as plain attributes on the ctx object.
        # ctx.save_for_backward() records each tensor's version counter; reading
        # the tensors back via ctx.saved_tensors compares these counters and
        # raises a RuntimeError if any tensor was modified in place. If we
        # stored them as plain attributes on the ctx object, we would bypass
        # this check, and the backward pass would silently compute incorrect
        # gradients using parameter values the forward pass never used.
        ctx.save_for_backward(*params)

        return total

    @staticmethod
    @_once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Re-evaluate each batch and accumulate its vector-Jacobian product.

        The total is the plain sum of the batch contributions. Consequently, the
        derivative of the total with respect to any single contribution is
        exactly 1. This means the gradient of the loss with respect to the
        total (``grad_output``) can be passed directly to every batch without
        modification. The chunking approach is valid due to this property, and
        this property holds only when the reduction is additive.

        Args:
            ctx: The autograd context containing the objects saved by
                :meth:`forward`.
            grad_output: The gradient of the loss with respect to the total.

        Returns:
            tuple: A tuple containing one gradient for each argument passed to
                :meth:`forward`, in the exact same order. The first three
                gradients are ``None`` because ``batch_fn``, ``setup_fn``, and
                ``batches`` are not tensors and do not require gradients.

        Raises:
            ValueError: If ``batch_fn`` returns a shape that differs from the
                shape of ``grad_output`` (the gradient of the loss with respect
                to the total).
            RuntimeError: If any tensor in ``params`` is disconnected from the
                computational graph across all batches.
        """
        params = ctx.saved_tensors
        param_grads = [torch.zeros_like(p) for p in params]
        # Track which parameters the computational graph reaches to prevent
        # returning false zeros. We do this by recording a boolean flag per
        # parameter. If a parameter is never reached by any batch, it would
        # otherwise finish with a zero gradient, which is indistinguishable
        # from a genuine mathematical zero. Tracking connectivity allows us to
        # raise a RuntimeError for unreachable parameters after the loop ends.
        connected = [False] * len(params)

        # Re-evaluate the batches with gradient recording explicitly enabled.
        # We do this by wrapping the loop in torch.enable_grad(). PyTorch's
        # autograd engine runs all custom backward() methods with gradients
        # disabled by default. If we do not explicitly re-enable them,
        # batch_fn builds no graph, and autograd.grad fails with a
        # "element 0 of tensors does not require grad" RuntimeError.
        with torch.enable_grad():
            for batch in ctx.batches:
                if ctx.setup_fn is not None:
                    # setup_fn must run before every batch, inside
                    # enable_grad(). It re-establishes the computational graph
                    # edges between params and the system's internal state.
                    #
                    # The full forward-pass data flow is:
                    #   nn.Parameter --(setup_fn)--> optical system's Variable
                    #       --(ray trace & reduce)--> total --(merit fn)--> loss
                    #
                    # For example, OpticalSystemModule._sync_params_to_problem()
                    # in wrappers.py copies the current values from the
                    # optimizer's nn.Parameter tensors (e.g., a lens radius)
                    # into the optical system's Variable objects. This copy may
                    # involve intermediate operations (such as scaling),
                    # creating graph nodes between params and the Variable
                    # objects. After each batch, retain_graph=False frees those
                    # nodes. Without a fresh setup_fn call, the next batch
                    # traces through the freed nodes and autograd.grad raises
                    # "Trying to backward through the graph a second time".
                    # The enable_grad() context is required so that these
                    # re-established edges are tracked by autograd; outside it,
                    # they would be detached.
                    ctx.setup_fn()

                contribution = ctx.batch_fn(batch)
                _check_contribution_shape(
                    contribution,
                    grad_output,
                    "grad_output (the gradient of the loss with respect to the total)",
                )

                # PyTorch's autograd engine applies the gradients that this
                # method returns. Calling contribution.backward() here would
                # apply them a second time, and every gradient would silently
                # come out doubled. autograd.grad returns the gradients
                # instead of applying them, and requires its inputs to be
                # named. That is why chunked_vjp takes params as an explicit
                # argument.
                vjps = torch.autograd.grad(
                    contribution,
                    params,
                    grad_outputs=grad_output,
                    # Frees this batch's graph as the gradient is computed, so
                    # only one graph exists at a time. With retain_graph=True,
                    # every batch's graph would be retained and peak graph
                    # memory would grow with the number of batches.
                    retain_graph=False,
                    # Returns None for a param that this batch's graph does
                    # not reach. Without this flag set to True, autograd.grad
                    # raises instead. The graph for one batch may legitimately
                    # reach only some params. A param that no batch's graph
                    # reaches is an error.
                    allow_unused=True,
                )

                # Accumulate this batch's share into the parameter gradients,
                # and record which params were reached.
                for i, vjp in enumerate(vjps):
                    if vjp is not None:
                        param_grads[i] += vjp
                        connected[i] = True

        # Checked after the loop, since a param missed by one batch may still
        # be reached by another.
        unused = [i for i, ok in enumerate(connected) if not ok]
        if unused:
            raise RuntimeError(
                f"No gradient reached params at position(s) {unused}. They are "
                "not connected to the graph batch_fn builds, so their gradients "
                "would silently be zero. Check that batch_fn uses these tensors. "
                "If you pass a setup_fn, check that it writes them into the "
                "system without detaching them."
            )

        # Drops the references so the memory held by batch_fn, setup_fn and
        # batches can be freed when this method returns. Otherwise it is
        # freed only when the caller drops its last reference to the total,
        # because ctx is reachable from the total's grad_fn. batch_fn usually
        # refers to the full array of rays, so holding batch_fn holds that
        # array too.
        ctx.batch_fn = None
        ctx.setup_fn = None
        ctx.batches = None

        return (None, None, None, *param_grads)


def chunked_vjp(
    batch_fn: Callable[[Any], torch.Tensor],
    batches: Sequence[Any],
    params: Sequence[torch.Tensor],
    *,
    setup_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Differentiable sum of ``batch_fn`` over ``batches``, with bounded graph size.

    ``chunked_vjp`` returns the total as an ordinary autograd tensor. Calling
    ``.backward()`` on a loss computed from the total produces the same
    gradients for ``params`` as plain non-chunked autograd, up to
    floating-point rounding. The module docstring describes the separability
    property and the three stages that keep peak graph memory bounded.

    Args:
        batch_fn: Maps one element of ``batches`` to that batch's contribution
            to the total. Must return the same shape on every call, since the
            total is the plain sum of the contributions.
        batches: The batches of rays to reduce over. A typical choice is to
            generate all the rays once, then pass slices into that array.
            Slices keep the ray set fixed, so the result does not change with
            batch size. Each batch is used once in the forward pass and once
            in the backward pass. If the forward pass consumes a batch, the
            backward pass sees it empty and the gradient comes back zero,
            with no error raised.
        params: The tensors to differentiate with respect to. Each must
            require grad and be reachable from the graph ``batch_fn`` builds.
            ``params`` itself must be a sequence, even when it holds only one
            tensor. Passing a single tensor instead raises ``ValueError``.
        setup_fn: Called before every batch, in both passes. Use it when the
            system being traced keeps its own copy of ``params`` and needs
            the current values before each trace. Must be idempotent.

    Returns:
        torch.Tensor: The total, the sum of every batch's contribution.

    Raises:
        RuntimeError: Raised by this call if torch is not installed or the
            active backend is not torch. Raised during the later
            ``.backward()`` call if a tensor in ``params`` is unreachable from
            every graph ``batch_fn`` builds.
        ValueError: Raised by this call if ``batches`` is empty, or if
            ``params`` is empty, is a single tensor, or holds a tensor that
            does not require grad. Raised by this call or during the later
            ``.backward()`` call if the contributions ``batch_fn`` returns
            differ in shape.

    Warning:
        Valid only for a reduction that is additive over rays, and only for a
        reproducible ``batch_fn``. Otherwise the gradients are incorrect,
        usually with no error raised. The module docstring explains what is
        detected and why additivity cannot be checked.

    Example:
        >>> total = chunked_vjp(
        ...     render_batch,
        ...     [slice(i, i + 10_000) for i in range(0, 1_000_000, 10_000)],
        ...     params=[radius],
        ... )
        >>> loss = criterion(total, target)
        >>> loss.backward()
    """
    if torch is None:
        raise RuntimeError(
            "chunked_vjp requires the 'torch' package. Install PyTorch to use "
            "this function."
        )
    if be.get_backend() != "torch":
        raise RuntimeError(
            "chunked_vjp requires the 'torch' backend, but the active backend "
            f"is '{be.get_backend()}'. This feature accumulates gradients "
            "through PyTorch autograd and has no numpy equivalent; call "
            "optiland.backend.set_backend('torch') first."
        )

    # Copied into a list so that a generator can be iterated by both passes.
    # Without the copy, the forward pass would exhaust the generator and the
    # backward pass would run no batches. backward's connectivity check would
    # then raise.
    batches = list(batches)
    if not batches:
        raise ValueError("batches is empty; there is nothing to reduce over.")

    # Rejects a single tensor passed as params. Checked before params is
    # converted to a tuple, because after that conversion a tuple built from
    # a bare tensor is indistinguishable from a correct sequence.
    #
    # A tensor is itself a sequence, so passing one directly as params
    # iterates into its row views instead of failing. Those views require
    # grad and appear in the graph batch_fn builds, so every later check
    # passes, including backward's connectivity check. The gradients then
    # accumulate onto the row views and are discarded with them. The tensor
    # the caller passed keeps a .grad of None, so every optimizer.step()
    # leaves it unchanged and that parameter is never optimized.
    if isinstance(params, torch.Tensor):
        raise ValueError(
            "params must be a sequence of tensors, but a single tensor was "
            "given. Iterating it differentiates with respect to its rows "
            "instead of the tensor itself, and no gradient reaches the tensor "
            "you passed. Write params=[tensor] instead of params=tensor."
        )

    params = tuple(params)
    if not params:
        raise ValueError("params is empty; there is nothing to accumulate.")

    no_grad = [i for i, p in enumerate(params) if not p.requires_grad]
    if no_grad:
        raise ValueError(
            f"params at position(s) {no_grad} do not require grad, so no "
            "gradient can be accumulated for them. Call requires_grad_(True) "
            "on them, or leave them out of params."
        )

    # The tensors in params are passed to _ChunkedVJP.apply() as separate
    # arguments. Autograd records each tensor argument of apply(), but not
    # the elements of a list argument. Gradients flow back only to what
    # autograd records. If params were passed as one list, none of its
    # tensors would be recorded, and the returned total would not require
    # grad. The caller's .backward() would then fail with "element 0 of
    # tensors does not require grad".
    return _ChunkedVJP.apply(batch_fn, setup_fn, batches, *params)
