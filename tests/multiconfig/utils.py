"""Shared helpers for the multi-configuration tests."""

from __future__ import annotations

from contextlib import contextmanager

import optiland.backend as be


@contextmanager
def deepcopy_safe():
    """Build without autograd history on torch.

    Deep-copying an optic whose tensors carry autograd history (e.g. a
    thickness set through a pickup) fails in torch with grad mode enabled,
    independently of multi-configuration links. Chained configurations are
    therefore built with grad mode off; it is restored afterwards.
    """
    if be.get_backend() != "torch" or not be.grad_mode.requires_grad:
        yield
        return
    be.grad_mode.disable()
    try:
        yield
    finally:
        be.grad_mode.enable()
