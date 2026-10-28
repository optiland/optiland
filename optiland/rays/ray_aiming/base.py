"""
Base Ray Aimer Module

This module defines the abstract base class for ray aiming algorithms.

Inside an evaluation epoch (:mod:`optiland._epoch`) the optic state is frozen,
so aim solutions are memoized there, for every aimer, by this base class.

Kramer Harrison, 2025
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland import _epoch

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from optiland._types import ScalarOrArrayT
    from optiland.optic import Optic

# True while an aimer's ``aim_rays`` runs. Aimers nested inside another aimer
# (paraxial seeds, wrapped aimers) pass through, so only entry calls memoize.
_aiming: ContextVar[bool] = ContextVar("optiland_ray_aiming_active", default=False)


class _UnkeyableError(Exception):
    """Raised when an ``aim_rays`` input has no epoch memo key."""


def _input_key(value: Any, keep: list) -> Hashable:
    """Epoch memo key for one ``aim_rays`` input.

    Pupil and field arrays are rebuilt for every trace, so arrays are keyed by
    content (dtype, shape and bytes), which identifies the sample set exactly.
    Tensors that require gradients are keyed by identity instead, so a result
    is never attached to another leaf's graph.

    Args:
        value: A field, wavelength or pupil input.
        keep: Objects keyed by ``id`` are appended here to keep them alive.

    Returns:
        A hashable key.

    Raises:
        _UnkeyableError: If the value cannot be keyed.
    """
    if isinstance(value, tuple | list):
        return tuple(_input_key(item, keep) for item in value)
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if hasattr(value, "detach"):  # torch tensor
        if value.requires_grad:
            keep.append(value)
            return _epoch.array_key(value)
        data = value.detach().cpu().numpy()
        return (
            "tensor",
            str(value.dtype),
            str(value.device),
            data.shape,
            data.tobytes(),
        )
    try:
        hash(value)
    except TypeError as exc:
        raise _UnkeyableError from exc
    # The type is part of the key, so 0 and 0.0 do not share an entry.
    return (type(value).__name__, value)


def _grad_state() -> tuple[bool, bool]:
    """Torch gradient modes, which decide whether results carry a graph."""
    if be.get_backend() != "torch":
        return (False, False)
    import torch

    return (be.grad_mode.requires_grad, torch.is_grad_enabled())


def _copy_result(result: Any) -> Any:
    """Copy each array of an aim result so callers never share memo storage."""
    if not isinstance(result, tuple):
        return result
    return tuple(be.copy(item) if hasattr(item, "shape") else item for item in result)


def _epoch_memoized(aim_rays: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an ``aim_rays`` implementation with the per-epoch memo.

    Outside an epoch, for aimer calls nested in another aimer, and for calls
    with extra arguments (such as an explicit ``initial_guess``) the
    implementation runs unchanged. The memo key is ``(aimer, field coords,
    wavelength, pupil sample identity)``; the aimer is the epoch owner.

    Args:
        aim_rays: The subclass's ``aim_rays`` implementation.

    Returns:
        The wrapped method.
    """

    @functools.wraps(aim_rays)
    def wrapper(
        self: BaseRayAimer,
        fields: Any,
        wavelengths: Any,
        pupil_coords: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        epoch = _epoch.current()
        if epoch is None or args or kwargs or _aiming.get():
            token = _aiming.set(True)
            try:
                return aim_rays(
                    self, fields, wavelengths, pupil_coords, *args, **kwargs
                )
            finally:
                _aiming.reset(token)

        keep: list = []
        try:
            key = (
                "BaseRayAimer.aim_rays",
                _input_key(fields, keep),
                _input_key(wavelengths, keep),
                _input_key(pupil_coords, keep),
                _grad_state(),
            )
        except _UnkeyableError:
            key = None

        fresh: list = []

        def compute() -> Any:
            token = _aiming.set(True)
            try:
                result = aim_rays(self, fields, wavelengths, pupil_coords)
            finally:
                _aiming.reset(token)
            fresh.append(result)
            return _copy_result(result)

        if key is None:
            compute()
            return fresh[0]
        stored = epoch.get_or_compute(self, key, compute, keep=tuple(keep))
        # A miss hands the caller the computed arrays; a hit hands out copies.
        return fresh[0] if fresh else _copy_result(stored)

    wrapper._epoch_memoized = True
    return wrapper


class BaseRayAimer(ABC):
    """
    Abstract base class for ray aiming algorithms.

    This class defines the interface for calculating the initial ray coordinates
    and direction cosines required to aim rays at a specific pupil coordinate
    on the stop surface.

    Every subclass's ``aim_rays`` is memoized per evaluation epoch, keyed by
    the aimer, field coordinates, wavelength and pupil sample identity. Outside
    an epoch it runs unchanged.

    Attributes:
        optic (Optic): The optical system to trace.
        kwargs (dict): Additional parameters for the aiming algorithm.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap the subclass's ``aim_rays`` with the per-epoch memo."""
        super().__init_subclass__(**kwargs)
        aim_rays = cls.__dict__.get("aim_rays")
        if aim_rays is not None and not getattr(aim_rays, "_epoch_memoized", False):
            cls.aim_rays = _epoch_memoized(aim_rays)

    def __init__(self, optic: Optic, **kwargs: Any) -> None:
        """
        Initialize the BaseRayAimer.

        Args:
            optic (Optic): The optical system instance.
            **kwargs: Specific parameters for the aiming algorithm.
        """
        self.optic = optic
        self.kwargs = kwargs

    @abstractmethod
    def aim_rays(
        self,
        fields: tuple[ScalarOrArrayT, ScalarOrArrayT],
        wavelengths: ScalarOrArrayT,
        pupil_coords: tuple[ScalarOrArrayT, ScalarOrArrayT],
    ) -> tuple[
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
        ScalarOrArrayT,
    ]:
        """
        Calculate ray starting coordinates and direction cosines.

        Args:
            fields: Normalized field coordinates (Hx, Hy).
            wavelengths: Wavelengths for the rays.
            pupil_coords: Normalized pupil coordinates (Px, Py).

        Returns:
            Tuple containing:
                - x: Starting x-coordinate on the object/entrance surface.
                - y: Starting y-coordinate on the object/entrance surface.
                - z: Starting z-coordinate on the object/entrance surface.
                - L: Direction cosine L.
                - M: Direction cosine M.
                - N: Direction cosine N.
        """
