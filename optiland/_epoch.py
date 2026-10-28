"""Evaluation epochs: memoize per-design work for one frozen optic state.

An evaluation epoch is a short-lived memo store opened by an optimizer
evaluator around a single design evaluation, after the variables have been
written and the optics updated, and before the residuals are read. Inside the
epoch the optic state is frozen by contract, so results that depend only on
that state (material indices, paraxial quantities, ...) can be computed once
and reused without recomputing state fingerprints.

Rules:
    - Only evaluators open epochs. Library users never need to.
    - Memo keys are ``(id(owner), qualname, key)``. Each entry holds strong
      references to the owner and to any keyed arrays, so their ids cannot be
      recycled while the epoch is alive.
    - The active epoch lives in a :class:`~contextvars.ContextVar`, so epochs
      are per-thread and safe for GUI worker threads.
    - Torch tensors are reused within one graph, so autograd stays correct.
      Array keys include each tensor's ``requires_grad`` flag.
    - Debug guard: with ``OPTILAND_EPOCH_CHECK=1`` the optics passed to
      :func:`evaluation_epoch` are fingerprinted on entry and exit, and a
      mismatch raises. Writing a variable while an epoch is open also raises.

Kramer Harrison, 2026
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator

T = TypeVar("T")

CHECK_ENV = "OPTILAND_EPOCH_CHECK"


class EpochViolationError(RuntimeError):
    """Raised by the debug guard when optic state changes inside an epoch."""


class EvaluationEpoch:
    """Memo store valid for one frozen optic state.

    Attributes:
        hits: Number of lookups answered from the memo.
        misses: Number of lookups that ran ``compute``.
    """

    def __init__(self) -> None:
        self._memo: dict[tuple, tuple[Any, tuple, Any]] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._memo)

    def get_or_compute(
        self,
        owner: object,
        key: Hashable,
        compute: Callable[[], T],
        *,
        keep: tuple = (),
    ) -> T:
        """Return the memoized value for ``(owner, key)``, computing it once.

        Args:
            owner: Object the value belongs to. Keyed by ``id`` and kept alive
                for the lifetime of the epoch.
            key: Hashable key, unique per owner. Callers namespace it, for
                example with a qualified method name.
            compute: Zero-argument callable producing the value on a miss.
            keep: Extra objects whose ``id`` appears in ``key``; they are kept
                alive with the entry so the ids cannot be recycled.

        Returns:
            The memoized or freshly computed value.
        """
        memo_key = (id(owner), key)
        entry = self._memo.get(memo_key)
        if entry is not None:
            self.hits += 1
            return entry[2]
        self.misses += 1
        value = compute()
        self._memo[memo_key] = (owner, keep, value)
        return value


_current: ContextVar[EvaluationEpoch | None] = ContextVar(
    "optiland_evaluation_epoch", default=None
)


def current() -> EvaluationEpoch | None:
    """Return the epoch open in this context, or None."""
    return _current.get()


def check_enabled() -> bool:
    """True when the ``OPTILAND_EPOCH_CHECK`` debug guard is switched on."""
    return os.environ.get(CHECK_ENV, "") not in ("", "0")


def assert_writable() -> None:
    """Raise if optic state is written while an epoch is open (debug mode only).

    Raises:
        EpochViolationError: If an epoch is open and the debug guard is on.
    """
    if _current.get() is not None and check_enabled():
        raise EpochViolationError(
            "Optic state was written while an evaluation epoch is open; "
            "write variables before opening the epoch."
        )


def _jsonable(value: Any) -> Any:
    """Convert to_dict output (which may hold arrays) into JSON-safe data."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    return repr(value)


def fingerprint(optic: Any) -> str:
    """Hash the serialized state of an optic for the debug guard.

    Args:
        optic: Any object with a ``to_dict`` method.

    Returns:
        A hex digest of the canonical JSON form of ``optic.to_dict()``.
    """
    data = json.dumps(_jsonable(optic.to_dict()), sort_keys=True, default=repr)
    return hashlib.sha256(data.encode()).hexdigest()


@contextmanager
def evaluation_epoch(optics: Iterable[Any] = ()) -> Iterator[EvaluationEpoch]:
    """Open an epoch; nested calls reuse the outer epoch.

    Args:
        optics: Optics whose state the epoch freezes. Used only by the
            ``OPTILAND_EPOCH_CHECK`` debug guard.

    Yields:
        The active epoch.

    Raises:
        EpochViolationError: In debug mode, if an optic's fingerprint differs
            between epoch entry and exit.
    """
    outer = _current.get()
    if outer is not None:
        yield outer
        return

    check = check_enabled()
    optics = tuple(optics)
    before = [fingerprint(optic) for optic in optics] if check else []
    epoch = EvaluationEpoch()
    token = _current.set(epoch)
    try:
        yield epoch
    finally:
        _current.reset(token)
    if check:
        after = [fingerprint(optic) for optic in optics]
        if after != before:
            raise EpochViolationError("Optic state changed inside an evaluation epoch.")


def array_key(value: Any) -> tuple | None:
    """Epoch-local identity key for an array argument, or None for non-arrays.

    Valid only inside an epoch, where the caller keeps ``value`` alive.
    """
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return (
        "array-id",
        id(value),
        tuple(shape),
        bool(getattr(value, "requires_grad", False)),
    )


def _default_key(args: tuple, kwargs: dict) -> tuple[Hashable, tuple]:
    """Key positional and keyword args; arrays by identity, kept alive."""
    keep = []

    def one(value: Any) -> Hashable:
        key = array_key(value)
        if key is None:
            return value
        keep.append(value)
        return key

    key = (
        tuple(one(arg) for arg in args),
        tuple((name, one(arg)) for name, arg in sorted(kwargs.items())),
    )
    return key, tuple(keep)


def epoch_memo(
    key_fn: Callable[..., Hashable] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Method decorator: memoize per epoch, passthrough outside one.

    Args:
        key_fn: Optional ``key_fn(*args, **kwargs)`` returning a hashable key
            for the call (``self`` excluded). By default arguments are used
            as-is, with arrays keyed by identity, shape and ``requires_grad``.
            A custom ``key_fn`` must not key on the ``id`` of temporaries.

    Returns:
        The decorator.
    """

    def decorator(method: Callable[..., T]) -> Callable[..., T]:
        qualname = method.__qualname__

        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            epoch = _current.get()
            if epoch is None:
                return method(self, *args, **kwargs)
            if key_fn is None:
                key, keep = _default_key(args, kwargs)
            else:
                key, keep = key_fn(*args, **kwargs), ()
            try:
                hash(key)
            except TypeError:
                return method(self, *args, **kwargs)
            return epoch.get_or_compute(
                self,
                (qualname, key),
                lambda: method(self, *args, **kwargs),
                keep=keep,
            )

        return wrapper

    return decorator
