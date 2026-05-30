"""Observer Mixin

Provides a reusable subscribe/notify observer pattern. Surfaces use this
to propagate material-change events to downstream surfaces.

deepcopy contract: _subscribers is cleared on copy. The new object starts
with no subscribers. SurfaceGroup._rewire_observers() re-establishes
subscriptions after any deepcopy of a SurfaceGroup.

Kramer Harrison, 2026
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class ObserverMixin:
    """Mixin providing subscribe/notify observer pattern.

    Classes inheriting from this mixin gain:
      - ``subscribe(callback)``   — register a callback
      - ``unsubscribe(callback)`` — deregister a callback
      - ``_notify()``             — call all live subscribers

    deepcopy contract: ``_subscribers`` is cleared on copy. The owning
    container (e.g. SurfaceGroup) is responsible for re-wiring subscriptions
    after copy via ``_rewire_observers()``.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable] = []

    def subscribe(self, callback: Callable) -> None:
        """Register a callback to be called when this object changes.

        Args:
            callback: Zero-argument callable to invoke on change.
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Remove a previously registered callback.

        Args:
            callback: The callable to remove. No-op if not registered.
        """
        self._subscribers = [cb for cb in self._subscribers if cb != callback]

    def _notify(self) -> None:
        """Call all subscribers. Dead callables are silently dropped."""
        live = []
        for cb in self._subscribers:
            try:
                cb()
                live.append(cb)
            except ReferenceError:
                pass
        self._subscribers = live

    def __deepcopy__(self, memo: dict) -> ObserverMixin:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_subscribers":
                object.__setattr__(result, k, [])
            else:
                object.__setattr__(result, k, copy.deepcopy(v, memo))
        return result
