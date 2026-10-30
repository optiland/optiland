"""Ray-Aiming Policies for Finite Differences

Decision V14: how ray aiming behaves while a finite-difference evaluator
computes Jacobian columns.

* :class:`ExactAim` (default) re-aims every column. The evaluation epoch keeps
  this cheap: each column aims each pupil sample set once.
* :class:`FrozenAim` reuses the aim solutions of the baseline evaluation for
  every column. Merit and residual evaluations stay exact; only the Jacobian
  is approximated, by ignoring how a variable moves the aim. Useful with
  costly iterative aiming when the variables sit far from the stop.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from optiland.optimization.failure import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Hashable

    from optiland._epoch import EvaluationEpoch


class AimPolicy(ABC):
    """Strategy for ray aiming during Jacobian columns."""

    #: Name accepted by ``aim_policy=``.
    name: str = ""

    @abstractmethod
    def prepare_column(
        self, epoch: EvaluationEpoch, baseline: EvaluationEpoch | None
    ) -> None:
        """Prepare the epoch of one Jacobian column before it is evaluated.

        Args:
            epoch: The fresh epoch the column will be evaluated in.
            baseline: The epoch of the baseline evaluation at the same ``x``,
                or None when it is not available.
        """

    @property
    def needs_baseline(self) -> bool:
        """Whether columns need the baseline evaluation's epoch."""
        return False


class ExactAim(AimPolicy):
    """Re-aim rays for every Jacobian column (exact; the default)."""

    name = "exact"

    def prepare_column(
        self, epoch: EvaluationEpoch, baseline: EvaluationEpoch | None
    ) -> None:
        """Leave the column epoch empty, so every column aims afresh."""


class FrozenAim(AimPolicy):
    """Reuse the baseline aim solutions for every Jacobian column."""

    name = "frozen"

    def prepare_column(
        self, epoch: EvaluationEpoch, baseline: EvaluationEpoch | None
    ) -> None:
        """Seed the column epoch with the baseline's aim solutions."""
        if baseline is not None:
            epoch.adopt(baseline, _is_aim_entry)

    @property
    def needs_baseline(self) -> bool:
        """Columns copy the aim solutions of the baseline epoch."""
        return True


def _is_aim_entry(key: Hashable) -> bool:
    from optiland.rays.ray_aiming.base import AIM_MEMO_TAG

    return isinstance(key, tuple) and bool(key) and key[0] == AIM_MEMO_TAG


_POLICIES: dict[str, type[AimPolicy]] = {
    ExactAim.name: ExactAim,
    FrozenAim.name: FrozenAim,
}


def resolve_aim_policy(policy: str | AimPolicy | None) -> AimPolicy:
    """Return the policy named by ``policy`` (or ``policy`` itself).

    Args:
        policy: ``"exact"``, ``"frozen"``, an :class:`AimPolicy` instance, or
            None for the default ``"exact"``.

    Returns:
        An :class:`AimPolicy`.

    Raises:
        ConfigurationError: If the name is unknown.
    """
    if policy is None:
        return ExactAim()
    if isinstance(policy, AimPolicy):
        return policy
    cls: Any = _POLICIES.get(str(policy).lower())
    if cls is None:
        raise ConfigurationError(
            f"Unknown aim_policy {policy!r}; expected one of {sorted(_POLICIES)}."
        )
    return cls()
