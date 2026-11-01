"""Residual Blocks

A :class:`ResidualBlock` returns many residual rows from one vectorized call,
for per-ray merit terms where one ``Operand`` per row would be too slow
(decision V16). Rows are already weighted, so they enter the merit as they
are: ``sum_squared = sum(operand rows ** 2) + sum(block rows ** 2)``.

A block declares its rays as :class:`BundleRequest` objects. The batched
evaluator merges them into the per-wavelength super-bundles, so block rays share
one trace with the ray operands; without batching each request is traced on its
own. Either way the block reads its rays from a :class:`BlockContext`.

Ray-set re-validation (decision V12): native drivers call :meth:`~ResidualBlock.
drop_rows` on rows whose rays failed. A dropped row is exactly 0 until
:meth:`~ResidualBlock.restore_rows`, at the end of the run, so the row count
never changes.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland import _epoch

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from optiland._types import BEArray


@dataclass(frozen=True, eq=False)
class BundleRequest:
    """Rays one block needs, at one field point and wavelength. *Experimental.*

    Attributes:
        optic: The optic to trace.
        wavelength: Wavelength in micrometers.
        Hx: Normalized x field coordinate.
        Hy: Normalized y field coordinate.
        Px: Normalized x pupil coordinates, 1-D.
        Py: Normalized y pupil coordinates, 1-D.
        attrs: Per-surface ray attributes the block reads, from ``x``, ``y``,
            ``z``, ``L``, ``M``, ``N``, ``opd`` and ``intensity``.
    """

    optic: Any
    wavelength: float
    Hx: float
    Hy: float
    Px: BEArray
    Py: BEArray
    attrs: tuple[str, ...] = ("x", "y")

    @property
    def size(self) -> int:
        """Number of rays."""
        return int(be.size(self.Px))

    @cached_property
    def key(self) -> Hashable:
        """Identity of the rays: requests with equal keys share a bundle segment."""
        return (
            id(self.optic),
            float(self.wavelength),
            float(self.Hx),
            float(self.Hy),
            np.asarray(be.to_numpy(self.Px), dtype=float).tobytes(),
            np.asarray(be.to_numpy(self.Py), dtype=float).tobytes(),
        )


class BlockContext:
    """Traced rays of the current evaluation, by request. *Experimental.*

    Args:
        reads: ``{request key: (view, start, stop)}``. A view exposes the
            ``(surfaces, rays)`` arrays as attributes. A request whose trace
            failed maps to None.
    """

    def __init__(self, reads: dict[Hashable, tuple[Any, int, int] | None]) -> None:
        self._reads = reads

    def read(self, request: BundleRequest, attr: str, surface: int = -1) -> BEArray:
        """Return one ray attribute on one surface.

        Args:
            request: The request whose rays to read.
            attr: Ray attribute, one of ``request.attrs``.
            surface: Surface index; -1 is the image surface.

        Returns:
            A 1-D array of ``request.size`` values, all ``NaN`` if the trace
            failed.
        """
        entry = self._reads.get(request.key)
        if entry is None:
            return be.full((request.size,), be.nan)
        view, start, stop = entry
        return getattr(view, attr)[surface, start:stop]


class ResidualBlock(ABC):
    """A vectorized group of residual rows of the merit. *Experimental.*

    Subclasses implement :meth:`size`, :meth:`residuals`,
    :meth:`trace_requests` and :meth:`describe`. Add one with
    ``OptimizationProblem.add_residual_block``.
    """

    _dropped: frozenset[int] = frozenset()

    @abstractmethod
    def size(self) -> int:
        """Return the number of residual rows."""

    @abstractmethod
    def residuals(self, context: BlockContext) -> BEArray:
        """Return the rows at the current design, already weighted.

        Args:
            context: The traced rays of this evaluation.

        Returns:
            A 1-D array of :meth:`size` rows. Rows of failed rays are ``NaN``.
        """

    @abstractmethod
    def trace_requests(self) -> list[BundleRequest]:
        """Return the rays this block reads from the context."""

    @abstractmethod
    def describe(self) -> dict:
        """Return one summary row for the info printers."""

    @property
    def dropped_rows(self) -> np.ndarray:
        """Sorted indices of the rows dropped by ray-set re-validation."""
        return np.array(sorted(self._dropped), dtype=int)

    def drop_rows(self, rows: Sequence[int] | np.ndarray) -> None:
        """Hold rows at exactly 0 until :meth:`restore_rows`.

        Args:
            rows: Row indices local to this block.
        """
        self._dropped = self._dropped | {int(row) for row in rows}

    def restore_rows(self) -> None:
        """Bring back every dropped row."""
        self._dropped = frozenset()

    def evaluate(self, context: BlockContext) -> BEArray:
        """Return :meth:`residuals` with dropped rows at 0, never raising.

        An exception inside the block yields ``NaN`` rows (opt v2 Section 8.4).

        Args:
            context: The traced rays of this evaluation.

        Returns:
            A 1-D array of :meth:`size` rows.
        """
        try:
            rows = be.ravel(self.residuals(context))
        except _epoch.EpochViolationError:
            raise  # debug guard: a contract breach is a bug, not a bad design
        except Exception:
            return be.full((self.size(),), be.nan)
        if rows.shape[0] != self.size():
            raise ValueError(
                f"{type(self).__name__}.residuals returned {rows.shape[0]} rows; "
                f"size() is {self.size()}."
            )
        if not self._dropped:
            return rows
        keep = np.ones(self.size())
        keep[self.dropped_rows] = 0.0
        return be.where(be.array(keep) > 0.5, rows, be.zeros_like(rows))


def evaluate_blocks(blocks: list[ResidualBlock], context: BlockContext) -> BEArray:
    """Return the rows of all blocks, in order."""
    rows = [block.evaluate(context) for block in blocks]
    return rows[0] if len(rows) == 1 else be.concatenate(rows)


def _trace_request(request: BundleRequest) -> tuple[Any, int, int] | None:
    """Trace one request on its own; None when the trace fails."""
    count = request.size
    try:
        request.optic.trace_generic(
            be.full((count,), float(request.Hx)),
            be.full((count,), float(request.Hy)),
            request.Px,
            request.Py,
            request.wavelength,
        )
        surfaces = request.optic.surfaces
        data = {name: getattr(surfaces, name) for name in request.attrs}
    except Exception:
        return None
    return SimpleNamespace(**data), 0, count


def direct_context(blocks: list[ResidualBlock]) -> BlockContext:
    """Trace the requests of ``blocks`` one by one, for unbatched evaluation."""
    reads: dict[Hashable, tuple[Any, int, int] | None] = {}
    for block in blocks:
        for request in block.trace_requests():
            if request.key not in reads:
                reads[request.key] = _trace_request(request)
    return BlockContext(reads)
