"""Per-surface booking of the flux a surface removes from the trace.

A mirror below unit reflectance multiplies a ray's flux by ``R``. A coating
with ``R + T < 1`` multiplies it by a weight whose expectation is ``R + T``.
A BSDF lobe returns a weight that is a fraction of the incident flux by
:class:`~optiland.nonsequential.bsdf.base.BaseBSDF`'s own contract. All three
remove flux from the trace, and none of them had a destination in the
conservation identity, so ``flux_conservation_error`` reported a defect equal
to the loss on a scene the engine traced correctly.

:class:`LedgerBooking` gives that flux a bin. It follows the counter pattern
:class:`~optiland.nonsequential.components.absorbing.AbsorbingComponent`
already uses -- a plain float on the component, reset at the start of a trace
and read once at the end -- so the loss is available per surface and not only
as a total.
"""

from __future__ import annotations

import optiland.backend as be
from optiland.backend.utils import to_numpy


class LedgerBooking:
    """Mixin: a surface books the flux it removes from the trace.

    Attributes:
        coating_loss: Flux this surface removed as mirror, coating or BSDF
            lobe loss over the current trace [W].
    """

    def reset_ledger(self) -> None:
        """Start a fresh trace's books."""
        self._coating_loss: float = 0.0

    def book_loss(self, flux, fraction, hit_mask) -> None:
        """Book ``flux * fraction`` on the hit rays as coating loss.

        Call this with the flux as it stands *before* the weight is applied,
        so the booked watts are the ones the multiply is about to remove.

        Args:
            flux: Per-ray flux before the weight is applied, shape (N,).
            fraction: Per-ray fraction of that flux this surface removes,
                shape (N,). Zero where the surface is lossless.
            hit_mask: Per-ray mask of rays interacting with this surface,
                shape (N,).
        """
        if not hasattr(self, "_coating_loss"):
            self.reset_ledger()
        lost = be.where(hit_mask, flux * fraction, be.zeros_like(flux))
        # One host read per booking, at the same point in the bounce as the
        # AbsorbingComponent counters and the bulk-absorption tally; the
        # ledger is a diagnostic, so it is read detached and never feeds back
        # into the ray state.
        self._coating_loss += float(to_numpy(be.sum(lost)))

    @property
    def coating_loss(self) -> float:
        """Flux booked into the coating bin over the current trace [W]."""
        if not hasattr(self, "_coating_loss"):
            self.reset_ledger()
        return self._coating_loss
