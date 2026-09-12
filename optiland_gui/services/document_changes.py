"""Classified, GUI-local document notifications independent of job activity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .job_records import DocumentToken

CHANGE_CATEGORIES = frozenset(
    {"optical", "structure", "replacement", "metadata", "presentation", "polarization"}
)
OPTICAL_CATEGORIES = frozenset({"optical", "structure", "replacement", "polarization"})


@dataclass(frozen=True)
class DocumentChange:
    """One committed edit, possibly assembled from a nested transaction."""

    token: DocumentToken
    edit_token: DocumentToken
    categories: frozenset[str]
    surface_indices: frozenset[int] = frozenset()
    columns: frozenset[int] = frozenset()

    @property
    def affects_optics(self) -> bool:
        return bool(self.categories & OPTICAL_CATEGORIES)

    @property
    def structural(self) -> bool:
        return bool(self.categories & {"structure", "replacement"})
