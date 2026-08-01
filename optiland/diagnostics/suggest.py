"""Did-you-mean suggestions for string-keyed lookups.

Kramer Harrison, 2026
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def did_you_mean(value: str, candidates: Iterable[str], n: int = 3) -> str:
    """Suggest close matches for a misspelled string-keyed lookup.

    Args:
        value: The value the caller supplied that was not found.
        candidates: The valid values it could have meant.
        n: Maximum number of suggestions to include.

    Returns:
        A string of the form ``" Did you mean: N-SF11, N-SF10?"``, or an
        empty string if nothing is close enough to suggest.

    Example::

        >>> did_you_mean("NSF11", ["N-SF11", "N-SF10", "N-BK7"])
        ' Did you mean: N-SF11, N-SF10?'
    """
    matches = difflib.get_close_matches(value, list(candidates), n=n, cutoff=0.6)
    if not matches:
        return ""
    return f" Did you mean: {', '.join(matches)}?"
