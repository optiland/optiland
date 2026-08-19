"""Did-you-mean suggestions for string-keyed lookups.

This module is deliberately dependency-free so that any part of the package
may import it without risking a circular import. The public entry point is
re-exported as :func:`optiland.diagnostics.suggest.did_you_mean`.

Kramer Harrison, 2026
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["did_you_mean", "options_hint"]


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
    if not isinstance(value, str):
        return ""
    pool = [str(c) for c in candidates]
    matches = difflib.get_close_matches(value, pool, n=n, cutoff=0.6)
    if not matches:
        # Fall back to a case-insensitive pass, which catches the common
        # "epd" vs. "EPD" mistake that difflib scores below the cutoff.
        lowered = {c.lower(): c for c in pool}
        ci = difflib.get_close_matches(value.lower(), list(lowered), n=n, cutoff=0.6)
        matches = [lowered[m] for m in ci]
    if not matches:
        return ""
    return f" Did you mean: {', '.join(matches)}?"


def options_hint(value: str, candidates: Iterable[str], max_listed: int = 12) -> str:
    """Build a trailing hint listing valid options, plus a did-you-mean.

    Args:
        value: The value the caller supplied that was not found.
        candidates: The valid values it could have meant.
        max_listed: Above this many candidates the full list is omitted and
            only the did-you-mean suggestion is returned, to keep the
            message readable.

    Returns:
        A string of the form ``" Valid options are: a, b, c. Did you
        mean: b?"``, beginning with a space, or ``""`` if there is nothing
        useful to say.

    Example::

        >>> options_hint("epd", ["EPD", "imageFNO"])
        ' Valid options are: EPD, imageFNO. Did you mean: EPD?'
    """
    pool = [str(c) for c in candidates]
    suggestion = did_you_mean(value, pool)
    if not pool:
        return suggestion
    if len(pool) > max_listed:
        return suggestion
    return f" Valid options are: {', '.join(pool)}.{suggestion}"
