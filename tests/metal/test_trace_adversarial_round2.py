"""Regression and lock tests closing verify-round 2 findings.

WP0 creates this file with one sanity test so the definition-of-done path
exists even for a round that produces no findings.  The wave-3 fix lane for
round 2 owns it from then on: every finding it closes lands here as a named
regression test (plan 0.1, 4/WP0), or as a documented-limit lock test.

Never widen a tolerance to make a test here pass (plan 0.2.2).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")


def test_round_file_present():
    """Placeholder so round 2 always has a collectable test file."""
    assert True
