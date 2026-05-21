"""Tests for prescription formatting helpers."""

from __future__ import annotations

import math

from optiland.prescription._formatting import fmt_float, fmt_infinity, safe_eval


def test_fmt_float_basic():
    assert fmt_float(3.14159) == "3.1416"


def test_fmt_float_decimals():
    assert fmt_float(1.0, decimals=6) == "1.000000"


def test_fmt_float_none():
    assert fmt_float(None) == "N/A"


def test_fmt_float_unit():
    assert fmt_float(50.0, unit="mm") == "50.0000 mm"


def test_fmt_infinity_finite():
    assert fmt_infinity(50.0) == "50.0000"


def test_fmt_infinity_inf():
    assert fmt_infinity(math.inf) == "∞"


def test_fmt_infinity_neg_inf():
    assert fmt_infinity(-math.inf) == "∞"


def test_fmt_infinity_large():
    assert fmt_infinity(1e13) == "∞"


def test_safe_eval_normal():
    result = safe_eval(lambda: 49.999)
    assert result == "49.9990"


def test_safe_eval_exception():
    def boom():
        raise ValueError("no aperture")

    assert safe_eval(boom) == "N/A"


def test_safe_eval_infinity():
    assert safe_eval(lambda: math.inf) == "∞"


def test_safe_eval_no_infinity():
    assert safe_eval(lambda: 1.0, infinity=False) == "1.0000"
