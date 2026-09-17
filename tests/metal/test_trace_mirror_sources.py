"""WP0: the mirrored-source fingerprint table (plan 3.7, design 8.3).

These tests are the alarm that tells a future upstream merge it has moved a
function the Metal kernel copies expression-for-expression.  They are
fork-local by design: upstream CI cannot run Metal.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import ast  # noqa: E402
import inspect  # noqa: E402
import textwrap  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from optiland.backend.torch_backend.metal import trace_mirror as TM  # noqa: E402

_MIRRORED = [fp for fp in TM.FINGERPRINTS if fp.klass == TM.MIRRORED]
_CONTRACT = [fp for fp in TM.FINGERPRINTS if fp.klass == TM.CONTRACT]

_REPO = Path(__file__).resolve().parents[2]
_VALUE_TEST_FILES = (
    _REPO / "tests" / "metal" / "test_trace_adapters.py",
    _REPO / "tests" / "metal" / "test_trace_batch.py",
)


def test_table_is_non_empty_and_split_into_two_classes():
    assert len(_MIRRORED) == 63
    assert len(_CONTRACT) == 16
    assert len(TM.FINGERPRINTS) == 79
    assert {fp.klass for fp in TM.FINGERPRINTS} == {TM.MIRRORED, TM.CONTRACT}


def test_qualnames_are_unique():
    names = [fp.qualname for fp in TM.FINGERPRINTS]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("fp", _MIRRORED, ids=lambda fp: fp.qualname)
def test_fingerprints_match(fp):
    """The live Python source still hashes to the recorded digest."""
    live = TM.source_digest(fp.qualname)
    assert live == fp.sha256, (
        f"Python `{fp.qualname}` changed; re-verify MSL `{fp.msl_function}` in "
        "kernels/trace.metal, then run `python -m "
        "optiland.backend.torch_backend.metal.trace_mirror --update --verified "
        f'"{fp.qualname}=<why it is still mirrored>"`'
    )


@pytest.mark.parametrize(
    "identity", TM.STRUCTURAL_IDENTITIES, ids=lambda fn: fn.__name__
)
def test_structural_identities(identity):
    assert identity() is True, (
        f"structural identity {identity.__name__} no longer holds"
    )


def test_fingerprint_ignores_docstrings_and_formatting():
    """Reformatting and docstring edits must not flap the digest."""
    qualname = "optiland.geometries.newton_raphson:_nz_threshold"
    original = inspect.getsource(TM._resolve(qualname))
    baseline = TM.source_digest(qualname)

    def digest_of(src: str) -> str:
        tree = TM._strip_docstring(ast.parse(textwrap.dedent(src)))
        import hashlib

        return hashlib.sha256(("ast:" + ast.dump(tree)).encode("utf-8")).hexdigest()

    assert digest_of(original) == baseline

    # (a) a different docstring
    tree = ast.parse(textwrap.dedent(original))
    fn = tree.body[0]
    assert isinstance(fn.body[0], ast.Expr), (
        "probe assumes the function has a docstring"
    )
    fn.body[0] = ast.Expr(value=ast.Constant(value="completely different docstring"))
    assert digest_of(ast.unparse(tree)) == baseline

    # (b) added comments and blank lines
    commented = original.replace("\n", "\n    # noise\n", 1)
    assert digest_of(commented) == baseline

    # (c) a real expression change must NOT match
    changed = original.replace("return", "return 1.0 *", 1)
    assert digest_of(changed) != baseline


def test_contract_rows_are_not_hashed():
    """CONTRACT rows carry no digest and name the value test that guards them."""
    for fp in _CONTRACT:
        assert fp.sha256 is None, f"{fp.qualname} is CONTRACT but carries a digest"
        assert fp.msl_function.startswith("test_"), fp.msl_function

    present = [p for p in _VALUE_TEST_FILES if p.exists()]
    if not present:
        pytest.skip(
            "WP2/WP7 have not landed tests/metal/test_trace_adapters.py or "
            "test_trace_batch.py yet; the required value-test names are frozen in "
            "NOTES/fused-trace-research/status.md (WP0 section)"
        )
    text = "\n".join(p.read_text(encoding="utf-8") for p in present)
    missing = sorted(
        {fp.msl_function for fp in _CONTRACT if fp.msl_function not in text}
    )
    assert not missing, (
        "CONTRACT rows name value tests that do not exist in "
        f"{[p.name for p in present]}: {missing}"
    )


def test_mirrored_rows_name_an_msl_function_or_host_site():
    for fp in _MIRRORED:
        assert fp.msl_function, fp.qualname
        assert not fp.msl_function.startswith("test_"), fp.qualname


def test_check_all_detects_injected_change(monkeypatch):
    """A one-expression edit to one mirrored source is reported, and only it."""
    target = "optiland.geometries.newton_raphson:_nz_threshold"
    victim = TM._resolve(target)
    original = inspect.getsource(victim)
    tampered = original.replace("return", "return 1.0 *", 1)
    assert tampered != original

    real_getsource = inspect.getsource

    def fake_getsource(obj):
        if obj is victim:
            return tampered
        return real_getsource(obj)

    monkeypatch.setattr(TM.inspect, "getsource", fake_getsource)
    problems = TM.check_all()
    assert len(problems) == 1, problems
    assert target in problems[0]
    assert "nz_threshold" in problems[0]


def test_check_all_reports_a_renamed_source(monkeypatch):
    """A rename (qualname gone) is drift, not a crash."""
    real_resolve = TM._resolve
    target = _MIRRORED[0].qualname

    def fake_resolve(qualname):
        if qualname == target:
            raise AttributeError("renamed upstream")
        return real_resolve(qualname)

    monkeypatch.setattr(TM, "_resolve", fake_resolve)
    problems = TM.check_all()
    assert len(problems) == 1, problems
    assert "cannot be resolved" in problems[0]


def test_check_all_detects_a_broken_structural_identity(monkeypatch):
    def broken():
        return False

    broken.__name__ = "broken_identity"
    monkeypatch.setattr(TM, "STRUCTURAL_IDENTITIES", (broken,))
    problems = TM.check_all()
    assert problems == ["structural identity broken_identity no longer holds"]


def test_check_all_is_clean_on_this_checkout():
    assert TM.check_all() == []


def test_table_hash_is_stable_and_sensitive():
    first = TM.table_hash()
    assert first == TM.table_hash()
    assert len(first) == 64
