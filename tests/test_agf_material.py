"""AGF glass catalogs: reading the records and evaluating all 13 formulas.

The catalog entries here are written for these tests; no vendor data are used.
Each formula is checked against its definition in the OpticStudio manual,
written out independently below.
"""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.materials import AGFMaterial, BaseMaterial
from optiland.materials.agf import evaluate_agf_formula, find_agf_catalog, read_agf
from tests.utils import assert_allclose

_WAVES = [0.45, 0.5875618, 0.9]

_SCHOTT = [2.27, -0.01, 0.012, 2e-4, -1e-5, 1e-6]
_SELLMEIER_3 = [1.0, 0.006, 0.2, 0.02, 0.9, 100.0, 0.01, 0.05]


def _sellmeier(c, w):
    w2 = w * w
    return math.sqrt(1 + sum(c[i] * w2 / (w2 - c[i + 1]) for i in range(0, len(c), 2)))


def _powers(c, powers, w):
    return math.sqrt(sum(a * w**p for a, p in zip(c, powers, strict=True)))


# (formula, coefficients, reference n(λ) from the manual's equation)
_FORMULAS = [
    (1, _SCHOTT, lambda c, w: _powers(c, [0, 2, -2, -4, -6, -8], w)),
    (2, [1.04, 0.006, 0.23, 0.02, 1.01, 103.6], _sellmeier),
    (
        3,
        [1.5, 0.005, 2e-4, -1e-3, 1e-5, -1e-7],
        lambda c, w: (
            c[0]
            + c[1] / (w**2 - 0.028)
            + c[2] / (w**2 - 0.028) ** 2
            + c[3] * w**2
            + c[4] * w**4
            + c[5] * w**6
        ),
    ),
    (
        4,
        [0.1, 1.0, 0.1, 0.002, 0.05],
        lambda c, w: math.sqrt(
            1 + c[0] + c[1] * w**2 / (w**2 - c[2] ** 2) + c[3] / (w**2 - c[4] ** 2)
        ),
    ),
    (5, [1.5, 0.005, 4e-4], lambda c, w: c[0] + c[1] / w + c[2] / w**3.5),
    (6, _SELLMEIER_3, _sellmeier),
    (
        7,
        [2.2, 0.01, 0.02, 0.01],
        lambda c, w: math.sqrt(c[0] + c[1] / (w**2 - c[2]) - c[3] * w**2),
    ),
    (
        8,
        [2.0, 0.3, 0.03, 0.01],
        lambda c, w: math.sqrt(c[0] + c[1] * w**2 / (w**2 - c[2]) - c[3] * w**2),
    ),
    (
        9,
        [1.2, 1.0, 0.01, 0.5, 100.0],
        lambda c, w: math.sqrt(
            c[0] + c[1] * w**2 / (w**2 - c[2]) + c[3] * w**2 / (w**2 - c[4])
        ),
    ),
    (
        10,
        _SCHOTT + [1e-7, -1e-8],
        lambda c, w: _powers(c, [0, 2, -2, -4, -6, -8, -10, -12], w),
    ),
    (11, _SELLMEIER_3 + [0.02, 0.08], _sellmeier),
    (
        12,
        _SCHOTT + [1e-4, -1e-5],
        lambda c, w: _powers(c, [0, 2, -2, -4, -6, -8, 4, 6], w),
    ),
    (
        13,
        [2.27, -0.01, 1e-4, 0.012, 2e-4, -1e-5, 1e-6, 1e-7, -1e-8],
        lambda c, w: _powers(c, [0, 2, 4, -2, -4, -6, -8, -10, -12], w),
    ),
]


def _entry(name, formula, coefficients, low=0.3, high=2.5):
    cd = " ".join(f"{c:.9E}" for c in coefficients + [0.0] * (10 - len(coefficients)))
    return (
        f"NM {name} {formula} 0 1.500000 60.000000 0 1 -1\n"
        f"GC written for the tests\n"
        f"ED 7.1 0 2.5 0 0 0\n"
        f"CD {cd}\n"
        f"TD 0 0 0 0 0 0 20\n"
        f"OD -1 -1 -1 -1 -1 -1\n"
        f"LD {low} {high}\n"
        f"IT 0.4 0.9 10\n"
    )


def _write_catalog(tmp_path, entries, name="TEST.AGF", encoding="ascii"):
    path = tmp_path / name
    path.write_text("CC catalog written for the tests\n" + "".join(entries), encoding)
    return str(path)


@pytest.mark.parametrize(
    "formula,coefficients,reference", _FORMULAS, ids=[f"f{f[0]}" for f in _FORMULAS]
)
def test_formula_matches_manual(
    set_test_backend, tmp_path, formula, coefficients, reference
):
    path = _write_catalog(tmp_path, [_entry("G1", formula, coefficients)])
    material = AGFMaterial.from_file(path, "G1")
    expected = [reference(coefficients, w) for w in _WAVES]
    assert_allclose(material.n(be.array(_WAVES)), expected, rtol=1e-12)
    assert_allclose(material.n(_WAVES[1]), expected[1], rtol=1e-12)


def test_all_formula_codes_are_covered():
    assert sorted(f[0] for f in _FORMULAS) == list(range(1, 14))


def test_read_records(tmp_path):
    path = _write_catalog(
        tmp_path,
        [_entry("G1", 2, _FORMULAS[1][1]), _entry("G2", 1, _SCHOTT, 0.36, 1.1)],
    )
    glasses = read_agf(path)
    assert list(glasses) == ["G1", "G2"]
    g2 = glasses["G2"]
    assert g2.formula == 1
    assert g2.catalog == "TEST"
    assert g2.nd == 1.5
    assert g2.vd == 60.0
    assert g2.wavelength_range == (0.36, 1.1)
    assert g2.coefficients[:6] == _SCHOTT
    assert g2.comment == "written for the tests"


def test_utf16_catalog(tmp_path):
    path = _write_catalog(tmp_path, [_entry("G1", 1, _SCHOTT)], encoding="utf-16")
    assert read_agf(path)["G1"].coefficients[:6] == _SCHOTT


def test_formula_code_written_as_float(tmp_path):
    entry = _entry("G1", 1, _SCHOTT).replace("NM G1 1 ", "NM G1 1.00000000E+00 ")
    path = _write_catalog(tmp_path, [entry])
    assert read_agf(path)["G1"].formula == 1


def test_lookup_is_exact_but_ignores_case(tmp_path):
    path = _write_catalog(tmp_path, [_entry("N-G1", 1, _SCHOTT)])
    assert AGFMaterial.from_file(path, "n-g1").name == "N-G1"
    with pytest.raises(ValueError, match="not in"):
        AGFMaterial.from_file(path, "G1")


def test_bad_records_fail_early():
    with pytest.raises(ValueError, match="Unknown AGF dispersion formula"):
        AGFMaterial("X", 14, _SCHOTT)
    with pytest.raises(ValueError, match="needs 8 coefficients"):
        AGFMaterial("X", 6, _SCHOTT)
    with pytest.raises(ValueError, match="Unknown"):
        evaluate_agf_formula(0, _SCHOTT, 0.5)


def test_material_properties(set_test_backend, tmp_path):
    path = _write_catalog(tmp_path, [_entry("G1", 1, _SCHOTT, 0.36, 1.1)])
    material = AGFMaterial.from_file(path, "G1")
    assert material.display_name == "G1"
    assert material.catalog == "TEST"
    assert material.spectral_range() == (0.36, 1.1)
    assert_allclose(material.k(be.array(_WAVES)), [0.0, 0.0, 0.0])
    assert_allclose(material.k(0.5), 0.0)
    n = [
        _powers(_SCHOTT, [0, 2, -2, -4, -6, -8], w)
        for w in (0.5875618, 0.4861327, 0.6562725)
    ]
    assert_allclose(material.abbe(), (n[0] - 1) / (n[1] - n[2]), rtol=1e-10)


def test_serialization_round_trip(set_test_backend, tmp_path):
    path = _write_catalog(tmp_path, [_entry("G1", 13, _FORMULAS[12][1])])
    material = AGFMaterial.from_file(path, "G1")
    restored = BaseMaterial.from_dict(material.to_dict())
    assert isinstance(restored, AGFMaterial)
    assert restored == material
    assert_allclose(restored.n(be.array(_WAVES)), material.n(be.array(_WAVES)))


def test_cache_follows_coefficient_edits(set_test_backend, tmp_path):
    path = _write_catalog(tmp_path, [_entry("G1", 1, _SCHOTT)])
    material = AGFMaterial.from_file(path, "G1")
    before = float(material.n(0.5))
    assert float(material.n(0.5)) == before
    material.coefficients[0] += 0.1
    assert_allclose(
        material.n(0.5), _powers(material.coefficients[:6], [0, 2, -2, -4, -6, -8], 0.5)
    )


def test_latin1_catalog_blank_lines_and_bad_numbers(tmp_path):
    # Older catalogs may be Latin-1 (here a comment with "ü"), have blank
    # lines, and carry placeholders that are not numbers; those read as 0.
    entry = _entry("G1", 1, _SCHOTT).replace(
        "GC written for the tests", "GC Glas für die Tests\n"
    )
    entry = entry.replace("LD 0.3 2.5", "LD 0.3 -")
    path = tmp_path / "LATIN.AGF"
    path.write_bytes(("CC catalog\n\n" + entry).encode("latin-1"))
    glass = read_agf(str(path))["G1"]
    assert glass.comment == "Glas für die Tests"
    assert glass.wavelength_range == (0.3, 0.0)
    assert glass.coefficients[:6] == _SCHOTT


def test_missing_folders_are_skipped(tmp_path):
    path = _write_catalog(tmp_path, [_entry("G1", 1, _SCHOTT)], name="CAT.AGF")
    missing = str(tmp_path / "does-not-exist")
    assert find_agf_catalog("cat", [missing, str(tmp_path)]) == path
    assert find_agf_catalog("cat", [missing]) is None
