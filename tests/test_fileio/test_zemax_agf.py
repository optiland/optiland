"""Zemax import with GCAT catalogs read from the user's AGF files.

``load_zemax_file(path, agf_dirs=...)`` (or ``OPTILAND_AGF_PATH``) reads a
GCAT catalog from its AGF file when one is found, and evaluates its glasses
with the catalog's own formula and coefficients. Catalogs with no AGF file use
the material database as before. The catalogs and prescriptions here are
written for these tests; no vendor data are used.
"""

from __future__ import annotations

import math
import warnings

import pytest

import optiland.backend as be
from optiland.fileio import load_zemax_file, save_zemax_file
from optiland.materials import (
    AbbeMaterial,
    AGFMaterial,
    Material,
    OptilandMaterialWarning,
)
from optiland.materials.agf import AGF_PATH_VARIABLE
from tests.test_fileio.test_zemax_glass_resolution import _PRESCRIPTION
from tests.utils import assert_allclose

_WL_D = 0.5875618

# A made-up Sellmeier 1 glass; nd = 1.6727.
_SF5 = [1.52481889, 0.011254756, 0.187085527, 0.0588995392, 1.42729015, 129.141675]


def _sellmeier(c, w):
    w2 = w * w
    return math.sqrt(1 + sum(c[i] * w2 / (w2 - c[i + 1]) for i in range(0, 6, 2)))


def _agf(name: str, coefficients: list[float], formula: int = 2) -> str:
    cd = " ".join(f"{c:.9E}" for c in coefficients + [0.0] * (10 - len(coefficients)))
    nd = _sellmeier(coefficients, _WL_D)
    return (
        "CC catalog written for the tests\n"
        f"NM {name} {formula} 0 {nd:.6f} 32.2 0 1 -1\n"
        f"CD {cd}\n"
        "LD 0.35 2.5\n"
    )


def _write_agf(directory, catalog: str, text: str) -> None:
    directory.mkdir(exist_ok=True)
    (directory / catalog).write_text(text, encoding="ascii")


def _zmx(tmp_path, glas: str, gcat: str = "GCAT SCHOTT") -> str:
    path = tmp_path / "glass.zmx"
    path.write_text(_PRESCRIPTION.format(gcat=gcat, glas=glas), encoding="utf-8")
    return str(path)


def _glass(optic):
    return optic.surfaces[1].material_post


def _nd(material) -> float:
    return float(be.to_numpy(be.atleast_1d(material.n(_WL_D))).ravel()[0])


@pytest.fixture(autouse=True)
def _no_agf_path(monkeypatch):
    monkeypatch.delenv(AGF_PATH_VARIABLE, raising=False)


def test_glass_comes_from_the_agf_catalog(tmp_path, set_test_backend):
    agf = tmp_path / "agf"
    _write_agf(agf, "SCHOTT.AGF", _agf("SF5", _SF5))
    glass = _glass(load_zemax_file(_zmx(tmp_path, "GLAS SF5 0 0"), agf_dirs=agf))
    assert isinstance(glass, AGFMaterial)
    assert glass.catalog == "SCHOTT"
    expected = [_sellmeier(_SF5, w) for w in (0.48, _WL_D, 0.65)]
    assert_allclose(glass.n(be.array([0.48, _WL_D, 0.65])), expected, rtol=1e-12)


def test_agf_catalog_replaces_the_database_catalog(tmp_path, set_test_backend):
    # N-BK7 is in the database's Schott catalog but not in this SCHOTT.AGF,
    # so it is modelled, as OpticStudio would fail to find it there.
    agf = tmp_path / "agf"
    _write_agf(agf, "SCHOTT.AGF", _agf("SF5", _SF5))
    path = _zmx(tmp_path, "GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0")
    with pytest.warns(OptilandMaterialWarning, match="N-BK7 is not in the GCAT"):
        glass = _glass(load_zemax_file(path, agf_dirs=[agf]))
    assert type(glass) is AbbeMaterial


def test_catalogs_without_agf_use_the_database(tmp_path, set_test_backend):
    agf = tmp_path / "agf"
    _write_agf(agf, "mycat.agf", _agf("MY1", _SF5))
    path = _zmx(
        tmp_path, "GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0", "GCAT MYCAT SCHOTT"
    )
    glass = _glass(load_zemax_file(path, agf_dirs=[agf]))
    assert isinstance(glass, Material)
    assert glass.material_data["filename"] == "glass/schott/N-BK7.yml"

    glass = _glass(
        load_zemax_file(_zmx(tmp_path, "GLAS MY1 0 0", "GCAT MYCAT SCHOTT"), agf)
    )
    assert isinstance(glass, AGFMaterial)
    assert glass.catalog == "mycat"


def test_without_agf_folders_nothing_changes(tmp_path, set_test_backend):
    glass = _glass(load_zemax_file(_zmx(tmp_path, "GLAS N-BK7 0 0")))
    assert isinstance(glass, Material)


def test_environment_variable(tmp_path, monkeypatch, set_test_backend):
    agf = tmp_path / "agf"
    _write_agf(agf, "SCHOTT.AGF", _agf("SF5", _SF5))
    monkeypatch.setenv(AGF_PATH_VARIABLE, str(agf))
    assert isinstance(
        _glass(load_zemax_file(_zmx(tmp_path, "GLAS SF5 0 0"))), AGFMaterial
    )
    # An explicit empty list switches the AGF lookup off.
    with pytest.warns(OptilandMaterialWarning, match="SF5 is not in the GCAT"):
        load_zemax_file(
            _zmx(tmp_path, "GLAS SF5 0 0 1.6727 32.2 0 0 0 0 0 0"), agf_dirs=[]
        )


def test_first_folder_wins(tmp_path, set_test_backend):
    first, second = tmp_path / "a", tmp_path / "b"
    other = [c * 1.01 for c in _SF5]
    _write_agf(first, "SCHOTT.AGF", _agf("SF5", _SF5))
    _write_agf(second, "SCHOTT.AGF", _agf("SF5", other))
    glass = _glass(load_zemax_file(_zmx(tmp_path, "GLAS SF5 0 0"), [first, second]))
    assert_allclose(_nd(glass), _sellmeier(_SF5, _WL_D))
    glass = _glass(load_zemax_file(_zmx(tmp_path, "GLAS SF5 0 0"), [second, first]))
    assert_allclose(_nd(glass), _sellmeier(other, _WL_D))


def test_unknown_catalog_warning_names_the_agf_folders(tmp_path, set_test_backend):
    path = _zmx(tmp_path, "GLAS X1 0 0 1.6 40 0 0 0 0 0 0", "GCAT NOSUCHCAT")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_zemax_file(path, agf_dirs=[tmp_path])
    messages = [str(w.message) for w in caught]
    assert any(
        "NOSUCHCAT is not in the material database or the AGF" in m for m in messages
    )
    assert any("X1 is not in the GCAT catalogs" in m for m in messages)


def test_round_trip_keeps_the_agf_glass(tmp_path, set_test_backend):
    agf = tmp_path / "agf"
    _write_agf(agf, "SCHOTT.AGF", _agf("SF5", _SF5))
    optic = load_zemax_file(_zmx(tmp_path, "GLAS SF5 0 0"), agf_dirs=agf)
    out = tmp_path / "out.zmx"
    save_zemax_file(optic, str(out))
    text = out.read_text(encoding="utf-8")
    assert "GCAT SCHOTT" in text
    assert "GLAS SF5 0 0 " in text
    glass = _glass(load_zemax_file(str(out), agf_dirs=agf))
    assert isinstance(glass, AGFMaterial)
    assert_allclose(_nd(glass), _nd(_glass(optic)))
