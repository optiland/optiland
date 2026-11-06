"""Zemax glasses resolve strictly against the file's GCAT catalogs.

``load_zemax_file`` looks each GLAS name up in the declared catalogs by exact
name only. A name found in several of them is settled by the nd/vd on the GLAS
line, then by GCAT order. A glass found in none of them becomes a model glass
from nd/vd, with a warning; MIL-number names and explicit model glasses do too,
without one. There is no fuzzy search across all catalogs. The prescriptions
here are written for these tests.
"""

from __future__ import annotations

import pytest
from numpy.testing import assert_allclose

import optiland.backend as be
from optiland.fileio import load_zemax_file, save_zemax_file
from optiland.materials import (
    AbbeMaterial,
    IdealMaterial,
    Material,
    OptilandMaterialWarning,
)
from optiland.optic import Optic

_WL_D = 0.5875618

_PRESCRIPTION = """VERS 221026 46 20120530 20120530
MODE SEQ
NAME glass
UNIT MM X W X CM MR CPMM
ENPD 10
{gcat}
FTYP 0 0 1 1 0 0 0 1
XFLN 0
YFLN 0
FWGN 1
WAVM 1 0.5875618 1
PWAV 1
SURF 0
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ INFINITY
SURF 1
  STOP
  TYPE STANDARD
  CURV 0.01 0 0 0 0 ""
  DISZ 5
  {glas}
SURF 2
  TYPE STANDARD
  CURV -0.01 0 0 0 0 ""
  DISZ 95
SURF 3
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ 0
"""


def _load(tmp_path, glas: str, gcat: str | None = "GCAT SCHOTT"):
    path = tmp_path / "glass.zmx"
    path.write_text(_PRESCRIPTION.format(gcat=gcat or "", glas=glas), encoding="utf-8")
    return load_zemax_file(str(path))


def _glass(optic):
    return optic.surfaces[1].material_post


def _catalog_file(material) -> str:
    return material.material_data["filename"]


def _nd(material) -> float:
    return float(be.to_numpy(be.atleast_1d(material.n(_WL_D))).ravel()[0])


class TestCatalogGlasses:
    def test_exact_name_in_declared_catalog(self, tmp_path, set_test_backend):
        glass = _glass(_load(tmp_path, "GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0"))
        assert isinstance(glass, Material)
        assert _catalog_file(glass) == "glass/schott/N-BK7.yml"
        # The reference names the catalog, so a re-export keeps it in GCAT.
        assert glass.reference == "schott"

    def test_first_gcat_catalog_wins_without_nd_vd(self, tmp_path, set_test_backend):
        # F2 is in both CDGM and Schott; with no nd/vd, GCAT order decides.
        glass = _glass(_load(tmp_path, "GLAS F2 0 0", gcat="GCAT CDGM SCHOTT"))
        assert _catalog_file(glass) == "glass/cdgm/F2.yml"
        glass = _glass(_load(tmp_path, "GLAS F2 0 0", gcat="GCAT SCHOTT CDGM"))
        assert _catalog_file(glass) == "glass/schott/F2.yml"

    def test_nd_vd_pick_between_declared_catalogs(self, tmp_path, set_test_backend):
        # Schott F2 is nd 1.62004, vd 36.37; CDGM F2 is nd 1.61293, vd 36.99.
        # A file lists catalogs once, so the GLAS line's nd/vd say which F2
        # the surface used, whatever the GCAT order.
        glass = _glass(
            _load(
                tmp_path,
                "GLAS F2 0 0 1.62004 36.37 0 0 0 0 0 0",
                gcat="GCAT CDGM SCHOTT",
            )
        )
        assert _catalog_file(glass) == "glass/schott/F2.yml"

    def test_undeclared_catalog_is_not_searched(self, tmp_path, set_test_backend):
        # Hikari has K5 and a bare Material("K5") finds it; optiland's Schott
        # catalog does not, so K5 under GCAT SCHOTT is modelled from nd/vd.
        with pytest.warns(OptilandMaterialWarning, match="K5 is not in the GCAT"):
            glass = _glass(_load(tmp_path, "GLAS K5 0 0 1.52249 59.48 0 0 0 0 0 0"))
        assert type(glass) is AbbeMaterial
        assert_allclose(_nd(glass), 1.52249, atol=1e-6)


class TestNoFuzzyMatch:
    def test_baf2_is_not_the_crystal(self, tmp_path, set_test_backend):
        # Fuzzy search turned BAF2 into the BaF2 crystal (main/BaF2/Kaiser.yml,
        # n = 1.413 at the d-line), far from the nd 1.57 on the GLAS line.
        with pytest.warns(OptilandMaterialWarning, match="BAF2 is not in the GCAT"):
            glass = _glass(_load(tmp_path, "GLAS BAF2 0 0 1.57082 53.20 0 0 0 0 0 0"))
        assert type(glass) is AbbeMaterial
        assert_allclose(_nd(glass), 1.57082, atol=1e-6)

    def test_sf5_is_not_its_lead_free_replacement(self, tmp_path, set_test_backend):
        # Fuzzy search turned SF5 into N-SF5 (nd 1.67271), a different glass.
        with pytest.warns(OptilandMaterialWarning, match="SF5 is not in the GCAT"):
            glass = _glass(_load(tmp_path, "GLAS SF5 0 0 1.67270 32.21 0 0 0 0 0 0"))
        assert type(glass) is AbbeMaterial
        assert_allclose(_nd(glass), 1.67270, atol=1e-6)

    def test_no_gcat_models_every_glass(self, tmp_path, set_test_backend):
        with pytest.warns(OptilandMaterialWarning, match="N-BK7 is not in the GCAT"):
            glass = _glass(
                _load(tmp_path, "GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0", gcat=None)
            )
        assert type(glass) is AbbeMaterial

    def test_unknown_catalog_warns(self, tmp_path, set_test_backend):
        with pytest.warns(OptilandMaterialWarning, match="catalog NOSUCHCAT"):
            glass = _glass(
                _load(
                    tmp_path,
                    "GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0",
                    gcat="GCAT NOSUCHCAT SCHOTT",
                )
            )
        assert _catalog_file(glass) == "glass/schott/N-BK7.yml"

    def test_missing_glass_without_nd_vd_raises(self, tmp_path, set_test_backend):
        with pytest.raises(ValueError, match="no nd/vd"):
            _load(tmp_path, "GLAS NOSUCHGLASS 0 0")


class TestModelGlasses:
    def test_explicit_model_glass(self, recwarn, tmp_path, set_test_backend):
        glass = _glass(_load(tmp_path, "GLAS ___BLANK 1 0 1.6 40 0 0 0 0 0 0"))
        assert type(glass) is AbbeMaterial
        assert_allclose(_nd(glass), 1.6, atol=1e-6)
        assert not [w for w in recwarn if w.category is OptilandMaterialWarning]

    def test_model_flag_skips_the_catalogs(self, tmp_path, set_test_backend):
        # Solve type 1 marks a model glass even when the name is a catalog one.
        glass = _glass(_load(tmp_path, "GLAS N-BK7 1 0 1.6 40 0 0 0 0 0 0"))
        assert type(glass) is AbbeMaterial
        assert_allclose(_nd(glass), 1.6, atol=1e-6)

    @pytest.mark.parametrize("name", ["517642", "517.642"])
    def test_mil_number(self, name, recwarn, tmp_path, set_test_backend):
        # nd and vd come from the name, not from the line.
        glass = _glass(_load(tmp_path, f"GLAS {name} 0 0 1.7 30 0 0 0 0 0 0"))
        assert type(glass) is AbbeMaterial
        assert_allclose(_nd(glass), 1.517, atol=1e-6)
        assert_allclose(float(be.to_numpy(glass.abbe)[0]), 64.2, atol=1e-6)
        assert not [w for w in recwarn if w.category is OptilandMaterialWarning]

    def test_zero_abbe_is_undispersed(self, tmp_path, set_test_backend):
        glass = _glass(_load(tmp_path, "GLAS ___BLANK 1 0 1.5 0 0 0 0 0 0 0"))
        assert type(glass) is IdealMaterial
        assert_allclose(_nd(glass), 1.5)


class TestRoundTrip:
    def test_material_without_reference_reloads(self, tmp_path, set_test_backend):
        # The writer exports the catalog a bare Material resolved to, so the
        # strict reader finds the same glass again.
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(
            index=1,
            radius=50,
            thickness=5,
            material=Material("N-BK7", catalog="schott"),
            is_stop=True,
        )
        optic.surfaces.add(index=2, radius=-50, thickness=95)
        optic.surfaces.add(index=3)
        optic.set_aperture("EPD", 10)
        optic.fields.set_type("angle")
        optic.fields.add(y=0)
        optic.wavelengths.add(0.5875618, is_primary=True)

        out = tmp_path / "round_trip.zmx"
        save_zemax_file(optic, str(out))
        assert "GCAT SCHOTT" in out.read_text(encoding="utf-8")
        glass = _glass(load_zemax_file(str(out)))
        assert _catalog_file(glass) == "glass/schott/N-BK7.yml"
