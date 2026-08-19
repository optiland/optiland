"""Tests for PR2 material system features:
- MaterialRegistry singleton, list_catalogs, list_materials
- Programmatic registration (register / register_file / load_catalog)
- Conflict / shadow warnings
- MaterialCatalog discovery UX
- Serialization: to_dict includes catalog/match_policy; from_dict warns on missing catalog
- Round-trip serialize → deserialize
- Material.__repr__
"""

from __future__ import annotations

import pathlib
import textwrap
import warnings

import pytest
import yaml

from optiland.materials import (
    Material,
    MaterialCatalog,
    MaterialRegistry,
    MatchPolicy,
    OptilandMaterialWarning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_registry() -> MaterialRegistry:
    """Return a new, isolated registry instance for test isolation."""
    reg = object.__new__(MaterialRegistry)
    reg.__init__()  # type: ignore[misc]
    return reg


def _minimal_yaml_data(name: str = "TestGlass") -> dict:
    """Minimal refractiveindex.info-format YAML payload (tabulated n)."""
    return {
        "REFERENCE": f"Test reference for {name}",
        "DATA": [
            {
                "type": "tabulated n",
                "data": textwrap.dedent(
                    """\
                    0.4 1.52
                    0.55 1.51
                    0.7 1.505
                    """
                ),
            }
        ],
    }


# ---------------------------------------------------------------------------
# Registry — singleton
# ---------------------------------------------------------------------------


class TestRegistrySingleton:
    def test_instance_returns_same_object(self):
        """Repeated .instance() calls return the same registry."""
        r1 = MaterialRegistry.instance()
        r2 = MaterialRegistry.instance()
        assert r1 is r2

    def test_built_in_df_non_empty(self):
        """built_in_df contains the built-in catalog entries."""
        df = MaterialRegistry.instance().built_in_df
        assert len(df) > 1000

    def test_built_in_df_has_expected_columns(self):
        """built_in_df has the expected schema columns."""
        df = MaterialRegistry.instance().built_in_df
        for col in ["name", "filename", "filename_no_ext", "category_name", "reference"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Registry — list_catalogs / list_materials
# ---------------------------------------------------------------------------


class TestRegistryDiscovery:
    def test_list_catalogs_non_empty(self):
        """list_catalogs returns a non-empty sorted list."""
        cats = MaterialRegistry.instance().list_catalogs()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_list_catalogs_includes_expected(self):
        """list_catalogs includes well-known manufacturer catalogs."""
        cats = MaterialRegistry.instance().list_catalogs()
        for expected in ("schott", "ohara", "hikari"):
            assert expected in cats, f"Missing catalog: {expected}"

    def test_list_catalogs_sorted(self):
        """list_catalogs returns values in sorted order."""
        cats = MaterialRegistry.instance().list_catalogs()
        assert cats == sorted(cats)

    def test_list_materials_no_filter(self):
        """list_materials() with no filter returns all materials."""
        mats = MaterialRegistry.instance().list_materials()
        assert len(mats) > 1000

    def test_list_materials_filtered(self):
        """list_materials('schott') returns only Schott glasses."""
        mats = MaterialRegistry.instance().list_materials("schott")
        assert len(mats) > 10
        assert "N-BK7" in mats

    def test_list_materials_sorted(self):
        """list_materials returns values in sorted order."""
        mats = MaterialRegistry.instance().list_materials("schott")
        assert mats == sorted(mats)


# ---------------------------------------------------------------------------
# Registry — programmatic registration
# ---------------------------------------------------------------------------


class TestRegistryProgrammatic:
    def test_register_resolves(self):
        """Programmatically registered material resolves via resolve()."""
        reg = _fresh_registry()
        data = _minimal_yaml_data("MyGlass")
        reg.register("MyGlass", "internal", data)
        path = reg.resolve("MyGlass", catalog="internal", match_policy=MatchPolicy.STRICT)
        assert pathlib.Path(path).exists()

    def test_register_material_returns_correct_n(self):
        """Programmatically registered material yields correct refractive index."""
        reg = _fresh_registry()
        data = _minimal_yaml_data("FlatGlass")
        reg.register("FlatGlass", "internal", data)

        path, _ = reg._resolve_with_row(
            "FlatGlass", "internal", None, MatchPolicy.STRICT, None, None
        )
        from optiland.materials.material_file import MaterialFile
        mf = MaterialFile(path)
        n = mf.n(0.55)
        assert abs(float(n) - 1.51) < 0.01

    def test_register_shadow_builtin_warns(self):
        """Shadowing a built-in entry emits OptilandMaterialWarning at register time."""
        reg = _fresh_registry()
        data = _minimal_yaml_data("N-BK7")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.register("N-BK7", "schott", data)
        mat_warns = [x for x in w if issubclass(x.category, OptilandMaterialWarning)]
        assert len(mat_warns) == 1
        assert "shadows" in str(mat_warns[0].message).lower()

    def test_register_shadow_user_warns(self):
        """Overwriting an existing user entry emits OptilandMaterialWarning."""
        reg = _fresh_registry()
        data = _minimal_yaml_data("MyGlass")
        reg.register("MyGlass", "internal", data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.register("MyGlass", "internal", data)
        mat_warns = [x for x in w if issubclass(x.category, OptilandMaterialWarning)]
        assert len(mat_warns) == 1

    def test_same_name_different_catalog_no_conflict(self):
        """Same name in different catalogs coexist without warnings."""
        reg = _fresh_registry()
        data = _minimal_yaml_data("SharedName")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.register("SharedName", "cat1", data)
            reg.register("SharedName", "cat2", data)
        mat_warns = [x for x in w if issubclass(x.category, OptilandMaterialWarning)]
        assert len(mat_warns) == 0


# ---------------------------------------------------------------------------
# Registry — register_file
# ---------------------------------------------------------------------------


class TestRegistryRegisterFile:
    def test_register_file_resolves(self, tmp_path):
        """register_file() loads a YAML file and makes it resolvable."""
        catalog_dir = tmp_path / "my_catalog"
        catalog_dir.mkdir()
        glass_file = catalog_dir / "TestGlass.yml"
        glass_file.write_text(
            yaml.dump(_minimal_yaml_data("TestGlass")), encoding="utf-8"
        )

        reg = _fresh_registry()
        reg.register_file(glass_file)
        path = reg.resolve("TestGlass", catalog="my_catalog", match_policy=MatchPolicy.STRICT)
        assert pathlib.Path(path).exists()

    def test_register_file_catalog_from_parent_dir(self, tmp_path):
        """Catalog name is inferred from the parent directory of the YAML file."""
        catalog_dir = tmp_path / "special_cat"
        catalog_dir.mkdir()
        glass_file = catalog_dir / "GlassX.yml"
        glass_file.write_text(yaml.dump(_minimal_yaml_data("GlassX")), encoding="utf-8")

        reg = _fresh_registry()
        reg.register_file(glass_file)
        assert "special_cat" in reg.list_catalogs()


# ---------------------------------------------------------------------------
# Registry — load_catalog
# ---------------------------------------------------------------------------


class TestRegistryLoadCatalog:
    def test_load_catalog_directory(self, tmp_path):
        """load_catalog() ingests all YAML files in a directory."""
        cat_dir = tmp_path / "batch_cat"
        cat_dir.mkdir()
        for name in ("GlassA", "GlassB", "GlassC"):
            (cat_dir / f"{name}.yml").write_text(
                yaml.dump(_minimal_yaml_data(name)), encoding="utf-8"
            )

        reg = _fresh_registry()
        reg.load_catalog(cat_dir)
        mats = reg.list_materials("batch_cat")
        assert "GlassA" in mats
        assert "GlassB" in mats
        assert "GlassC" in mats

    def test_load_catalog_nonexistent_dir_noop(self):
        """load_catalog() on a nonexistent directory is a no-op (no error)."""
        reg = _fresh_registry()
        reg.load_catalog("/nonexistent/path/xyz123")  # should not raise


# ---------------------------------------------------------------------------
# Registry — user wins on conflict
# ---------------------------------------------------------------------------


class TestRegistryUserWins:
    def test_user_wins_over_builtin(self):
        """User-registered entry overrides built-in with same (name, catalog)."""
        reg = _fresh_registry()

        # Register a custom N-BK7 in schott catalog with n=1.99
        custom_data = {
            "REFERENCE": "custom",
            "DATA": [{"type": "tabulated n", "data": "0.4 1.99\n0.55 1.99\n0.7 1.99\n"}],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.register("N-BK7", "schott", custom_data)

        path, _ = reg._resolve_with_row(
            "N-BK7", "schott", None, MatchPolicy.STRICT, None, None
        )
        from optiland.materials.material_file import MaterialFile
        mf = MaterialFile(path)
        n = mf.n(0.55)
        assert abs(float(n) - 1.99) < 0.01, "User entry should shadow built-in"


# ---------------------------------------------------------------------------
# MaterialCatalog
# ---------------------------------------------------------------------------


class TestMaterialCatalog:
    def test_available_returns_list(self):
        """MaterialCatalog.available() returns a list of strings."""
        cats = MaterialCatalog.available()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_available_includes_schott(self):
        """MaterialCatalog.available() includes schott."""
        assert "schott" in MaterialCatalog.available()

    def test_list_returns_materials(self):
        """MaterialCatalog('schott').list() returns a non-empty list."""
        mats = MaterialCatalog("schott").list()
        assert len(mats) > 0
        assert "N-BK7" in mats

    def test_search_finds_nbk7(self):
        """MaterialCatalog.search('bk7') returns N-BK7 among results."""
        results = MaterialCatalog("schott").search("bk7")
        assert len(results) > 0
        assert "N-BK7" in results

    def test_get_returns_material(self):
        """MaterialCatalog.get('N-BK7') returns a Material instance."""
        glass = MaterialCatalog("schott").get("N-BK7")
        assert isinstance(glass, Material)
        assert abs(glass.n(0.5876) - 1.5168) < 0.001

    def test_repr(self):
        """MaterialCatalog.__repr__ includes the catalog name."""
        cat = MaterialCatalog("schott")
        assert "schott" in repr(cat)


# ---------------------------------------------------------------------------
# Serialization — to_dict / from_dict
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_includes_catalog(self):
        """to_dict() includes the 'catalog' field."""
        m = Material("N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        d = m.to_dict()
        assert d["catalog"] == "schott"

    def test_to_dict_includes_match_policy(self):
        """to_dict() includes the 'match_policy' field as a string."""
        m = Material("N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        d = m.to_dict()
        assert d["match_policy"] == "best"

    def test_to_dict_catalog_none_when_unset(self):
        """to_dict() emits catalog=None when no catalog was given."""
        m = Material("N-BK7", match_policy=MatchPolicy.BEST)
        d = m.to_dict()
        assert d["catalog"] is None

    def test_from_dict_missing_catalog_warns(self):
        """from_dict() emits DeprecationWarning when catalog field is absent."""
        data = {"name": "N-BK7", "type": "Material"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Material.from_dict(data)
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warns) >= 1
        assert "catalog" in str(dep_warns[0].message)

    def test_from_dict_with_catalog_no_deprecation_warning(self):
        """from_dict() with catalog field does not emit DeprecationWarning."""
        data = {
            "name": "N-BK7",
            "catalog": "schott",
            "match_policy": "best",
            "type": "Material",
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Material.from_dict(data)
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warns) == 0

    def test_round_trip(self):
        """Serialize → deserialize preserves name, catalog, and match_policy."""
        original = Material("N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        d = original.to_dict()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            restored = Material.from_dict(d)
        assert restored.name == original.name
        assert restored._catalog == original._catalog
        assert restored._match_policy == original._match_policy

    def test_round_trip_optical_values(self):
        """Round-trip preserves the computed refractive index."""
        original = Material("N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        d = original.to_dict()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            restored = Material.from_dict(d)
        assert abs(original.n(0.55) - restored.n(0.55)) < 1e-6


# ---------------------------------------------------------------------------
# Material.__repr__
# ---------------------------------------------------------------------------


class TestMaterialRepr:
    def test_repr_includes_name(self):
        m = Material("N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        assert "N-BK7" in repr(m)

    def test_repr_includes_catalog(self):
        m = Material("N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        assert "schott" in repr(m)

    def test_repr_includes_wavelength_range(self):
        m = Material("N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        r = repr(m)
        assert "µm" in r or "um" in r.lower() or "λ" in r

    def test_repr_no_catalog_omits_catalog(self):
        m = Material("N-BK7", match_policy=MatchPolicy.BEST)
        r = repr(m)
        assert "catalog" not in r


# ---------------------------------------------------------------------------
# Group-aware discovery (plan: MaterialCatalog group API)
# ---------------------------------------------------------------------------


class TestGroupAwareDiscovery:
    def test_groups_contains_expected(self):
        """groups() contains at minimum the four primary group names."""
        groups = MaterialCatalog.groups()
        assert {"glass", "main", "organic", "other"}.issubset(set(groups))

    def test_available_returns_only_glass(self):
        """available() returns glass manufacturer names, not element symbols."""
        cats = MaterialCatalog.available()
        assert "schott" in cats
        assert "ohara" in cats
        assert "Ag" not in cats
        assert "methane" not in cats

    def test_available_is_sorted(self):
        """available() returns values in sorted order."""
        cats = MaterialCatalog.available()
        assert cats == sorted(cats)

    def test_available_excludes_pure_element_symbols(self):
        """available() does not contain unambiguous element/compound symbols."""
        glass_cats = set(MaterialCatalog.available())
        # These catalog_dir values only ever appear in the 'main' group
        for pure_element in ("Ag", "Au", "Cu", "Ge", "Pt", "W"):
            assert pure_element not in glass_cats

    def test_catalog_list_nonempty_and_sorted(self):
        """MaterialCatalog('schott').list() is non-empty and sorted."""
        mats = MaterialCatalog("schott").list()
        assert len(mats) > 0
        assert mats == sorted(mats)

    def test_catalog_search_bk7(self):
        """search('bk7') returns names containing the BK7 substring."""
        results = MaterialCatalog("schott").search("BK7")
        assert len(results) > 0
        assert all("BK7" in r.upper() for r in results)

    def test_catalog_get_returns_material(self):
        """get('N-BK7') returns a Material instance."""
        glass = MaterialCatalog("schott").get("N-BK7")
        assert isinstance(glass, Material)

    def test_registry_list_groups_matches_catalog_groups(self):
        """MaterialRegistry.list_groups() equals MaterialCatalog.groups()."""
        assert MaterialRegistry.instance().list_groups() == MaterialCatalog.groups()
