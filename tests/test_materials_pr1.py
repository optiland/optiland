"""Tests for PR1 material system foundation:
- catalog= kwarg on Material
- MatchPolicy enum
- robust_search deprecation
- MaterialSpec dataclass
- MaterialFactory extensions (MaterialSpec, 3-tuple, dict)
"""

from __future__ import annotations

import warnings

import pytest

from optiland.materials import (
    Material,
    MaterialSpec,
    MatchPolicy,
    OptilandMaterialWarning,
)
from optiland.surfaces.factories.material_factory import MaterialFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factory():
    """Return a fresh MaterialFactory with a minimal stub surface_group."""

    class _FakeSurface:
        material_post = Material("N-BK7", match_policy=MatchPolicy.BEST)

    class _FakeSurfaceGroup:
        num_surfaces = 2
        surfaces = [_FakeSurface(), _FakeSurface()]

    return MaterialFactory(), _FakeSurfaceGroup()


# ---------------------------------------------------------------------------
# catalog= kwarg
# ---------------------------------------------------------------------------


class TestCatalogKwarg:
    def test_material_catalog_kwarg_resolves(self):
        """Material('N-BK7', catalog='schott') resolves without error."""
        m = Material("N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        assert m.n(0.55) > 1.4

    def test_material_catalog_exact_match_no_warning(self):
        """Exact catalog match emits no OptilandMaterialWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Material("N-BK7", catalog="schott", match_policy=MatchPolicy.WARN)
        material_warnings = [
            x for x in w if issubclass(x.category, OptilandMaterialWarning)
        ]
        assert len(material_warnings) == 0

    def test_material_catalog_invalid_raises(self):
        """Unknown catalog raises ValueError."""
        with pytest.raises(ValueError, match="No catalog"):
            Material("N-BK7", catalog="nonexistent_catalog_xyz")

    def test_material_catalog_ohara_resolves(self):
        """catalog='ohara' restricts to Ohara glasses."""
        m = Material("S-BSM2", catalog="ohara", match_policy=MatchPolicy.BEST)
        assert m.n(0.55) > 1.0

    def test_material_catalog_stored_as_attribute(self):
        """The catalog value is stored on the instance."""
        m = Material("N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        assert m._catalog == "schott"


# ---------------------------------------------------------------------------
# catalog fuzzy fallback warning
# ---------------------------------------------------------------------------


class TestCatalogFuzzyWarning:
    def test_material_catalog_fuzzy_warning_emitted(self):
        """Fuzzy match within a catalog always emits OptilandMaterialWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # "N-BK" is not an exact match for "N-BK7" → fuzzy fallback
            Material("N-BK", catalog="schott", match_policy=MatchPolicy.WARN)
        mat_warns = [x for x in w if issubclass(x.category, OptilandMaterialWarning)]
        assert len(mat_warns) >= 1

    def test_material_catalog_fuzzy_strict_raises(self):
        """match_policy='strict' raises on fuzzy-within-catalog."""
        with pytest.raises(ValueError, match="No exact match"):
            Material("N-BK", catalog="schott", match_policy=MatchPolicy.STRICT)


# ---------------------------------------------------------------------------
# MatchPolicy without catalog
# ---------------------------------------------------------------------------


class TestMatchPolicy:
    def test_match_policy_best_no_warning(self):
        """match_policy='best' suppresses OptilandMaterialWarning on fuzzy match."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Material("N-BK", match_policy=MatchPolicy.BEST)
        mat_warns = [x for x in w if issubclass(x.category, OptilandMaterialWarning)]
        assert len(mat_warns) == 0

    def test_match_policy_warn_emits_warning(self):
        """match_policy='warn' emits OptilandMaterialWarning on fuzzy match."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Material("N-BK", match_policy=MatchPolicy.WARN)
        mat_warns = [x for x in w if issubclass(x.category, OptilandMaterialWarning)]
        assert len(mat_warns) >= 1

    def test_match_policy_strict_raises(self):
        """match_policy='strict' raises ValueError on fuzzy match."""
        with pytest.raises(ValueError, match="No exact match"):
            Material("N-BK", match_policy=MatchPolicy.STRICT)

    def test_match_policy_string_values_accepted(self):
        """MatchPolicy accepts string values ('best', 'warn', 'strict')."""
        assert MatchPolicy("best") == MatchPolicy.BEST
        assert MatchPolicy("warn") == MatchPolicy.WARN
        assert MatchPolicy("strict") == MatchPolicy.STRICT

    def test_exact_match_no_warning_default_policy(self):
        """Exact name match never emits OptilandMaterialWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Material("N-BK7")
        mat_warns = [x for x in w if issubclass(x.category, OptilandMaterialWarning)]
        assert len(mat_warns) == 0


# ---------------------------------------------------------------------------
# robust_search deprecation
# ---------------------------------------------------------------------------


class TestRobustSearchDeprecated:
    def test_robust_search_true_emits_deprecation(self):
        """Passing robust_search=True emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Material("N-BK7", robust_search=True)
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warns) == 1
        assert "robust_search" in str(dep_warns[0].message)

    def test_robust_search_false_emits_deprecation(self):
        """Passing robust_search=False emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # DeprecationWarning emitted before resolution; ValueError may also
            # be raised for ambiguous matches under STRICT — ignore it here.
            try:
                Material("N-BK7", robust_search=False)
            except ValueError:
                pass
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warns) == 1

    def test_robust_search_true_maps_to_best(self):
        """robust_search=True maps to MatchPolicy.BEST (silent fuzzy)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Material("N-BK", robust_search=True)
        mat_warns = [x for x in w if issubclass(x.category, OptilandMaterialWarning)]
        assert len(mat_warns) == 0

    def test_robust_search_false_maps_to_strict(self):
        """robust_search=False maps to MatchPolicy.STRICT."""
        with pytest.raises(ValueError, match="No exact match"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                Material("N-BK", robust_search=False)

    def test_robust_search_none_no_deprecation(self):
        """robust_search=None (default) does not emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Material("N-BK7")
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warns) == 0


# ---------------------------------------------------------------------------
# from_dict backward compat
# ---------------------------------------------------------------------------


class TestFromDictCompat:
    def test_from_dict_robust_search_true(self):
        """Old dict with robust_search=True: no robust_search DeprecationWarning,
        but a catalog DeprecationWarning is expected (PR2 behavior)."""
        data = {
            "type": "Material",
            "name": "N-BK7",
            "reference": None,
            "robust_search": True,
            "min_wavelength": None,
            "max_wavelength": None,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = Material.from_dict(data)
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        # One DeprecationWarning for missing catalog field (PR2); none for robust_search
        assert all("catalog" in str(dw.message) for dw in dep_warns)
        assert m.n(0.55) > 1.4

    def test_from_dict_robust_search_false(self):
        """Old dict with robust_search=False loads as STRICT policy."""
        # Use a glass with a unique exact match so STRICT doesn't raise here.
        data = {
            "type": "Material",
            "name": "ZERODUR",
            "reference": None,
            "robust_search": False,
            "min_wavelength": None,
            "max_wavelength": None,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Material.from_dict(data)
        assert m._match_policy == MatchPolicy.STRICT

    def test_from_dict_match_policy_field(self):
        """New dict with match_policy and catalog fields loads without warnings."""
        data = {
            "type": "Material",
            "name": "N-BK7",
            "catalog": "schott",
            "reference": None,
            "match_policy": "best",
            "min_wavelength": None,
            "max_wavelength": None,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = Material.from_dict(data)
        dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warns) == 0
        assert m._match_policy == MatchPolicy.BEST

    def test_from_dict_catalog_field(self):
        """New dict with catalog field loads correctly."""
        data = {
            "type": "Material",
            "name": "N-BK7",
            "catalog": "schott",
            "reference": None,
            "match_policy": "warn",
            "min_wavelength": None,
            "max_wavelength": None,
        }
        m = Material.from_dict(data)
        assert m._catalog == "schott"


# ---------------------------------------------------------------------------
# MaterialSpec
# ---------------------------------------------------------------------------


class TestMaterialSpec:
    def test_material_spec_to_material_returns_material(self):
        """MaterialSpec.to_material() returns a Material instance."""
        spec = MaterialSpec(name="N-BK7", catalog="schott")
        m = spec.to_material()
        assert isinstance(m, Material)

    def test_material_spec_to_material_correct_index(self):
        """MaterialSpec.to_material() resolves to correct glass."""
        spec = MaterialSpec(name="N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        m = spec.to_material()
        assert abs(m.n(0.5876) - 1.5168) < 0.001

    def test_material_spec_frozen(self):
        """MaterialSpec is immutable (frozen dataclass)."""
        spec = MaterialSpec(name="N-BK7")
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "other"  # type: ignore[misc]

    def test_material_spec_hashable(self):
        """MaterialSpec can be used as a dict key / in a set."""
        spec1 = MaterialSpec(name="N-BK7", catalog="schott")
        spec2 = MaterialSpec(name="N-BK7", catalog="schott")
        assert hash(spec1) == hash(spec2)
        s = {spec1, spec2}
        assert len(s) == 1

    def test_material_spec_to_dict(self):
        """MaterialSpec.to_dict() round-trips correctly."""
        spec = MaterialSpec(
            name="N-BK7",
            catalog="schott",
            match_policy=MatchPolicy.BEST,
        )
        d = spec.to_dict()
        assert d["name"] == "N-BK7"
        assert d["catalog"] == "schott"
        assert d["match_policy"] == "best"

    def test_material_spec_from_dict(self):
        """MaterialSpec.from_dict() deserializes correctly."""
        d = {
            "name": "N-BK7",
            "catalog": "schott",
            "match_policy": "warn",
        }
        spec = MaterialSpec.from_dict(d)
        assert spec.name == "N-BK7"
        assert spec.catalog == "schott"
        assert spec.match_policy == MatchPolicy.WARN

    def test_material_spec_from_dict_defaults(self):
        """MaterialSpec.from_dict() applies sensible defaults for missing keys."""
        spec = MaterialSpec.from_dict({"name": "N-BK7"})
        assert spec.catalog is None
        assert spec.match_policy == MatchPolicy.WARN


# ---------------------------------------------------------------------------
# MaterialFactory extensions
# ---------------------------------------------------------------------------


class TestMaterialFactoryExtensions:
    def test_factory_accepts_material_spec(self):
        """MaterialFactory accepts MaterialSpec input."""
        factory, sg = _make_factory()
        spec = MaterialSpec(name="N-BK7", catalog="schott", match_policy=MatchPolicy.BEST)
        mat_pre, mat_post = factory.create(1, spec, sg)
        assert mat_post is not None
        assert mat_post.n(0.55) > 1.4

    def test_factory_accepts_3_tuple(self):
        """MaterialFactory accepts (name, reference, catalog) 3-tuple."""
        factory, sg = _make_factory()
        mat_pre, mat_post = factory.create(1, ("N-BK7", None, "schott"), sg)
        assert mat_post is not None
        assert mat_post.n(0.55) > 1.4

    def test_factory_accepts_2_tuple_unchanged(self):
        """Existing 2-tuple (name, reference) still works."""
        factory, sg = _make_factory()
        mat_pre, mat_post = factory.create(1, ("N-BK7", None), sg)
        assert mat_post is not None

    def test_factory_accepts_dict(self):
        """MaterialFactory accepts a dict (MaterialSpec.from_dict path)."""
        factory, sg = _make_factory()
        spec_dict = {
            "name": "N-BK7",
            "catalog": "schott",
            "match_policy": "best",
        }
        mat_pre, mat_post = factory.create(1, spec_dict, sg)
        assert mat_post is not None
        assert mat_post.n(0.55) > 1.4

    def test_factory_3_tuple_invalid_length_raises(self):
        """MaterialFactory raises on a tuple with invalid length."""
        factory, sg = _make_factory()
        with pytest.raises(ValueError, match="2 or 3 elements"):
            factory.create(1, ("a", "b", "c", "d"), sg)

    def test_factory_accepts_material_spec_instance_directly(self):
        """MaterialFactory uses BaseMaterial instance directly."""
        from optiland.materials import IdealMaterial

        factory, sg = _make_factory()
        ideal = IdealMaterial(n=1.5, k=0.0)
        mat_pre, mat_post = factory.create(1, ideal, sg)
        assert mat_post is ideal
