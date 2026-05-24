# spec_materials.md — Material System Overhaul

**Status:** Draft  
**Date:** 2026-05-21  
**Scope:** `optiland/materials/`, `optiland/surfaces/factories/material_factory.py`, serialization layer

---

## 1. Motivation and Goals

The current `Material("N-BK7")` lookup is fuzzy-only: it applies Levenshtein distance across a 3,200-entry refractiveindex.info CSV. The caller cannot disambiguate by manufacturer (Schott vs Ohara vs generic), warnings are unclear, there is no discovery API, and the `reference` parameter is misunderstood as a catalog filter when it is actually a citation string. Users have no way to add custom materials without forking the database.

**Goals:**

1. Add `catalog=` (optional, keyword-only) to `Material` for manufacturer-scoped lookup.
2. Replace the opaque `robust` flag with a three-value `MatchPolicy` enum.
3. Introduce `MaterialSpec` as the canonical type-safe spec for surface material assignment.
4. Introduce `MaterialRegistry` (global singleton, zero cost on existing paths) and `MaterialCatalog` (view/proxy).
5. Support user-defined custom materials via both file and programmatic registration.
6. Emit a custom `OptilandMaterialWarning` on ambiguous matches; update serialization to warn on missing catalog.
7. **API frozen**: all changes are additive. No positional-signature changes to `Material`.

---

## 2. Decisions Log

| Question | Decision |
|---|---|
| Catalog granularity | `catalog` maps to `category_name` in `catalog_nk.csv` (e.g. `"schott"`, `"ohara"`) |
| `reference` param | Keep; add `catalog` as a separate keyword-only parameter |
| Match mode when catalog given | Exact case-insensitive first; fall back to fuzzy with warning |
| `robust` parameter | Replace with `MatchPolicy` enum (`"best"`, `"warn"`, `"strict"`); `"warn"` is the only default, no global override |
| `MaterialSpec` | New dataclass; `MaterialFactory` accepts it alongside legacy string/tuple |
| MaterialCatalog vs Registry | `MaterialRegistry` is global singleton; `MaterialCatalog` is a lightweight proxy view into it |
| User catalog format | Exact refractiveindex.info YAML; hybrid registration (file + programmatic) |
| Conflict resolution | User-registered entry wins; emit `OptilandMaterialWarning` |
| Serialization compat | Old files lacking `catalog` field load fine; emit `DeprecationWarning` |
| Warning mechanism | Custom `OptilandMaterialWarning(UserWarning)` via `warnings.warn` |
| PR phasing | Two PRs: (1) Foundation — `MaterialSpec`, `catalog` kwarg, `MatchPolicy`; (2) Features — `MaterialCatalog`, `MaterialRegistry`, user catalog, serialization compat |
| Material optimization | Out of scope |

---

## 3. New Files and Classes

### 3.1 `optiland/materials/warnings.py`

```python
class OptilandMaterialWarning(UserWarning):
    """Emitted when a material lookup resolves via fuzzy match or has ambiguity."""
```

Users can silence with:

```python
import warnings
warnings.filterwarnings("ignore", category=OptilandMaterialWarning)
```

---

### 3.2 `optiland/materials/material_spec.py`

#### `MatchPolicy` (StrEnum)

```python
class MatchPolicy(str, enum.Enum):
    BEST   = "best"    # Silent best-match; no warning emitted
    WARN   = "warn"    # Default: warn when fuzzy match used (edit distance > 0)
    STRICT = "strict"  # Raise ValueError on any non-exact or ambiguous match
```

`"warn"` is the system-wide default. There is no global override — callers opt into `"best"` per instantiation.

#### `MaterialSpec` (dataclass)

```python
@dataclasses.dataclass(frozen=True)
class MaterialSpec:
    name: str
    catalog: str | None = None
    reference: str | None = None
    match_policy: MatchPolicy = MatchPolicy.WARN
    min_wavelength: float | None = None
    max_wavelength: float | None = None

    def to_material(self) -> "Material":
        """Resolve this spec to a concrete Material instance."""
        ...

    def to_dict(self) -> dict:
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialSpec":
        ...
```

`MaterialSpec` is `frozen=True` so it is hashable and safe to cache. It is the canonical way to specify a material in surface-assignment code. The legacy string and tuple formats remain supported in `MaterialFactory`.

---

### 3.3 `optiland/materials/registry.py` — `MaterialRegistry`

`MaterialRegistry` is a **global singleton** that owns the complete lookup table for both built-in and user-registered materials. It wraps the existing `catalog_nk.csv` load (currently done ad-hoc in `Material`) and adds user catalog state.

**Critical performance constraint:** the singleton must not add any overhead to existing `Material("N-BK7")` call paths. All additional state is layered on top; the hot path (built-in CSV lookup) must remain as fast as today.

#### Class structure

```python
class MaterialRegistry:
    """Global singleton. Access via MaterialRegistry.instance()."""

    # ---- construction ----
    @classmethod
    def instance(cls) -> "MaterialRegistry": ...

    # ---- built-in catalog ----
    @property
    def built_in_df(self) -> pd.DataFrame:
        """Lazy-loaded, cached once per process."""
        ...

    # ---- user catalog ----
    def register(
        self,
        name: str,
        catalog: str,
        data: dict,           # refractiveindex.info YAML payload as dict
    ) -> None:
        """Register a material programmatically. Warns if shadowing a built-in."""
        ...

    def register_file(self, path: str | Path) -> None:
        """Load a single refractiveindex.info-format YAML file.
        
        The catalog name is inferred from the YAML's REFERENCE field or
        from the parent directory name if the REFERENCE field is absent.
        The material name is inferred from the filename (without extension).
        """
        ...

    def load_catalog(self, directory: str | Path) -> None:
        """Load all YAML files found in directory.
        
        If a catalog.csv index exists in the directory (same schema as the
        built-in catalog_nk.csv), it is used. Otherwise, each YAML file is
        individually registered via register_file().
        """
        ...

    # ---- resolution ----
    def resolve(
        self,
        name: str,
        catalog: str | None = None,
        reference: str | None = None,
        match_policy: MatchPolicy = MatchPolicy.WARN,
        min_wavelength: float | None = None,
        max_wavelength: float | None = None,
    ) -> str:
        """Return the absolute path to the resolved YAML data file.
        
        Raises ValueError if no match found or match_policy='strict' and
        the match is not exact.
        Emits OptilandMaterialWarning if match_policy='warn' and fuzzy match used.
        """
        ...

    # ---- discovery ----
    def list_catalogs(self) -> list[str]:
        """Return sorted unique catalog names (built-in + user-registered)."""
        ...

    def list_materials(self, catalog: str | None = None) -> list[str]:
        """Return sorted material names, optionally filtered to one catalog."""
        ...
```

#### Auto-discovery of user catalogs

On the **first call to `MaterialRegistry.instance()`**, the registry checks for the directory `~/.optiland/catalogs/`. If it exists, `load_catalog()` is called on each subdirectory. This is a best-effort load; failures emit `OptilandMaterialWarning` and are skipped, never crashing the import.

#### Lookup algorithm

```
resolve(name, catalog, reference, match_policy, min_wl, max_wl):

1. Build candidate DataFrame:
   a. Start with built-in_df (user entries are merged on top, shadowing by (name, catalog) key).
   b. If catalog is given: filter rows where category_name == catalog (case-insensitive).
   c. If reference is given: further filter by reference field.
   d. If min/max wavelength given: filter by wavelength range columns.

2. If catalog is given:
   a. Try exact case-insensitive match on (name, filename_no_ext, category_name).
   b. If unique exact match found → return path (no warning).
   c. If no exact match → fall through to fuzzy within the filtered candidate set.
      On fuzzy hit: always emit OptilandMaterialWarning (regardless of match_policy)
      showing the resolved name and catalog.
   d. If match_policy='strict' and no exact match → raise ValueError.

3. If catalog is None (or after step 2c falls through):
   a. Apply Levenshtein distance across candidates (current algorithm).
   b. If best match has edit distance == 0 → return path (no warning).
   c. If edit distance > 0:
      - match_policy='best'  → return path silently.
      - match_policy='warn'  → emit OptilandMaterialWarning; return path.
      - match_policy='strict'→ raise ValueError listing top candidates.

4. If no candidates → raise ValueError.
```

**User-registered entries shadow built-ins** for the same `(name, catalog)` key. When shadowing occurs, `OptilandMaterialWarning` is emitted at **registration time** (not at lookup time).

---

### 3.4 `optiland/materials/catalog.py` — `MaterialCatalog`

`MaterialCatalog` is a thin, read-only proxy into `MaterialRegistry`. It holds no state beyond a catalog name string.

```python
class MaterialCatalog:
    """Read-only view into one catalog within the MaterialRegistry.
    
    Usage:
        MaterialCatalog.available()         # list all catalog names
        MaterialCatalog("schott").list()    # list glass names in Schott catalog
        MaterialCatalog("schott").search("bk7")  # fuzzy search within catalog
        MaterialCatalog("schott").get("N-BK7")   # return a Material instance
    """

    def __init__(self, catalog: str) -> None: ...

    @classmethod
    def available(cls) -> list[str]:
        """Return sorted list of all catalog names (built-in + user)."""
        return MaterialRegistry.instance().list_catalogs()

    def list(self) -> list[str]:
        """Return sorted list of material names in this catalog."""
        return MaterialRegistry.instance().list_materials(self._catalog)

    def search(self, name: str, n: int = 10) -> list[str]:
        """Return top-n fuzzy matches for name within this catalog."""
        ...

    def get(self, name: str, match_policy: MatchPolicy = MatchPolicy.WARN) -> "Material":
        """Instantiate a Material from this catalog by name."""
        return Material(name, catalog=self._catalog, match_policy=match_policy)
```

`MaterialCatalog` is exported from `optiland.materials` but **not** re-exported in `optiland.__init__`.

---

## 4. Changes to Existing Classes

### 4.1 `Material` (`optiland/materials/material.py`)

#### Signature change (additive only)

```python
class Material(MaterialFile):
    def __init__(
        self,
        name: str,
        reference: str | None = None,              # unchanged positional-2
        robust_search: bool | None = None,         # deprecated; None = use match_policy
        min_wavelength: float | None = None,
        max_wavelength: float | None = None,
        *,                                          # keyword-only boundary
        catalog: str | None = None,                # NEW
        match_policy: MatchPolicy = MatchPolicy.WARN,  # NEW; replaces robust_search
    ) -> None:
```

**Backward compat:** `reference` stays as positional-2. `catalog` is keyword-only so it cannot collide with existing call sites. `robust_search` is deprecated but still accepted; passing it emits `DeprecationWarning` and internally maps:
- `robust_search=True` → `match_policy=MatchPolicy.BEST` (silent)
- `robust_search=False` → `match_policy=MatchPolicy.STRICT`

The resolution logic is delegated to `MaterialRegistry.instance().resolve(...)`. The existing `_retrieve_file` private method is kept as an internal shim during transition and can be removed in PR2.

#### `to_dict` / `from_dict`

```python
def to_dict(self) -> dict:
    return {
        "type": "Material",
        "name": self._name,
        "reference": self._reference,  # None if not set
        "catalog": self._catalog,      # None if not set — NEW field
        "match_policy": self._match_policy.value,  # NEW field
        "robust_search": None,         # omitted / always None in new files
        "min_wavelength": ...,
        "max_wavelength": ...,
        "propagation_model": ...,
    }

@classmethod
def from_dict(cls, data: dict) -> "Material":
    if "catalog" not in data or data["catalog"] is None:
        warnings.warn(
            f"Material '{data['name']}' loaded from file has no 'catalog' field. "
            "Re-save the lens file to record catalog information. "
            "Lookup will fall back to fuzzy search.",
            DeprecationWarning,
            stacklevel=2,
        )
    ...
```

---

### 4.2 `MaterialFactory` (`optiland/surfaces/factories/material_factory.py`)

The factory's `create()` method signature and return type are unchanged. The supported `material_spec` types are extended:

| Input type | Existing? | Behavior |
|---|---|---|
| `BaseMaterial` instance | Yes | Used directly |
| `MaterialSpec` dataclass | **NEW** | `spec.to_material()` |
| `str` `"air"` | Yes | `IdealMaterial(1.0, 0.0)` |
| `str` `"mirror"` | Yes | reflection: both sides same material |
| `str` other | Yes | `Material(spec)` with default `match_policy=MatchPolicy.WARN` |
| `tuple[str, str]` | Yes | `Material(name, reference)` |
| `tuple[str, str, str]` | **NEW** | `Material(name, reference, catalog=catalog)` |
| `dict` | **NEW** | `MaterialSpec.from_dict(spec).to_material()` |

The 3-tuple form `(name, reference, catalog)` is the minimal extension for existing users who use tuple-based surface assignment.

---

## 5. User Catalog

### 5.1 File-based (persistent)

Users place refractiveindex.info-format YAML files under `~/.optiland/catalogs/<catalog_name>/`. The directory name becomes the `catalog` value. On registry first-access, each subdirectory is ingested via `load_catalog()`.

Example:
```
~/.optiland/catalogs/
  my_company/
    my_glass_A.yml    # refractiveindex.info YAML
    my_glass_B.yml
    catalog.csv       # optional index (same schema as catalog_nk.csv)
```

After this, `Material("my_glass_A", catalog="my_company")` resolves directly.

### 5.2 Programmatic (session-scoped)

```python
from optiland.materials import MaterialRegistry
import numpy as np

registry = MaterialRegistry.instance()
registry.register(
    name="MyGlass",
    catalog="internal",
    data={                          # refractiveindex.info YAML payload
        "REFERENCE": "Internal measurement, 2024",
        "DATA": [{
            "type": "tabulated n",
            "data": "0.4 1.52\n0.55 1.51\n0.7 1.505\n"
        }]
    }
)

# Now resolvable this session:
m = Material("MyGlass", catalog="internal")
```

### 5.3 Conflict rules

- User entry with same `(name, catalog)` as a built-in: **user wins**; `OptilandMaterialWarning` emitted at `register()` time.
- User entry with same `name` but **different** catalog: no conflict; both coexist.
- Case: two user registrations with same `(name, catalog)`: second call overwrites; `OptilandMaterialWarning` emitted.

### 5.4 YAML format

User YAML files must follow the exact refractiveindex.info schema. This enables direct copy-paste from refractiveindex.info without conversion. Optiland reads the same fields it reads from built-in data (formula type, coefficients, tabulated data, SPECS for `nd`/`Vd`, thermal dispersion). Unknown YAML keys are silently ignored.

---

## 6. Discovery UX

### Finding catalogs

```python
from optiland.materials import MaterialCatalog

# What catalogs exist?
MaterialCatalog.available()
# ['ami', 'cdgm', 'hikari', 'hoya', 'infrared', 'nikon', 'ohara',
#  'schott', 'sumita', 'vitron', ...]

# What glasses are in Schott?
MaterialCatalog("schott").list()
# ['F2', 'F5', 'FK5', 'K10', 'N-BK10', 'N-BK7', 'N-F2', ...]

# Fuzzy search within a catalog
MaterialCatalog("schott").search("bk7")
# ['N-BK7', 'N-BK10', 'N-K5', ...]

# Get a Material directly
glass = MaterialCatalog("schott").get("N-BK7")
```

### On the `Material` repr

`Material.__repr__` is updated to show resolved name, catalog, and wavelength range:

```
Material(name='N-BK7', catalog='schott', λ=[0.31µm, 2.50µm])
```

---

## 7. Warning Inventory

| Condition | Warning class | Message pattern |
|---|---|---|
| Fuzzy match used (edit dist > 0) with `match_policy='warn'` | `OptilandMaterialWarning` | `"Material 'nbk7' resolved to 'N-BK7' (catalog='schott') via fuzzy match."` |
| Catalog given, no exact match, fell back to fuzzy | `OptilandMaterialWarning` | `"No exact match for 'NBK7' in catalog 'schott'; resolved to 'N-BK7'. Use exact name to silence."` |
| User registration shadows built-in | `OptilandMaterialWarning` | `"User-registered material 'N-BK7' (catalog='schott') shadows a built-in entry."` |
| `robust_search` kwarg used | `DeprecationWarning` | `"robust_search is deprecated; use match_policy='strict' or match_policy='best'."` |
| Deserializing Material with no `catalog` field | `DeprecationWarning` | `"Material 'N-BK7' loaded from file has no 'catalog' field. Re-save to record catalog."` |

---

## 8. Serialization Format

### New format (PR1+)

```json
{
  "type": "Material",
  "name": "N-BK7",
  "catalog": "schott",
  "reference": null,
  "match_policy": "warn",
  "min_wavelength": null,
  "max_wavelength": null,
  "propagation_model": { ... }
}
```

### Legacy format (pre-spec)

```json
{
  "type": "Material",
  "name": "N-BK7",
  "reference": "Schott 2023 data sheet",
  "robust_search": true,
  "min_wavelength": null,
  "max_wavelength": null,
  "propagation_model": { ... }
}
```

`from_dict` handles both: `robust_search` is mapped to `match_policy` if present; missing `catalog` emits `DeprecationWarning`; missing `match_policy` defaults to `MatchPolicy.WARN`.

---

## 9. Module Exports

### `optiland/materials/__init__.py` additions

```python
from optiland.materials.warnings import OptilandMaterialWarning
from optiland.materials.material_spec import MaterialSpec, MatchPolicy
from optiland.materials.catalog import MaterialCatalog    # PR2
from optiland.materials.registry import MaterialRegistry  # PR2
```

**`optiland/__init__.py`** is not modified. `MaterialCatalog`, `MaterialRegistry`, `MaterialSpec`, and `MatchPolicy` are available via `optiland.materials` only.

---

## 10. SOLID Compliance Notes

| Principle | How satisfied |
|---|---|
| SRP | `MaterialRegistry` owns lookup state; `MaterialCatalog` owns discovery UX; `Material` owns instantiation + caching; `MaterialSpec` owns parameter bundling |
| OCP | New `MatchPolicy`, `catalog`, and user catalogs extend behavior without modifying `BaseMaterial` or `MaterialFile` |
| LSP | All new classes are standalone; no existing subclass contracts are altered |
| ISP | `MaterialCatalog` is a narrow read-only interface; `MaterialRegistry` exposes separate registration and resolution interfaces |
| DIP | `Material` will delegate resolution to `MaterialRegistry.instance()` (an abstract dependency) rather than directly querying the CSV |

---

## 11. PR Phasing

### PR1 — Foundation (additive only, zero removals)

Files changed:
- `optiland/materials/warnings.py` (new)
- `optiland/materials/material_spec.py` (new)
- `optiland/materials/material.py` — add `catalog` kwarg, `match_policy`, deprecate `robust_search`; keep `_retrieve_file` shim
- `optiland/surfaces/factories/material_factory.py` — accept `MaterialSpec`, 3-tuple
- `optiland/materials/__init__.py` — export `MaterialSpec`, `MatchPolicy`, `OptilandMaterialWarning`
- Tests for all of the above

**What PR1 does NOT do:** introduce `MaterialRegistry` or `MaterialCatalog`. Material resolution still goes through the existing `_retrieve_file` path (now wrapped to honor `catalog` and `match_policy`).

### PR2 — Features

Files changed:
- `optiland/materials/registry.py` (new) — `MaterialRegistry` singleton
- `optiland/materials/catalog.py` (new) — `MaterialCatalog`
- `optiland/materials/material.py` — delegate `_retrieve_file` to `MaterialRegistry.instance().resolve()`; add `from_dict` `DeprecationWarning` for missing catalog; update `to_dict` to emit `catalog`; update `__repr__`
- `optiland/materials/__init__.py` — export `MaterialCatalog`, `MaterialRegistry`
- Tests for registry, catalog, user registration, conflict resolution, serialization compat

---

## 12. Tests Required

### PR1

- `test_material_catalog_kwarg`: `Material("N-BK7", catalog="schott")` resolves correctly
- `test_material_catalog_exact_match_no_warning`: no warning when exact catalog match
- `test_material_catalog_fuzzy_warning`: warning emitted when fuzzy-within-catalog fallback used
- `test_match_policy_best_no_warning`: `match_policy="best"` suppresses fuzzy warning
- `test_match_policy_strict_raises`: `match_policy="strict"` raises on fuzzy match
- `test_robust_search_deprecated`: passing `robust_search` emits `DeprecationWarning`
- `test_material_spec_to_material`: `MaterialSpec.to_material()` returns correct `Material`
- `test_material_factory_accepts_material_spec`: factory accepts `MaterialSpec` input
- `test_material_factory_accepts_3_tuple`: factory accepts `(name, ref, catalog)` 3-tuple

### PR2

- `test_registry_singleton`: repeated `.instance()` returns same object
- `test_registry_list_catalogs`: all expected catalog names present
- `test_registry_list_materials_filtered`: materials filtered correctly by catalog
- `test_registry_register_programmatic`: user-registered material resolves correctly
- `test_registry_register_shadow_warning`: shadowing built-in emits `OptilandMaterialWarning`
- `test_registry_register_file`: YAML file loaded and resolved correctly
- `test_registry_load_catalog_dir`: directory of YAML files all loaded
- `test_registry_user_wins_conflict`: user entry returned when same (name, catalog) as built-in
- `test_material_catalog_available`: `MaterialCatalog.available()` returns list of strings
- `test_material_catalog_list`: `.list()` returns non-empty list for known catalog
- `test_material_catalog_search`: `.search("bk7")` returns "N-BK7" in results
- `test_material_catalog_get`: `.get("N-BK7")` returns a `Material` instance
- `test_serialization_new_format`: `to_dict` includes `catalog` field
- `test_serialization_legacy_load_warning`: `from_dict` without `catalog` emits `DeprecationWarning`
- `test_serialization_round_trip`: serialize → deserialize preserves name/catalog/match_policy

---

## 13. Non-Goals (explicit)

- Material optimization / discrete glass variables — deferred
- GUI glass-picker dialog — deferred (separate spec)
- Zemax `.agf` import — deferred
- `optiland/__init__.py` re-exports for `MaterialCatalog` or `MaterialRegistry`
- Thermal dispersion for user-registered materials (existing `MaterialFile` logic handles this already if SPECS section is present in YAML)
