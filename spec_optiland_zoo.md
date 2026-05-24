# `optiland-zoo` Implementation Specification

**Repository:** `github.com/HarrisonKramer/optiland-zoo`  
**Status:** Pre-launch — full bootstrap before first public push  
**Author:** Harrison Kramer  
**Date:** 2026-05-23

---

## 1. Overview and Goals

`optiland-zoo` is a curated, self-validating catalog of optical lens designs in Optiland JSON format. It serves three audiences:

1. **Optical engineers** who want a starting-point design for a given specification (EFL, F/#, FOV, category).
2. **Researchers and ML practitioners** who need a clean, validated corpus of real lens designs.
3. **Optiland contributors** who want reference designs for testing new analysis features.

The zoo is a **first-class Python package** (`pip install optiland-zoo`) with a programmatic catalog API, a GitHub Pages browsing gallery, and a fully automated CI/CD pipeline. Correctness and provenance are non-negotiable: every design in the zoo has been successfully ray-traced using its own field/pupil definitions, and every design is unambiguously attributed to its source.

---

## 2. Data Scope

### 2.1 Source Data

Source data lives at `C:\Users\kdani\Documents\Python_Scripts\LensAI\data` and contains ~9,500 files. The following categories are **included**:

| Source directory | Category slug | Approximate count |
|---|---|---|
| `Photographic lenses - prime/` | `prime` | ~1,980 |
| `Photographic lenses - zoom/` | `zoom` | ~615 |
| `Microscope objectives/` | `microscope` | ~196 |
| `Eyepieces/` | `eyepiece` | ~167 |
| `Telescopes/` | `telescopes` | ~181 |
| `Projectors/` | `projector` | ~50 |

The following are **excluded**:

- **`Augmented_Lenses/`** — 5,573 hex-named synthetic/perturbed variants. Not canonical designs.
- **`Cleaned_Spherical_Lenses/`** — Revised copies of designs already present in the included categories. Including both would introduce duplicates.

### 2.2 Format Priority and Deduplication

Source files appear in `.json`, `.zmx`, `.len`, `.seq`, and other formats. When multiple formats represent the same design (same base filename):

**JSON wins.** If a `<name>.json` already exists for a design, do not convert `<name>.zmx`. Only convert non-JSON formats when no JSON counterpart is present.

### 2.3 ZMX/LEN Conversion Strategy

Run Optiland's Zemax importer on all `.zmx` and `.len` files that have no JSON counterpart. Use **best-effort conversion**:

- If the importer succeeds and produces a loadable `Optic`, write the resulting JSON.
- If the importer raises any exception, record the failure in `scripts/conversion_failures.csv` (columns: `source_file`, `error_type`, `error_message`) and skip the file.
- Conversion failures are not fatal — they are tracked for future revisitation.

The conversion script does **not** rename files. The output JSON filename matches the source filename stem: `CH321571_Example01P.zmx` → `CH321571_Example01P.json`.

---

## 3. Repository Structure

```
optiland-zoo/
├── data/
│   ├── prime/
│   ├── zoom/
│   ├── microscope/
│   ├── eyepiece/
│   ├── telescopes/
│   └── projector/
│
├── docs/
│   ├── assets/
│   │   └── featured/          # ~50–100 curated showcase PNGs (committed to main)
│   ├── overrides/             # MkDocs theme overrides and Jinja macros
│   └── index.md               # Gallery home (auto-generated from registry.json)
│
├── scripts/
│   ├── bootstrap.py           # One-time ingestion: convert + validate + populate registry
│   ├── generate_registry.py   # Incremental: validate changed files, update registry.json
│   ├── generate_gallery.py    # Regenerate MkDocs Markdown pages from registry.json
│   ├── convert_zmx.py         # ZMX/LEN → Optiland JSON converter
│   └── validate_design.py     # Single-design validation logic (Level 3)
│
├── optiland_zoo/
│   ├── __init__.py
│   ├── catalog.py             # Public API: load(), search(), list()
│   ├── registry.py            # Registry loader and CatalogEntry dataclass
│   └── _version.py
│
├── tests/
│   └── test_catalog.py
│
├── registry.json              # Master metadata index (auto-generated, committed)
├── mkdocs.yml
├── pyproject.toml
├── CREDITS.md                 # Full attribution manifest
├── CONTRIBUTING.md
├── LICENSE                    # Repo-level MIT license (for scripts and package code)
└── .github/
    └── workflows/
        ├── validate.yml       # PR validation
        └── publish.yml        # Post-merge registry rebuild + Pages deploy
```

---

## 4. Attribution and Licensing

**The JSON design files are not modified.** Attribution lives in companion files and a top-level manifest.

### 4.1 Per-Category LICENSE Files

Each `data/<category>/` directory contains a `LICENSE` file specifying the license covering the designs in that directory. For designs sourced from lens-designs.com:

```
MIT License
Copyright (c) 2014, Daniel J. Reiley
...
```

For designs from other sources (e.g., public patents, other repositories), the appropriate license or public-domain declaration is used.

### 4.2 CREDITS.md

`CREDITS.md` at the repo root maps each source category and file group to its provenance:

```markdown
## lens-designs.com (MIT, Copyright © 2014 Daniel J. Reiley)
Source: https://lens-designs.com
License: MIT
Applies to: data/prime/, data/zoom/, data/microscope/, data/eyepiece/, data/telescopes/, data/projector/
...
```

### 4.3 Contributor Requirement

When a contributor adds a new design via PR, they must:
1. Place the JSON in the correct `data/<category>/` directory.
2. Confirm in their PR description that the design is under a compatible open-source license.

CI checks that new JSON files are accompanied by a license declaration in the PR description (via a PR template checklist), but does not block the build on absence of a sidecar file.

---

## 5. Registry Schema (`registry.json`)

`registry.json` is an array of `CatalogEntry` objects. Every field is documented and typed. The schema is **extensible by design** — new fields can be added without breaking existing consumers.

```jsonc
[
  {
    // --- Identity ---
    "slug": "CH321571_Example01P",          // filename stem (no extension)
    "name": "Cooke Triplet 50mm f/4",       // from Optiland JSON name field, or slug if absent
    "category": "prime",                     // one of: prime, zoom, microscope, eyepiece, telescopes, projector
    "source_file": "CH321571_Example01P.json",
    "data_path": "data/prime/CH321571_Example01P.json",

    // --- First-order parameters ---
    "efl_mm": 50.0,
    "fnum": 4.0,
    "fov_max_deg": 20.0,
    "num_elements": 3,
    "num_surfaces": 6,
    "wavelengths_um": [0.486, 0.587, 0.656],

    // --- Seidel aberrations (at primary wavelength) ---
    "seidel": {
      "S1": 0.002,   // spherical aberration
      "S2": -0.001,  // coma
      "S3": 0.0005,  // astigmatism
      "S4": 0.0001,  // field curvature
      "S5": -0.0003  // distortion
    },

    // --- Performance metrics (traced at design's own fields/pupils) ---
    "rms_spot_on_axis_um": 1.2,
    "rms_spot_0p7_field_um": 3.5,
    "rms_spot_full_field_um": 8.1,

    // --- Provenance ---
    "license": "MIT",
    "source_url": "https://lens-designs.com",
    "copyright": "Copyright (c) 2014, Daniel J. Reiley",

    // --- Validation state ---
    "checksum": "sha256:abc123...",    // SHA-256 of the JSON file contents
    "valid": true,
    "validation_errors": [],

    // --- Assets ---
    "image_path": null,               // null unless a featured PNG exists
    "featured": false,

    // --- Extension point ---
    "extra": {}                        // arbitrary dict for future fields
  }
]
```

### 5.1 Extensibility

`extra` is a free-form dict. Future additions (MTF, Strehl, glass catalog names, patent numbers) go here first. When a field stabilizes and is present for >80% of designs, it gets promoted to a top-level field in the next minor version of the schema.

---

## 6. Validation Pipeline

### 6.1 Level 3 Validation

Every design undergoes full real-ray-trace validation using its **own field and pupil definitions** (not a standardized set). This respects the design's operating conditions and produces physically meaningful metrics.

Validation steps for a single design:

1. **Parse**: `Optic.from_json(path)` — must not raise.
2. **Paraxial check**: EFL, F/#, and BFL must all be finite and positive (or, for reflective systems, finite).
3. **Real ray trace**: Trace rays at on-axis, 0.7-field, and full-field using the design's defined fields and entrance pupil. Compute RMS spot radius at each field point.
4. **Seidel extraction**: Compute first five Seidel coefficients via `optiland`'s paraxial aberration analysis.
5. **Record**: Write all metrics and `valid=True` to registry. If any step raises or produces non-finite values, write `valid=False` and record the error string.

### 6.2 Hash-Based Caching (Skip Re-Validation)

The SHA-256 of each design's JSON content is stored in `registry.json` as `checksum`. During incremental runs, a design is **skipped** if its current file hash matches the stored checksum. This means Level 3 validation runs at most once per design per content change — even across CI runs.

```python
# pseudocode in generate_registry.py
for entry in registry:
    current_hash = sha256(path.read_bytes())
    if current_hash == entry["checksum"]:
        continue  # no change, skip
    entry.update(validate_design(path))
    entry["checksum"] = current_hash
```

### 6.3 Validation Script Interface

`scripts/validate_design.py` is a standalone script for single-design testing:

```bash
python scripts/validate_design.py data/prime/CH321571_Example01P.json
```

Prints a JSON summary of validation results to stdout.

---

## 7. Python Package — `optiland_zoo`

### 7.1 Installation

```bash
pip install optiland-zoo
```

Published to PyPI. JSON data files are bundled with the package (`package_data` includes `data/**/*.json` and `registry.json`). No network access required for basic use.

### 7.2 Version Coupling

`pyproject.toml` pins an explicit lower bound on optiland:

```toml
[project]
dependencies = [
    "optiland >= 0.9.0",  # updated on each optiland API change that affects loading
]
```

`optiland-zoo` version is bumped whenever the optiland pin changes. The zoo's `CHANGELOG.md` documents which optiland version each zoo release targets.

### 7.3 Public API

```python
from optiland_zoo import catalog
```

#### `catalog.load(slug: str) -> optiland.Optic`

Load a single design by slug (filename stem). Returns a fully-initialized `Optic` object.

```python
lens = catalog.load("CH321571_Example01P")
```

Raises `KeyError` if the slug is not found. Raises `optiland.LoadError` if the JSON fails to parse (should not happen for validated entries, but possible after optiland upgrades).

---

#### `catalog.list() -> list[CatalogEntry]`

Return all valid registry entries as a list of `CatalogEntry` dataclasses.

```python
entries = catalog.list()
# or as a DataFrame:
import pandas as pd
df = pd.DataFrame([e.to_dict() for e in catalog.list()])
```

---

#### `catalog.search(...) -> list[CatalogEntry]`

Filter the catalog by one or more criteria.

```python
results = catalog.search(
    category="prime",
    efl_range=(45.0, 55.0),
    fnum_max=2.0,
    fov_min_deg=10.0,
    valid_only=True,   # default True
)
```

All parameters are optional and combinable. Returns a list of matching `CatalogEntry` objects.

---

#### `CatalogEntry` dataclass

```python
@dataclass
class CatalogEntry:
    slug: str
    name: str
    category: str
    efl_mm: float
    fnum: float
    fov_max_deg: float
    num_elements: int
    num_surfaces: int
    wavelengths_um: list[float]
    seidel: dict[str, float]
    rms_spot_on_axis_um: float
    rms_spot_0p7_field_um: float
    rms_spot_full_field_um: float
    license: str
    valid: bool
    image_path: str | None
    extra: dict         # forward-compatible extension point

    def load(self) -> "optiland.Optic":
        """Convenience shortcut for catalog.load(self.slug)."""

    def to_dict(self) -> dict:
        """Serialize to a plain dict (registry.json-compatible)."""
```

### 7.4 SOLID Design Notes

- **Single Responsibility**: `catalog.py` is only the public API. `registry.py` handles loading and searching. `validate_design.py` handles validation. Each is independently testable.
- **Open/Closed**: New metadata fields are added to `CatalogEntry.extra` without changing the class. When promoted to top-level, add with a default value — no breaking change.
- **Liskov/Interface**: `CatalogEntry` is a pure dataclass. Subclasses (e.g., `AnnotatedEntry` with human notes) can extend without breaking consumers.
- **Dependency Inversion**: `catalog.py` depends on the `Registry` abstraction, not a specific file path. Tests can inject a mock registry.

---

## 8. GitHub Pages Gallery

### 8.1 Technology

**MkDocs + Material theme** with custom Jinja macros. This is consistent with Optiland's documentation tooling and provides built-in search, mobile responsiveness, and a polished look with minimal custom CSS.

### 8.2 Build Process

`scripts/generate_gallery.py` reads `registry.json` and writes Markdown files into `docs/`:

- `docs/index.md` — Featured showcase gallery (50–100 curated designs with images, one or two per category)
- `docs/catalog/index.md` — Full searchable inventory using a MkDocs plugin or a JavaScript table (DataTables or similar lightweight lib)
- `docs/catalog/<category>.md` — Per-category pages (prime, zoom, microscope, etc.)

Gallery pages are regenerated on every post-merge CI run and committed to the `gh-pages` branch (not `main`).

### 8.3 Gallery Card Fields

Each design card displays, in order:

| Field | Source |
|---|---|
| Name | `registry.name` |
| Category tag | `registry.category` |
| EFL (mm) | `registry.efl_mm` |
| F/# | `registry.fnum` |
| Max FOV (°) | `registry.fov_max_deg` |
| Thumbnail | `registry.image_path` (thumbnail if available, placeholder otherwise) |
| Seidel S1–S5 | `registry.seidel` |

Card rendering is driven by a single Jinja macro (`docs/overrides/main.html`) that takes a `CatalogEntry` dict. Adding a new field to the card requires only updating the macro and `CatalogEntry`. No other code changes needed.

### 8.4 Image Strategy

**`main` branch (committed):** `docs/assets/featured/` contains ~50–100 hand-picked showcase PNGs, selected by the maintainer (representative designs per category, best performers, historically notable). These are ~80KB each, totalling ~5–8MB. Acceptable to commit directly.

**`gh-pages` branch (CI-generated):** Thumbnails for all valid designs are generated during the Pages build step. These are rendered at 400×200px and stored only in the gh-pages branch artefact, never in `main`. This keeps `main` lean.

**Thumbnail generation:** During the CI Pages build, any design without an existing thumbnail in the build cache is rendered using `optiland`'s 2D layout plot, saved as a 400×200 JPEG at 75% quality (~8–15KB each). The build tool tracks a manifest of `{slug: content_hash}` to skip unchanged designs.

---

## 9. CI/CD Workflows

### 9.1 Workflow: `validate.yml` (on PR)

**Trigger:** Any PR targeting `main`.

**Steps:**
1. Checkout PR branch.
2. Set up Python, install `optiland` (pinned version).
3. Identify changed/added JSON files in `data/` using `git diff --name-only origin/main`.
4. For each changed JSON:
   a. Check if the file's SHA-256 matches `registry.json` checksum.
   b. If changed (or new): run `scripts/validate_design.py`. 
   c. If `valid=False`: fail the job with the error output.
5. Pass/fail status reported as a PR check.

**Cost:** Only validates changed files. A PR adding 1 new design runs Level 3 validation on exactly 1 design (~30–60s total). A PR modifying 10 designs runs validation on 10.

### 9.2 Workflow: `publish.yml` (on push to `main`)

**Trigger:** Push to `main` (i.e., merged PR).

**Steps:**
1. Checkout `main`.
2. Run `scripts/generate_registry.py`:
   - Iterate all JSON files in `data/`.
   - For each file: compute SHA-256, compare to stored checksum.
   - Validate only changed/new files (Level 3, hash-skipping all others).
   - Write updated `registry.json`.
3. Run `scripts/generate_gallery.py` to rebuild MkDocs Markdown sources from the updated registry.
4. Run `mkdocs build` to produce the static site.
5. Deploy site to `gh-pages` branch using `peaceiris/actions-gh-pages`.
6. If `registry.json` changed: commit it back to `main` using the `github-actions[bot]` identity.

### 9.3 Manual Trigger: Full Regeneration

Both workflows support `workflow_dispatch` with an optional `force_regenerate: true` input. When set, the hash-skip logic is bypassed and all designs are re-validated and all thumbnails regenerated. Used when upgrading the pinned optiland version.

---

## 10. Bootstrap Script (`scripts/bootstrap.py`)

This is the **one-time offline script** that populates the zoo before the first public push. It must be run by the maintainer locally.

### 10.1 Steps

```
1. For each category in (prime, zoom, microscope, eyepiece, telescopes, projector):
   a. Scan source directory for JSON files → copy to data/<category>/
   b. Scan source directory for ZMX/LEN files with no JSON counterpart
      → run convert_zmx.py → on success, write to data/<category>/
      → on failure, append to conversion_failures.csv
   c. Deduplicate: if JSON was already produced, skip ZMX
   d. Write per-category LICENSE file

2. Run generate_registry.py with force_regenerate=True
   (validates every design, computes all metrics, writes registry.json)

3. Hand-select ~50–100 featured designs (one or two per category,
   based on highest Strehl / most historically notable)

4. Generate featured PNGs:
   python scripts/generate_featured_images.py --slugs <comma-separated>

5. Commit everything to main, push.
```

### 10.2 Expected Outcomes

| Metric | Expected value |
|---|---|
| Source files scanned | ~3,900 |
| Conversion failures (ZMX) | Unknown; tracked in `conversion_failures.csv` |
| Valid designs in registry | Estimated 70–90% of scanned |
| Featured PNGs committed | 50–100 |
| `registry.json` size | ~5–15MB |

---

## 11. Contributor Workflow

Adding a new design:

1. Fork the repo.
2. Place `<design_name>.json` in `data/<category>/`.
3. Open a PR. The PR template checklist requires:
   - [ ] License confirmed (MIT, CC0, or other compatible open license)
   - [ ] Design loads without error in current optiland release
   - [ ] Category directory is correct
4. CI validates the new JSON (Level 3). If validation fails, CI fails and the PR is blocked.
5. On merge, CI rebuilds registry and gallery automatically.

No sidecar `.meta.json` required from contributors. The registry is auto-populated from the JSON content.

---

## 12. Phased Implementation Plan

### Phase 0: Infrastructure Setup (before any data)
- Initialize `optiland-zoo` repo with full directory structure
- Write `pyproject.toml`, `mkdocs.yml`, `.gitignore`
- Write skeleton `optiland_zoo/` package with stub API
- Write `CONTRIBUTING.md`, `LICENSE`, `CREDITS.md` stubs
- Set up GitHub Actions workflow files (non-functional until data exists)

### Phase 1: Bootstrap Script
- Implement `scripts/convert_zmx.py` (wraps optiland Zemax importer, logs failures)
- Implement `scripts/validate_design.py` (Level 3 validation, outputs JSON summary)
- Implement `scripts/bootstrap.py` (orchestrates conversion + validation + registry write)
- Run bootstrap locally against source data at `LensAI/data/`
- Produce initial `registry.json` and `conversion_failures.csv`

### Phase 2: Python Package
- Implement `CatalogEntry`, `Registry`, `catalog.load()`, `catalog.list()`, `catalog.search()`
- Write `tests/test_catalog.py` (load a known design, search filter correctness, invalid slug raises)
- Set up PyPI publishing in `publish.yml`

### Phase 3: Gallery
- Implement `scripts/generate_gallery.py`
- Write MkDocs Jinja macros for gallery cards
- Configure `mkdocs.yml` for Material theme + search
- Generate and review featured PNGs, commit to `docs/assets/featured/`
- Wire up gh-pages deployment in CI

### Phase 4: CI Hardening
- Finalize `validate.yml` with hash-skip logic and `workflow_dispatch` override
- Finalize `publish.yml` with registry commit-back
- End-to-end test: add a dummy design via PR, verify full pipeline fires correctly

### Phase 5: Public Launch
- Finalize `CREDITS.md` with all provenance entries
- Write `README.md` (featured gallery section, quick-start, contributor guide link)
- Tag `v0.1.0`, publish to PyPI
- Push to GitHub

---

## 13. File Conventions and Standards

- All Python code follows the same standards as `optiland/`: `from __future__ import annotations`, Google-style docstrings, type hints on all public functions, Ruff-enforced (E, F, UP, B, SIM, I).
- `registry.json` is minified (no extra whitespace) to keep file size down and diff noise low.
- JSON design files are committed as-is (no reformatting or field additions).
- Featured PNG images are 800×400px, white background, `tight` layout, 150 DPI.
- Thumbnail images (gh-pages only) are 400×200px JPEG at 75% quality.

---

## 14. Open Questions / Future Work

The following items are deferred and tracked in `extra` or future milestones:

- **MTF curves** per design (requires standardized spatial frequency range per category)
- **Glass catalog cross-reference** (link material names to Schott/Ohara datasheets)
- **Patent number field** (for designs sourced from patent literature)
- **Optiland version compatibility matrix** (which zoo entries break on optiland upgrades)
- **Hugging Face Dataset mirror** (for ML use cases, not a priority for v1)
- **Sub-categories** (e.g., `prime/double-gauss`, `prime/cooke-triplet`) — defer until count justifies it
