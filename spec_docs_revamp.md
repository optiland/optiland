# Optiland Documentation Revamp — Implementation Spec

**Status:** Draft  
**Scope:** All user-facing docs (README, RTD site, notebooks, developer's guide)  
**Out of scope:** API reference docstring audit, ML/LensAI notebooks (12a–12e), NSQ raytracing tutorial

---

## 1. Guiding Decisions

| Decision | Choice |
|---|---|
| Canonical source | RST (RTD-first); LEARNING_GUIDE.md deleted |
| Notebook arc | Progressive where logical; no forced connections |
| Gallery rendering | nbsphinx + thumbnail metadata; no browser execution |
| Developer guide sync | Aspirational/conceptual; rewritten as contribution onramp |
| README scope | Middle ground — strip duplication, keep GitHub-unique sections |
| Onboarding | New "Start Here" RTD page routing 4 personas |
| Cheat sheet | Split: conceptual glossary + task-oriented API snippet page |
| Quickstart | Expand to 5-minute complete tour |
| New notebooks | Fully authored (mixed: concept block → code block) |
| Stale notebooks | Audit and update all for current API |
| ML tutorials | External LensAI links retained as-is |
| Replite shells | Keep index-only; no expansion |
| API reference | Out of scope |
| CONTRIBUTING.md | Keep separate; cross-link with dev guide |

---

## 2. Target Personas

The "Start Here" page addresses all four:

1. **Optics student / first-timer** — knows optics basics, no prior Python library experience; needs hand-holding from zero
2. **Optical engineer (practitioner)** — migrating from Zemax/CODE V; wants fast productivity
3. **Computational researcher** — ML/differentiable optics; cares about PyTorch backend and autograd
4. **Software contributor / extender** — adding surfaces, analysis types, or integrating Optiland into a larger system

---

## 3. File-Level Change Inventory

### 3.1 Files to Delete

| File | Reason |
|---|---|
| `docs/LEARNING_GUIDE.md` | RST is canonical; README already links to RTD |

### 3.2 Files to Rewrite

| File | Change |
|---|---|
| `README.md` | Strip duplicated sections; keep GitHub-unique content |
| `docs/quickstart.rst` | Expand to 5-minute complete tour |
| `docs/cheat_sheet.rst` | Rewrite as task-oriented API snippet reference |
| `docs/index.rst` | Add "Start Here" to toctree; update captions |
| `docs/learning_guide.rst` | Add new notebooks; reorder intro; add new sections 13 and 14 |
| `docs/developers_guide/*.rst` | Add "How to extend this" sections to each architectural page |
| `CONTRIBUTING.md` | Add cross-link to RTD developer's guide |

### 3.3 Files to Create (RST)

| File | Purpose |
|---|---|
| `docs/start_here.rst` | Persona-routing onboarding page |
| `docs/glossary.rst` | Conceptual definitions (extracted from old cheat_sheet.rst) |
| `docs/developers_guide/extension_recipes.rst` | Step-by-step contribution recipes (standalone companion to dev guide) |

### 3.4 New Notebooks to Author

| File | Section | Title |
|---|---|---|
| `docs/examples/Tutorial_1g_Material_Catalog_and_Registry.ipynb` | §1 | Material Catalog & Registry |
| `docs/examples/Tutorial_1h_Prescription_Generator.ipynb` | §1 |Prescription Generator |
| `docs/examples/Tutorial_5e_Differentiable_Optimization.ipynb` | §5 | Differentiable Lens Optimization |

### 3.5 Notebooks to Audit and Update

All 48 existing notebooks in `docs/examples/` must be verified against the current API and fixed if broken. Priority order for audit:

**Critical (beginner path — touched first):**
- Tutorial_1a, 1b, 1c, 1d, 1e, 1f
- Tutorial_2a, 2b, 2c, 2d

**High:**
- Tutorial_3a, 3b, 3c
- Tutorial_4a, 4b, 4c
- Tutorial_5a, 5b, 5c, 5d

**Medium:**
- Tutorial_6a through 6i (skip 6g — does not exist)
- Tutorial_7a through 7f
- Tutorial_8a, 8b
- Tutorial_9a, 9b
- Tutorial_10a, 10b, 10c
- Tutorial_11a

**Lower:**
- ML notebooks: Singlet_RF, Ray_Path_Failure, Double_Gauss_Surrogate, SR_GAN, RL_aspheric, Misalignment_Prediction (LensAI copies — flag if broken, note they are external-canonical)
- Standalone: Scan_Lens_System_for_UV, Example_Optimization_Using_Reciprocal_Radii (see §8 for disposition)

---

## 4. README Rewrite

### 4.1 Sections to Keep (with edits)

| Section | Changes |
|---|---|
| Badges | Keep as-is |
| Header image + GUI image | Keep |
| Introduction (2-para) | Keep; no material changes |
| Installation | Keep; verify CUDA version is current |
| Core Capabilities table | Keep (unique: dense, scannable) |
| Roadmap | Keep (GitHub-specific; not on RTD) |
| Get Involved | Keep (badge links; GitHub-specific) |
| Contributing | Keep 1-para pointer to CONTRIBUTING.md |
| License | Keep |
| Contact and Support | Keep |

### 4.2 Sections to Remove or Condense

| Section | Action |
|---|---|
| Contents (ToC) | Remove — GitHub auto-renders anchor links |
| "Documentation" (single-line paragraph) | Collapse into Introduction paragraph |
| "Learning Guide" (full numbered list) | **Remove** — fully duplicated in RTD; replace with 4-link summary block pointing to RTD |
| Quickstart (numbered list pointing to RTD) | Replace with single paragraph + 3 links: Quickstart, Gallery, Learning Guide |

### 4.3 Replacement Quickstart Block (README)

```markdown
**Get started in 5 minutes:**
```python
pip install optiland
```
→ [5-minute quickstart](https://optiland.readthedocs.io/en/latest/quickstart.html) · [Example Gallery](https://optiland.readthedocs.io/en/latest/gallery/introduction.html) · [Full Learning Guide](https://optiland.readthedocs.io/en/latest/learning_guide.html)
```

---

## 5. New Page: `docs/start_here.rst`

**Purpose:** First stop for every newcomer. Routes to the right content without reading everything.

### 5.1 Structure

```
Start Here
==========

Short intro paragraph: What is Optiland? Who uses it?

.. rubric:: Choose Your Path

[4 persona cards — each with a brief role description, 3-4 recommended pages/notebooks, and a primary CTA link]
```

### 5.2 Persona Cards

**Optics Student / First-Timer**
- Goal: understand Optiland basics, trace rays, visualize a lens
- Recommended path:
  1. Installation → `installation.rst`
  2. Tutorial 1a — Optiland for Beginners
  3. Tutorial 1b — Lens Properties
  4. Tutorial 3a — Aberration Analyses
- CTA: "Start with Tutorial 1a →"

**Optical Engineer (Practitioner)**
- Goal: get productive fast, import existing designs, run analyses
- Recommended path:
  1. 5-minute Quickstart → `quickstart.rst`
  2. API Cheat Sheet → `cheat_sheet.rst`
  3. Tutorial 9a — Edmund Optics Catalogue (file import)
  4. Tutorial 5c — Optimization Case Study
- CTA: "Go to the Quickstart →"

**Computational Researcher**
- Goal: PyTorch backend, autograd, differentiable optimization
- Recommended path:
  1. Tutorial 1f — Differentiable Ray Tracing Hello World
  2. Tutorial 5e — Differentiable Optimization Deep-Dive (NEW)
  3. Developer's Guide: Configurable Backend
  4. Tutorial 5b — Advanced Optimization
- CTA: "Start with Tutorial 1f →"

**Software Contributor / Extender**
- Goal: add new surfaces/analysis/operands, understand codebase
- Recommended path:
  1. Developer's Guide: Architecture
  2. Developer's Guide: Extension Recipes (NEW)
  3. Tutorial 10a — Custom Surface Types
  4. Tutorial 10b — Custom Coating Types
- CTA: "Read the Developer's Guide →"

### 5.3 Toctree Placement

Add `start_here` as the **first entry** in the Getting Started toctree in `index.rst`, before `installation`.

---

## 6. Cheat Sheet Split

### 6.1 `docs/cheat_sheet.rst` — Task-Oriented API Snippet Reference

Rewrite as 20 canonical copy-paste snippets covering the 80% tasks. Each snippet:
- 1-line "what this does" label
- 3–12 lines of runnable Python
- No prose paragraphs — headers + code blocks only

**Snippet categories (ordered by frequency of use):**

1. Install and import
2. Load a sample lens
3. Build a simple singlet from scratch
4. Add a surface
5. Set aperture / field / wavelength
6. Switch backend (NumPy ↔ PyTorch)
7. Draw the lens (2D)
8. Draw the lens (3D)
9. Trace rays manually
10. Spot diagram
11. Ray fan plot
12. Wavefront / Zernike decomposition
13. PSF and MTF
14. Paraxial properties (EFL, f/#, pupil positions)
15. Define an optimization variable
16. Define an operand
17. Run local optimization
18. Run global optimization
19. Save / load system (JSON)
20. Generate a prescription report

### 6.2 `docs/glossary.rst` — Conceptual Definitions

Migrate the existing conceptual content from the current `cheat_sheet.rst` here. Structure:

```
Glossary
========

Optic
  The central container...

SurfaceGroup
  ...

Surface
  ...

Material
  ...

Geometry
  ...

Aperture
  ...

Fields
  ...

Wavelengths
  ...

Coordinate System
  ...

Apodization
  ...

Backend
  ...
```

### 6.3 Linking Strategy

- `cheat_sheet.rst` header: "New to these concepts? See the :ref:`glossary` first."
- `glossary.rst` header: "Want runnable examples? See the :ref:`cheat_sheet`."
- Both appear in the Getting Started toctree (cheat sheet first, then glossary).
- `index.rst` Getting Started section becomes: `installation`, `start_here`, `quickstart`, `cheat_sheet`, `glossary`, `gui_quickstart`.

---

## 7. Quickstart Rewrite

**Goal:** A self-contained 5-minute tour. Reader goes from zero to seeing real output.

### 7.1 Structure

```
Quickstart — Your First 5 Minutes
==================================

1. Install  (pip install snippet)
2. Hello, World  (CookeTriplet → draw3D)
3. Build from Scratch  (2-surface singlet, 5–8 lines)
4. Trace Rays  (lens.trace(...) → show x/y coords)
5. Spot Diagram  (SpotDiagram(lens).view())
6. One-Step Optimization  (define variable + operand + optimize)
7. Save and Load  (lens.save / optic.Optic.load)
8. What Next?  (→ Start Here page with persona paths)
```

Each section: 1-sentence context + code block + expected output description (no actual output images; code is runnable). Total page length: ~150 lines RST.

---

## 8. Learning Guide RST Changes

### 8.1 Intro Paragraph Rewrite

Replace the current flat intro sentence with:

```
This guide is Optiland's primary learning path. Tutorials are grouped thematically and follow a progressive arc where concepts from earlier sections inform later ones. Each notebook is self-contained and runnable; you do not need to execute prior notebooks to run any given one. New to Optiland? Start with :ref:`start_here` to find the path that fits your goals.
```

### 8.2 Section 1 — Introduction to Optiland (Updated)

Add two new tutorials:

| Tutorial | Status | Topic |
|---|---|---|
| 1a | existing (update) | Optiland for Beginners |
| 1b | existing (update) | Lens Properties |
| 1c | existing (update) | Saving and Loading |
| 1d | existing (UPDATE for new catalog API) | Material Database |
| 1e | existing (update) | Non-Rotationally Symmetric Systems |
| 1f | existing (update) | Differentiable Ray Tracing Hello World |
| **1g** | **NEW** | Material Catalog & Registry |
| **1h** | **NEW** | Prescription Generator |

### 8.3 Section 5 — Optimization (Updated)

Add Tutorial 5e:

| Tutorial | Status | Topic |
|---|---|---|
| 5a | existing (update) | Simple Optimization |
| 5b | existing (update) | Advanced Optimization |
| 5c | existing (update) | Optimization Case Study (Cooke Triplet) |
| 5d | existing (update) | User-Defined Optimization Metrics |
| **5e** | **NEW** | Differentiable Lens Optimization (PyTorch) |

### 8.4 Standalone Gallery Notebooks

Two notebooks exist in `docs/examples/` but are not in the learning guide:
- `Scan_Lens_System_for_UV.ipynb`
- `Example_Optimization_Using_Reciprocal_Radii.ipynb`

**Decision:** Add these to the gallery (not the learning guide — they are application examples, not tutorials). Place in `gallery/real_world_projects` or a new `gallery/advanced_examples` page. Update the gallery RST accordingly.

### 8.5 Tutorial 6g Gap

Tutorial 6g does not exist. The jump from 6f → 6h is intentional or was left as a placeholder. **No action required** — do not renumber existing tutorials.

### 8.6 Section 12 — Machine Learning

Keep as external LensAI links. Add a note at the top of the section:

```
.. note::
   The following tutorials are hosted in the `LensAI repository <https://github.com/HarrisonKramer/LensAI>`_. For in-repo differentiable modeling examples, see :ref:`Tutorial_1f` and :ref:`Tutorial_5e`.
```

---

## 9. New Notebook Specifications

All new notebooks use **mixed format**: brief conceptual intro per section (1–3 sentences explaining the "why"), then runnable code.

### 9.1 Tutorial 1g — Material Catalog & Registry

**File:** `docs/examples/Tutorial_1g_Material_Catalog_and_Registry.ipynb`  
**Prerequisites:** Tutorial 1d (Material Database)  
**Optiland modules:** `optiland.materials` (MaterialRegistry, Catalog, MatchPolicy, MaterialSpec)  

**Section outline:**

1. **Concept: From flat database to structured catalog**  
   - What problem the new catalog/registry system solves over flat lookups  
   - Code: `from optiland.materials import MaterialRegistry; registry = MaterialRegistry()`

2. **Built-in catalogs**  
   - Listing available catalogs  
   - Code: `registry.list_catalogs()`; iterate entries

3. **Looking up a material by name**  
   - Code: `mat = registry.get("N-BK7")` → inspect `n`, `k`, dispersion curve

4. **MatchPolicy: controlling catalog search behavior**  
   - Explain exact vs. fuzzy matching, catalog priority ordering  
   - Code: construct `MaterialSpec` with a `MatchPolicy`; show different results

5. **Creating a user catalog**  
   - Code: define a custom glass (name, Sellmeier coefficients or nd/Vd), register it, retrieve it

6. **Using catalog materials in a lens system**  
   - Code: build a 2-surface system, assign `N-BK7` and `N-F2` via the registry; verify `n` at design wavelength

7. **Grouping and filtering**  
   - Code: filter catalog by Abbe number range; find all glasses near a target nd/Vd

### 9.2 Tutorial 1h — Prescription Generator

**File:** `docs/examples/Tutorial_1h_Prescription_Generator.ipynb`  
**Prerequisites:** Tutorial 1a (Beginners)  
**Optiland modules:** `optiland.Prescription` (exported at top level)  

**Section outline:**

1. **What is a prescription?**  
   - Optical system description: surfaces, radii, thicknesses, materials, apertures, fields, Seidel aberrations  
   - Code: `from optiland import Prescription`

2. **Generating a prescription for a sample lens**  
   - Code: load CookeTriplet, `p = Prescription(lens)`, `p.view()` (console Rich output)

3. **Prescription output fields**  
   - Explain each section: system overview, first-order properties, surface table, Seidel aberrations  
   - Code: access `p.data` dict programmatically

4. **Saving to plain text**  
   - Code: `p.save("cooke_triplet.txt")`

5. **Exporting to PDF**  
   - Code: `p.save("cooke_triplet.pdf")` — note dependency requirements

6. **Styling in the console**  
   - Code: `p.view(style="dark")` vs default

7. **Custom prescription: user-defined lens**  
   - Code: build a doublet from scratch, generate prescription; read off EFL and pupil positions from the report

### 9.3 Tutorial 5e — Differentiable Lens Optimization

**File:** `docs/examples/Tutorial_5e_Differentiable_Optimization.ipynb`  
**Prerequisites:** Tutorial 1f (DRT Hello World), Tutorial 5a (Simple Optimization)  
**Optiland modules:** `optiland.backend`, `optiland.optimization`, PyTorch  

**Section outline:**

1. **Why differentiable optimization?**  
   - Gradient-based vs. merit function (SciPy) — tradeoffs  
   - Code: `import optiland.backend as be; be.set_backend("torch")`

2. **Setting up a differentiable lens**  
   - Code: build a simple singlet with `requires_grad=True` parameters

3. **Computing gradients of an optical metric**  
   - Code: run `lens.trace(...)`, compute spot size as a tensor, call `.backward()`; inspect `.grad` on radii

4. **Custom loss function**  
   - Code: define `loss = rms_spot_size(lens)`, show how to use it in a torch optimizer loop

5. **Full gradient descent optimization loop**  
   - Code: PyTorch `optim.Adam` loop over radii and thicknesses; plot loss curve

6. **Comparing to SciPy optimizer**  
   - Code: solve the same problem with `LeastSquaresOptimizer`; compare convergence speed and result quality

7. **Multi-field differentiable optimization**  
   - Code: loss over 3 fields simultaneously; gradient clipping for stability

8. **Practical considerations**  
   - Numerical precision on GPU, parameter bounds, when to use autograd vs. merit functions

---

## 10. Developer's Guide Rewrite

### 10.1 Scope

Each of the 18 RST pages in `docs/developers_guide/` retains its conceptual content but gains a new **"How to extend this"** section at the bottom. The guide remains aspirational (can lag the code); the extension sections give contributors a concrete starting point.

### 10.2 Extension Section Template

Each page's "How to extend this" section follows this template:

```rst
How to Extend This
------------------

**Scenario:** Add a new [X] to Optiland.

**Step 1:** Create a new file in ``optiland/<module>/my_new_X.py``.
**Step 2:** Subclass ``Base<X>`` and implement ``method_a(...)`` and ``method_b(...)``.
**Step 3:** Register in ``optiland/<module>/__init__.py``.
**Step 4:** Add tests in ``tests/test_<module>/test_my_new_X.py``.

For a complete worked example, see :ref:`Tutorial_10a`.
```

### 10.3 Pages Requiring New Extension Sections

| Page | Scenario to cover |
|---|---|
| `surface_overview.rst` | Add a new surface geometry type |
| `interaction_models.rst` | Add a new surface interaction model |
| `geometry_overview.rst` | Add a new geometry class |
| `analysis_framework.rst` | Add a new analysis class |
| `optimization_framework.rst` | Add a new optimization operand |
| `tolerancing_framework.rst` | Add a custom tolerance sensitivity class |
| `configurable_backend.rst` | Add a backend-agnostic utility function |
| `visualization_framework.rst` | Add a new 2D or 3D renderer |

### 10.4 New Page: `docs/developers_guide/extension_recipes.rst`

A companion page listing short step-by-step recipes for the 8 most common contribution scenarios. Each recipe is 10–20 lines: what file to create, what to subclass, what methods to implement, and where to register. Cross-linked from each architecture page's "How to extend this" section and from `CONTRIBUTING.md`.

### 10.5 CONTRIBUTING.md Cross-Link

Add to the top of `CONTRIBUTING.md`:

```markdown
> For a deep dive into Optiland's architecture and step-by-step extension recipes, see the [Developer's Guide](https://optiland.readthedocs.io/en/latest/developers_guide/introduction.html) on Read the Docs.
```

---

## 11. `docs/index.rst` Changes

### 11.1 Getting Started Toctree (Updated)

```rst
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   start_here
   installation
   quickstart
   cheat_sheet
   glossary
   gui_quickstart
```

### 11.2 No Other Structural Changes

The main toctree structure (Core Functionalities, Example Gallery, Learning Guide, Developer's Guide, Contributing, API Reference, Authors, License, References) is retained. The gallery section is unchanged.

---

## 12. Notebook Audit Checklist

For each notebook, the auditor must verify:

- [ ] All imports resolve against current `optiland` public API
- [ ] No `from optiland.X import Y` where `Y` has been renamed or moved
- [ ] Material names match the current catalog (no legacy lookup strings that no longer resolve)
- [ ] Sample lens calls (`CookeTriplet()`, `ReverseTelephoto()`, etc.) work
- [ ] No deprecated backend calls (pre-refactor `be.array` signatures, old `set_backend` spellings)
- [ ] All cell outputs are cleared and cells run sequentially without error
- [ ] Kernel metadata specifies Python 3 (not a pinned version)

**Known likely-stale areas:**
- Tutorial 1d: material lookup API changed with catalog refactor — verify all `Material(...)` and database calls
- Tutorial 7e (Glass Expert): `GlassExpert` API may have changed with material catalog merge
- Tutorial 9a/9b: Zemax/OSLO file import paths; verify `optiland.fileio` imports are current

---

## 13. Implementation Order

Execute in this order to minimize rework:

### Phase 1 — Structural (no content authoring)
1. Delete `docs/LEARNING_GUIDE.md`
2. Update `README.md` (trim learning guide section, add 3-link quickstart block)
3. Update `docs/index.rst` (add start_here, glossary to toctree)
4. Create `docs/glossary.rst` (migrate content from cheat_sheet.rst)
5. Rewrite `docs/cheat_sheet.rst` as task-oriented snippet reference

### Phase 2 — New Pages
6. Create `docs/start_here.rst` (persona cards)
7. Rewrite `docs/quickstart.rst` (5-minute tour)
8. Create `docs/developers_guide/extension_recipes.rst`
9. Add "How to extend this" sections to 8 dev guide pages
10. Update `CONTRIBUTING.md` cross-link

### Phase 3 — New Notebooks
11. Author `Tutorial_1g_Material_Catalog_and_Registry.ipynb`
12. Author `Tutorial_1h_Prescription_Generator.ipynb`
13. Author `Tutorial_5e_Differentiable_Optimization.ipynb`
14. Update `docs/learning_guide.rst` to include new notebooks

### Phase 4 — Notebook Audit
15. Audit and fix beginner notebooks (1a–1f) for API correctness
16. Audit and fix raytracing and analysis notebooks (2a–4c)
17. Audit and fix optimization and advanced notebooks (5a–11a)
18. Add Scan_Lens_System_for_UV and Example_Optimization_Using_Reciprocal_Radii to gallery RST

### Phase 5 — Verification
19. Verify all toctree links resolve (no missing RST files)
20. Verify all notebook cross-references in RST point to existing files
21. Spot-check rendered HTML (local `make html`) for broken links and images
