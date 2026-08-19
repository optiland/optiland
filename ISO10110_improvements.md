# ISO 10110 Drawing Improvements

Items identified from code review and the ISO 10110 standard.
✅ = implemented and committed.

---

## A — Quick fixes (notation.py / drawing.py)

### A1 · Multiplication sign in ISO codes ✅
- Was: `1/ 1x0.16`, `5/ 2x0.04; L1x0.001`
- Now: `1/ 1×0.16`, `5/ 2×0.04; L1×0.001` (U+00D7 ×)
- `_fix_mul()` helper normalises ASCII x→× in all `fmt_*` methods

### A2 · Abbe number symbol ✅
- Was: `Vd = 64.17`
- Now: `νd = 64.17` (ν = U+03BD Greek nu)
- Also: nd / νd annotations now include reference wavelength per §5.10.1

### A3 · Remove spurious λ prefix from 7/ coating ✅
- Was: `7/ λ AR 400-700 nm, R<0.5%`
- Now: `7/ AR 400-700 nm, R<0.5%`
- ISO 10110-17 / ISO 9211 format has no λ prefix

### A4 · Surface index labels on drawing view ✅
- S1/S2 labels placed at the callout bar position on the cross-section

---

## B — Notation extensions (notation.py + spec.py)

### B1 · 3/ test wavelength note ✅
- ISO 10110-5 fringes are defined at 546.07 nm
- `SurfaceSpec.test_wavelength` field: if set and ≠ 546 nm, appended to 3/ code
- Default 546.07 nm always shown in fringe output per ISO 10110-5:2016 §4.3

### B2 · 4/ full centration format ✅
- Was: `4/ 0.5'`
- Now: `4/ σ'(δ)` where δ = lateral decentration in mm (optional)
- `SurfaceSpec.centration_decentration` field added

### B3 · Chamfer with angle ✅
- `SurfaceSpec.chamfer` + `SurfaceSpec.chamfer_angle` fields
- Rendered as `Schutzfase <width> × <angle>°`

### B4 · Glass catalog name ✅
- Now rendered as `N-BK7 (SCHOTT)` — manufacturer in parentheses
- `_mat_display_name()` helper reads catalog from material database

---

## C — Drawing view improvements

### C1 · Surface number on view ✅
- S1/S2/… labels on the cross-section view (see A4)

### C2 · Optical axis label ✅
- Optical axis carries "opt. axis" or "opt./rot. axis" label at the left end

### C3 · Edge thickness dimensioning ✅
- ET shown above the rim with extension lines and bracket notation
- Reference dimension shown as `(ET mm)` per ISO 10110-1

### C4 · Radius of curvature on drawing view ✅
- R shown only in the spec table — correct per ISO practice

---

## D — Title block

### D1 · "DIM. IN mm" placement ✅
- In row 2 (notes row) per ISO 7200

### D2 · Units for ISO codes ✅
- Fringes in 3/: dimensionless ✓
- Arc-minutes in 4/: ' symbol ✓
- mm in 1/ bubble grade: implied ✓

### D3 · Missing fields in title block ✅
- Sheet number (X of Y) now shown
- Organisation field propagated to title block

---

## E — Conformance improvements implemented from ISO 10110 standard PDFs

### E1 · ISO 10110-18:2019 homogeneity class designators ✅
- Legacy integer grades 0–5 mapped to NH100, NH040, NH010, NH004, NH002, NH001
- `SurfaceSpec.nh_class` field accepts any NH designator; validated against known set
- Striae: `striae_density` (1–5) and `striae_shadowgraph` (A–D) fields added

### E2 · 3/ form error — always show λ in fringe mode ✅
- Per ISO 10110-5:2016 §4.3: fringe measurements always require wavelength annotation
- `fmt_form_error()` always appends `; λ=<wl> nm` for fringe units
- nm mode (form_unit="nm") omits wavelength annotation (absolute units)

### E3 · ISO 10110-8:2019 surface texture (8/ code) ✅
- `SurfaceSpec.roughness` field with docstring matching §5 format rules
- `fmt_roughness()` returns `"8/ P2"` etc., or `""` when unset (optional code)
- P-grade validation: advisory warning for P5+ (only P1–P4 per Table 1 §4.3.3)
- Valid forms: `"P"`, `"P1"`–`"P4"`, `"G"`, `"-1/Rq 0.002"`, `"0.002-1/Rq 0.002"`
- Both matplotlib and DXF renderers use `fmt_roughness()` consistently

### E4 · ISO Table 1 code order ✅
- 8/ now appears after 6/ and before 13/ and 15/ in spec column
  (numerical order: 3, 4, 5, 6, 8, 13, 15)
- Previously 8/ was appended last, after 15/

### E5 · Per-component material specs for cemented lenses ✅
- `DrawingSpec.set_material(element_idx, component_idx, **kwargs)` enables
  independent 0/, 1/, 2/ grades per glass in a cemented doublet
- `DrawingSpec.get_material_spec(element_idx, component_idx)` with fallback
  to element-level spec for backward compat
- Both renderers (mpl, DXF) removed the old `if mi == 0` restriction;
  each glass column now shows its own quality codes
- YAML persistence under `material_specs` key

---

## F — Outstanding / To investigate

### F1 · 7/ spec table row position ✅
- ISO 10110-1 Table 1 numerical order: 3, 4, 5, 6, **7**, 8, 13, 15
- Was: 7/ placed before all numbered codes (non-standard)
- Now: 7/ placed after 6/ and before 8/ in both renderers

### F2 · 13/ wavefront format advisory ✅
- `SurfaceSpec.__post_init__` now warns when `wavefront_deformation` doesn't
  match the ISO 10110-14 `A`, `A(B)`, or `A(B/C)` numeric format.

### F3 · 15/ cement interfaces only ✅
- 15/ suppressed on outer surfaces unless explicitly set by user
- Shown automatically for cement interfaces `(0 < si < n−1)`

### F4 · Aspheric coefficient notation ✅ (no change needed)
- `_surface_header_lines()` renders `A2, A4, A6, …` matching ISO 10110-12:
  `coefficients[i]` → `A_{2(i+1)}` for EvenAsphere, verified correct.

### F5 · Pyramid error placement — separate row after 4/ ✅ (acceptable)
- ISO 10110-6 §5.3 places pyramid error as an additional element; current
  placement as a separate row adjacent to 4/ is acceptable.

### F6 · Cement interface 3/, 4/, 5/ codes — each surface independent ✅
- Each surface (including cement interfaces) can carry independent
  3/, 4/, 5/ specs via `DrawingSpec.set_surface(surface_index, ...)`.
  This is the correct ISO behaviour; no change needed.

### F7 · ISO 9211 coating code format — intentionally free-form
- Coating callouts vary widely across standards (ISO 9211, MIL-C, …).
  Free-form string is intentional; validation would add friction without
  clear benefit for the typical use case.

---

## G — Further DXF / renderer improvements

### G1 · DXF spec table row order ✅
- 7/ coating now placed after 6/ per ISO 10110-1 Table 1 (both renderers)

### G2 · DXF λ-circle symbol ✅
- Encircled-λ symbol now uses actual λ (U+03BB) rather than the ASCII 'L'

### G3 · DXF Øe brackets ✅
- Effective aperture brackets added to DXF cross-section, matching mpl

### G4 · DXF axis label and cement surface labels ✅
- "opt./rot. axis" text label added at left of optical axis in DXF
- S2, S3, … labels below axis for cement surfaces in DXF

### G5 · DXF sharp-edge "0" markers ✅
- Sharp-edge "0" symbols rendered in DXF callouts when sharp_edge=True

### G6 · DXF Unicode symbols ✅
- Physical diameter: Ø (U+00D8) instead of plain O
- Øe labels: Øe instead of Oe
- General tolerance line: Ø±, CT± with proper symbols
- General tolerance now includes 5/ imperfections (both renderers)

### G7 · 13/ wavefront format advisory ✅
- Warning when wavefront_deformation doesn't match ISO 10110-14 A(B/C) format

### G8 · DXF κ preserved in asphere headers ✅
- Removed spurious κ→k replacement: DXF R2010 TEXT entities encode Unicode
  via ezdxf; ISO 10110-12 κ (U+03BA) now appears correctly on asphere drawings

### G9 · DXF chamfer uses × and ° ✅
- Chamfer row now uses × (U+00D7) and ° (U+00B0), matching mpl renderer
  and ISO typographic conventions; removed dead Ø\u2091→Oe substitution

### G10 · DXF first-angle projection symbol ✅
- Two concentric circles + truncated cone + centre line added to DXF title
  block (ISO 10110-1 §5.11.4); text positions corrected to match mpl layout

### G11 · DXF alternating hatch for cemented lenses ✅
- Per ISO 128-50 §4.2.3: each glass component now gets its own HATCH entity
  at alternating 45°/135° angles, matching the matplotlib renderer behaviour

---

## H — DXF quality and ISO 128 conformance

### H1 · DXF arrowheads on dimension lines ✅
- All dimension lines (CT, Ø, ET, component thicknesses) now carry filled
  SOLID arrowhead triangles at both ends per ISO 129-1 dimensioning rules
- `_dxf_arrowhead(msp, tip, toward, layer)` helper function added

### H2 · DXF optical axis linetype — ISO 128-20 dash-double-dot ✅
- Changed from single-dot CENTER `[5,-2,0.5,-2]` to double-dot OPTICAL
  `[8,-2,1,-2,1,-2]`, matching the matplotlib `(0,(8,2,1,2,1,2))` phantom
  line and ISO 128-20 axis line type
- Renamed linetype "CENTER" → "OPTICAL" to avoid conflict with the AutoCAD
  standard CENTER (single-dot) definition
- Rotational axis (when separate) uses DASHDOT `[8,-2,1,-2]` (single-dot)
  for visual distinction

### H3 · DXF f′ prime annotation ✅
- f' ASCII → f′ (U+2032 prime), matching the matplotlib renderer

---

## I — Surface header and notation parity

### I1 · TOROID header Rx / Ry labels ✅
- Was: `R\u2090 …` (Rₐ — wrong Unicode subscript-a) and `R_y …`
- Now: `Rx …` and `Ry …` — consistent with BICONIC notation and ISO 10110-12

### I2 · DXF Øₑ subscript-e symbol ✅
- DXF spec table and dimension labels now use Ø + U+2091 (ₑ subscript e),
  matching the mpl renderer which uses `Ø\u2091`
- Affected: `_dxf_spec_table` Øe row and `_dxf_dimensions` Øe bracket label

### I3 · r_tolerance on ASPH and CYL vertex radius line ✅
- Per ISO 10110-12 §5.3.1 the vertex radius (and its tolerance) must appear
  in the surface header; standard spheres already carried r_tolerance
- ASPH and CYL now include r_tolerance in both mpl (`$R tol$`) and DXF
  (`R tol` plain-text) modes; TOROID and BICONIC omitted (two independent
  radii; single-field r_tolerance is ambiguous)

---

## J — General tolerance and helper improvements

### J1 · ISO 10110-11 §4.1 element size fix ✅
- Was: `_iso10110_11_defaults(math.hypot(phys_d, ct))` — used diagonal, not diameter
- Now: `_iso10110_11_defaults(phys_d)` — uses physical outer diameter per ISO 10110-11 §4.1
  "The default tolerances are determined by the largest dimension of the element"
- For lenses the largest dimension is the outer diameter; diagonal was slightly
  over-conservative at category boundaries
- Function parameter renamed `diagonal` → `element_size` with docstring

### J3 · Integer formatting in fmt_birefringence / fmt_centration / fmt_pyramid_error ✅
- Was: `0/ 5.0 nm/cm`, `4/ 3.0′`, `pyr 2.0′` (float artifacts from internal coercion)
- Now: `0/ 5 nm/cm`, `4/ 3′`, `pyr 2′` — clean integer representation
- `fmt_form_error()` already had this via `_fmtv()`; now applied consistently

### J2 · Test coverage for recent fixes ✅
- `TestDxfTitleBlockEmDash`: verifies U+2014 em dash (not ASCII `-`) in DXF
  title block when `drawn_by`/`approved_by` are empty (regression test for the
  title-block fix from session H)
- `TestTolHelpers`: unit tests for `_tol_plain()` and `_tol_math()` helper
  functions; confirms plain-text mode strips U+2212 → ASCII hyphen-minus
- `TestIso1011Defaults`: confirms the correct default tolerance category is
  selected for a 22 mm diameter element (≤30 mm: Ø±0.10, CT±0.15)

### J4 · DXF `$EXTMIN`/`$EXTMAX` paper-boundary extents ✅
- ezdxf calls `update_extents()` during `write()`, overwriting any header values
  set before saving; `Vec3(0,0,0)` is also falsy so the origin corner was never
  written even when set on `modelspace.dxf.extmin`
- Fix: monkey-patch `doc.update_extents` on the instance in `ElementDrawing.generate()`
  so the paper-boundary values `(0,0)–(pw,ph)` always survive to the saved file
- `TestDxfExtents`: two tests verify A4-portrait and A3-landscape `$EXTMAX`

---

## K — ISO 10110-12 conic constant completeness

### K1 · BICONIC and CYL/TOROID conic constant display ✅
- ISO 10110-12 requires all non-zero surface parameters to appear in the header
- Was: BICONIC showed only Rx/Ry; CYL/TOROID showed only R (no κ for base curve)
- Now: BICONIC appends `κx = …` / `κy = …` lines when the respective conic constant
  is non-zero; CYL and TOROID append `κ = …` when the base-curve conic constant
  (`k_yz`) is non-zero
- Both mpl and DXF renderers benefit (both call `_surface_header_lines()`)
- Four new tests: `test_biconic_conic_constants_shown_when_nonzero`,
  `test_biconic_conic_constants_omitted_when_zero`,
  `test_cyl_conic_constant_shown_when_nonzero`,
  `test_toroid_conic_constant_shown_when_nonzero`

---

## L — Forbes Q polynomial surface header

### L1 · Forbes Q radial-term coefficients in ASPH header ✅
- `ForbesQNormalSlopeGeometry` was grouped under ASPH but its coefficients
  were silently omitted: the code tried `g.coefficients` (which doesn't exist)
  and caught the `AttributeError`; Forbes Q uses `g.radial_terms` dict instead
- Now: after the EvenAsphere/OddAsphere A-coefficient block, a separate try/except
  reads `g.radial_terms` and appends:
  - `ρmax = <norm_radius>` (normalization radius — essential to interpret a_m values)
  - `a0 = …`, `a1 = …`, … for each non-zero term at order m
- ISO 10110-12 has no standard Forbes polynomial notation; this display follows
  Forbes 2007/2011 paper notation (a_m coefficients for Q_m(u²) terms)
- Two new tests: `test_forbes_q_radial_terms_shown`,
  `test_forbes_q_radial_terms_empty`

---

## M — Grating surface header completeness (ISO 10110-16)

### M1 · Grating frequency and diffraction order in header ✅
- Was: GRAT header only showed type symbol (LG, CG, …) and "GRAT" — no
  frequency or order information
- Now: header appends `<freq> l/mm` (frequency = 1000/period_µm) and
  `m = ±<order>` per ISO 10110-16 §5.3
- Applies to both `PlaneGrating` and `StandardGratingGeometry`
- `period_µm=0` or missing attribute handled gracefully via try/except
- Three updated/new tests: `test_plane_grating` (updated to verify frequency +
  order), `test_plane_grating_standard_geometry` (new)

---

## N — Notation validation improvements

### N1 · ISO 10110-5 §4.2 RSI component C ≤ B validation ✅
- ISO 10110-5 §4.2 defines C as the *rotationally-symmetric component* of the
  total irregularity B; by definition C ≤ B — a larger C is physically impossible
- Added advisory `warnings.warn` when `rotationally_symmetric > irregularity`
- Two new tests: `test_rsi_exceeds_irregularity_warns` and
  `test_rsi_equal_irregularity_no_warning` (C == B is valid — 100% RSI)

### N2 · ISO 10110-6 δ without σ validation ✅
- ISO 10110-6: lateral decentration δ appears only inside `4/ σ′(δ)`;
  without σ the parenthesis is absent and δ is silently dropped.
- Added advisory `warnings.warn` in `SurfaceSpec.__post_init__` when
  `centration_decentration` is set but `centration` is not
- Parallel to the existing RSI-without-irregularity warning
- New test: `test_decentration_without_centration_warns`

### N3 · OddAsphere ASPH header test coverage ✅
- No code change needed: `coefficients[i] → A_{i+1}` was already correct
  for OddAsphere (`z = conic_base + Σ Ci·r^(i+1)`)
- Two new tests: `test_odd_asphere_coefficients` (verifies A2, A4 shown,
  A1/A3 omitted for zero values) and `test_odd_asphere_no_ρmax`

### N4 · Documentation fixes ✅
- `ElementSpec.nh_class` docstring now lists the full preferred NH series
  (NH040, NH010, NH004, NH002) that was previously absent; corrects the
  misleading "NH001" as the only modern designator
- `SurfaceSpec` class docstring: added 6/ section, corrected 7/ ISO
  reference from `ISO 10110-17` to `ISO 10110-9`

---

## O — Grating drawing completeness (ISO 10110-16)

### O1 · Groove orientation angle φ in GRAT header ✅
- ISO 10110-16 Table 2: groove orientation angle φ must appear in the
  notation when grooves are not aligned with the default direction
- Both `PlaneGrating` and `StandardGratingGeometry` carry `groove_orientation_angle`
  in radians; converted to degrees and shown as `φ = <angle>°` when non-zero
- Zero angle omitted (default grooves perpendicular to x-axis, no annotation needed)
- New test: `test_grating_groove_orientation_angle`

### O2 · StandardGratingGeometry curved substrate R and κ ✅
- `StandardGratingGeometry` has a spherical/conic substrate (radius R, conic κ)
  that was previously invisible in the GRAT header
- Now shows `R <value>` (with r_tolerance when set) and `κ = …` (when non-zero)
  between the GRAT type identifier and the grating frequency line
- New test: `test_standard_grating_curved_substrate`

---

## P — ElementSpec coercion and documentation

### P1 · striae_density string coercion ✅
- `ElementSpec.__post_init__` was missing `int()` coercion for `striae_density`
  before the `1 <= ... <= 5` range check; YAML values arrive as strings and
  `1 <= "3" <= 5` raises TypeError
- Added coercion: `self.striae_density = int(self.striae_density)`
- `fmt_birefringence` docstring corrected: was "ISO 10110-18 §5.2.2" (wrong
  part), now "ISO 10110-2 §5.2" (correct — birefringence defined in part 2)
- New test: `test_striae_density_string_coercion`

---

## Q — ASPH conic constant omit-when-zero

### Q1 · ASPH κ omitted when zero ✅
- ISO 10110-12 §4.3.1: "the conic coefficient κ shall be given when it is non-zero"
- Was: κ line always appended in ASPH header (even for κ = 0)
- BICONIC, CYL, TOROID already had the `if k_val != 0.0` guard — ASPH was missing it
- Fix: added `if k_val != 0.0` check to the ASPH branch in `_surface_header_lines()`
- A spherical asphere (κ = 0) now cleanly shows only R, polynomial A-coefficients, and
  (for Forbes Q) ρmax + a_m terms — no spurious "κ = +0" line
- Two new tests: `test_even_asphere_conic_zero_omitted` (κ=0 → no κ line),
  `test_even_asphere_conic_nonzero_shown` (κ=–1 → κ line present)

---

## T — Storage-time normalisation and round-trip robustness

### T1 · ASCII 'x' → × normalised at storage time ✅
- Was: `_fix_mul()` called only at render time in `fmt_bubbles()` / `fmt_imperfections()`;
  stored field values kept ASCII `x`, so YAML files emitted `"1x0.16"` instead of `"1×0.16"`
- Fix: `ElementSpec.__post_init__` now applies `_fix_mul()` to `bubbles`; 
  `SurfaceSpec.__post_init__` applies it to `imperfections`, `coating_imperfections`,
  `scratches`, and `assembly_imperfections` — before any R5 advisory check
- The `_fix_mul()` calls in `fmt_*()` methods are retained as safety nets for any value
  that bypasses `__post_init__`
- Four new tests verifying stored field values, YAML output, and round-trip behaviour:
  `test_imperfection_fields_normalized_at_storage`,
  `test_bubbles_normalized_at_storage`,
  `test_yaml_multiplication_sign_normalized`,
  `test_form_unit_nm_survives_roundtrip`

---

## U — Fork sync + API/architecture follow-ups from issue #458 (2026-07-16)

Folded in from the now-retired `ISO10110_PLAN.md`, whose remaining open items turned
out to already be implemented (see audit below), plus new items from the latest
round of comments on #458 (avjj / HarrisonKramer / laser0376).

### U1 · Rebased onto current upstream `origin/master` ✅
- Branch was 90 commits ahead of a `master` that was itself 84+ commits behind
  `HarrisonKramer/optiland` (now `optiland/optiland`) upstream. Rebased cleanly
  (no conflicts across new files); fixed the one real breakage: upstream added its
  own `[project.optional-dependencies]` table (prescription-console/pdf, gui, torch)
  which collided with this branch's `manufacturing` table (TOML forbids duplicate
  tables) — merged into one table.
- Upstream deprecated `Optic.surface_group` in favour of `Optic.surfaces` (removal
  in v0.7.0); updated the one call site in `elements.py::identify_elements()`.

### U2 · `ezdxf` genuinely optional (HarrisonKramer, Phase 1.1 from the old plan) ✅
- `ezdxf>=1.0.0` was listed in both the hard `dependencies` list and the
  `manufacturing` optional-extras group — not actually optional. Removed from hard
  deps; `drawing.py` already guards the import with a helpful `ImportError`, so no
  code change was needed there.

### U3 · Input validation (Phase 1.2 from the old plan) ✅ (already done, audited)
- `SurfaceSpec.__post_init__`: numeric coercion + `centration ≥ 0` and
  `ca_diameter > 0` checks already present.
- `ElementSpec.__post_init__`: `birefringence ≥ 0`, `nh_class` membership,
  `striae_density` range, `striae_shadowgraph` membership all already present.

### U4 · `DrawingSpec.set_title()` (laser0376's API-shape suggestion) ✅
- laser0376 proposed separate setters for geometry/surface/material/title
  parameters. `set_surface()`/`set_element()`/`set_material()` already matched
  that shape; the one gap was title-block fields (`part_number`, `revision`,
  `drawn_by`, `approved_by`, `notes`) living only inside `set_element()`/
  `ElementSpec` mixed with optical-quality fields.
- Added `DrawingSpec.set_title(element_index, **kwargs)`: restricted to the
  title-block field subset, merges into any spec already set by `set_element()`/
  `set_material()` rather than replacing it (call `set_element()` first if using
  both — it still fully replaces the stored spec, same as before). `ElementSpec`
  itself is unchanged, so YAML schema and renderers needed no changes.

### U5 · Pydantic validation (HarrisonKramer) — reversed, see §V1
- This was initially skipped (dataclasses + manual validation, to avoid a new
  hard dependency) without asking the user first. On review the user asked for
  pydantic explicitly, per HarrisonKramer's original request — see §V1, which
  adopts it.

### U6 · Renderer isolation (HarrisonKramer) — mostly done, gaps closed in §V2/§V3
- `drawing.py` already separated `_MatplotlibRenderer` / `_DxfRenderer` /
  `ElementDrawing` into distinct classes, but two real gaps surfaced on closer
  review: the per-surface/per-material spec-table assembly logic was
  duplicated between the two renderers (§V2), and `ezdxf` was still imported
  eagerly at module scope, undermining the "optional manufacturing extra"
  from §U2 (§V3).

---

## V — Follow-up architecture review (2026-07-16, same day as §U)

The user asked whether every review-comment topic was actually addressed, and
pushed back on two calls made in §U without asking first. Re-reviewed the code
directly rather than trusting the earlier summary; found the gaps below were
real, not just theoretical.

### V1 · Adopt pydantic for SurfaceSpec/ElementSpec ✅
- Both are now `pydantic.BaseModel` (`extra="forbid"`) instead of `@dataclass`.
  `__post_init__` bodies became `@model_validator(mode="after")` methods;
  business-rule checks (ranges, R5 advisories, warnings) are unchanged, but the
  manual `float()`/`int()` coercion loops are gone — pydantic's field typing
  already coerces numeric-looking strings, so a new field no longer needs to
  be remembered in a separate coercion list.
- `pydantic_core.ValidationError` subclasses `ValueError`, so existing
  `pytest.raises(ValueError, match=...)` tests kept working unchanged.
- `set_surface()`/`set_element()`/`set_material()` in `spec.py` no longer
  pre-filter kwargs before construction — unknown/typo'd keyword names now
  raise immediately instead of being silently dropped, same as `set_title()`
  already did.
- Found and fixed a real latent bug this surfaced: `ElementSpec.diameter_tolerance`/
  `ct_tolerance` and `SurfaceSpec.r_tolerance`/`ca_tolerance` were typed
  str-only, but the renderers' `_tol_math()`/`_tol_plain()` helpers always
  accepted bare numbers too (existing smoke tests pass `diameter_tolerance=0.05`).
  Widened all four to `str | float` to match actual usage.
- `to_dict()`/`from_dict()` now wrap `model_dump()`/`model_fields`; `from_dict()`
  stays tolerant of unknown keys (loading a YAML file saved by an older schema
  version shouldn't crash), unlike the strict direct-call API.

### V2 · Deduplicate renderer spec-table assembly ✅
- `_MatplotlibRenderer` and `_DxfRenderer` each rebuilt the identical "which
  ISO 10110 codes, in what order" logic for the per-surface (3/,4/,5/,6/,7/,
  8/,13/,15/,pyr) and per-material (0/,1/,2/) spec-table rows. The commit
  history already showed the cost — several past fixes needed both blocks
  edited in parallel.
- Extracted `_numbered_code_rows()` and `_material_code_rows()` as shared
  helpers (`drawing.py`, above the geometry-helpers section) so a future
  standards revision touching table order/contents is a one-place edit. Text
  formatting (mathtext vs. plain, encircled-λ drawing) stays in each renderer.
- Verified behavior-preserving: regenerated example drawings and diffed
  against the prior commit — only ezdxf's own nondeterministic timestamp/GUID/
  layout-ordering noise changed, zero content differences.

### V3 · Make ezdxf import lazy, genuinely optional ✅
- §U2 moved `ezdxf` to `[project.optional-dependencies].manufacturing` in
  `pyproject.toml`, but `drawing.py` still imported it at module top-level, so
  `import optiland.iso10110` (triggered unconditionally by `__init__.py`)
  required ezdxf regardless of whether DXF output was ever used.
- Worse, `ISO10110Report.generate()` — the documented entry point for every
  PDF/PNG workflow — eagerly called `ElementDrawing.generate()` (DXF-specific)
  for every element, so even a pure-PDF workflow indirectly required ezdxf.
- Fix: removed the module-level `import ezdxf` from `drawing.py`; the two
  places that actually need it (`ElementDrawing.generate()`, module-level
  `dxf_to_png()`) import it lazily inside the function body, with a clear
  `ImportError` pointing at `pip install optiland[manufacturing]`.
  `ISO10110Report.generate()` now only constructs `ElementDrawing` instances
  without building each one's DXF document; `save_dxf()` still lazily calls
  `ElementDrawing.generate()` itself, unchanged.
- New `TestOptionalEzdxfDependency` blocks `import ezdxf` via monkeypatching
  `builtins.__import__` and verifies the full `generate()`/`save_pdf()`/
  `save_png()` workflow works, while `save_dxf()` raises a clear error naming
  the extra.

---

## W — Fixes from Axel's PR review comments (2026-07-26)

### W1 · Fixed deprecated Optic API calls in Tutorial_10a ✅
- `lens.add_surface()`, `set_field_type()`, `add_field()`, `add_wavelength()`,
  `image_solve()`, `update_paraxial()`, and `.surface_group` all triggered
  `DeprecationWarning` (removal in v0.7.0). Switched to `lens.surfaces.add()`,
  `lens.fields.set_type()`/`add()`, `lens.wavelengths.add()`,
  `lens.updater.image_solve()`/`update_paraxial()`, `lens.surfaces`. Verified
  identical numeric output before/after.
- Also regenerated `docs/examples/iso10110_output/` by re-running the
  notebook — these tracked artifacts had drifted from the notebook's own spec
  (they were actually last produced by `regen_iso10110.py`, a since-removed
  separate script that set `coating=` values the notebook never did). The
  notebook is now the single source of truth for these outputs.

### W2 · Fixed diameter shown for surfaces with a physical aperture (e.g. RadialAperture) ✅
- Root cause in `LensElement.semi_aperture()` (`elements.py`): it only
  consulted `surf.semi_aperture` (populated solely by
  `optic.updater.update_paraxial()`) or, failing that, a raw paraxial
  marginal-ray estimate — completely ignoring any physical aperture
  (`surf.aperture`, e.g. `RadialAperture`) configured on the surface. This
  broke in two ways: (1) if `update_paraxial()` was never called (a common
  workflow), the aperture was silently ignored entirely; (2) even when it
  was called, `update_paraxial()`'s own logic takes `max(ray_height,
  aperture_extent)`, which shows too *large* a diameter when the aperture is
  the smaller, vignetting factor — exactly backwards for a fabrication
  drawing, which must show the true, smaller mechanical edge.
- Fix: `semi_aperture()` now checks `surf.aperture` first and uses its
  `.extent` directly whenever finite, before falling back to
  `surf.semi_aperture` or the paraxial estimate — mirroring the same
  priority `optiland.visualization.system.surface` already gives
  `surface.aperture.extent` when drawing the system layout.
- Five new tests: aperture respected with/without `update_paraxial()` called,
  the vignetting case (aperture smaller than ray height), the no-aperture
  fallback (unchanged), and `OffsetRadialAperture`'s off-axis extent.

### W3 · Made font sizes and border margin user-configurable via `DrawingStyle` ✅
- Added `optiland/iso10110/style.py`: `DrawingStyle` (pydantic `BaseModel`,
  `extra="forbid"`) with 14 named multiplicative scale factors (grouped by
  semantic role — axis labels, surface index labels, the sharp-edge "0"
  symbol, primary/component dimension callouts, spec-table header/body, the
  encircled-λ glyph, the bottom reference note, the EFL annotation, and the
  three title-block text roles) plus one absolute field, `border_margin`
  (mm). Each renderer's own already-tuned base value per role is unchanged;
  a scale factor of `1.0` (the default everywhere) reproduces the built-in
  appearance exactly, since matplotlib (pt) and DXF (mm) were never on a
  common unit scale to begin with — inventing a pt↔mm conversion would have
  been fictitious, not a restatement of the existing design.
- Threaded `style: DrawingStyle | None = None` through `_Geo`,
  `_MatplotlibRenderer`, `_DxfRenderer`, `ElementDrawing`, `draw_element()`,
  and `ISO10110Report`, defaulting to `DrawingStyle()` everywhere.
- Verified behavior-preserving: regenerated example drawings with the
  default style and diffed against the prior commit — zero content
  differences (only ezdxf's own nondeterministic noise). Ten new tests
  cover defaults, unknown-field/non-positive-value rejection, that overrides
  actually change the rendered mpl fontsize and DXF text height for a given
  role, that `border_margin` shifts the DXF layout, and that
  `ISO10110Report` forwards one shared style instance to every element's
  drawing.
- Tutorial_10a gained a new "§7 Customising Font Sizes and Layout" section
  demonstrating `DrawingStyle` on the cemented-doublet example.

## X — Maintainer review: renderer internals + example output bloat (2026-07-31)

### X1 · Dropped generated example output from git ✅
- `docs/examples/iso10110_output/` (19 tracked files, ~2.0MB — mostly the
  text-based `.dxf` files, which alone contributed ~9,500–13,000 diff lines
  each) and the orphaned, unreferenced `docs/examples/iso10110_drawings/`
  (6 files from an earlier PR #458 iteration, generated by a lens config
  that doesn't appear in any current notebook) were removed from git.
  Investigating what actually reproduces these files found that 6 of the 19
  files in `iso10110_output/` (the per-element `.png` previews) aren't
  produced by any cell in `Tutorial_11a_ISO_10110_Drawings.ipynb` at all —
  they were leftovers from the same removed `regen_iso10110.py` script noted
  in §W1, i.e. already stale before this cleanup.
- The notebook's own stored cell outputs (rendered inline via `nbsphinx`,
  which does not re-execute notebooks at doc-build time per `docs/conf.py`)
  already show the drawings visually in the docs, so no illustrative sample
  was kept in git — regenerating locally via the notebook remains the way
  to get fresh copies of these files.
- Added a `.gitignore` entry for `docs/examples/iso10110_output/` so
  re-running the notebook locally doesn't dirty the tree.

### X2 · Split `_MatplotlibRenderer.render()` into per-section methods ✅
- `render()` was ~760 lines doing borders, lens outline, hatching, dimension
  lines, spec table, title block, and callouts all in one method, with
  comment dividers already marking each section. Extracted each into its
  own private method (`_draw_borders`, `_draw_lens_outline`, `_draw_axes`,
  `_draw_sharp_edge_symbols`, `_draw_aperture_brackets`,
  `_draw_surface_finish_callouts`, `_draw_dimension_lines`,
  `_draw_spec_table`, `_draw_reference_annotation`, `_draw_efl_annotation`,
  `_draw_title_block`); `render()` is now a thin orchestrator. Pure code
  motion — the handful of values computed in one section and needed by a
  later one (`sa_e`/rim points from the lens outline, `phys_d` from the
  dimension lines) are threaded through as explicit parameters/return
  values instead of being stashed on `self`.

### X3 · Shared renderer interface — `_BaseRenderer` with format-specific primitives ✅
- Added `_BaseRenderer`, subclassed by both `_MatplotlibRenderer` and
  `_DxfRenderer`. It declares a small set of drawing primitives
  (`_prim_rect`, `_prim_hline`/`_prim_vline`, `_prim_line`, `_prim_curve`,
  `_prim_polygon`, `_prim_circle`, `_prim_text`, `_prim_dim_arrow`) plus a
  few text-formatting hooks (`_tol_fmt`, `_fmt_line`, `_fmt_dim_text`,
  `_axis_margin_mm`); each concrete renderer implements only these,
  translating an abstract `role` string into its own format's styling
  (matplotlib line weight/zorder/color; DXF layer/height). Every layout
  section *except* the lens outline/glass-hatch fill (an irreducible
  rendering-mechanism difference — matplotlib draws a solid fill plus a
  custom clipped-line hatch overlay, DXF uses an `ezdxf` `HATCH` entity with
  a named `ANSI31` pattern; unifying either would change that renderer's
  visual output) now lives once in `_BaseRenderer`, driven by those
  primitives, instead of being duplicated per format.
- Also resolved the curve/rim geometry triplication flagged by the
  maintainer: `_Geo` gained `curve_at()`/`rim_at()` (sa-parameterized
  variants of the existing `curve()`/`rim()`), and both renderers' own
  `_curve_sa`/`_rim_sa`-style closures were deleted in favor of calling
  `_Geo` directly.
- Verification: rather than trust visual inspection, built a byte-exact
  regression harness — generate the full PDF/PNG/DXF output set from a
  fixed spec with `PYTHONHASHSEED=0` (needed because ezdxf's internal
  `CLASSES` table ordering is otherwise sensitive to Python's per-process
  hash-randomization, an unrelated pre-existing nondeterminism unmasked by
  running the generator repeatedly), normalize the small set of genuinely
  nondeterministic fields (PDF `/CreationDate`, DXF Julian-date/GUID/version
  stamps), and diff against a reference captured before this refactor
  started. Ran after every migrated section, not just once at the end.
- That process surfaced several **pre-existing divergences between the two
  renderers that predate this refactor** and had to be preserved exactly
  rather than "fixed" while unifying the layout code around them:
  - The physical-outer-diameter (OD) dimension's bottom witness line was
    anchored to the *front* surface's rim in DXF but the *rear* surface's
    rim in matplotlib (both use the rear surface for the top witness line).
  - The OD dimension arrow's `xy`/`xytext` were in the opposite order from
    the center-thickness/edge-thickness/component-thickness arrows in the
    original matplotlib code — an inconsistency in the original authoring,
    not a semantic difference, but one that changes the rendered PDF bytes.
  - The optical/rotational axis line extends 15mm past the lens outline in
    DXF vs. 10mm in matplotlib.
  - Several dimension-line labels sit at different absolute offsets from
    their dimension line per format (DXF needs more clearance for its
    larger absolute text height).
  - The DXF spec-table coating-row text is vertically centered on the
    encircled-λ symbol, while matplotlib top-aligns it to the row.
  These are now documented inline at each `role`-specific branch in the
  primitive implementations, rather than living as silent, undocumented
  drift between two independently-hand-tuned code paths.

### X4 · Split `drawing.py` into per-concern modules ✅
- `drawing.py` was 2.6k lines bundling shared geometry/formatting helpers,
  the shared renderer base class, both concrete renderers, and the public
  `ElementDrawing`/`draw_element`/`dxf_to_png` API in one file — the
  outlier in a package where every other concern already gets its own
  module (`spec.py`, `style.py`, `elements.py`, `notation.py`, `report.py`).
  Split it, matching that convention:
  - `_geometry.py` — `_Geo`, tolerance parsing/formatting, ISO spec-table
    row builders, symbol-transform math, paper sizes, and the shared
    layout constants (`_SPEC_H`, `_TTL_H`, etc.).
  - `_base_renderer.py` — `_BaseRenderer` (the primitive-driven shared
    layout from §X3).
  - `_mpl_renderer.py` — `_MatplotlibRenderer` plus its own
    `_draw_glass_hatch()` helper.
  - `_dxf_renderer.py` — `_DxfRenderer`, the DXF layer-name constants, and
    its own `_dxf_arrowhead()` helper.
  - `drawing.py` — now just 248 lines: the public `ElementDrawing`,
    `draw_element()`, `dxf_to_png()`, importing renderers from the modules
    above. The public API (`optiland.iso10110.__init__` only ever imported
    `ElementDrawing`/`draw_element` from here) is unaffected.
- `tests/test_iso10110.py` reached into several now-relocated private
  helpers directly (`_mat_display_name`, `_surface_header_lines`,
  `_tol_math`/`_tol_plain`, `_iso10110_11_defaults`, `_dxf_arrowhead`) via
  `from optiland.iso10110.drawing import ...`; updated those ~38 import
  lines to the new module paths, nothing else in the test file touched.
- Verified with the same byte-exact harness as §X3 (pure file
  reorganization, so this was mostly a formality, but confirmed no
  import-order/circular-import side effect changed behavior) — zero diff
  against the pre-refactor reference.
