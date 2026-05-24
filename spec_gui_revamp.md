# Optiland GUI Revamp — Implementation Specification

**Branch:** `gui/revamp`
**Target:** Commercial-grade PySide6 GUI competing with Zemax OpticStudio and CODE V
**Deployment:** `pip install optiland` only — no bundling, no installer
**Architecture constraint:** Strict SOLID compliance; new features must not mutate existing panel/service APIs

---

## Goals

1. Close the feature gap between the GUI and the Python API (tolerancing, missing analyses, solves, prescription, backend control)
2. Elevate visual and interaction quality to commercial-grade polish
3. Introduce genuinely differentiating features unavailable in Zemax (live first-order panel with differentiable gradients, torch backend integration in the UI)

**Out of scope:** Multi-configuration editor, OSLO file support, coating design UI, propagation/diffraction panels, polarization (Jones/Mueller) dedicated dock.

---

## Phase 0 — Prerequisites / Infrastructure Changes

These changes must land before subsequent phases depend on them.

### 0.1 Backend Control in Status Bar

**Location:** Persistent right-hand segment of the main window status bar.

**Display:** Three chips, always visible:
- `Backend: NumPy` / `Backend: PyTorch` — clickable
- `Precision: float64` / `float32` — clickable (PyTorch only; hidden for NumPy)
- `Grad: off` / `Grad: on` — clickable (PyTorch only; controls `requires_grad`)

**Interaction:**
- Clicking any chip opens a small popover (not a full dialog) with the full set of options.
- Switching backend triggers a confirmation dialog: _"Switching backend clears all cached analysis results and resets the optimization state. Continue?"_
- On confirm: call `optiland.backend.set_backend(...)`, emit `connector.backend_changed` signal, clear all analysis result caches, mark any pinned analysis windows as stale, reset optimizer state.
- Precision and grad mode changes do NOT require confirmation — they apply immediately and emit `connector.backend_settings_changed`.

**New signal on `OptilandConnector`:**
```python
backend_changed = Signal(str)          # 'numpy' | 'torch'
backend_settings_changed = Signal()    # precision or grad mode changed
```

**Settings persistence:** Backend, precision, and grad mode are stored in QSettings under `[backend]` group; restored on launch.

---

## Phase 1 — Lens Editor Improvements

**Files affected:** `optiland_gui/lens_editor.py`

### 1.1 Optimization Variable Indicators

When the optimization panel has variables defined, rows in the lens editor table whose radius, thickness, conic, or material correspond to a defined variable are marked with a small colored indicator in the row header (left gutter). Suggested: a filled circle in the theme's accent color.

- The `optimization_panel` emits a `variables_changed` signal (new) carrying a set of `(surface_index, param_name)` tuples.
- The lens editor subscribes and repaints row headers accordingly.
- The gutter column is 12 px wide; no impact on column layout.

### 1.2 Cell Validity Color Coding

Apply background tinting to cells that contain physically invalid or suspicious values. Use the existing theme system (light/dark aware).

| Condition | Color |
|---|---|
| Negative airspace (thickness < 0 for a lens surface) | Light red tint |
| Missing material on a refractive surface | Amber tint |
| Radius of curvature = 0 (singularity) | Light red tint |
| Very large radius (> 1e9) — numerically risky | Light amber tint |

Validation runs after every cell edit via a `_validate_row(row_idx)` method. No external API call required — validation is pure geometry logic from surface data already in the model.

### 1.3 Solve Exposure in the Lens Editor

The API has five solve types via `SolveFactory`:
- `marginal_ray_height` — thickness solve
- `chief_ray_height` — thickness solve
- `quick_focus` — thickness solve (auto-focus)
- `marginal_ray_angle` — curvature solve
- `chief_ray_angle` — curvature solve

**UX:**
- Right-click on a **Thickness** cell → context menu includes `Set Solve...`
- Right-click on a **Radius** cell → context menu includes `Set Solve...`
- `Set Solve...` opens a small modal dialog:
  - Dropdown: solve type (filtered by parameter type — thickness types for thickness, curvature types for radius)
  - For height-based solves: numeric input for target height
  - `Apply` / `Remove Solve` / `Cancel` buttons
- Cells with an active solve display a small lock icon (🔒 or custom SVG) in the cell and show the computed value in italics to distinguish from user-entered values.
- Solves serialize via the existing `SolveManager` / `BaseSolve.to_json()` mechanism that is already part of the optic JSON format.

**Implementation:** `SolveManager` is already on each surface (`optic.surface_group.surfaces[i].solve_manager`). The GUI calls `SolveFactory.create_solve(optic, solve_type, surface_idx, ...)` and attaches the result via `solve_manager.set_solve(param, solve_instance)`.

### 1.4 Contextual Right-Click Menu

Right-click on any row in the lens editor surface table opens a context menu:
- **Insert Surface Above / Below**
- **Delete Surface**
- **Duplicate Surface**
- **Set Solve...** (for Radius or Thickness, as above)
- **Surface Properties...** (opens existing geometry parameter dialog)
- _(separator)_
- **Copy Row** / **Paste Row**

---

## Phase 2 — Live System Properties Panel

**New file:** `optiland_gui/system_info_panel.py`
**New dock:** "System Info" — default position: bottom-right, below system properties.

### 2.1 Panel Structure

A two-column `QTableWidget` with read-only rows, grouped by section with bold header rows:

```
FIRST-ORDER PROPERTIES
  Effective Focal Length       25.000 mm
  Back Focal Length            22.341 mm
  Front Focal Length          -22.341 mm
  F/#                          2.00
  Numerical Aperture           0.249
  Entrance Pupil Diameter      12.500 mm
  Entrance Pupil Position       0.000 mm
  Exit Pupil Diameter           8.200 mm
  Exit Pupil Position          40.200 mm
  Paraxial Magnification        0.000
  Paraxial Image Height         3.600 mm

SYSTEM SUMMARY
  Surfaces (excl. OBJ/IMG)       8
  Total Track                  55.000 mm
  Fields                          3
  Wavelengths                     3
  Aperture Type             Float by stop
  Aperture Value                 12.5

GRADIENT SENSITIVITY  [optional section — see 2.2]
  ∂EFL/∂R₁                   +0.0234 mm/mm
  ...
```

Data is computed by calling `optic.paraxial` and `optic.surface_group` properties directly — no new API required.

### 2.2 Real-Time Update

- Subscribes to `connector.optic_changed` signal.
- Recomputes on every optic change. The paraxial computation is fast (< 1 ms for typical systems) so no debounce is needed.
- If paraxial computation raises an exception (e.g. invalid system), the affected rows show `—` and a tooltip with the error message.

### 2.3 Gradient Sensitivity Section (PyTorch backend only)

When the torch backend is active AND grad mode is `on`, an optional "Gradient Sensitivity" section appears at the bottom of the table.

**User control:** A `▶ Gradient Sensitivity` collapsible section header with a gear icon. Clicking the header expands/collapses the section. When expanded, a dropdown at the top of the section lets the user select which scalar metric to differentiate:
- EFL, BFL, F/#, NA, Entrance Pupil Diameter, Paraxial Magnification, Image Height

**Computation:**
1. Clone the nominal optic.
2. Set `requires_grad=True` on selected surface parameters.
3. Compute the selected metric via torch.
4. Call `.backward()`.
5. Read `.grad` from each parameter tensor.
6. Display as `∂metric/∂param` rows grouped by surface.

**Performance:** Runs in a `QThreadPool` worker. Shows a spinner in the section header during computation. Result is cached and only invalidated when optic changes or metric selection changes. Not triggered if torch is not active or grad is off — section is hidden entirely in that case.

**Implementation note:** The gradient computation must be isolated to avoid polluting the live optic's autograd graph. Use `torch.no_grad()` context for all other optic operations; gradient computation happens on a detached clone.

---

## Phase 3 — Tolerancing Dock

**New file:** `optiland_gui/tolerancing_panel.py`
**New service:** `optiland_gui/services/tolerancing_service.py`
**New dock:** "Tolerancing" — default position: tab-stacked with Optimization panel.

### 3.1 Panel Structure

Three tabs mirroring the Optimization panel pattern:

#### Tab 1: Perturbations

A table with columns:
| # | Type | Surface | Min | Max | Distribution | σ / Steps |
|---|---|---|---|---|---|---|

- **Type** dropdown: `radius`, `thickness`, `conic`, `refractive_index`, `x_tilt`, `y_tilt`, `x_decenter`, `y_decenter`
- **Surface**: integer spinner (0-indexed, clamped to surface count)
- **Min / Max**: float inputs
- **Distribution**: `Scalar` | `Uniform` | `Normal` | `Range`
- **σ / Steps**: context-sensitive — σ for Normal, steps count for Range, hidden for Scalar/Uniform

Buttons below table: `+ Add Row`, `− Remove Row`, `Clear All`

This maps directly to `Tolerancing.add_perturbation(variable_type, sampler, surface_idx=...)` where `sampler` is one of `ScalarSampler`, `RangeSampler`, or `DistributionSampler`.

#### Tab 2: Compensators & Operands

Two sub-sections:

**Compensators** — same variable picker as optimization Variables tab:
- Surface, parameter type, bounds
- Maps to `Tolerancing.add_compensator(variable_type, surface_idx=..., min_val=..., max_val=...)`

**Operands (Metrics)** — same operand picker as optimization Operands tab:
- Operand type, inputs, target
- Maps to `Tolerancing.add_operand(operand_type, input_data={})`
- At least one operand required to run

#### Tab 3: Run

**Run type selector:**
- `Monte Carlo` (uses `MonteCarlo` runner)
- `Sensitivity Analysis` (uses `SensitivityAnalysis` runner)

**Monte Carlo settings:** Number of samples (int, default 1000), seed (optional int), compensate on each run (bool checkbox).

**Sensitivity settings:** Delta value (float, default 0.001), normalize (bool).

**Run / Cancel buttons** with a `QProgressBar`. Progress fed from `tolerancing_service` via Qt signals.

**Results area** (below the run controls):
- Monte Carlo: histogram plots (one per operand) using matplotlib, embedded in a `QScrollArea`. Summary statistics (mean, std, P10, P90, Cpk if nominal is known) shown above each histogram.
- Sensitivity: table of `∂operand/∂perturbation` values, sortable by column.
- `Export DataFrame...` button: opens a save-file dialog for `.csv` or `.pkl` export of the raw pandas DataFrame returned by the runner.

### 3.2 Tolerancing Service

`tolerancing_service.py` pattern mirrors `optimization_service.py`:

```python
class TolerancingService(QObject):
    progress_updated = Signal(int, int)   # current, total
    run_completed = Signal(object)        # pd.DataFrame or dict
    run_failed = Signal(str)
    run_cancelled = Signal()

    def run_monte_carlo(self, tolerancing: Tolerancing, n_samples: int, seed: int | None, compensate: bool) -> None: ...
    def run_sensitivity(self, tolerancing: Tolerancing, delta: float, normalize: bool) -> None: ...
    def cancel(self) -> None: ...
```

The service wraps `MonteCarlo.run()` and `SensitivityAnalysis.run()` in a `QRunnable` worker with a cancellation flag checked between samples.

### 3.3 State Persistence

Tolerancing specs are **ephemeral** (consistent with the Optimization panel). State is not saved to the optic JSON file. Users are expected to re-enter tolerance specs per session, or use the Python terminal for scripted setup.

---

## Phase 4 — Missing Analysis Types + Pinned Windows

### 4.1 Analysis Types to Add

Add the following to the analysis panel dropdown. All follow the existing `BaseAnalysis` pattern and fit into the current `analysis_runner.py` / `analysis_panel.py` framework.

| Analysis Type | API Class | Notes |
|---|---|---|
| Through-Focus MTF | `analysis.ThroughFocusMTF` | Settings: frequency, field, focus range, steps |
| MTF vs Field | `analysis.MTFvsField` | Settings: frequency, num fields |
| Radiant Intensity | `analysis.RadiantIntensity` | Settings: wavelength, field |
| Incoherent Irradiance | `analysis.IncoherentIrradiance` | Settings: wavelength, grid size, field |
| PSF | `analysis.PSF` | Settings: wavelength, field, grid size |
| Image Simulator | `analysis.ImageSimulator` | Settings form includes a file-picker button for the input image (PNG/JPG); image path stored in analysis config dict |
| Jones Pupil | `analysis.JonesPupil` | Settings: wavelength, field, polarization state (Jones vector inputs Ex, Ey as complex floats) |

For **ImageSimulator**, the settings form renders:
- `Input Image:` label + `[Browse...]` button + a path display label
- Standard field/wavelength selectors

For **JonesPupil**, the settings form renders:
- Field / wavelength selectors
- `Ex_real`, `Ex_imag`, `Ey_real`, `Ey_imag` float inputs (Jones vector components)

### 4.2 Pinned Analysis Windows (Hybrid Model)

**Current behavior:** Single Analysis dock, one analysis at a time, dropdown replaces current result.

**New behavior:**

1. A `📌 Pin` button is added to the analysis panel toolbar (next to the run button).
2. Clicking `Pin` while results are displayed creates a new `PinnedAnalysisWindow`:
   - A `QDockWidget` subclass with a custom title bar showing the analysis name and a close button.
   - Spawned and added to the main window as a floating dock by default (user can dock it anywhere).
   - Contains a copy of the current matplotlib figure (deep copy of the `Figure` object).
   - Stores the analysis type + settings used to produce it.
3. When `connector.optic_changed` fires, all pinned windows apply a **stale watermark**: a semi-transparent `QLabel` overlay reading `"Results outdated — click to refresh"` with a refresh icon, positioned center-overlay using `QStackedLayout`.
4. Clicking the overlay triggers a background re-run of that analysis with its stored settings, removes the overlay on completion.
5. Pinned windows are NOT persisted across sessions.

**New class:** `optiland_gui/widgets/pinned_analysis_window.py`

```python
class PinnedAnalysisWindow(QDockWidget):
    def __init__(self, analysis_type: str, settings: dict, figure: Figure, connector: OptilandConnector): ...
    def mark_stale(self) -> None: ...
    def refresh(self) -> None: ...
```

**SOLID compliance:** `PinnedAnalysisWindow` depends on `OptilandConnector` and `AnalysisRunner` via constructor injection. The `AnalysisPanel` creates pinned windows via a factory function injected at panel construction time — it does not import `PinnedAnalysisWindow` directly, avoiding a circular dependency.

---

## Phase 5 — General Quality & Polish

### 5.1 Persistent Status Bar

The main window status bar (bottom) is divided into three zones:

**Left zone** — system summary (updates on `optic_changed`):
```
Surfaces: 8   |   Fields: 3   |   Wavelengths: 3   |   EFL: 25.000 mm   |   F/#: 2.00
```
These values come from the same paraxial computation as the System Info panel. If paraxial fails, show `EFL: —`.

**Center zone** — transient operation status (cleared after 3 s):
```
Analysis complete (0.42 s)    [success in theme accent color]
Optimization converged        [success]
File saved                    [success]
Error: singular matrix        [error in red]
```
This replaces the current toast system for operations that have a natural status bar representation. Toasts are retained for non-operation events (e.g., theme switched, file loaded).

**Right zone** — backend chips (Phase 0).

### 5.2 Welcome Screen

A `WelcomeDialog` shown on first launch and accessible via `Help > Welcome Screen`.

Layout: two-column card grid.
- **Left column:** Recent Files list (max 10, from QSettings), each as a clickable card with filename and last-modified date.
- **Right column:** Sample gallery cards (subset of existing samples), Quick-start templates (placeholder for now), link to documentation.

A `Don't show on startup` checkbox in the bottom-left corner. State stored in QSettings.

**New file:** `optiland_gui/widgets/welcome_dialog.py`

### 5.3 Keyboard Shortcut Completeness

Audit and fill gaps. Required shortcuts:

| Action | Shortcut |
|---|---|
| Run Analysis | `F5` |
| Pin Analysis Result | `Ctrl+Shift+P` |
| New File | `Ctrl+N` |
| Open File | `Ctrl+O` |
| Save | `Ctrl+S` |
| Save As | `Ctrl+Shift+S` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Command Palette | `Ctrl+K` |
| Toggle Theme | `Ctrl+Shift+T` |
| Add Surface | `Ctrl+Enter` (when LDE focused) |
| Delete Surface | `Delete` (when LDE row selected) |
| Run Optimization | `F7` |
| Run Tolerancing | `F8` |
| Show Keyboard Shortcuts | `Ctrl+?` |

`Help > Keyboard Shortcuts` opens a `QDialog` rendering a two-column table (Action / Shortcut) organized by category. The table is generated from the `ActionManager` registry, ensuring it stays in sync automatically.

### 5.4 Contextual Right-Click Menus (Throughout)

Beyond the lens editor (Phase 1.4), add right-click menus to:

**System Properties — Fields table:**
- Add Field, Remove Field, Duplicate Field

**System Properties — Wavelengths table:**
- Add Wavelength, Remove Wavelength, Set as Primary

**Analysis Panel — result plot area:**
- Copy Image to Clipboard, Save Image As..., Pin Result (same as pin button)

**Viewer Panel — 2D/3D tabs:**
- Copy Image to Clipboard, Save Image As..., Reset View

### 5.5 Prescription Report Access

A `Reports > Prescription Report...` menu action (new `Reports` top-level menu).

Opens a modal `QDialog` with:
- A `QTextEdit` (read-only, monospace font) showing the plain-text prescription via `optiland.Prescription(optic).render_text()`.
- Toolbar above the text: `Regenerate`, `Copy to Clipboard`, `Export PDF...`, `Export Text...`
- PDF export calls `optiland.Prescription(optic).render_pdf(filepath)`.

No live updates — this is a snapshot report generated on demand.

**New file:** `optiland_gui/widgets/prescription_dialog.py`

### 5.6 About Box

`Help > About Optiland` opens a `QDialog` showing:
- Optiland logo (SVG if available, else text)
- Version number (from `optiland.__version__`)
- Python version, PySide6 version, NumPy version, PyTorch version (if installed)
- Link to GitHub repository
- License information

---

## Architecture Decisions & Constraints

### SOLID Compliance Rules

1. **Single Responsibility:** Each panel/service handles one concern. `TolerancingPanel` owns the UI; `TolerancingService` owns the threading and API calls; `Tolerancing` (API) owns the math.
2. **Open/Closed:** New analysis types are registered in a `dict[str, AnalysisConfig]` data structure in `analysis_panel.py`, not in `if/elif` chains. Adding a new analysis type requires adding one entry to that dict, not modifying conditional logic.
3. **Liskov Substitution:** `PinnedAnalysisWindow` is a `QDockWidget` subclass — it must be substitutable wherever a dock widget is used.
4. **Interface Segregation:** Services expose narrow signal/slot interfaces. `PinnedAnalysisWindow` receives only `OptilandConnector` signals it cares about, not the full connector object. Pass specific signals, not the connector reference, where possible.
5. **Dependency Inversion:** Panels depend on service abstractions (injected at construction time via `panel_manager.py`). No panel imports a concrete service class directly.

### Threading Model

- All heavy computations (tolerancing MC, gradient sensitivity, optimization, analysis) run in `QRunnable` workers via `QThreadPool`.
- Workers communicate results back to the UI via `Signal` on a `QObject` (services pattern already in use).
- Cancellation uses a `threading.Event` flag polled between samples/iterations — no `QThread.terminate()` calls.

### Analysis Config Registry

Replace the current `if/elif` analysis dispatch in `analysis_panel.py` with a declarative registry:

```python
@dataclass
class AnalysisConfig:
    display_name: str
    api_class: type
    settings_form: type   # QWidget subclass that builds the settings form
    default_settings: dict

ANALYSIS_REGISTRY: dict[str, AnalysisConfig] = {
    "spot_diagram": AnalysisConfig(...),
    "through_focus_mtf": AnalysisConfig(...),
    # ...
}
```

New analysis types are registered by appending to `ANALYSIS_REGISTRY`. The panel iterates the registry to populate the dropdown and construct forms — no conditional logic.

### JSON / State Persistence Summary

| Feature | Persisted? | Where |
|---|---|---|
| Optic geometry | Yes | Optic JSON (existing) |
| Solve locks | Yes | Embedded in surface JSON (via `SolveManager.to_json()`) |
| Optimization problem | No | Ephemeral (existing behavior) |
| Tolerance specs | No | Ephemeral (consistent with optimization) |
| Analysis results | No | Ephemeral |
| Pinned windows | No | Ephemeral |
| Backend settings | Yes | QSettings `[backend]` group |
| Recent files | Yes | QSettings `[recent_files]` group |
| Welcome dialog suppressed | Yes | QSettings `[ui]` group |
| Window layout | Yes | QSettings (existing) |

---

## Implementation Phases & Order

| Phase | Deliverable | Depends On |
|---|---|---|
| 0 | Backend control status bar chips | Nothing |
| 1 | Lens editor: validity colors, variable indicators, solve exposure, right-click menu | Phase 0 |
| 2 | System Info / live first-order panel | Phase 0 |
| 3 | Tolerancing dock + service | Phase 0 |
| 4a | Analysis registry refactor + 7 new analysis types | Nothing |
| 4b | Pinned analysis windows | Phase 4a |
| 5a | Status bar system summary | Phase 2 (shares paraxial compute) |
| 5b | Welcome screen | Nothing |
| 5c | Keyboard shortcut completeness + cheatsheet dialog | Phase 0–4 complete (so shortcuts are final) |
| 5d | Context menus throughout | Phase 1 (LDE), Phase 4a (analysis) |
| 5e | Prescription dialog | Nothing |
| 5f | About box | Nothing |

---

## Files To Create (New)

```
optiland_gui/tolerancing_panel.py
optiland_gui/services/tolerancing_service.py
optiland_gui/system_info_panel.py
optiland_gui/widgets/pinned_analysis_window.py
optiland_gui/widgets/welcome_dialog.py
optiland_gui/widgets/prescription_dialog.py
optiland_gui/widgets/solve_dialog.py
optiland_gui/widgets/keyboard_shortcuts_dialog.py
optiland_gui/widgets/about_dialog.py
```

## Files To Modify (Existing)

```
optiland_gui/main_window.py              # add new docks, menus, status bar zones, welcome dialog
optiland_gui/panel_manager.py            # register new panels
optiland_gui/action_manager.py           # new actions, keyboard shortcuts
optiland_gui/optiland_connector.py       # backend_changed signal, backend_settings_changed signal
optiland_gui/lens_editor.py              # validity colors, solve exposure, variable indicators, right-click
optiland_gui/analysis_panel.py           # registry refactor, pin button, 7 new analysis types
optiland_gui/services/analysis_runner.py # pass through to registry-based dispatch
```
