# Copilot Instructions for Optiland

## Repository Overview

**Optiland** is an open-source optical design library in Python for designing, analyzing, and optimizing optical systems. It provides comprehensive tools for ray tracing, optical system definition, aberration analysis, optimization, and visualization.

### Key Facts
- **Version**: 0.6.0
- **Languages**: Python 3.10-3.13
- **Size**: ~25 core modules, 1700+ tests, comprehensive documentation
- **Dependencies**: numpy, scipy, matplotlib, vtk, pandas, pyyaml, tabulate, numba, requests, seaborn
- **Optional extras**: `gui` (PySide6), `torch` (PyTorch backend), `dev` (pytest, codecov)

## Build and Validation Workflow

### Required Tools
- **Package Manager**: `uv` (modern Python package manager) - **ALWAYS install first**
- **Linter/Formatter**: `ruff` (replaces black, flake8, isort) 
- **Testing**: `pytest` with coverage support
- **Build**: `hatchling` backend via `uv build`

### Essential Setup Commands (Run in Order)
```bash
# 1. Install uv package manager first
pip install uv

# 2. Set up development environment and install all dependencies
uv sync --dev

# 3. Install additional development tools
pip install ruff pre-commit  # Only if not available via uv
```

### Core Development Commands

#### Linting and Formatting
```bash
# Check code quality (ALWAYS run before committing)
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Check formatting
ruff format --check .

# Apply formatting
ruff format .
```

#### Testing
```bash
# Run all tests (~5-10 minutes)
uv run pytest

# Run specific test module (~20-30 seconds for analysis modules)
uv run pytest tests/test_optic.py

# Run tests with coverage (for PRs)
uv run pytest --cov=optiland --cov-report=xml

# Quick test to verify core functionality
uv run pytest tests/test_optic.py tests/test_materials.py --quiet
```

#### Build and Package
```bash
# Build package (creates dist/ directory)
uv build

# Verify package builds successfully
ls dist/  # Should show .whl and .tar.gz files
```

### Pre-commit Hooks
The repository uses pre-commit hooks that automatically run ruff checks:
```bash
# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### Timing Expectations
- **uv sync --dev**: ~1-2 minutes (downloads large dependencies like vtk)
- **ruff check/format**: <5 seconds
- **pytest single module**: 20-30 seconds (analysis modules)
- **pytest full suite**: 5-10 minutes (1700+ tests)
- **uv build**: <10 seconds

### Common Issues and Workarounds
1. **Network timeouts**: uv/pip may timeout downloading large dependencies (vtk ~100MB). Retry if needed.
2. **NumPy warnings**: Test runs show deprecation warnings about scalar conversion - these are non-blocking.
3. **Pre-commit failures**: If pre-commit installation fails due to network issues, use `ruff` directly.
4. **Group object lengths**: Use `.num_fields`, `.num_wavelengths`, `.num_surfaces` instead of `len()` on FieldGroup/WavelengthGroup/SurfaceGroup objects.

### Environment Requirements
- **Memory**: Recommend ≥8GB RAM for large-scale simulations
- **Display**: For GUI (`optiland[gui]`), requires display server (X11/Wayland on Linux)
- **CUDA**: Optional for PyTorch backend acceleration (`optiland[torch]`)

## Project Architecture and Layout

### Core Modules (optiland/)
- **`optic/`**: Core `Optic` class - starting point for all optical systems
- **`rays/`**: Ray objects and generators for light propagation simulation  
- **`surfaces/`**: Building blocks for optical elements (lenses, mirrors)
- **`optimization/`**: Tools for optimizing optical systems
- **`analysis/`**: Analysis tools (spot diagrams, MTF, PSF, wavefront, aberrations)
- **`materials/`**: Material database and optical property calculations
- **`visualization/`**: 2D/3D plotting and system drawing capabilities
- **`fileio/`**: Import/export functionality for optical design files

### Key Configuration Files
- **`pyproject.toml`**: Main project configuration (dependencies, build, ruff settings)
- **`.pre-commit-config.yaml`**: Pre-commit hooks configuration
- **`uv.lock`**: Dependency lock file (DO NOT edit manually)

### Testing Structure (tests/)
- **1700+ tests** across all modules
- **Parametrized backends**: Most tests run with both numpy and torch backends
- **Coverage reporting**: Integrated with codecov
- **Test utilities**: `tests/utils.py` provides common test helpers

### Documentation (docs/)
- **Sphinx-based** with Read the Docs theme
- **Auto-generated API docs** from docstrings
- **Example gallery** with executable scripts
- **Developer guides** for contributors

### GitHub Workflows (.github/workflows/)
The CI pipeline runs:
1. **Lock file validation**: `uv lock --locked`
2. **Linting**: `uvx ruff check .`  
3. **Formatting**: `uvx ruff format --check .`
4. **Testing**: Multi-version tests (Python 3.10-3.13) with coverage
5. **Build**: `uv build` to verify package creation

## Development Best Practices

### Code Style (Enforced by Ruff)
- **Line length**: 88 characters maximum
- **Import order**: stdlib, third-party, local (with `from __future__ import annotations`)
- **Docstrings**: Google style for all public functions/classes
- **Type hints**: Required for new code using `typing-extensions`

### Making Changes
1. **Always run `uv sync --dev` first** to ensure environment is up to date
2. **Use `ruff check . --fix && ruff format .`** before committing
3. **Run relevant tests** for the area you're modifying
4. **Update documentation** if adding new features or changing APIs
5. **Check `git status`** to ensure only intended files are staged

### Common Patterns
- **Backend abstraction**: Code uses `optiland.backend` for numpy/torch compatibility
- **Parametrized testing**: Tests run with `backend=numpy` parameter
- **Modular design**: Each major component is self-contained
- **Material handling**: Use material database in `optiland/database/`

## File Structure Reference

### Repository Root
```
optiland/
├── optiland/           # Main package
├── optiland_gui/       # GUI application (PySide6)
├── tests/              # Test suite (1700+ tests)
├── docs/               # Sphinx documentation
├── pyproject.toml      # Project configuration
├── uv.lock            # Dependency lock file
└── README.md          # Project overview
```

### Key Entry Points
- **Main API**: `optiland.optic.Optic` - primary interface for optical systems
- **GUI**: `optiland_gui.run_gui:main` - graphical interface
- **Examples**: `docs/examples/` - Jupyter notebook tutorials and examples

### Quick API Test
```python
# Basic optical system creation (use for testing changes)
import numpy as np
from optiland import optic

system = optic.Optic()
system.surfaces.add(index=0, thickness=np.inf)
system.surfaces.add(index=1, thickness=7, radius=20.0, is_stop=True, material='N-SF11')
system.surfaces.add(index=2, thickness=23.0)
system.surfaces.add(index=3)
system.set_aperture(aperture_type='EPD', value=20)
system.fields.set_type(field_type='angle')
system.fields.add(y=0)
system.wavelengths.add(value=0.587, is_primary=True)

# Verify system properties
print(f"Surfaces: {system.surface_group.num_surfaces}")
print(f"Fields: {system.fields.num_fields}")
print(f"Wavelengths: {system.wavelengths.num_wavelengths}")
```

## Validation Steps for Changes

### Before Committing
```bash
# 1. Lint and format
ruff check . --fix && ruff format .

# 2. Run tests for modified areas
uv run pytest tests/test_[relevant_module].py

# 3. Verify build works
uv build
```

### For Pull Requests
```bash
# Run full test suite with coverage
uv run pytest --cov=optiland --cov-report=xml

# Verify formatting is consistent
ruff format --check .

# Check for linting issues
ruff check .
```

### Known TODO Items and Limitations
- Lens surface overlap detection (visualization/system/lens.py:34)
- Decentering implementation in file converters (fileio/converters.py:237)  
- Vectorization improvements in tolerancing (tolerancing/perturbation.py:128)
- Polychromatic MTF support (mtf/geometric.py:156)

## Trust These Instructions

These instructions are comprehensive and tested. Only search for additional information if:
- The instructions are incomplete for your specific task
- You encounter errors not covered in the "Common Issues" section
- You need details about specific modules not covered here

The build and test commands listed here are verified to work correctly. Follow them exactly to minimize setup time and avoid common pitfalls.