System Requirements
===================

Optiland is a Python-based library, and it is compatible with Python versions **3.11 through 3.14**. 

Operating System and Hardware
-----------------------------

Python is a cross-platform language, and Optiland is expected to work on Windows, macOS, and Linux systems.
Testing is performed on Linux (Ubuntu latest) and Windows 11. 

The library does not impose specific hardware requirements. However, for large-scale simulations or complex optimization tasks, we recommend using a machine with:

- **CPU**: Multi‑core processor  
- **Memory**: ≥ 8 GB RAM  
- **GPU** (optional):  
  - Required for **CUDA** acceleration with the PyTorch backend.  
  - If you plan to use GPU‑accelerated ray tracing or differentiable optics, install a CUDA‑enabled PyTorch build manually (see :ref:`installation`).  
  - Without a GPU, you can still use the PyTorch backend in CPU‑only mode (installed via `optiland[torch]`) or the NumPy backend.  

Dependencies
------------

The following Python libraries are required to use Optiland. These dependencies are automatically installed when you install Optiland via `pip`:

- `numpy` (for vectorized ray tracing)
- `scipy` (for optimization routines)
- `matplotlib` (for basic visualization)
- `vtk` (for advanced 3D visualization)
- `pandas` (for data manipulation)
- `pyyaml` (for reading and writing YAML files)
- `tabulate` (for tabular data formatting)
- `numba` (for just-in-time compilation)
- `requests` (for HTTP requests)
- `seaborn` (for statistical data visualization)

You can view the complete list of dependencies in the `pyproject.toml` file of the repository.
