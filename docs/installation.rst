.. _installation:

.. meta::
   :description: Install Optiland with pip, add optional extras for PyTorch, the GUI, reports and drawings, set up GPU support, or build from source.

Installation
============

Optiland supports Python 3.11 and newer. For most people, one command is
enough:

.. code-block:: console

   pip install optiland

That gives you the full sequential and non-sequential engines on the NumPy
backend, plus 2D (Matplotlib) and 3D (VTK) plotting. Everything else is
optional, so add it when you need it.

.. tip::

   Installing into a fresh virtual environment (``python -m venv .venv``, a
   conda environment, or ``uv venv``) keeps Optiland and its dependencies
   separate from the rest of your system. It is not required, but it does
   save headaches later.

Optional extras
---------------

Extras are installed with square brackets, for example
``pip install "optiland[torch,gui]"``. (The quotes keep shells like zsh from
interpreting the brackets.)

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Extra
     - What it adds
   * - ``torch``
     - CPU-only PyTorch, for the differentiable backend
       (:doc:`Tutorial 7a <examples/Tutorial_7a_Differentiable_Ray_Tracing_Hello_World>`).
       For an NVIDIA GPU, see `GPU support`_ below.
   * - ``gui``
     - The Optiland desktop GUI (PySide6). See the :doc:`gui_quickstart`.
   * - ``prescription``
     - Prescription reports in the console (``rich``) and as PDF
       (``reportlab``). ``prescription-console`` and ``prescription-pdf``
       install only one of the two.
   * - ``manufacturing``
     - DXF export of ISO 10110 fabrication drawings (``ezdxf``). PDF and PNG
       drawings work without it.

GPU support
-----------

The ``torch`` extra installs the CPU-only PyTorch build from PyPI. For GPU
acceleration, install Optiland first and then a CUDA build of PyTorch that
matches your NVIDIA driver. For example, for CUDA 12.6:

.. code-block:: console

   pip install optiland
   pip install torch --index-url https://download.pytorch.org/whl/cu126

The right command depends on your platform and driver, so check the
`PyTorch installation selector <https://pytorch.org/get-started/locally/>`_.
On Apple Silicon or a non-NVIDIA GPU, use the CPU-only ``torch`` extra.

Once PyTorch is installed, switch backends in your code:

.. code-block:: python

   import optiland.backend as be

   be.set_backend("torch")
   be.set_device("cuda")   # or "cpu"

Installing from source
----------------------

To get the latest development version, or to contribute, clone the
repository and install it in editable mode:

.. code-block:: console

   git clone https://github.com/optiland/optiland.git
   cd optiland
   pip install -e .               # core only
   pip install -e ".[torch]"      # with PyTorch
   pip install -e . --group dev   # with test and lint tools (pip 25.1 or newer)

If you use `uv <https://docs.astral.sh/uv/>`_, ``uv sync --group dev`` (and
``--extra torch`` for PyTorch) sets up the same development environment. The
:doc:`contributing` guide covers the rest of the development workflow.

Check that it works
-------------------

.. code-block:: python

   from importlib.metadata import version
   from optiland.samples.objectives import ReverseTelephoto

   print(version("optiland"))
   lens = ReverseTelephoto()
   lens.draw()      # 2D layout; lens.draw3D() opens an interactive 3D view

If a figure with a lens and rays appears, you are all set. Head to the
:doc:`quickstart` next.

Troubleshooting
---------------

- **The first import is slow.** Optiland compiles and caches some functions
  the first time it runs, so the first import can take a few seconds. Later
  imports are faster.
- **"No module named 'torch'".** The ``torch`` extra did not install, often
  because your package index cannot reach PyTorch's wheels. Install PyTorch
  directly as shown in `GPU support`_.
- **No 3D window appears.** ``draw3D()`` needs a desktop session. On a remote
  machine or in CI, use ``draw()`` for 2D plots.
- **Something else?** Search the
  `GitHub issues <https://github.com/optiland/optiland/issues>`_ or ask in
  `GitHub Discussions <https://github.com/optiland/optiland/discussions>`_.
  Someone has probably hit it before.
