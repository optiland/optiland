.. _installation:

Installation
============

Optiland supports Python 3.10 and above and can be installed via pip with optional extras or can be built from source. Choose the installation method that best fits your needs.

Quick Start (Core + CPU‑only PyTorch)
-------------------------------------

To install Optiland with PyTorch support (CPU only) in one step:

.. code-block:: console

   pip install optiland[torch]

This command installs the Optiland core plus the latest PyTorch wheels (CPU‑only) from PyPI.

Standard Install (Core Only)
----------------------------

If you only need the Optiland core (no PyTorch):

.. code-block:: console

   pip install optiland

Standard Install with GUI
----------------------------

If you wish to use the Optiland GUI:

.. code-block:: console

   pip install optiland[gui]

Standard Install with Prescription Reports
------------------------------------------

To use the new prescription report generator functionality:

*   **All Prescription features** (Console Rich-text and PDF rendering):

    .. code-block:: console

       pip install optiland[prescription]

*   **Console-only** (Requires only `rich`):

    .. code-block:: console

       pip install optiland[prescription-console]

*   **PDF-only** (Requires only `reportlab`):

    .. code-block:: console

       pip install optiland[prescription-pdf]

Development Extras
------------------

To install Optiland's development dependencies-pytest, codecov, and linting tools-use:

.. code-block:: console

   pip install optiland[dev]

GPU‑Enabled PyTorch (Manual Install)
------------------------------------

If you require GPU acceleration, you must install a CUDA‑enabled PyTorch build manually. First install Optiland (core or with dev extras if desired), then follow PyTorch’s official install instructions. For example, for CUDA 11.8:

.. code-block:: console

   pip install optiland
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Note that this command installs PyTorch with CUDA 11.8. This command may not be correct for your system. Refer to https://pytorch.org/get-started/locally/ for the latest installation instructions for your platform.

Installing from Source
----------------------

To clone and install the latest development version:

1. **Clone the repository**  

   .. code-block:: console

      git clone https://github.com/optiland/optiland.git

2. **Change the directory**

   .. code-block:: console

      cd optiland

3. **Install with optional extras**  

   - Core only:  

     .. code-block:: console

        pip install .

   - With PyTorch support (CPU‑only):  

     .. code-block:: console

        pip install .[torch]

   - With development dependencies:  

     .. code-block:: console

        pip install .[dev]


Verify Your Installation
------------------------

After installation, verify that Optiland imports correctly:

.. code-block:: python

   import optiland

Optionally, generate and render a sample lens:

.. code-block:: python

   from optiland.samples.objectives import ReverseTelephoto
   lens = ReverseTelephoto()
   lens.draw3D()

.. note::
   - The first import may take a few seconds as modules and JIT-compiled code are cached.
   - If you see “Module 'torch' not found” after installing optiland[torch], ensure your environment’s PyPI index can reach the official PyTorch packages, or install PyTorch manually as shown above.
   - For any other issues, please consult our GitHub Issues page.
