Installation
============

To get started with Optiland, ensure you have a compatible Python version installed.
Then, you can install Optiland directly from PyPI with:

.. code-block:: bash

   pip install optiland

For the latest development version, you can clone the repository and install it locally in editable mode:

.. code-block:: bash

   git clone https://github.com/optiland/optiland.git
   cd optiland
   pip install -e . --group dev   # editable install with test and lint tools

With `uv <https://docs.astral.sh/uv/>`_, ``uv sync --group dev`` does the same (add
``--extra torch`` to run the PyTorch-backend tests).

Installation details are also outlined in the main :ref:`installation` section of the documentation.
