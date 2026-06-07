import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../optiland/"))

project = "Optiland"
current_year = datetime.now().year
copyright = f"2024-{current_year}, Kramer Harrison & contributors"
author = "Kramer Harrison"
release = "0.5.8"

extensions = [
    "jupyterlite_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_gallery.gen_gallery",
]

add_module_names = False  # Remove module names from class and function names
autosummary_generate = True  # Automatically generate summaries

templates_path = ["_templates"]
modindex_common_prefix = ["optiland."]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"navigation_depth": 2}

sphinx_gallery_conf = {
    "examples_dirs": "examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # gallery output directory
}

autodoc_mock_imports = [
    "numpy",
    "yaml",
    "scipy",
    "matplotlib",
    "numba",
    "pandas",
    "vtk",
    "torch",
]

pygments_style = "sphinx"

# Allow notebook execution errors to not abort the build.
# Some gallery notebooks have no committed outputs and fail on execution
# (e.g. torch notebooks that require a GPU or specific environment).
# This prevents those failures from crashing the whole build.
nbsphinx_allow_errors = True

# Autodoc configuration: include only public members by default
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "inherited-members": True,
}

# Jupyterlite configuration
jupyterlite_bind_ipynb_suffix = False
