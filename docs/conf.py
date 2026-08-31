"""Sphinx configuration for the Optiland documentation.

The file is organised in numbered sections:

1. Project metadata and version
2. Extensions
3. Autodoc, autosummary and Napoleon
4. Notebooks (nbsphinx) and JupyterLite
5. HTML theme and branding (PyData Sphinx Theme)
6. Canonical URLs, sitemap, 404 page and page metadata
7. Source and edit links
8. Environment-specific settings and build hooks

Environment variables (all optional):

OPTILAND_DOCS_BASE_URL
    Canonical base URL of the published documentation. Defaults to the
    first-party location ``https://www.optiland.org/docs/``.
OPTILAND_DOCS_VERSION
    Version label used when the ``optiland`` package metadata is unavailable
    (for example when Sphinx runs against an uninstalled checkout).
OPTILAND_DOCS_GIT_SHA
    Source commit recorded in ``_meta/build.json``. Falls back to ``git``.
OPTILAND_DOCS_SKIP_JUPYTERLITE
    Set to ``1`` to skip the JupyterLite build. Intended only for local
    iteration on hosts without an emscripten toolchain (for example Windows,
    where micromamba cannot create ``emscripten-wasm32`` environments). The
    ``replite`` directive then renders a placeholder note. Never set in CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from urllib.parse import urlparse

from docutils import nodes
from docutils.parsers.rst import Directive, directives

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent

# The API stubs under docs/api/ document most modules by their in-package name
# (``analysis.distortion`` rather than ``optiland.analysis.distortion``), which
# relies on the package directory itself being importable. Keep both entries.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "optiland"))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


# -- 1. Project metadata and version ------------------------------------------

project = "Optiland"
author = "Kramer Harrison"
copyright = (
    f"2024-{datetime.now(timezone.utc).year}, Kramer Harrison & contributors"
)


def _resolve_release() -> str:
    """Return the version of the package being documented.

    The version is never hardcoded. It comes from the installed package
    metadata (``versioningit`` derives it from git tags, so a build from
    ``master`` yields an honest ``X.Y.Z.postN+g<sha>`` development label),
    falling back to ``OPTILAND_DOCS_VERSION`` and then ``git describe``.
    """
    try:
        return package_version("optiland")
    except PackageNotFoundError:
        pass
    if os.environ.get("OPTILAND_DOCS_VERSION"):
        return os.environ["OPTILAND_DOCS_VERSION"]
    described = _git("describe", "--tags", "--always", "--dirty")
    return described.removeprefix("v") if described else "dev"


release = _resolve_release()
version = ".".join(release.split(".")[:2])
git_sha = (
    os.environ.get("OPTILAND_DOCS_GIT_SHA") or _git("rev-parse", "HEAD") or "unknown"
)

# -- 2. Extensions -------------------------------------------------------------

SKIP_JUPYTERLITE = _env_flag("OPTILAND_DOCS_SKIP_JUPYTERLITE")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_last_updated_by_git",
    "sphinx_sitemap",
    "notfound.extension",
    "nbsphinx",
]
if not SKIP_JUPYTERLITE:
    extensions.insert(0, "jupyterlite_sphinx")

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "_contents",  # staging directory created by jupyterlite-sphinx
    "jupyterlite/**",  # mock packages for the in-browser kernel, not pages
]

# -- 3. Autodoc, autosummary and Napoleon --------------------------------------

autosummary_generate = True
add_module_names = False
modindex_common_prefix = ["optiland."]

# Heavyweight scientific / visualization dependencies are mocked so that the
# documentation can be built without them (API pages only need signatures and
# docstrings).
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

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    # Document inherited members, except those inherited from Python builtins
    # and stdlib bases: their docstrings are not reStructuredText and produce
    # parser warnings (for example ``int.to_bytes`` on ``IntEnum`` subclasses).
    "inherited-members": (
        "object, int, float, str, bytes, tuple, list, dict, set, frozenset, "
        "Enum, IntEnum, StrEnum, Exception, BaseException"
    ),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render Google-style ``Attributes:`` sections as inline ``:ivar:`` fields rather
# than separate ``.. attribute::`` object descriptions. This avoids "duplicate
# object description" warnings for dataclasses whose fields are documented both
# by the docstring Attributes section and by autodoc's member enumeration.
napoleon_use_ivar = True
# Render Google-style ``Methods:`` sections as a plain field list. Napoleon's
# default turns them into ``.. method::`` object descriptions, which collide
# with the members autodoc already documents (the same duplicate warnings).
napoleon_custom_sections = [("Methods", "params_style")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
intersphinx_timeout = 30

# -- 4. Notebooks and JupyterLite ----------------------------------------------

# Notebooks are committed together with their outputs and are never executed
# during a documentation build: builds stay deterministic and do not need a
# kernel or the optional heavyweight dependencies.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# The in-browser environment for the "Try Optiland" page is defined by
# docs/environment.yml; jupyterlite-xeus builds it with micromamba as part of
# the Sphinx build.
jupyterlite_bind_ipynb_suffix = False

# -- 5. HTML theme and branding ------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"Optiland {release} documentation"
html_short_title = "Optiland documentation"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_css_files = [
    (
        "https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;"
        "9..40,500;9..40,600;9..40,700&family=Outfit:wght@500;600;700"
        "&family=JetBrains+Mono:wght@400;500&display=swap"
    ),
    "optiland-docs.css",
]
html_last_updated_fmt = "%Y-%m-%d"
html_show_sourcelink = True
html_sourcelink_suffix = ""  # link to the real .ipynb / .rst sources
html_sidebars = {"index": [], "404": []}

html_theme_options = {
    "logo": {
        "image_light": "_static/logo-mark.png",
        "image_dark": "_static/logo-mark.png",
        "text": "Optiland",
        "alt_text": "Optiland",
        "link": "https://www.optiland.org/",
    },
    # Header: one link per top-level section of the root toctree.
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["search-button-field", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "external_links": [
        {"name": "Community", "url": "https://www.optiland.org/forum"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/optiland/optiland",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/optiland/",
            "icon": "fa-brands fa-python",
        },
    ],
    # Navigation.
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": True,
    "show_toc_level": 2,
    "show_prev_next": True,
    "back_to_top_button": True,
    "search_bar_text": "Search the docs (Ctrl+K)",
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page", "sourcelink"],
        "index": [],
    },
    # Source links (see html_context below).
    "use_edit_page_button": True,
    # Footer.
    "footer_start": ["copyright", "optiland-build-info"],
    "footer_center": [],
    "footer_end": ["optiland-footer-links"],
    # Code highlighting.
    "pygments_light_style": "a11y-high-contrast-light",
    "pygments_dark_style": "a11y-high-contrast-dark",
}

# -- 6. Canonical URLs, sitemap, 404 page and page metadata --------------------

html_baseurl = (
    os.environ.get("OPTILAND_DOCS_BASE_URL", "https://www.optiland.org/docs/")
    .strip()
    .rstrip("/")
    + "/"
)

sitemap_url_scheme = "{link}"
sitemap_locales = [None]
sitemap_show_lastmod = True
sitemap_excludes = ["search.html", "genindex.html", "py-modindex.html", "404.html"]

# The 404 page can be served at any depth, so its links must be absolute
# under the public path prefix (``/docs/`` in production).
notfound_urls_prefix = urlparse(html_baseurl).path or "/"
notfound_template = "page.html"

optiland_description = (
    "Optiland is an open-source Python framework for optical design, analysis "
    "and optimization, with NumPy and differentiable PyTorch backends."
)

# -- 7. Source and edit links --------------------------------------------------

html_context = {
    "github_user": "optiland",
    "github_repo": "optiland",
    "github_version": "master",
    "doc_path": "docs",
    "default_mode": "auto",
    # Consumed by _templates/layout.html and _templates/components/*.html.
    "optiland_release": release,
    "optiland_git_sha": git_sha,
    "optiland_base_url": html_baseurl,
    "optiland_site_url": "https://www.optiland.org/",
    "optiland_description": optiland_description,
}

# -- 8. Environment-specific settings and build hooks --------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_exclude = ".linenos, .gp, .go"


class _RepliteUnavailable(Directive):
    """Placeholder for ``replite`` when the JupyterLite build is skipped."""

    has_content = True
    option_spec = {
        name: directives.unchanged
        for name in (
            "kernel",
            "toolbar",
            "width",
            "height",
            "theme",
            "execute",
            "prompt",
            "prompt_color",
            "new_tab",
            "new_tab_button_text",
            "search_params",
        )
    }

    def run(self):
        text = (
            "The interactive shell is not available in this local build "
            "(OPTILAND_DOCS_SKIP_JUPYTERLITE is set). It is included in "
            "published builds."
        )
        return [nodes.note("", nodes.paragraph(text=text))]


def _write_build_metadata(app, exception) -> None:
    """Record the source commit and version of every HTML build."""
    if exception is not None or app.builder.name not in {"html", "dirhtml"}:
        return
    meta_dir = Path(app.outdir) / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_repository": "optiland/optiland",
        "source_commit": git_sha,
        "docs_version": release,
        "canonical_base_url": html_baseurl,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sphinx_builder": app.builder.name,
        "jupyterlite": not SKIP_JUPYTERLITE,
    }
    (meta_dir / "build.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def setup(app) -> None:
    app.connect("build-finished", _write_build_metadata)
    if SKIP_JUPYTERLITE:
        app.add_directive("replite", _RepliteUnavailable)
