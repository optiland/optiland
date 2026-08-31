.. _documentation_operations:

Documentation Site Operations
=============================

This page is the runbook for the documentation site at
https://www.optiland.org/docs/. It describes how the site is built, deployed,
validated and rolled back. Writing documentation itself is covered in
:doc:`../contributing`.

Architecture
------------

.. code-block:: text

   optiland/optiland (docs/, notebooks, docstrings)
        │  pull request / push to master
        ▼
   GitHub Actions "Docs" workflow
        │  Sphinx build → validate_build.py → prepare_vercel_output.py
        ▼
   Dedicated Vercel project (the docs origin, static files only)
        │  external rewrite  /docs/:path*  →  <origin>/:path*
        ▼
   https://www.optiland.org/docs/…   (optiland/optiland-website, Next.js)

- **Single source of truth.** Every page comes from ``docs/`` in
  ``optiland/optiland`` (``.rst`` pages, ``.ipynb`` notebooks committed with
  their outputs) or from Python docstrings (API reference via ``autodoc`` /
  ``autosummary``). The website repository holds routing only; it never
  contains documentation content.
- **Canonical URLs.** ``docs/conf.py`` sets ``html_baseurl`` to
  ``https://www.optiland.org/docs/``; every page carries a matching
  ``<link rel="canonical">``, the sitemap uses the same base, and the origin
  publishes a ``robots.txt`` that disallows crawling so only first-party URLs
  are indexed.
- **Build identity.** Every build writes ``_meta/build.json`` (source commit,
  version, build time). The footer of every page links to it.
- **Version.** ``release`` is derived from the installed package metadata
  (``versioningit``), so a build from ``master`` is labelled with the honest
  development version (for example ``0.6.2.post49+g<sha>``). It is never
  hardcoded.
- **Code highlighting.** Colours come from the ``a11y-high-contrast``
  Pygments styles selected in ``docs/conf.py`` plus a few overrides in
  ``docs/_static/optiland-docs.css``. Pygments only marks ``def``/``class``
  names, so the local extension ``docs/_ext/optiland_pygments.py`` extends the
  ``python``, ``pycon`` and ``ipython3`` lexers to tag function and method
  calls as well; it is registered like any other extension in ``conf.py``.

The workflow
------------

``.github/workflows/docs.yml`` runs:

``build``
   On every pull request that touches ``docs/**``, ``optiland/**``,
   ``optiland_gui/**``, ``pyproject.toml``, ``uv.lock``, ``.readthedocs.yaml``
   or the docs tooling, and on every push to ``master`` (API pages depend on
   docstrings, so master is always rebuilt). Steps: create the environment
   from ``docs/build-environment.yml`` with micromamba, build with
   ``scripts/docs/build_docs.py``, validate with
   ``scripts/docs/validate_build.py``, package with
   ``scripts/docs/prepare_vercel_output.py``, upload the ``docs-site``
   artifact.
``screenshots``
   Renders representative pages (landing light/dark, quickstart on a phone
   viewport, an API page, a notebook, search) with Playwright and uploads
   them as the ``docs-screenshots`` artifact for review. Non-blocking.
``preview``
   For pull requests from branches of this repository: deploys the validated
   artifact as a Vercel preview and smoke-tests it. Forks never receive
   secrets; they still get the full build and validation.
``deploy``
   On push to ``master``: asserts the artifact was built from the pushed
   commit, deploys it to the docs origin project with
   ``vercel deploy --prebuilt --prod``, then runs
   ``scripts/docs/smoke_test.py`` against the origin alias and against
   ``https://www.optiland.org/docs/`` (with retries while CDN caches
   refresh). A failed build never reaches deployment; the previous production
   deployment stays live. The ``docs-production`` concurrency group prevents
   an older run from finishing after a newer one.
``linkcheck`` and ``production-smoke``
   Weekly (and on manual dispatch): ``sphinx -b linkcheck`` for external
   links (third-party failures are reported, first-party failures are
   blocking) and a smoke test of the production route.

What blocks a deployment
------------------------

``scripts/docs/validate_build.py`` fails the build on:

- missing required pages or assets (``index.html``, ``searchindex.js``,
  ``objects.inv``, ``sitemap.xml``, ``404.html``, ``_meta/build.json``, the
  key narrative pages, the JupyterLite bundle);
- any internal ``href``/``src`` that does not resolve inside the build,
  including anchors;
- duplicate element ids on a page;
- canonical URLs or sitemap entries not rooted at the first-party base;
- absolute links that escape ``/docs/``;
- links to ``optiland.readthedocs.io``;
- a stale or unknown version string;
- a partial sitemap (a sign of an incremental build);
- any Sphinx warning that is not matched by
  ``scripts/docs/warnings-allowlist.txt``.

The allowlist is intentionally short and every entry carries a reason. Fix
warnings at the source; extend the allowlist only when the warning cannot be
fixed in this repository. The validator reports allowlist patterns that no
longer match so they can be removed.

One-time setup
--------------

Vercel
   Create a dedicated project for the docs origin (static, no framework
   preset, no build command; deployments arrive prebuilt from CI). Note the
   ids of the project and of the owning team or personal account: running
   ``vercel link`` against the project writes both to ``.vercel/project.json``
   as ``projectId`` and ``orgId``. Assign a stable production alias, for
   example ``optiland-docs.vercel.app``.
GitHub
   Create the ``docs-production`` and ``docs-preview`` environments and add
   the secrets ``VERCEL_TOKEN``, ``VERCEL_ORG_ID`` (the ``orgId``; a personal
   account has one too, no team required), ``VERCEL_DOCS_PROJECT_ID`` (the
   ``projectId``) and ``DOCS_ORIGIN_URL`` (the production alias, without a
   trailing slash). Restrict ``docs-production`` to the ``master`` branch.
Website
   In ``optiland/optiland-website`` set ``DOCS_ORIGIN`` (Preview and
   Production environments) to the production alias. ``next.config.mjs``
   rewrites ``/docs/:path*`` to it, redirects ``/docs`` to ``/docs/`` and every
   legacy ``/tutorials/*`` URL to its canonical page
   (``config/legacy-doc-redirects.json``), and ``middleware.ts`` excludes
   ``/docs`` so documentation requests never touch Supabase.

Rollback
--------

Documentation content or build regression
   Revert the offending commit on ``master``; the next workflow run redeploys.
   To restore the previous deployment immediately without a rebuild, promote
   it in the Vercel dashboard or run ``vercel rollback`` against the docs
   project (``vercel rollback --token=… --scope=<team>``; pick the previous
   production deployment from ``vercel ls``). The website needs no change.
Routing regression on optiland.org
   Redeploy the website with the previous ``next.config.mjs`` (or unset
   ``DOCS_ORIGIN`` to disable the proxy). Legacy redirects are plain config
   and can be reverted the same way.
Read the Docs fallback
   The Read the Docs project keeps building from the same ``docs/conf.py`` and
   ``docs/build-environment.yml`` during the stabilization period and can be
   linked publicly again if a severe problem occurs.

Legacy URLs
-----------

- ``https://www.optiland.org/tutorials/<slug>`` (the former website
  tutorials) redirect permanently, in one hop, to the canonical notebook
  page; the manifest with the reason for each mapping lives in the website
  repository (``config/legacy-doc-redirects.json``) and is validated by
  ``scripts/validate-legacy-redirects.py`` there.
- After the stabilization window, configure Read the Docs redirects so that
  ``/en/latest/*`` and ``/en/stable/*`` land on ``https://www.optiland.org/docs/*``
  (forced exact redirects that strip the ``/en/<version>/`` prefix).

Local builds
------------

See :doc:`../contributing` for the day-to-day workflow. The CI-equivalent
commands are::

    python scripts/docs/build_docs.py --fresh
    python scripts/docs/validate_build.py docs/_build/html \
        --warnings docs/_build/warnings.log \
        --allowlist scripts/docs/warnings-allowlist.txt

On hosts where ``jupyterlite-xeus`` cannot build the WebAssembly kernel
environment (for example Windows), add ``--skip-jupyterlite`` to the build
and ``--allow-missing-jupyterlite`` to the validator. Published builds always
include JupyterLite.

Versioning
----------

``/docs/`` follows ``master`` and is redeployed on every merge. Versioned
snapshots (``/docs/stable/``, ``/docs/0.6/``) are a later phase; do not
publish such paths before the deployment and redirect strategy for them
exists.
