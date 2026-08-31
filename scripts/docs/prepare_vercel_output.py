"""Package a validated Sphinx HTML tree as a Vercel Build Output API artifact.

The result is a ``.vercel/output`` directory that ``vercel deploy --prebuilt``
uploads verbatim: the static site under ``static/`` plus a ``config.json``
with cache and security headers and a 404 route. Page content is copied
without modification.

Usage::

    python scripts/docs/prepare_vercel_output.py \
        --source docs/_build/html --output .vercel/output
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

EXCLUDED_DIRS = {".doctrees", "__pycache__", ".ipynb_checkpoints"}
EXCLUDED_FILES = {".buildinfo"}
REQUIRED = [
    "index.html",
    "404.html",
    "searchindex.js",
    "objects.inv",
    "sitemap.xml",
    "_meta/build.json",
]

SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "strict-origin-when-cross-origin",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
}

# Cache policy (see the documentation spec, section 10.3). Static theme assets
# are not content-hashed, so nothing is marked immutable.
HTML_CACHE = "public, max-age=0, s-maxage=600, stale-while-revalidate=86400"
META_CACHE = "public, max-age=0, s-maxage=300, stale-while-revalidate=300"
ASSET_CACHE = "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800"

ROBOTS_TXT = (
    "# The docs origin is an implementation detail: the canonical location of\n"
    "# every page is https://www.optiland.org/docs/ (see <link rel=canonical>).\n"
    "User-agent: *\n"
    "Disallow: /\n"
)


def build_config() -> dict:
    return {
        "version": 3,
        "routes": [
            {
                "src": "^/_meta/build\\.json$",
                "headers": {"cache-control": META_CACHE, **SECURITY_HEADERS},
                "continue": True,
            },
            {
                "src": "^/(searchindex\\.js|objects\\.inv|sitemap\\.xml)$",
                "headers": {"cache-control": HTML_CACHE, **SECURITY_HEADERS},
                "continue": True,
            },
            {
                "src": "^/(_static|_images|_sources|lite)/.*$",
                "headers": {"cache-control": ASSET_CACHE, **SECURITY_HEADERS},
                "continue": True,
            },
            {
                "src": "^/.*\\.html$",
                "headers": {"cache-control": HTML_CACHE, **SECURITY_HEADERS},
                "continue": True,
            },
            {
                "src": "^/$",
                "headers": {"cache-control": HTML_CACHE, **SECURITY_HEADERS},
                "continue": True,
            },
            {
                "src": "^/.*$",
                "headers": {**SECURITY_HEADERS},
                "continue": True,
            },
            {"handle": "filesystem"},
            {"src": "^/.*$", "status": 404, "dest": "/404.html"},
        ],
    }


def copy_tree(source: Path, target: Path) -> tuple[int, int]:
    files = 0
    total = 0
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            raise SystemExit(f"refusing to package symlink: {rel}")
        if path.is_dir():
            continue
        if path.name in EXCLUDED_FILES:
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        files += 1
        total += path.stat().st_size
    return files, total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("docs/_build/html"))
    ap.add_argument("--output", type=Path, default=Path(".vercel/output"))
    ap.add_argument("--no-robots", action="store_true", help="Do not add the origin robots.txt")
    args = ap.parse_args(argv)

    source: Path = args.source.resolve()
    output: Path = args.output.resolve()
    if not source.is_dir():
        print(f"ERROR: source directory not found: {source}")
        return 2
    missing = [rel for rel in REQUIRED if not (source / rel).is_file()]
    if missing:
        print("ERROR: source build is missing required files: " + ", ".join(missing))
        return 1

    if output.exists():
        shutil.rmtree(output)
    static = output / "static"
    static.mkdir(parents=True)

    files, total = copy_tree(source, static)
    if not args.no_robots:
        (static / "robots.txt").write_text(ROBOTS_TXT, encoding="utf-8")
        files += 1
        total += len(ROBOTS_TXT)
    (output / "config.json").write_text(json.dumps(build_config(), indent=2) + "\n", encoding="utf-8")

    meta = json.loads((static / "_meta" / "build.json").read_text(encoding="utf-8"))
    print(f"packaged {files} files, {total:,} bytes -> {output}")
    print(f"source commit {meta.get('source_commit')} version {meta.get('docs_version')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
