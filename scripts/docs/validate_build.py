"""Validate a built Sphinx HTML tree before it is deployed.

The checks are static: nothing is fetched over the network. Every internal
``href``/``src`` target is resolved against the build tree, canonical URLs
and the sitemap are checked against the first-party base URL, and the Sphinx
warning log is compared with an explicit allowlist.

Usage::

    python scripts/docs/validate_build.py docs/_build/html \
        --base-url https://www.optiland.org/docs/ \
        --warnings docs/_build/warnings.log \
        --allowlist scripts/docs/warnings-allowlist.txt \
        --report docs/_build/validation.json

Exit status is non-zero when any error is found. Run with ``--help`` for all
options.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

REQUIRED_FILES = [
    "index.html",
    "search.html",
    "genindex.html",
    "py-modindex.html",
    "searchindex.js",
    "objects.inv",
    "sitemap.xml",
    "404.html",
    "_meta/build.json",
    "start_here.html",
    "installation.html",
    "quickstart.html",
    "how_do_i.html",
    "learning_guide.html",
    "api/api_introduction.html",
    "developers_guide/introduction.html",
    "gallery/introduction.html",
    "try_it.html",
]
JUPYTERLITE_FILES = [
    "lite/index.html",
    "lite/repl/index.html",
    "lite/jupyter-lite.json",
]
FORBIDDEN_HOSTS = ("optiland.readthedocs.io", "readthedocs.org/projects/optiland")
LINK_ATTRS = {
    "a": ("href",),
    "link": ("href",),
    "script": ("src",),
    "img": ("src",),
    "iframe": ("src",),
    "source": ("src",),
    "video": ("src", "poster"),
    "audio": ("src",),
    "object": ("data",),
    "embed": ("src",),
}
EXTERNAL_SCHEMES = (
    "http:",
    "https:",
    "mailto:",
    "tel:",
    "data:",
    "javascript:",
    "blob:",
)


@dataclass
class Page:
    path: Path
    rel: str
    ids: Counter = field(default_factory=Counter)
    links: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (tag, attr, value)
    canonical: str | None = None
    title: str | None = None
    has_main: bool = False
    text: str = ""


class _Parser(HTMLParser):
    def __init__(self, page: Page) -> None:
        super().__init__(convert_charrefs=True)
        self.page = page
        self._in_title = False
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if "id" in attrs_d and attrs_d["id"]:
            self.page.ids[attrs_d["id"]] += 1
        if tag == "a" and attrs_d.get("name"):
            self.page.ids[attrs_d["name"]] += 1
        for attr in LINK_ATTRS.get(tag, ()):
            value = attrs_d.get(attr)
            if value:
                self.page.links.append((tag, attr, value))
        if tag == "link" and attrs_d.get("rel", "").lower() == "canonical":
            self.page.canonical = attrs_d.get("href")
        if tag == "title":
            self._in_title = True
        if tag == "main" or attrs_d.get("id") == "main-content":
            self.page.has_main = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.page.title = (self.page.title or "") + data
        self._text.append(data)

    def close(self):
        super().close()
        self.page.text = " ".join(self._text)


def parse_pages(root: Path) -> dict[str, Page]:
    pages: dict[str, Page] = {}
    for path in sorted(root.rglob("*.html")):
        # Skip the JupyterLite app and theme-internal template fragments
        # shipped under _static (e.g. pydata's webpack-macros.html).
        if ".doctrees" in path.parts or path.parts[len(root.parts)] in {
            "lite",
            "_static",
        }:
            continue
        rel = path.relative_to(root).as_posix()
        page = Page(path=path, rel=rel)
        parser = _Parser(page)
        try:
            parser.feed(path.read_text(encoding="utf-8", errors="replace"))
            parser.close()
        except Exception as exc:  # pragma: no cover - defensive
            page.text = ""
            page.title = None
            page.links.append(("parser", "error", str(exc)))
        pages[rel] = page
    return pages


def resolve_local(page_rel: str, value: str) -> tuple[str | None, str | None]:
    """Return (target_rel_path, fragment) for a local link, or (None, None)."""
    # nbsphinx emits gallery thumbnail paths with the OS separator, so builds
    # made on Windows contain backslashes; browsers treat them as slashes.
    value = value.strip().replace("\\", "/")
    if not value or value.startswith(EXTERNAL_SCHEMES) or value.startswith("//"):
        return None, None
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        return None, None
    fragment = parsed.fragment or None
    path = unquote(parsed.path)
    if not path:
        return page_rel, fragment  # fragment-only link
    if path.startswith("/"):
        return "/" + path.lstrip("/"), fragment  # absolute: checked separately
    base = PurePosixPath(page_rel).parent
    joined = PurePosixPath(base, path)
    parts: list[str] = []
    for part in joined.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts), fragment


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("build_dir", type=Path)
    ap.add_argument(
        "--base-url",
        default="https://www.optiland.org/docs/",
        help="Canonical base URL (with trailing slash)",
    )
    ap.add_argument("--warnings", type=Path, help="Sphinx warning log written with -w")
    ap.add_argument(
        "--allowlist",
        type=Path,
        help="File of regexes for tolerated warnings (one per line, # comments)",
    )
    ap.add_argument("--report", type=Path, help="Write a JSON report to this path")
    ap.add_argument(
        "--forbid-text",
        action="append",
        default=["Optiland 0.5.8"],
        help="Page text that must not appear (repeatable)",
    )
    ap.add_argument(
        "--allow-missing-jupyterlite",
        action="store_true",
        help="Do not require the JupyterLite bundle (local builds only)",
    )
    ap.add_argument("--min-searchindex-bytes", type=int, default=100_000)
    args = ap.parse_args(argv)

    root: Path = args.build_dir.resolve()
    base_url = args.base_url.rstrip("/") + "/"
    url_prefix = urlparse(base_url).path or "/"
    errors: list[str] = []
    infos: list[str] = []

    if not root.is_dir():
        print(f"ERROR: build directory not found: {root}")
        return 2

    # 1. Required files -------------------------------------------------------
    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            errors.append(f"missing required file: {rel}")
    if not args.allow_missing_jupyterlite:
        for rel in JUPYTERLITE_FILES:
            if not (root / rel).is_file():
                errors.append(f"missing JupyterLite file: {rel}")
        if (root / "lite").is_dir() and not any((root / "lite").rglob("*.wasm")):
            errors.append(
                "JupyterLite bundle contains no WebAssembly kernel assets (*.wasm)"
            )
    searchindex = root / "searchindex.js"
    if (
        searchindex.is_file()
        and searchindex.stat().st_size < args.min_searchindex_bytes
    ):
        errors.append(
            f"searchindex.js is suspiciously small ({searchindex.stat().st_size} bytes)"
        )

    # 2. Build metadata -------------------------------------------------------
    meta_path = root / "_meta" / "build.json"
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"_meta/build.json is not valid JSON: {exc}")
        for key in (
            "source_repository",
            "source_commit",
            "docs_version",
            "canonical_base_url",
            "built_at_utc",
        ):
            if not meta.get(key):
                errors.append(f"_meta/build.json is missing '{key}'")
        if meta.get("canonical_base_url") and meta["canonical_base_url"] != base_url:
            errors.append(
                "_meta/build.json canonical_base_url "
                f"{meta['canonical_base_url']!r} != {base_url!r}"
            )
        if meta.get("docs_version") in {"0.5.8", "unknown", "0.0.0+unknown"}:
            errors.append(
                "_meta/build.json reports a stale or unknown version: "
                f"{meta.get('docs_version')!r}"
            )
        if meta.get("jupyterlite") is False and not args.allow_missing_jupyterlite:
            errors.append("_meta/build.json reports the JupyterLite build was skipped")

    # 3. Pages ---------------------------------------------------------------
    pages = parse_pages(root)
    infos.append(f"parsed {len(pages)} HTML pages")
    all_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    duplicate_ids = 0
    broken_links = 0
    checked_links = 0
    for rel, page in pages.items():
        if page.title is None or not page.title.strip():
            errors.append(f"{rel}: no <title>")
        if not page.has_main:
            errors.append(f"{rel}: no <main> element (malformed page?)")
        for tag, _attr, value in page.links:
            if tag == "parser":
                errors.append(f"{rel}: HTML parse error: {value}")
        dupes = [i for i, n in page.ids.items() if n > 1]
        if dupes:
            duplicate_ids += len(dupes)
            errors.append(
                f"{rel}: duplicate element id(s): {', '.join(sorted(dupes)[:5])}"
            )
        if rel != "404.html":
            if not page.canonical:
                errors.append(f"{rel}: missing <link rel=canonical>")
            elif not page.canonical.startswith(base_url):
                errors.append(
                    f"{rel}: canonical {page.canonical!r} is not rooted at {base_url}"
                )
        for needle in args.forbid_text:
            if needle in page.text:
                errors.append(f"{rel}: forbidden text {needle!r} found in page")
        for tag, attr, value in page.links:
            if tag == "parser":
                continue
            lowered = value.lower()
            if any(host in lowered for host in FORBIDDEN_HOSTS):
                errors.append(f"{rel}: link to Read the Docs: {value}")
                continue
            target, fragment = resolve_local(rel, value)
            if target is None:
                continue
            checked_links += 1
            if target.startswith("/"):
                if rel == "404.html" and target.startswith(url_prefix):
                    target = target[len(url_prefix) :]
                else:
                    errors.append(
                        f"{rel}: absolute internal link escapes {url_prefix}: {value}"
                    )
                    broken_links += 1
                    continue
            if target == "" or target.endswith("/"):
                target = target + "index.html"
            if target not in all_files:
                errors.append(f"{rel}: broken <{tag} {attr}> target: {value}")
                broken_links += 1
                continue
            # JupyterLite anchors are generated client-side; only flag anchors
            # on regular pages.
            if (
                fragment
                and target in pages
                and fragment not in pages[target].ids
                and not target.startswith("lite/")
            ):
                errors.append(f"{rel}: broken anchor #{fragment} in {target}")
                broken_links += 1
    infos.append(
        f"checked {checked_links} internal links; {broken_links} broken; "
        f"{duplicate_ids} duplicate ids"
    )

    # 4. Sitemap -------------------------------------------------------------
    sitemap = root / "sitemap.xml"
    if sitemap.is_file():
        locs = re.findall(
            r"<loc>\s*([^<\s]+)\s*</loc>", sitemap.read_text(encoding="utf-8")
        )
        if not locs:
            errors.append("sitemap.xml contains no <loc> entries")
        for loc in locs:
            if not loc.startswith(base_url):
                errors.append(f"sitemap.xml entry not rooted at base URL: {loc}")
                continue
            path = unquote(loc[len(base_url) :]) or "index.html"
            if path not in all_files:
                errors.append(f"sitemap.xml entry has no file: {loc}")
        infos.append(f"sitemap.xml lists {len(locs)} URLs")
        # sphinx-sitemap only sees pages written in the current run, so an
        # incremental build produces a partial sitemap. Deployed builds are
        # always fresh; enforce coverage so a partial sitemap never ships.
        # sphinx-sitemap skips viewcode source pages (_modules/) and the four
        # pages excluded in conf.py (search, genindex, py-modindex, 404).
        expected = max(
            sum(1 for rel in pages if not rel.startswith("_modules/")) - 4, 1
        )
        if len(locs) < 0.95 * expected:
            errors.append(
                f"sitemap.xml covers {len(locs)} of ~{expected} pages; "
                "rebuild from scratch (sphinx -E)"
            )

    # 5. Warnings vs allowlist ----------------------------------------------
    unused_allow: list[str] = []
    if args.warnings:
        patterns: list[re.Pattern] = []
        if args.allowlist and args.allowlist.is_file():
            for line in args.allowlist.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(re.compile(line))
        used = set()
        unexpected: list[str] = []
        if args.warnings.is_file():
            for line in args.warnings.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if not line.strip():
                    continue
                for pat in patterns:
                    if pat.search(line):
                        used.add(pat.pattern)
                        break
                else:
                    unexpected.append(line)
        else:
            infos.append(f"warning log not found: {args.warnings}")
        for line in unexpected:
            errors.append(f"unexpected Sphinx warning: {line}")
        unused_allow = [p.pattern for p in patterns if p.pattern not in used]
        infos.append(
            f"warnings: {len(unexpected)} unexpected, {len(used)} allowlist "
            f"patterns matched, {len(unused_allow)} unused"
        )

    # 6. Report --------------------------------------------------------------
    report = {
        "build_dir": str(root),
        "base_url": base_url,
        "pages": len(pages),
        "errors": errors,
        "info": infos,
        "unused_allowlist_patterns": unused_allow,
        "build_metadata": meta,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for line in infos:
        print(f"INFO: {line}")
    for pattern in unused_allow:
        print(f"INFO: allowlist pattern never matched (consider removing): {pattern}")
    for line in errors:
        print(f"ERROR: {line}")
    print(f"\n{len(errors)} error(s) in {root}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
