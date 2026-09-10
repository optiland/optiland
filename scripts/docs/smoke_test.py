"""Smoke-test a deployed documentation site over HTTP.

Checks a handful of representative pages and assets, the build identity in
``_meta/build.json``, canonical URLs, the 404 page, and (for the first-party
route) that responses are publicly cacheable and never redirect to Read the
Docs.

Usage::

    python scripts/docs/smoke_test.py --base-url https://www.optiland.org/docs/ \
        --expect-commit <sha> --first-party
    python scripts/docs/smoke_test.py --base-url https://<origin>.vercel.app/ \
        --expect-commit <sha>
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request

PAGES = [
    "",
    "start_here.html",
    "installation.html",
    "quickstart.html",
    "how_do_i.html",
    "learning_guide.html",
    "examples/Tutorial_1a_Optiland_for_Beginners.html",
    "api/api_introduction.html",
    "api/api_optic.html",
    "developers_guide/introduction.html",
    "search.html",
    "try_it.html",
]
ASSETS = [
    "searchindex.js",
    "objects.inv",
    "sitemap.xml",
    "_static/optiland-docs.css",
    "lite/repl/index.html",
    "lite/jupyter-lite.json",
]
UA = "optiland-docs-smoke-test/1.0"


def _decode(headers: dict, body: bytes) -> bytes | None:
    """Undo a Content-Encoding; None when the body cannot be decoded."""
    enc = headers.get("Content-Encoding") or headers.get("content-encoding") or ""
    enc = enc.strip().lower()
    if enc in ("", "identity"):
        return body
    if enc == "gzip":
        return gzip.decompress(body)
    if enc == "br":
        try:
            import brotli  # type: ignore[import-not-found]
        except ImportError:
            try:
                import brotlicffi as brotli  # type: ignore[import-not-found,no-redef]
            except ImportError:
                return None
        return brotli.decompress(body)
    return None


def fetch(url: str, timeout: float = 30.0):
    # A unique query string bypasses CDN caches so every check observes the
    # current deployment rather than a previously cached variant (the edge may
    # otherwise serve a compressed cached copy regardless of Accept-Encoding).
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(
        f"{url}{sep}smoke={int(time.time() * 1000)}",
        headers={"User-Agent": UA, "Accept-Encoding": "identity"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            headers = dict(resp.headers)
            return resp.status, headers, _decode(headers, resp.read()), resp.geturl()
    except urllib.error.HTTPError as exc:
        headers = dict(exc.headers)
        return exc.code, headers, _decode(headers, exc.read()), exc.geturl()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", required=True)
    ap.add_argument(
        "--expect-commit", help="Expected source_commit in _meta/build.json"
    )
    ap.add_argument("--canonical-base", default="https://www.optiland.org/docs/")
    ap.add_argument(
        "--first-party",
        action="store_true",
        help="Also assert CDN-cacheable, auth-free responses",
    )
    args = ap.parse_args(argv)

    base = args.base_url.rstrip("/") + "/"
    errors: list[str] = []

    def check(
        rel: str, *, expect_status: int = 200, expect_html: bool = True
    ) -> tuple[int, dict, bytes]:
        url = base + rel
        try:
            status, headers, body, final = fetch(url)
        except Exception as exc:  # network failure
            errors.append(f"{url}: request failed: {exc}")
            return 0, {}, b""
        if status != expect_status:
            errors.append(f"{url}: HTTP {status} (expected {expect_status})")
        if "readthedocs" in final:
            errors.append(f"{url}: redirected to Read the Docs ({final})")
        if args.first_party:
            if not final.startswith(args.canonical_base):
                errors.append(f"{url}: final URL left the first-party route: {final}")
            cc = headers.get("Cache-Control", headers.get("cache-control", "")).lower()
            if "private" in cc or "no-store" in cc:
                errors.append(f"{url}: not cacheable: Cache-Control: {cc}")
            if "set-cookie" in {k.lower() for k in headers}:
                errors.append(f"{url}: response sets a cookie")
        if body is None:
            enc = headers.get("Content-Encoding") or headers.get("content-encoding")
            errors.append(
                f"{url}: cannot decode Content-Encoding {enc!r} (pip install brotli)"
            )
            return status, headers, b""
        if expect_html and status == expect_status:
            text = body.decode("utf-8", errors="replace")
            if "<html" not in text.lower():
                errors.append(f"{url}: response is not HTML")
            if "optiland.readthedocs.io" in text:
                errors.append(f"{url}: page links to Read the Docs")
            if rel and expect_status == 200:
                m = re.search(r'<link rel="canonical" href="([^"]+)"', text)
                if not m:
                    errors.append(f"{url}: no canonical link")
                elif not m.group(1).startswith(args.canonical_base):
                    errors.append(
                        f"{url}: canonical {m.group(1)} "
                        f"not rooted at {args.canonical_base}"
                    )
        return status, headers, body

    for rel in PAGES:
        check(rel)
    for rel in ASSETS:
        check(rel, expect_html=False)

    status, _, body = check("_meta/build.json", expect_html=False)
    if status == 200:
        try:
            meta = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            errors.append("_meta/build.json is not valid JSON")
            meta = {}
        print(f"build metadata: {json.dumps(meta)}")
        if args.expect_commit and meta.get("source_commit") != args.expect_commit:
            errors.append(
                "_meta/build.json source_commit "
                f"{meta.get('source_commit')} != expected {args.expect_commit}"
            )

    # 404 handling keeps users inside the docs experience.
    status, _, body = check("this-page-does-not-exist.html", expect_status=404)
    if status == 404:
        text = body.decode("utf-8", errors="replace")
        if "Page not found" not in text or "bd-header" not in text:
            errors.append("404 response does not render the documentation 404 page")

    for line in errors:
        print(f"ERROR: {line}")
    print(f"\n{len(errors)} error(s) for {base}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
