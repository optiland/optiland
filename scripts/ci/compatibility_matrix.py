"""Read, resolve and report the weekly compatibility matrix (CI tooling only)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / ".github/compatibility-matrix.toml"
PRIMARY = {"numpy", "numba", "llvmlite", "scipy"}
STAGES = ("system", "environment", "resolve", "install", "verify", "tests")


def load_matrix(path=MANIFEST):
    """Validate configuration without importing any application dependencies."""
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if set(config) != {"settings", "rows"}:
        raise ValueError("Expected only settings and rows in the matrix")
    settings = config["settings"]
    limits = {
        "max_parallel": 256,
        "timeout_minutes": 360,
        "artifact_retention_days": 90,
    }
    if set(settings) != {"runner", *limits}:
        raise ValueError("Unknown or missing matrix settings")
    if settings["runner"] != "ubuntu-latest":
        raise ValueError("This workflow supports ubuntu-latest x64 CPU runners")
    for name, maximum in limits.items():
        if type(settings[name]) is not int or not 1 <= settings[name] <= maximum:
            raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    rows = config["rows"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError("Expected between 1 and 256 matrix rows")
    seen = set()
    fields = {"id", "purpose", "python", "source", "torch"}
    for row in rows:
        if not isinstance(row, dict) or not fields <= row.keys():
            raise ValueError("Each row needs id, purpose, python, source and torch")
        if row.keys() - fields - {"packages"}:
            raise ValueError(f"Unknown fields in row {row['id']!r}")
        if not all(isinstance(row[key], str) for key in fields):
            raise ValueError("Row metadata must contain strings")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", row["id"]) or row["id"] in seen:
            raise ValueError(f"Invalid or duplicate row ID: {row['id']!r}")
        seen.add(row["id"])
        if not row["purpose"].strip() or "\n" in row["purpose"]:
            raise ValueError(f"{row['id']}: provide a one-line purpose")
        if not re.fullmatch(r"3\.\d+", row["python"]) or int(row["python"][2:]) < 11:
            raise ValueError(f"{row['id']}: Python must be a quoted minor >= 3.11")
        if row["source"] not in {"project-lock", "pinned"}:
            raise ValueError(f"{row['id']}: unknown dependency source")
        if row["torch"] not in {"cpu", "absent"}:
            raise ValueError(f"{row['id']}: Torch must be cpu or absent")
        packages = row.get("packages", {})
        if not isinstance(packages, dict):
            raise ValueError(f"{row['id']}: packages must be a table")
        if row["source"] == "project-lock":
            if packages:
                raise ValueError("Lock controls obtain all their pins from uv.lock")
        else:
            required = PRIMARY | ({"torch"} if row["torch"] == "cpu" else set())
            if not required <= packages.keys():
                raise ValueError(f"{row['id']}: missing primary package pins")
        if row["torch"] == "absent" and "torch" in packages:
            raise ValueError(f"{row['id']}: absent Torch cannot have a package pin")
        for name, version in packages.items():
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
                raise ValueError(f"Use a canonical package name: {name!r}")
            if not isinstance(version, str) or not re.fullmatch(
                r"\d+(?:\.\d+)+(?:\.post\d+)?", version
            ):
                raise ValueError(f"{name}: expected a quoted exact stable version")
    return config


def select_rows(config, selection=""):
    """Select stable IDs, preserving manifest order and rejecting typos."""
    if not selection.strip():
        return config["rows"]
    requested = [value.strip() for value in selection.split(",")]
    known = {row["id"] for row in config["rows"]}
    if len(set(requested)) != len(requested) or set(requested) - known:
        raise ValueError(f"Unknown, empty or duplicate selected IDs: {selection!r}")
    return [row for row in config["rows"] if row["id"] in requested]


def write_json(path, value):
    """Write readable report data with deterministic encoding."""
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def requested_versions(row, root=ROOT):
    """Read control versions from the lock; no duplicated control pins."""
    if row["source"] == "pinned":
        return row["packages"]
    with (root / "uv.lock").open("rb") as stream:
        packages = tomllib.load(stream)["package"]
    names = PRIMARY | ({"torch"} if row["torch"] == "cpu" else set())
    versions = {}
    for name in sorted(names):
        candidates = {p["version"] for p in packages if p["name"] == name}
        if len(candidates) != 1:
            raise ValueError(f"Lock control needs one unambiguous {name} version")
        versions[name] = candidates.pop()
    return versions


def run_logged(command, report, log_name, root=ROOT):
    """Keep command output and timings while propagating its failure."""
    start = time.monotonic()
    log_path = report / log_name
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(list(map(str, command))) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    record = {
        "command": list(map(str, command)),
        "exit_code": result.returncode,
        "seconds": round(time.monotonic() - start, 3),
    }
    write_json(report / (log_name + ".json"), record)
    if result.returncode:
        message = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        # Preserve the resolver failure on Windows consoles with narrow encodings.
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(message.encode(encoding, errors="backslashreplace").decode(encoding))
        raise subprocess.CalledProcessError(result.returncode, command)


def resolve(row, report, uv="uv", root=ROOT):
    """Resolve declared dependencies under row constraints without changing uv.lock."""
    if platform.python_version_tuple()[:2] != tuple(row["python"].split(".")):
        raise ValueError(f"Run this row with Python {row['python']}")
    report.mkdir(parents=True, exist_ok=True)
    versions = requested_versions(row, root)
    write_json(report / "requested.json", {"row": row, "versions": versions})
    lock_hash = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    constraints = report / "constraints.txt"
    if row["source"] == "project-lock":
        run_logged(
            [
                uv,
                "export",
                "--locked",
                "--all-extras",
                "--group",
                "dev",
                "--no-hashes",
                "--no-emit-project",
                "--output-file",
                constraints,
            ],
            report,
            "export.log",
            root,
        )
    else:
        constraints.write_text(
            "".join(f"{k}=={v}\n" for k, v in sorted(versions.items())),
            encoding="utf-8",
        )
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    extras = [
        name
        for name in project.get("optional-dependencies", {})
        if row["torch"] == "cpu" or name != "torch"
    ]
    command = [
        uv,
        "pip",
        "compile",
        "pyproject.toml",
        "--group",
        "dev",
        "--python",
        sys.executable,
        "--constraint",
        constraints,
        "--only-binary",
        ":all:",
        "--generate-hashes",
        "--upgrade",
        "--emit-index-annotation",
        "--prerelease",
        "disallow",
        "--output-file",
        report / "requirements.txt",
    ]
    for extra in sorted(extras):
        command.extend(["--extra", extra])
    if row["torch"] == "cpu":
        command.extend(["--torch-backend", "cpu"])
    run_logged(command, report, "resolve.log", root)
    if hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest() != lock_hash:
        raise ValueError("Resolution modified uv.lock")


def verify(row, report, root=ROOT, manifest=MANIFEST):
    """Check the real installed pins, source, optional backend and enabled JIT."""
    report.mkdir(parents=True, exist_ok=True)
    packages = {
        re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
        for d in metadata.distributions()
    }
    inventory = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": dict(sorted(packages.items())),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "runner_image": os.environ.get("RUNNER_IMAGE_VERSION", "local"),
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
    }
    write_json(report / "environment.json", inventory)
    if platform.python_version_tuple()[:2] != tuple(row["python"].split(".")):
        raise ValueError("Installed Python does not match the selected row")
    for name, version in requested_versions(row, root).items():
        expected = version + "+cpu" if name == "torch" else version
        if packages.get(name) != expected:
            raise ValueError(
                f"{name}: requested {expected}, installed {packages.get(name)}"
            )
    if row["torch"] == "absent" and (
        "torch" in packages or importlib.util.find_spec("torch") is not None
    ):
        raise ValueError("Torch must be absent, not merely unused")

    import numba
    import numpy as np

    import optiland
    import optiland.backend as be

    if not Path(optiland.__file__).resolve().is_relative_to(root.resolve()):
        raise ValueError("Optiland was not imported from the checked-out source")
    expected_backends = {"numpy", "torch"} if row["torch"] == "cpu" else {"numpy"}
    if set(be.list_available_backends()) != expected_backends:
        raise ValueError("Registered backends do not match the row")
    if row["torch"] == "cpu":
        import torch

        be.set_backend("torch")
        be.set_device("cpu")
        if torch.version.cuda is not None or str(be.array([1.0]).device) != "cpu":
            raise ValueError("Expected a CPU Torch build and CPU execution")
    be.set_backend("numpy")
    if numba.config.DISABLE_JIT:
        raise ValueError("Numba JIT must remain enabled")

    @numba.njit
    def probe(values):
        return (values * values).sum()

    if probe(np.array([1.0, 2.0, 3.0])) != 14 or not probe.nopython_signatures:
        raise ValueError("Numba nopython compilation/execution failed")


def record(row, report, outcomes):
    """Retain step conclusions even when dependency installation failed."""
    report.mkdir(parents=True, exist_ok=True)
    failed = [stage for stage in STAGES if outcomes.get(stage) != "success"]
    result = {
        "id": row["id"],
        "outcomes": outcomes,
        "status": f"failed: {failed[0]}" if failed else "passed",
    }
    write_json(report / "result.json", result)


def summarize(rows, reports):
    """Fail on missing/failed reports and show actual versions and test counts."""
    lines = [
        "## Weekly dependency compatibility",
        "",
        "| Row | Result | Python | NumPy | Numba | Torch | Tests / skips |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    failures = False
    environments = {}
    for row in rows:
        report = reports / row["id"]
        result_file = report / "result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        status = result.get("status", "missing report (runner/setup/timeout failure)")
        environment_file = report / "environment.json"
        environment = (
            json.loads(environment_file.read_text())
            if environment_file.exists()
            else {}
        )
        packages = environment.get("packages", {})
        environments[row["id"]] = packages
        junit = report / "junit.xml"
        counts = "unavailable"
        if junit.exists():
            suites = list(ET.parse(junit).getroot().iter("testsuite"))
            totals = {
                key: sum(int(s.get(key, "0")) for s in suites)
                for key in ("tests", "skipped", "failures", "errors")
            }
            counts = f"{totals['tests']} / {totals['skipped']}"
            if status == "passed" and (
                not totals["tests"] or totals["failures"] or totals["errors"]
            ):
                status = "failed: tests/collection"
        elif status == "passed":
            status = "failed: missing JUnit"
        if status == "passed" and not environment_file.exists():
            status = "failed: missing environment inventory"
        failures |= status != "passed"
        versions = [
            environment.get("python", "?"),
            packages.get("numpy", "?"),
            packages.get("numba", "?"),
            packages.get("torch", "absent" if packages else "?"),
        ]
        lines.append(
            f"| {row['id']} | {status} | " + " | ".join(versions) + f" | {counts} |"
        )
    differences = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:

            def pins(row):
                return {
                    k: v for k, v in row.get("packages", {}).items() if k != "torch"
                }

            if (
                left["python"] != right["python"]
                or not pins(left)
                or pins(left) != pins(right)
            ):
                continue
            a, b = environments[left["id"]], environments[right["id"]]
            if not a or not b:
                continue
            changed = {
                k: [a[k], b[k]]
                for k in a.keys() & b.keys()
                if k != "torch" and a[k] != b[k]
            }
            differences.append(
                {
                    "rows": [left["id"], right["id"]],
                    "changed": changed,
                    "only_left": sorted(a.keys() - b.keys() - {"torch"}),
                    "only_right": sorted(b.keys() - a.keys() - {"torch"}),
                }
            )
    lines.extend(
        [
            "",
            "Non-Torch inventory differences for matching primary pins:",
            "",
            "```json",
            json.dumps(differences, indent=2),
            "```",
            "Differences in transitive packages prevent an isolated Torch comparison.",
        ]
    )
    return "\n".join(lines) + "\n", failures


def main(argv=None):
    """Expose the same row selection and checks to maintainers and Actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "show",
            "matrix",
            "resolve",
            "verify",
            "record",
            "summary",
        ),
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--rows", default="")
    parser.add_argument("--row")
    parser.add_argument("--reports", type=Path, default=ROOT / "compatibility-reports")
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args(argv)
    config = load_matrix(args.manifest)
    rows = select_rows(config, args.rows)
    if args.command == "validate":
        print(f"Validated {len(rows)} compatibility rows")
    elif args.command == "show":
        print("| ID | Python | NumPy | Numba | llvmlite | SciPy | Torch |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            pins = requested_versions(row)
            values = [
                pins.get(k, "absent")
                for k in ("numpy", "numba", "llvmlite", "scipy", "torch")
            ]
            print(f"| {row['id']} | {row['python']} | " + " | ".join(values) + " |")
    elif args.command == "matrix":
        outputs = {
            "matrix": {
                "include": [{"id": r["id"], "python": r["python"]} for r in rows]
            },
            **config["settings"],
        }
        output = "".join(f"{k}={json.dumps(v)}\n" for k, v in outputs.items())
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(output)
        print(output, end="")
    elif args.command == "summary":
        summary, failed = summarize(rows, args.reports)
        print(summary)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(
                summary, encoding="utf-8"
            )
        return int(failed)
    else:
        if not args.row:
            parser.error(f"{args.command} requires --row")
        row = select_rows(config, args.row)
        if len(row) != 1:
            parser.error("--row must select exactly one row")
        row = row[0]
        report = args.reports.resolve() / row["id"]
        if args.command == "resolve":
            resolve(row, report, args.uv)
        elif args.command == "verify":
            verify(row, report, manifest=args.manifest)
        else:
            record(row, report, json.loads(os.environ["STEP_OUTCOMES"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
