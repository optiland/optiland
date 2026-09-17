"""Exercise configuration changes and failure reporting without scientific imports."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.ci import compatibility_matrix as cm


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.config = cm.load_matrix()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def load(self, config):
        with patch.object(cm.tomllib, "load", return_value=config):
            return cm.load_matrix()

    def test_script_entrypoint_and_exit_status(self):
        script = str(Path(cm.__file__))
        for arguments, status, message in (
            (["validate"], 0, "Validated"),
            (["summary", "--reports", str(self.root)], 1, "missing"),
        ):
            with (
                self.subTest(arguments=arguments),
                patch.object(sys, "argv", [script, *arguments]),
                patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}),
                redirect_stdout(io.StringIO()) as output,
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_path(script, run_name="__main__")
            self.assertEqual(exited.exception.code, status)
            self.assertIn(message, output.getvalue())

    def test_manifest_and_single_file_updates(self):
        self.assertTrue(self.config["rows"])
        row = copy.deepcopy(self.config["rows"][1])
        row["id"] = "another-combination"
        row["packages"]["pandas"] = "2.3.3"
        self.config["rows"] = [row]
        self.assertEqual(self.load(self.config)["rows"], [row])
        self.assertEqual(cm.select_rows(self.config, row["id"]), [row])

    def test_invalid_configurations(self):
        mutations = [
            lambda c: c.update(typo=True),
            lambda c: c["settings"].update(typo=1),
            lambda c: c["settings"].update(runner="windows-latest"),
            lambda c: c["settings"].update(max_parallel=True),
            lambda c: c["settings"].update(timeout_minutes=0),
            lambda c: c.update(rows=[]),
            lambda c: c.update(rows=["invalid"]),
            lambda c: c["rows"].append(c["rows"][0]),
            lambda c: c["rows"][0].update(id="bad/../id"),
            lambda c: c["rows"][0].pop("purpose"),
            lambda c: c["rows"][0].update(python=3.11),
            lambda c: c["rows"][0].update(python="3.10"),
            lambda c: c["rows"][0].update(purpose=""),
            lambda c: c["rows"][0].update(source="latest"),
            lambda c: c["rows"][0].update(torch="cuda"),
            lambda c: c["rows"][0].update(packages={"numpy": "2.3.5"}),
            lambda c: c["rows"][1].update(packages=[]),
            lambda c: c["rows"][1]["packages"].pop("numpy"),
            lambda c: c["rows"][1].update(torch="absent"),
            lambda c: c["rows"][1]["packages"].update(numpy="latest"),
            lambda c: c["rows"][1]["packages"].update(numpy=2.5),
            lambda c: c["rows"][1]["packages"].update(Bad_Name="1.0"),
            lambda c: c["rows"][0].update(typo="x"),
        ]
        for mutate in mutations:
            config = copy.deepcopy(self.config)
            mutate(config)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.load(config)

    def test_selection_is_strict_and_preserves_order(self):
        rows = self.config["rows"]
        self.assertEqual(cm.select_rows(self.config, "  "), rows)
        self.assertEqual(
            cm.select_rows(self.config, f"{rows[-1]['id']}, {rows[0]['id']}"),
            [rows[0], rows[-1]],
        )
        for selection in (
            "typo",
            f"{rows[0]['id']},",
            f"{rows[0]['id']},{rows[0]['id']}",
        ):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                cm.select_rows(self.config, selection)

    def test_requested_pins_and_lock_control(self):
        for row in self.config["rows"]:
            self.assertTrue(cm.PRIMARY <= cm.requested_versions(row).keys())
        locked = self.config["rows"][0]
        (self.root / "uv.lock").write_text("package = []\n")
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            cm.requested_versions(locked, self.root)

    def test_command_logging_preserves_failure(self):
        cm.run_logged([sys.executable, "-c", "print('installed')"], self.root, "ok.log")
        self.assertIn("installed", (self.root / "ok.log").read_text())
        with (
            redirect_stdout(io.StringIO()),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            cm.run_logged(
                [sys.executable, "-c", "raise SystemExit(7)"], self.root, "bad.log"
            )
        self.assertEqual(
            json.loads((self.root / "bad.log.json").read_text())["exit_code"], 7
        )

    def test_resolver_retains_pins_and_only_selected_extras(self):
        (self.root / "uv.lock").write_text("unchanged")
        (self.root / "pyproject.toml").write_text(
            '[project.optional-dependencies]\ntorch = ["torch"]\ngui = ["PySide6"]\n'
        )
        for row in (self.config["rows"][1], self.config["rows"][-1]):
            report = self.root / row["id"]
            with patch.object(
                cm.platform,
                "python_version_tuple",
                return_value=(*row["python"].split("."), "0"),
            ):
                with patch.object(cm, "run_logged") as run:
                    cm.resolve(row, report, root=self.root)
            command = run.call_args.args[0]
            self.assertIn("--constraint", command)
            self.assertIn("--generate-hashes", command)
            self.assertIn("gui", command)
            self.assertEqual("--torch-backend" in command, row["torch"] == "cpu")
            self.assertEqual("torch" in command, row["torch"] == "cpu")
            self.assertIn("numpy==", (report / "constraints.txt").read_text())
            self.assertEqual((self.root / "uv.lock").read_text(), "unchanged")

    def test_unicode_resolver_output_preserves_exit_status(self):
        output = io.BytesIO()
        console = io.TextIOWrapper(output, encoding="ascii")
        with (
            redirect_stdout(console),
            self.assertRaises(subprocess.CalledProcessError) as error,
        ):
            cm.run_logged(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(bytes.fromhex('e29c97')); raise SystemExit(7)",
                ],
                self.root,
                "unicode.log",
            )
        console.flush()
        self.assertEqual(error.exception.returncode, 7)
        self.assertIn(b"\\u2717", output.getvalue())
        self.assertIn("\u2717", (self.root / "unicode.log").read_text(encoding="utf-8"))

    def test_resolver_rejects_wrong_python(self):
        with patch.object(
            cm.platform, "python_version_tuple", return_value=("9", "9", "0")
        ):
            with self.assertRaisesRegex(ValueError, "Python"):
                cm.resolve(self.config["rows"][1], self.root)

    def test_control_export_and_unexpected_lock_mutation(self):
        (self.root / "uv.lock").write_text("unchanged")
        (self.root / "pyproject.toml").write_text("[project]\n")
        row = self.config["rows"][0]
        versions = cm.requested_versions(row)
        with patch.object(
            cm.platform, "python_version_tuple", return_value=("3", "11", "0")
        ):
            with patch.object(cm, "requested_versions", return_value=versions):
                with patch.object(cm, "run_logged") as run:
                    cm.resolve(row, self.root / "control", root=self.root)
                self.assertIn("export", run.call_args_list[0].args[0])
                self.assertIn("--locked", run.call_args_list[0].args[0])
                with patch.object(
                    cm,
                    "run_logged",
                    side_effect=lambda *a: (self.root / "uv.lock").write_text(
                        "modified"
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "modified uv.lock"):
                        cm.resolve(row, self.root / "mutation", root=self.root)

    def test_verifier_success_and_rejection_paths(self):
        # Controlled installed-package and backend interfaces let these failure
        # checks run even in the dependency-free workflow validation job. Real
        # NumPy/Numba/Torch execution is checked by every compatibility row.
        class Values:
            def __mul__(self, other):
                return self

            def sum(self):
                return 14

        def njit(function):
            function.nopython_signatures = ["compiled"]
            return function

        for torch_mode in ("cpu", "absent"):
            row = copy.deepcopy(self.config["rows"][1 if torch_mode == "cpu" else -1])
            packages = {
                k: v + ("+cpu" if k == "torch" else "")
                for k, v in row["packages"].items()
            }
            backends = ["numpy", "torch"] if torch_mode == "cpu" else ["numpy"]
            be = SimpleNamespace(
                list_available_backends=lambda: backends,
                set_backend=lambda name: None,
                set_device=lambda name: None,
                array=lambda values: SimpleNamespace(device="cpu"),
            )
            numba = SimpleNamespace(
                config=SimpleNamespace(DISABLE_JIT=False), njit=njit
            )
            modules = {
                "numba": numba,
                "numpy": SimpleNamespace(array=lambda values: Values()),
                "optiland": SimpleNamespace(
                    __file__=str(self.root / "optiland/__init__.py"), backend=be
                ),
                "optiland.backend": be,
                "torch": SimpleNamespace(version=SimpleNamespace(cuda=None)),
            }

            def distributions():
                return [
                    SimpleNamespace(metadata={"Name": k}, version=v)
                    for k, v in packages.items()
                ]

            report = self.root / torch_mode
            with (
                patch.dict(sys.modules, modules),
                patch.object(cm.metadata, "distributions", side_effect=distributions),
                patch.object(
                    cm.platform,
                    "python_version_tuple",
                    return_value=(*row["python"].split("."), "0"),
                ),
                patch.object(cm.subprocess, "check_output", return_value="abc123\n"),
                patch.object(cm.importlib.util, "find_spec", return_value=None) as find,
            ):
                cm.verify(row, report, root=self.root)
                self.assertTrue((report / "environment.json").exists())
                with patch.object(
                    cm.platform, "python_version_tuple", return_value=("9", "9", "0")
                ):
                    with self.assertRaisesRegex(ValueError, "Python"):
                        cm.verify(row, report, root=self.root)
                packages["numpy"] = "0.0.0"
                with self.assertRaisesRegex(ValueError, "numpy: requested"):
                    cm.verify(row, report, root=self.root)
                packages["numpy"] = row["packages"]["numpy"]
                modules["optiland"].__file__ = str(
                    self.root.parent / "elsewhere/__init__.py"
                )
                with self.assertRaisesRegex(ValueError, "checked-out source"):
                    cm.verify(row, report, root=self.root)
                modules["optiland"].__file__ = str(self.root / "optiland/__init__.py")
                with patch.object(be, "list_available_backends", return_value=[]):
                    with self.assertRaisesRegex(ValueError, "Registered backends"):
                        cm.verify(row, report, root=self.root)
                with patch.object(numba.config, "DISABLE_JIT", True):
                    with self.assertRaisesRegex(ValueError, "remain enabled"):
                        cm.verify(row, report, root=self.root)
                with patch.object(Values, "sum", return_value=0):
                    with self.assertRaisesRegex(ValueError, "nopython"):
                        cm.verify(row, report, root=self.root)
                if torch_mode == "cpu":
                    with patch.object(modules["torch"].version, "cuda", "13.0"):
                        with self.assertRaisesRegex(ValueError, "CPU Torch"):
                            cm.verify(row, report, root=self.root)
                    with patch.object(
                        be, "array", return_value=SimpleNamespace(device="cuda:0")
                    ):
                        with self.assertRaisesRegex(ValueError, "CPU Torch"):
                            cm.verify(row, report, root=self.root)
                else:
                    find.return_value = object()
                    with self.assertRaisesRegex(ValueError, "must be absent"):
                        cm.verify(row, report, root=self.root)
                    find.return_value = None
                    packages["torch"] = "2.14.0+cpu"
                    with self.assertRaisesRegex(ValueError, "must be absent"):
                        cm.verify(row, report, root=self.root)

    def test_cli_errors_and_actual_entrypoint(self):
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            self.assertEqual(cm.main(["matrix"]), 0)
            self.assertEqual(cm.main(["summary", "--reports", str(self.root)]), 1)
        for arguments in (
            ["record"],
            ["record", "--row", ",".join(r["id"] for r in self.config["rows"][:2])],
        ):
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                cm.main(arguments)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY"}
        }
        result = subprocess.run(
            [
                sys.executable,
                str(cm.ROOT / "scripts/ci/compatibility_matrix.py"),
                "matrix",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('matrix={"include":', result.stdout)

    def test_outcomes_and_summary_cannot_hide_failures(self):
        row = self.config["rows"][1]
        report = self.root / row["id"]
        success = dict.fromkeys(cm.STAGES, "success")
        cm.record(row, report, success)
        summary, failed = cm.summarize([row], self.root)
        self.assertTrue(failed)
        self.assertIn("missing JUnit", summary)
        (report / "junit.xml").write_text(
            '<testsuites><testsuite tests="2" skipped="1" failures="0" errors="0"/></testsuites>'
        )
        self.assertIn("missing environment", cm.summarize([row], self.root)[0])
        cm.write_json(
            report / "environment.json",
            {"python": row["python"], "packages": {"numpy": "2.4.6"}},
        )
        summary, failed = cm.summarize([row], self.root)
        self.assertFalse(failed)
        self.assertIn("2 / 1", summary)
        for stage in cm.STAGES:
            outcomes = {**success, stage: "failure"}
            cm.record(row, report, outcomes)
            summary, failed = cm.summarize([row], self.root)
            self.assertTrue(failed)
            self.assertIn(f"failed: {stage}", summary)
        cm.record(row, report, success)
        (report / "junit.xml").write_text(
            '<testsuites><testsuite tests="2" failures="1"/></testsuites>'
        )
        self.assertTrue(cm.summarize([row], self.root)[1])
        (report / "junit.xml").write_text(
            '<testsuites><testsuite tests="0"/></testsuites>'
        )
        self.assertTrue(cm.summarize([row], self.root)[1])
        (report / "result.json").unlink()
        self.assertIn("missing report", cm.summarize([row], self.root)[0])

    def test_summary_compares_inventories_without_row_id_switches(self):
        rows = copy.deepcopy(self.config["rows"][-2:])
        for i, row in enumerate(rows):
            row["id"] = f"renamed-{i}"
            report = self.root / row["id"]
            cm.record(row, report, dict.fromkeys(cm.STAGES, "success"))
            cm.write_json(
                report / "environment.json",
                {
                    "python": row["python"],
                    "packages": {"numpy": "2.5.3", "pandas": f"2.3.{i}"},
                },
            )
            (report / "junit.xml").write_text('<testsuite tests="3"/>')
        summary, failed = cm.summarize(rows, self.root)
        self.assertFalse(failed)
        self.assertIn('"pandas"', summary)
        self.assertIn('"renamed-0"', summary)

    def test_cli_matrix_and_reports(self):
        output = self.root / "output.txt"
        summary = self.root / "summary.md"
        with patch.dict(
            os.environ,
            {"GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary)},
        ):
            with redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(cm.main(["validate"]), 0)
                self.assertEqual(cm.main(["show"]), 0)
                self.assertEqual(cm.main(["matrix"]), 0)
                self.assertEqual(cm.main(["summary", "--reports", str(self.root)]), 1)
            self.assertIn("Validated", stream.getvalue())
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        self.assertEqual(
            len(json.loads(values["matrix"])["include"]), len(self.config["rows"])
        )
        self.assertTrue(summary.exists())
        row = self.config["rows"][0]
        with patch.dict(
            os.environ,
            {"STEP_OUTCOMES": json.dumps(dict.fromkeys(cm.STAGES, "success"))},
        ):
            self.assertEqual(
                cm.main(["record", "--row", row["id"], "--reports", str(self.root)]), 0
            )
        for command in ("resolve", "verify"):
            with patch.object(cm, command) as function:
                self.assertEqual(cm.main([command, "--row", row["id"]]), 0)
                function.assert_called_once()


if __name__ == "__main__":
    unittest.main()
