"""Snapshot of Optiland's public API, used to keep the public surface frozen.

Each line of the snapshot is ``<module>:<name><signature>``, where the
signature lists parameter names, kinds and default values. Annotations are left
out on purpose: refining a type hint does not change the calling contract.

A symbol is public when no part of its dotted path starts with an underscore.
Classes list every public method and property they expose, including those
inherited from other Optiland classes, so moving a method to a base class is
not reported as a removal.

Usage (from the repository root)::

    python -m scripts.guard.api_snapshot           # print the live snapshot
    python -m scripts.guard.api_snapshot --update  # rewrite the committed file
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import pkgutil
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = ROOT / "tests" / "api" / "public_api.txt"

# Local scratch space and maintenance scripts, not public API.
_SKIP_PREFIXES = (
    "optiland.jupyter",
    "optiland.scripts",
    "optiland.database.generate_csv_database",
)
_MEMORY_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _is_public(dotted: str) -> bool:
    return not any(part.startswith("_") for part in dotted.split("."))


def _iter_module_names() -> list[str]:
    import optiland

    names = ["optiland"]
    for info in pkgutil.walk_packages(optiland.__path__, prefix="optiland."):
        if _is_public(info.name) and not info.name.startswith(_SKIP_PREFIXES):
            names.append(info.name)
    return sorted(names)


def _signature(obj: object) -> str:
    """Return the parameter list of ``obj`` without annotations."""
    try:
        params = list(inspect.signature(obj).parameters.values())  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "(?)"
    parts: list[str] = []
    star_written = False
    for i, param in enumerate(params):
        kind = param.kind
        prev = params[i - 1].kind if i else None
        if prev is inspect.Parameter.POSITIONAL_ONLY and kind is not prev:
            parts.append("/")
        if kind is inspect.Parameter.VAR_POSITIONAL:
            star_written = True
            parts.append(f"*{param.name}")
            continue
        if kind is inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{param.name}")
            continue
        if kind is inspect.Parameter.KEYWORD_ONLY and not star_written:
            star_written = True
            parts.append("*")
        text = param.name
        if param.default is not inspect.Parameter.empty:
            text += "=" + _MEMORY_ADDRESS.sub("", repr(param.default))
        parts.append(text)
    if params and params[-1].kind is inspect.Parameter.POSITIONAL_ONLY:
        parts.append("/")
    return "(" + ", ".join(parts) + ")"


def _defined_in_optiland(cls: type, attr: str) -> object | None:
    """Return the raw attribute if the class that defines it is Optiland's."""
    for klass in cls.__mro__:
        if attr in klass.__dict__:
            if klass.__module__.split(".")[0] != "optiland":
                return None
            return klass.__dict__[attr]
    return None


def _class_lines(qualname: str, cls: type) -> list[str]:
    lines = [f"{qualname}{_signature(cls)}"]
    for attr in sorted(dir(cls)):
        if attr.startswith("_"):
            continue
        raw = _defined_in_optiland(cls, attr)
        if raw is None:
            continue
        if isinstance(raw, property):
            lines.append(f"{qualname}.{attr} [property]")
        elif isinstance(raw, (staticmethod, classmethod)):
            lines.append(f"{qualname}.{attr}{_signature(raw.__func__)}")
        elif inspect.isfunction(raw):
            lines.append(f"{qualname}.{attr}{_signature(raw)}")
    return lines


def _module_lines(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    lines: list[str] = []
    for name, obj in sorted(vars(module).items()):
        if name.startswith("_") or getattr(obj, "__module__", None) != module_name:
            continue
        qualname = f"{module_name}:{name}"
        if inspect.isclass(obj):
            lines.extend(_class_lines(qualname, obj))
        elif inspect.isfunction(obj):
            lines.append(f"{qualname}{_signature(obj)}")
    # Re-exports promised through ``__all__`` are public API too.
    for name in sorted(set(getattr(module, "__all__", ()))):
        lines.append(f"{module_name}:{name} [export]")
    return lines


def collect() -> tuple[list[str], list[str]]:
    """Import every public module and return ``(snapshot_lines, failed_modules)``.

    Modules that cannot be imported in the current environment (for example
    because an optional extra such as ``torch`` is missing) are returned in
    ``failed_modules`` instead of raising.
    """
    lines: list[str] = []
    failed: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for module_name in _iter_module_names():
            try:
                lines.extend(_module_lines(module_name))
            except Exception:  # any import failure means "skip this module"
                failed.append(module_name)
    return sorted(set(lines)), failed


def module_of(line: str) -> str:
    """Return the module part of a snapshot line."""
    return line.split(":", 1)[0]


def key_of(line: str) -> str:
    """Return the symbol part of a snapshot line (everything but the signature)."""
    return re.split(r"[( ]", line, maxsplit=1)[0]


def _parse_params(line: str) -> ast.arguments | None:
    sig = line[len(key_of(line)) :]
    if not sig.startswith("(") or sig == "(?)":
        return None
    try:
        func = ast.parse(f"def _f{sig}: pass").body[0]
    except SyntaxError:
        return None
    assert isinstance(func, ast.FunctionDef)
    return func.args


def _defaults_by_name(args: ast.arguments) -> dict[str, str | None]:
    positional = [*args.posonlyargs, *args.args]
    pos_defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    result = {
        a.arg: (ast.unparse(d) if d is not None else None)
        for a, d in zip(positional, pos_defaults, strict=True)
    }
    for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        result[a.arg] = ast.unparse(d) if d is not None else None
    return result


def is_compatible(old: str, new: str) -> bool:
    """Return True if every call valid against ``old`` is still valid on ``new``.

    Allowed: new trailing positional or keyword-only parameters that have a
    default, and a required parameter gaining a default. Anything else (removed
    or reordered parameters, changed defaults, dropped ``*args``/``**kwargs``)
    is a breaking change. Signatures that cannot be parsed must match exactly.
    """
    if old == new:
        return True
    old_args, new_args = _parse_params(old), _parse_params(new)
    if old_args is None or new_args is None:
        return False
    old_pos = [a.arg for a in [*old_args.posonlyargs, *old_args.args]]
    new_pos = [a.arg for a in [*new_args.posonlyargs, *new_args.args]]
    if new_pos[: len(old_pos)] != old_pos:
        return False
    if (old_args.vararg and not new_args.vararg) or (
        old_args.kwarg and not new_args.kwarg
    ):
        return False
    old_defaults, new_defaults = (
        _defaults_by_name(old_args),
        _defaults_by_name(new_args),
    )
    for name, default in old_defaults.items():
        if name not in new_defaults:
            return False
        if default is not None and new_defaults[name] != default:
            return False
    added = set(new_defaults) - set(old_defaults)
    return all(new_defaults[name] is not None for name in added)


def compare(
    snapshot: list[str], live: list[str], failed_modules: list[str]
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Compare the committed snapshot with the live API.

    Returns ``(removed, changed, added)``: snapshot lines whose symbol is gone,
    ``(old, new)`` pairs whose signature changed incompatibly, and live lines
    that are not in the snapshot. Symbols in ``failed_modules`` are skipped.
    """
    skip = set(failed_modules)
    old = {key_of(ln): ln for ln in snapshot if module_of(ln) not in skip}
    new = {key_of(ln): ln for ln in live}
    removed = [old[k] for k in sorted(old.keys() - new.keys())]
    changed = [
        (old[k], new[k])
        for k in sorted(old.keys() & new.keys())
        if not is_compatible(old[k], new[k])
    ]
    added = [new[k] for k in sorted(new.keys() - old.keys())]
    return removed, changed, added


def read_snapshot(path: Path = SNAPSHOT_PATH) -> list[str]:
    """Read the committed snapshot, ignoring blank lines and comments."""
    text = path.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln and not ln.startswith("#")]


def write_snapshot(lines: list[str], path: Path = SNAPSHOT_PATH) -> None:
    """Write ``lines`` to the committed snapshot file."""
    header = (
        "# Optiland public API snapshot. Generated; do not edit by hand.\n"
        "# Regenerate: python -m scripts.guard.api_snapshot --update\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="rewrite the snapshot")
    parser.add_argument("--json", action="store_true", help="print JSON for tools")
    args = parser.parse_args(argv)
    lines, failed = collect()
    if args.json:
        print(json.dumps({"lines": lines, "failed": failed}))
        return 0
    if failed:
        print(f"# not importable here: {', '.join(failed)}", file=sys.stderr)
    if args.update:
        if failed:
            print(
                "Refusing to update: install all extras (uv sync --dev --all-extras)"
                " so every module can be imported.",
                file=sys.stderr,
            )
            return 1
        write_snapshot(lines)
        print(f"Wrote {len(lines)} entries to {SNAPSHOT_PATH.relative_to(ROOT)}")
    else:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
