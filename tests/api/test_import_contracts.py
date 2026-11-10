"""Architectural import rules that keep the package layers apart.

* ``optiland`` (the core library) never imports ``optiland_gui``.
* ``optiland.backend`` is the bottom layer: it may import only itself and the
  private top-level helpers (``optiland._types`` and the like).
"""

from __future__ import annotations

from scripts.guard.files import imported_modules, tracked_python_files


def _violations(prefix: str, is_forbidden) -> list[str]:
    found = []
    for path in tracked_python_files(prefix):
        for line, name in imported_modules(path):
            if is_forbidden(name):
                found.append(f"{path.as_posix()}:{line} imports {name}")
    return found


def test_core_never_imports_gui():
    bad = _violations(
        "optiland/",
        lambda name: name == "optiland_gui" or name.startswith("optiland_gui."),
    )
    assert not bad, "optiland/ must not depend on optiland_gui/:\n" + "\n".join(bad)


def _backend_forbidden(name: str) -> bool:
    parts = name.split(".")
    if parts[0] != "optiland" or len(parts) == 1:
        return parts == ["optiland"]
    return parts[1] != "backend" and not parts[1].startswith("_")


def test_backend_is_the_bottom_layer():
    bad = _violations("optiland/backend/", _backend_forbidden)
    assert not bad, (
        "optiland/backend/ may only import optiland.backend and private optiland._* "
        "helpers:\n" + "\n".join(bad)
    )


def test_backend_rule_examples():
    assert _backend_forbidden("optiland.rays")
    assert _backend_forbidden("optiland")
    assert not _backend_forbidden("optiland.backend.utils")
    assert not _backend_forbidden("optiland._types")
    assert not _backend_forbidden("numpy")
