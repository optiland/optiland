"""GUI installation must exclude wheels that corrupt CPython singletons."""

from __future__ import annotations

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
import pytest


@pytest.mark.parametrize("python_version", ["3.11", "3.12", "3.13", "3.14"])
@pytest.mark.parametrize("qt_version", ["6.10.1", "6.11.2", "6.12.0"])
def test_gui_binding_resolution_by_python_version(python_version, qt_version):
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]
    requirements = [Requirement(value) for value in project["optional-dependencies"]["gui"]]
    bindings = [
        requirement for requirement in requirements
        if requirement.name.lower() == "pyside6"
        and (requirement.marker is None or requirement.marker.evaluate(
            {"python_version": python_version}
        ))
    ]
    assert bindings, "The GUI extra must still install a Qt binding"
    allowed = all(qt_version in requirement.specifier for requirement in bindings)
    assert allowed == (python_version != "3.11" or qt_version != "6.12.0")
