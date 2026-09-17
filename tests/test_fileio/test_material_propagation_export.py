"""Homogeneous file formats must not discard custom medium propagation."""

from __future__ import annotations

import pytest

from optiland.fileio import load_oslo_file, save_codev_file, save_oslo_file, save_zemax_file
from optiland.fileio.common import reject_unsupported_propagation
from optiland.materials import AbbeMaterial, IdealMaterial, Material
from optiland.propagation.grin import GRINPropagation
from optiland.propagation.homogeneous import HomogeneousPropagation


class CustomPropagation(HomogeneousPropagation):
    """Even a subclass may implement different physics from its base."""


@pytest.mark.parametrize("writer", [save_codev_file, save_zemax_file, save_oslo_file])
@pytest.mark.parametrize("kind", ["air", "catalog", "model"])
@pytest.mark.parametrize("side", ["material_pre", "material_post"])
@pytest.mark.parametrize("reflective", [False, True])
def test_export_checks_both_media_before_shortcuts(
    set_test_backend, writer, kind, side, reflective, lens_file, tmp_path
):
    optic = load_oslo_file(lens_file(), strict=True)
    material = {"air": lambda: IdealMaterial(1),
                "catalog": lambda: Material("N-BK7"),
                "model": lambda: AbbeMaterial(1.5, 60)}[kind]()
    material.propagation_model = CustomPropagation(material)
    surface = optic.surfaces[1]
    if side == "material_pre":
        optic.surfaces[0].material_post = material
    else:
        surface.material_post = material
    surface.interaction_model.is_reflective = reflective
    path = tmp_path / "existing.txt"
    path.write_bytes(b"previous design")
    with pytest.raises(NotImplementedError, match="custom propagation"):
        writer(optic, path)
    assert path.read_bytes() == b"previous design"


def test_grin_placeholder_is_not_an_exportable_homogeneous_medium():
    material = IdealMaterial(1.5, propagation_model=GRINPropagation())
    with pytest.raises(NotImplementedError, match="custom propagation"):
        reject_unsupported_propagation(material)


@pytest.mark.parametrize("material", [None, "air", "mirror"])
def test_legacy_specifications_have_no_custom_propagation(material):
    reject_unsupported_propagation(material)
