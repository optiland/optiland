"""The common plotting caller supports scalar-valued existing material models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from optiland.materials import AbbeMaterial, AbbeMaterialE, MaterialFile
from optiland.materials.material_utils import plot_nk
from tests.utils import assert_allclose


@pytest.mark.parametrize("kind", ["polynomial", "buchdahl", "e-line", "constant-file"])
def test_plot_existing_models_with_scalar_properties(kind, tmp_path, set_test_backend):
    if kind == "constant-file":
        path = tmp_path / "constant.yml"
        path.write_text(
            "DATA:\n- type: formula 5\n  coefficients: '1.5'\n", encoding="utf-8"
        )
        material = MaterialFile(path)
    elif kind == "e-line":
        material = AbbeMaterialE(1.5, 60)
    else:
        material = AbbeMaterial(1.5, 60, model=kind)
    try:
        _, (ax_n, ax_k) = plot_nk(material, wavelength_range=(0.4, 0.7), n_sample=9)
        for axis in [ax_n, ax_k]:
            assert axis.lines[0].get_ydata().shape == (9,)
        assert_allclose(ax_k.lines[0].get_ydata(), 0)
        if kind == "constant-file":
            assert_allclose(ax_n.lines[0].get_ydata(), 1.5)
    finally:
        plt.close("all")
