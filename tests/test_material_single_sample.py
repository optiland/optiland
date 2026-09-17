"""Existing file tables with one sample are constant under endpoint clamping."""

from __future__ import annotations

import optiland.backend as be
from optiland.materials import MaterialFile

from .utils import assert_allclose


def test_file_single_sample_is_constant_on_both_backends(set_test_backend, tmp_path):
    path = tmp_path / "single.yml"
    path.write_text('DATA:\n- type: tabulated nk\n  data: |\n    0.5 1.55 0.02\n')
    material = MaterialFile(str(path))
    waves = be.asarray([0.4, 0.5, 0.6])
    assert_allclose(material.n(waves), [1.55, 1.55, 1.55])
    assert_allclose(material.k(waves), [0.02, 0.02, 0.02])
