"""Lossy material conversions must fail before touching an export destination."""

from __future__ import annotations

import pytest

from optiland.fileio import load_oslo_file, save_oslo_file, save_zemax_file, save_codev_file
from optiland.materials import DataMaterial


@pytest.mark.parametrize('writer', [save_zemax_file, save_codev_file])
@pytest.mark.parametrize('index', [1.0, 1.5])
def test_inline_material_does_not_become_air_or_an_abbe_approximation(writer, index, lens_file, tmp_path):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.surfaces[1].material_post = DataMaterial.from_coefficients('formula 5', [index], name='N-BK7')
    output = tmp_path / 'untouched.txt'
    output.write_bytes(b'previous export')
    with pytest.raises(NotImplementedError, match='native JSON'):
        writer(optic, output)
    assert output.read_bytes() == b'previous export'


@pytest.mark.parametrize('kind', ['formula', 'extinction', 'negative', 'clamp'])
def test_oslo_only_encodes_lossless_positive_n_tables(kind, lens_file, tmp_path):
    optic = load_oslo_file(lens_file(), strict=True)
    if kind == 'formula':
        material = DataMaterial.from_coefficients('formula 5', [1.5])
    else:
        extinction = {'kind': 'tabulated_k', 'wavelengths_um': [.4, .8], 'values': [.01, .01]} if kind == 'extinction' else None
        material = DataMaterial.from_samples([.4, .8], [-1.6, -1.4] if kind == 'negative' else [1.6, 1.4], extinction=extinction, bounds='clamp' if kind == 'clamp' else 'raise')
    optic.surfaces[1].material_post = material
    output = tmp_path / 'untouched.len'
    output.write_bytes(b'previous export')
    with pytest.raises((ValueError, NotImplementedError)):
        save_oslo_file(optic, output)
    assert output.read_bytes() == b'previous export'
