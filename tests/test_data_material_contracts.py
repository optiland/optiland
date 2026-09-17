"""Owned optical definitions, independent physical checks and consumer contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

import optiland.backend as be
from optiland.materials import AbbeMaterial, AbbeMaterialE, BaseMaterial, DataMaterial, IdealMaterial, Material, MaterialFile, plot_nk
from optiland.visualization.info.material_formatter import MaterialFormatter
from tests.utils import assert_allclose


FORMULAS = [
    (1, [0, 1, .1], lambda w: np.sqrt(1 + w*w/(w*w-.01))),
    (2, [0, 1, .01], lambda w: np.sqrt(1 + w*w/(w*w-.01))),
    (3, [2, .1, -2], lambda w: np.sqrt(2 + .1/w**2)),
    (4, [1, 1, 2, .1, 2, 1, 2, .2, 2, .1, 2],
     lambda w: np.sqrt(1 + w*w/(w*w-.01) + w*w/(w*w-.04) + .1*w*w)),
    (5, [1.5, .01, -2], lambda w: 1.5 + .01/w**2),
    (6, [.5, .1, 10], lambda w: 1.5 + .1/(10-1/w**2)),
    (7, [1.5, .01, .001, .1, .2],
     lambda w: 1.5 + .01/(w*w-.028) + .001/(w*w-.028)**2 + .1*w*w + .2*w**4),
    (8, [.1, .01, .01, .01],
     lambda w: np.sqrt((1+2*(.1+.01*w*w/(w*w-.01)+.01*w*w))/(1-(.1+.01*w*w/(w*w-.01)+.01*w*w)))),
    (9, [2, .1, .01, .1, .1, .01],
     lambda w: np.sqrt(2+.1/(w*w-.01)+.1*(w-.1)/((w-.1)**2+.01))),
]


@pytest.mark.parametrize('number,coefficients,reference', FORMULAS)
def test_all_formulas_keep_analytic_model_and_match_file_adapter(number, coefficients, reference, tmp_path, set_test_backend):
    name = f'formula {number}'
    material = DataMaterial.from_coefficients(name, coefficients, wavelength_range=(.4, .8))
    path = tmp_path / 'material.yml'
    path.write_text(yaml.safe_dump({'DATA': [{'type': name, 'coefficients': ' '.join(map(str, coefficients)), 'wavelength_range': '.4 .8'}]}))
    file_material = MaterialFile(str(path))
    waves = np.array([[.45, .5], [.65, .75]])
    actual = material.n(be.asarray(waves))
    assert_allclose(actual, reference(waves))
    # Existing files keep column coefficients, owned data keeps plain tuples.
    assert_allclose(file_material.n(be.asarray(waves)), actual)
    restored = BaseMaterial.from_dict(material.to_dict())
    assert restored.definition == material.definition
    assert_allclose(restored.n(.55), reference(.55))
    with pytest.raises(ValueError, match='range'):
        material.n(.9)


def test_owned_inputs_outputs_and_metadata_are_independent(set_test_backend):
    waves, indices = [.8, .4], [1.4, 1.6]
    metadata = {'source': {'catalog': ['synthetic']}}
    material = DataMaterial.from_samples(waves, indices, metadata=metadata)
    waves[0], indices[0] = .1, 3
    metadata['source']['catalog'][0] = 'changed'
    serialized = material.to_dict()
    serialized['definition']['dispersion']['indices'][0] = 9
    material.metadata['source']['catalog'][0] = 'changed'
    assert_allclose(material.n(.5), 1.55)
    assert material.metadata == {'source': {'catalog': ['synthetic']}}
    with pytest.raises(FrozenInstanceError):
        material.definition.dispersion.indices = (2, 3)
    assert material.to_dict()['type'] == 'DataMaterial'
    assert BaseMaterial.from_dict(material.to_dict()).propagation_model.to_dict() == material.propagation_model.to_dict()


def test_native_dispatch_preserves_registered_propagation():
    from optiland.propagation.grin import GRINPropagation
    material = DataMaterial.from_coefficients('formula 5', [1.5], propagation_model=GRINPropagation())
    restored = BaseMaterial.from_dict(material.to_dict())
    assert isinstance(restored.propagation_model, GRINPropagation)


@pytest.mark.parametrize('metadata', [[1], {'a': float('nan')}, {'a': object()}])
def test_metadata_requires_finite_json(metadata):
    with pytest.raises(ValueError, match='metadata'):
        DataMaterial.from_samples([.4, .8], [1.6, 1.4], metadata=metadata)


def test_extinction_has_its_own_grid_and_bounds(set_test_backend):
    material = DataMaterial.from_samples([.4, .8], [-1.6, -1.4], extinction={
        'kind': 'tabulated_k', 'wavelengths_um': [.5, .7], 'values': [.01, .03]})
    assert_allclose(material.n(.6), -1.5)
    assert_allclose(material.k(.6), .02)
    assert material.spectral_range('n') == (.4, .8)
    assert material.spectral_range('k') == (.5, .7)
    assert_allclose(material.n(.45), -1.575)
    with pytest.raises(ValueError, match='range'):
        material.k(.45)
    with pytest.raises(ValueError, match='nonnegative'):
        DataMaterial.from_samples([.4, .8], [1.6, 1.4], extinction={
            'kind': 'tabulated_k', 'wavelengths_um': [.5, .7], 'values': [-.01, .03]})


@pytest.mark.parametrize('definition', [None, {}, {'dispersion': 1},
    {'dispersion': {'kind': 'unknown'}},
    {'dispersion': {'kind': 'tabulated', 'wavelengths_um': [.4, .8], 'indices': [1.6, 1.4], 'extra': True}},
    {'dispersion': {'kind': 'formula', 'formula': {}, 'coefficients': [1]}},
    {'dispersion': {'kind': 'formula', 'formula': 'formula 1', 'coefficients': [1, 2]}},
    {'dispersion': {'kind': 'formula', 'formula': 'formula 99', 'coefficients': [1]}},
    {'dispersion': {'kind': 'formula', 'formula': 'formula 5', 'coefficients': None}},
    {'dispersion': {'kind': 'formula', 'formula': 'formula 5', 'coefficients': [1], 'wavelength_range_um': [1, .5]}},
])
def test_invalid_optical_definitions_fail_at_construction(definition):
    with pytest.raises(ValueError):
        DataMaterial(definition)


def test_reject_unknown_extinction_kind_and_thermal_at_this_feature():
    definition = {'dispersion': {'kind': 'formula', 'formula': 'formula 5', 'coefficients': [1.5]}}
    with pytest.raises(ValueError, match='Thermal'):
        DataMaterial({**definition, 'thermal': {'kind': 'unknown'}})
    with pytest.raises(ValueError, match='extinction kind'):
        DataMaterial({**definition, 'extinction': {'kind': 'unknown', 'wavelengths_um': [.4, .8], 'values': [0, 0]}})


@pytest.mark.parametrize('wave', [0, -.1, float('nan'), float('inf')])
def test_invalid_query_is_rejected_for_both_properties(wave, set_test_backend):
    material = DataMaterial.from_coefficients('formula 5', [1.5])
    for query in [material.n, material.k]:
        with pytest.raises(ValueError, match='finite and positive'):
            query(wave)


def test_nonfinite_formula_result_is_rejected(set_test_backend):
    material = DataMaterial.from_coefficients('formula 3', [-1])
    with pytest.raises(ValueError, match='non-finite'):
        material.n(.5)


@pytest.mark.parametrize('kind', ['sampled', 'formula'])
def test_each_wavelength_keeps_its_gradient_on_repeated_queries(kind, set_test_backend):
    if be.get_backend() != 'torch':
        pytest.skip('Torch gradient contract')
    import torch

    material = (DataMaterial.from_samples([.4, .8], [1.6, 1.4]) if kind == 'sampled'
                else DataMaterial.from_coefficients('formula 5', [1.5, .01, -2]))
    material.n(.5)  # Warm the scalar cache before requesting independent gradients.
    for _ in range(2):
        waves = torch.tensor([[.5, .5], [.6, .6]], dtype=torch.float64, requires_grad=True)
        material.n(waves).sum().backward()
        expected = torch.full_like(waves, -.5) if kind == 'sampled' else -.02/waves.detach()**3
        torch.testing.assert_close(waves.grad, expected)
    with pytest.raises(ValueError):
        DataMaterial.from_coefficients('formula 5', torch.tensor([1.5], requires_grad=True))


def test_plotting_intersects_support_and_shows_units(set_test_backend):
    material = DataMaterial.from_samples([.4, .8], [1.6, 1.4], name='Owned samples', extinction={
        'kind': 'tabulated_k', 'wavelengths_um': [.5, .7], 'values': [.01, .03]})
    fig, (ax, _) = plot_nk(material, wavelength_range=(.3, .9), n_sample=8)
    assert_allclose(ax.lines[0].get_xdata(), np.linspace(.5, .7, 8))
    assert ax.get_title() == 'Owned samples'
    assert 'µm' in ax.get_xlabel()
    plt.close(fig)
    with pytest.raises(ValueError, match='No common'):
        plot_nk(material, wavelength_range=(.8, .9))
    with pytest.raises(ValueError, match='Specify wavelength_range'):
        plot_nk(IdealMaterial(1.5))
    fig, _ = plot_nk(IdealMaterial(1.5), wavelength_range=(.4, .8))
    plt.close(fig)


def test_labels_and_ranges_are_shared_by_information_display(set_test_backend):
    glasses = [(IdealMaterial(1), 'Air'), (IdealMaterial(1, .01), '1.0'),
        (AbbeMaterial(1.5, 60, model='buchdahl'), '1.5000, 60.00'),
        (AbbeMaterialE(1.5, 60), '1.5000, 60.00 (ne, Ve)'),
        (DataMaterial.from_samples([.4, .8], [1.6, 1.4], name='test'), 'test')]
    for glass, name in glasses:
        surface = SimpleNamespace(material_post=glass, interaction_model=SimpleNamespace(is_reflective=False))
        assert glass.display_name == MaterialFormatter.format(surface) == name
        with pytest.raises(ValueError, match='property'):
            glass.spectral_range('unknown')
    polynomial = AbbeMaterial(1.5, 60, model='polynomial')
    assert polynomial.spectral_range() == (.38, .75)
    assert AbbeMaterialE(1.5, 60).spectral_range() is None
    catalog = Material('N-BK7', catalog='schott')
    assert catalog.display_name == 'N-BK7'
    assert catalog.spectral_range()[0] > 0
