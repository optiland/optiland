"""Export must not round valid optical data across physical boundaries."""

from __future__ import annotations

import math

import pytest

from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.materials import DataMaterial


def test_export_preserves_an_angular_reference_just_below_ninety(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.fields.fields.clear()
    optic.fields.add(y=89.999999)
    optic.fields.add(y=1)
    output = tmp_path / "near-right-angle.len"
    save_oslo_file(optic, output)
    restored = load_oslo_file(output, strict=True)
    assert restored.fields[0].y == optic.fields[0].y
    assert math.tan(math.radians(restored.fields[1].y)) == pytest.approx(
        math.tan(math.radians(1)), rel=1e-12
    )


@pytest.mark.parametrize("medium,index", [("AIR", 1.0), ("GLA 1.5", 1.5)])
def test_export_preserves_na_below_the_object_medium_index(
    lens_file, tmp_path, set_test_backend, medium, index
):
    optic = load_oslo_file(lens_file(distance="100", system=medium), strict=True)
    value = index - 1e-8
    optic.set_aperture("objectNA", value)
    output = tmp_path / "near-hemisphere.len"
    save_oslo_file(optic, output)
    restored = load_oslo_file(output, strict=True)
    assert restored.aperture.value == value


def test_export_preserves_distinct_sampled_wavelengths(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(), strict=True)
    wavelengths = [0.50000000000001, 0.50000000000002]
    while optic.wavelengths:
        optic.wavelengths.remove(0)
    for index, wavelength in enumerate(wavelengths):
        optic.wavelengths.add(wavelength, is_primary=index == 0)
    optic.surfaces[1].material_post = DataMaterial.from_samples(wavelengths, [1.5, 1.500001])
    output = tmp_path / "nearby-wavelengths.len"
    save_oslo_file(optic, output)
    restored = load_oslo_file(output, strict=True)
    assert [wave.value for wave in restored.wavelengths] == wavelengths
    assert restored.surfaces[1].material_post.definition.dispersion.wavelengths_um == tuple(wavelengths)


def test_export_retains_small_sampled_dispersion_and_its_optical_path(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(system="WV .5 .6"), strict=True)
    indices = [1.0000000000001, 1.0000000000002]
    optic.surfaces[1].material_post = DataMaterial.from_samples([0.5, 0.6], indices)
    output = tmp_path / "small-dispersion.len"
    save_oslo_file(optic, output)
    restored = load_oslo_file(output, strict=True)
    material = restored.surfaces[1].material_post
    # The index difference is small but nonzero: collapsing both samples to
    # unity removes dispersion and their optical-path excess over vacuum.
    for wavelength, index in zip([0.5, 0.6], indices, strict=True):
        assert float(material.n(wavelength).item()) == index
