"""Independent OSLO configuration snapshots and declarative footer boundaries."""

from __future__ import annotations

import pytest

from optiland.fileio import load_oslo_file
from optiland.fileio.oslo.reader.converter import OsloToOpticConverter
from optiland.fileio.oslo.reader.parser import OsloDataParser
from tests.utils import assert_allclose


def configuration_lens(tmp_path, footer="", surface="", second="PK THM 1 0"):
    path = tmp_path / "configurations.len"
    path.write_text(
        'LEN NEW "configurations" 1 4; NAO .05; OBH .2; WV .55; WW 1; '
        "TH 20; NXT; TH 4; "
        + surface
        + "; NXT; TH -4; "
        + second
        + "; NXT; WV .4 .55 .8; GLA CONFIG 1.64 1.62 1.60; "
        "WV .55; TH 2; NXT; AIR; END 4\n" + footer
    )
    return path


def test_configuration_overrides_precede_pickups_and_dispersion(
    tmp_path, set_test_backend
):
    path = configuration_lens(
        tmp_path,
        "CFG NEW\nTH 1 2 7\nWV1 2 .45\nTH 1 3 9\nWV1 3 .75\nEND\n"
        "CFWT 1 2\nCFWT 2 .5\nCFAC 2 NO\nCFAC 3 YES\n",
    )
    data = OsloDataParser(path, strict=True).parse()
    assert list(data.configurations) == [1, 2, 3]
    assert data.configurations[1].weight == 2
    assert data.configurations[2].weight == 0.5
    assert data.configurations[2].active is False
    assert data.to_dict()["configurations"][3]["active"] is True
    optics = [load_oslo_file(path, strict=True, configuration=i) for i in (1, 2, 3)]
    for optic, thickness, wavelength in zip(
        optics, (4, 7, 9), (0.55, 0.45, 0.75), strict=True
    ):
        assert optic.surfaces[1].thickness == thickness
        assert optic.surfaces[2].thickness == -thickness
        assert optic.surfaces[2].geometry.cs.z.item() == thickness
        assert optic.surfaces[3].geometry.cs.z.item() == 0
        assert optic.primary_wavelength == wavelength
    indices = [
        optic.surfaces[3].material_post.n(optic.primary_wavelength).item()
        for optic in optics
    ]
    assert indices[1] > indices[0] > indices[2]
    optics[1].surfaces[1].thickness = 99
    assert optics[0].surfaces[1].thickness == 4
    assert data.surfaces[1]["TH"] == 4


def test_configuration_repositions_a_folded_leg(tmp_path, set_test_backend):
    path = configuration_lens(
        tmp_path, "CFG NEW; TH 1 2 -8; END", "RFL; TLA 45; BEN", "AIR"
    )
    optic = load_oslo_file(path, strict=True, configuration=2)
    assert_allclose(optic.surfaces[2].geometry.cs.y, -8)
    assert_allclose(optic.surfaces[2].geometry.cs.z, 0)


@pytest.mark.parametrize(
    "footer,message",
    [
        ("CFG NEW; TH 1 1 4; END", "configuration"),
        ("CFG NEW; TH 9 2 4; END", "surface"),
        ("CFG NEW; TH 1 2; END", "TH"),
        ("CFG NEW; TH 1 2 nan; END", "finite"),
        ("CFG NEW; WV1 2 -1; END", "positive"),
        ("CFG NEW; WV3 2 .7; END", "undefined wavelength"),
        ("CFG NEW; WW1 2 0; END", "primary wavelength"),
        ("CFG NEW; TH 1 2 4", "unterminated CFG"),
        ("CFG NEW; CFG NEW; END", "nested CFG"),
        ("CFG NEW; END; CFG NEW; END", "multiple CFG"),
        ("CFWT 1 -1", "nonnegative"),
        ("CFWT 0 1", "configuration"),
        ("CFAC 1 MAYBE", "YES or NO"),
        ("CFG NEW; WV1 2; END", "requires configuration and value"),
        ("CFG NEW; WV1002 2 .6; END", "exceeds 1001"),
        ("CFWT 1", "requires configuration and value"),
        ("CFG NEW; WW1 2 -1; END", "nonnegative"),
        ("CFG NEW; WW2 2 1; END", "undefined wavelength"),
        ("CFG NEW; END 1", "END takes no arguments"),
    ],
)
def test_invalid_configuration_records_fail(tmp_path, footer, message):
    with pytest.raises(ValueError, match=message):
        load_oslo_file(
            configuration_lens(tmp_path, footer), strict=True, configuration=2
        )


@pytest.mark.parametrize("configuration", [0, -1, 1.5, True, 8])
def test_invalid_configuration_selection_fails(tmp_path, configuration):
    with pytest.raises(ValueError, match="configuration"):
        load_oslo_file(configuration_lens(tmp_path), configuration=configuration)


def test_field_table_after_configuration_footer_is_preserved(tmp_path):
    path = configuration_lens(
        tmp_path, "CFG NEW; TH 1 2 6; END; RST NEW; F 1 .5 0 0 0 0 -1 1 -1 1 1; END"
    )
    optic = load_oslo_file(path, strict=True, configuration=2)
    assert_allclose(optic.fields.y_fields, [0.1])


def test_unknown_configuration_override_is_diagnosed(tmp_path):
    path = configuration_lens(tmp_path, "CFG NEW; TEM 2 30; END")
    with pytest.raises(ValueError, match="TEM.*configuration"):
        load_oslo_file(path, strict=True)
    with pytest.warns(UserWarning, match="TEM.*configuration"):
        load_oslo_file(path)


def test_non_declarative_configuration_stops_footer_import(tmp_path):
    path = configuration_lens(tmp_path, "CFG RUN; TH 1 2 9; END")
    with pytest.warns(UserWarning, match="only declarative CFG NEW"):
        optic = load_oslo_file(path)
    assert optic.surfaces[1].thickness == 4
    with pytest.raises(ValueError, match="only declarative CFG NEW"):
        load_oslo_file(path, strict=True)


@pytest.mark.parametrize(
    "footer,surface,message",
    [
        ("CFG NEW; TH 2 2 6; END", "", "pickup-controlled"),
        ("CFG NEW; TH 1 2 6; END", "PY 0", "CSLV"),
    ],
)
def test_configuration_rejects_unmapped_constraint_precedence(
    tmp_path, footer, surface, message
):
    with pytest.raises(ValueError, match=message):
        load_oslo_file(
            configuration_lens(tmp_path, footer, surface),
            strict=True,
            configuration=2,
        )


def test_implicit_configuration_and_added_spectrum_slot(tmp_path):
    path = configuration_lens(tmp_path, "CFG NEW; WV2 3 .65; WW2 3 .3; END; CFAC 2 NO")
    data = OsloDataParser(path, strict=True).parse()
    assert list(data.configurations) == [1, 2, 3]
    second = load_oslo_file(path, strict=True, configuration=2)
    third = load_oslo_file(path, strict=True, configuration=3)
    assert second.primary_wavelength == 0.55
    assert len(second.wavelengths) == 1
    assert len(third.wavelengths) == 2
    assert third.wavelengths[1].value == 0.65
    assert third.wavelengths[1].weight == 0.3


@pytest.mark.parametrize("kind", ["th", "ThM", "ln", "lNm"])
def test_configuration_rejects_mixed_case_pickup_conflicts(tmp_path, kind):
    pickup = f"PK {kind} 1" + (" 0" if kind.upper() in {"LN", "LNM"} else "")
    path = configuration_lens(tmp_path, "CFG NEW; TH 2 2 6; END", second=pickup)
    with pytest.raises(ValueError, match="pickup-controlled"):
        load_oslo_file(path, strict=True, configuration=2)


def test_reusing_converter_keeps_configuration_base(tmp_path, set_test_backend):
    path = configuration_lens(tmp_path, "CFG NEW; TH 1 2 7; WV1 2 .45; END")
    data = OsloDataParser(path, strict=True).parse()
    converter = OsloToOpticConverter(data, strict=True, configuration=2)
    alternate = converter.convert()
    converter.configuration = 1
    base = converter.convert()
    assert alternate.surfaces[1].thickness == 7
    assert base.surfaces[1].thickness == 4
    assert base.surfaces[2].thickness == -4
    assert base.primary_wavelength == 0.55
    assert converter.data is data


def test_failed_conversion_keeps_configuration_base(tmp_path):
    path = configuration_lens(tmp_path, "CFG NEW; TH 0 2 1e10; END")
    data = OsloDataParser(path, strict=True).parse()
    converter = OsloToOpticConverter(data, strict=True, configuration=2)
    with pytest.raises(ValueError, match="finite"):
        converter.convert()
    converter.configuration = 1
    assert converter.convert().primary_wavelength == 0.55
    assert converter.data is data
