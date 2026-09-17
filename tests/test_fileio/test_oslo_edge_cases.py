"""OSLO boundary validation, fallback behavior and less common round trips."""

from __future__ import annotations

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

import optiland.backend as be
from optiland.fields.field_types import AngleField, ObjectHeightField
from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.fileio.oslo.model import OsloDataModel
from optiland.fileio.oslo.reader.converter import OsloToOpticConverter
from optiland.fileio.oslo.reader.coordinates import surface_coordinates
from optiland.fileio.oslo.reader.parser import OsloDataParser
from optiland.fileio.oslo.reader.pickups import resolve_pickups
from optiland.fileio.oslo.writer.encoder import OpticToOsloEncoder
from optiland.fileio.oslo.writer.formatter import OsloDataFormatter
from optiland.materials import AbbeMaterial
from optiland.physical_apertures import BaseAperture, RectangularAperture
from optiland.rays import RealRays
from tests.utils import assert_allclose


@pytest.mark.parametrize(
    "command,message",
    [
        ("WW 0 0 0", "weights cannot all be zero"),
        ("TELE MAYBE", "TELE expects"),
        ("APCK MAYBE", "APCK expects"),
        ("AP -1", "radius must be nonnegative"),
        ("AS1", "one coefficient"),
        ("WV1002 .55", "bounded wavelength index"),
        ("WV2 .55 .6", "one value"),
        ("WV5 .55", "undefined wavelength slots"),
        ("GTO -1", "outside the declared lens"),
        ("GTO 4", "outside the declared lens"),
        ("END 2", "count differs from LEN"),
        ("PK AP 0 1", "invalid number of arguments"),
    ],
)
def test_parser_rejects_invalid_boundaries_with_source(lens_file, command, message):
    with pytest.raises(ValueError, match=message) as error:
        OsloDataParser(lens_file(surface=command), strict=True).parse()
    assert "edge.len:" in str(error.value)


@pytest.mark.parametrize("count", [0, 10001])
def test_surface_count_limit(tmp_path, count):
    path = tmp_path / "count.len"
    path.write_text(f'LEN NEW "invalid count" 1 {count}\n')
    with pytest.raises(ValueError, match="LEN surface count"):
        OsloDataParser(path).parse()


def test_nxt_cannot_append_beyond_declared_image(lens_file):
    with pytest.raises(ValueError, match="NXT exceeds LEN"):
        OsloDataParser(lens_file(image="NXT")).parse()


@pytest.mark.parametrize(
    "record,message",
    [
        ("F 1 0 0", "ten field-table values"),
        ("F 0 0 0 0 0 0 -1 1 -1 1 1", "positive index"),
        ("F 1 0 0 0 0 0 -1 1 -1 1 -1", "nonnegative weight"),
        ("F 1 0 0 0 0 0 1 -1 -1 1 1", "bounds must be increasing"),
        ("F 1 0 0 0 0 0 -1 1 0 0 1", "bounds must be increasing"),
    ],
)
def test_invalid_field_table(lens_file, record, message):
    with pytest.raises(ValueError, match=message):
        OsloDataParser(lens_file(footer=f"RST NEW\n{record}\nEND\n")).parse()


def test_asymmetric_field_pupil_warns_and_retains_field(lens_file, set_test_backend):
    path = lens_file(
        system="OBH 4",
        distance="100",
        footer=("RST NEW\nF 1 -.5 .25 0 0 0 -.5 1 -1 1 2\nEND\n"),
    )
    with pytest.warns(UserWarning, match="asymmetric field pupil"):
        optic = load_oslo_file(path)
    assert_allclose([optic.fields[0].x, optic.fields[0].y], [1, -2])
    assert_allclose([optic.fields[0].vx, optic.fields[0].vy], [0, 0])
    assert optic.fields[0].weight == 2
    with pytest.raises(ValueError, match="asymmetric field pupil"):
        load_oslo_file(path, strict=True)


@pytest.mark.parametrize(
    "command,message",
    [
        ("LMO NSS", "non-sequential groups"),
        ("ASP ZER 4", "asphere type ZER"),
        ("PK UNSUPPORTED 0", "pickup type UNSUPPORTED"),
    ],
)
def test_unsupported_prescriptions_warn_or_reject(lens_file, command, message):
    path = lens_file(surface=command)
    with pytest.warns(UserWarning, match=message):
        model = OsloDataParser(path).parse()
    assert model.diagnostics[0].command == command.split()[0]
    with pytest.raises(ValueError, match=message):
        load_oslo_file(path, strict=True)
    # A warned unsupported pickup must not be executed by the converter.
    with pytest.warns(UserWarning, match=message):
        optic = load_oslo_file(path)
    assert_allclose(optic.surfaces[1].geometry.radius, 20)


@pytest.mark.parametrize(
    "command,message",
    [
        ("GLA 0", "positive refractive indices"),
        ("GLA 1.5 1.6", "index/wavelength counts differ"),
        ("GSP -1", "positive spacing"),
        ("GOR 1", "positive spacing"),
        ("ASP ASR 1\nAS257 .001", "limit 256"),
        ("ASP ASR 1\nAS0 .001", "offset sag"),
        ("ASP ARA 1\nAS0 .001", "offset sag"),
        ("ASP ASX 1\nCVX .01", "even YZ asphere"),
        ("GC 2", "preceding surface"),
        ("RCO 2", "unavailable surface"),
        ("TLA 10\nTLB 20\nBEN", "single-axis local mirror"),
        ("DCY 1\nTH 1e20", "infinite thickness"),
    ],
)
def test_converter_rejects_invalid_optical_data(lens_file, command, message):
    with pytest.raises(ValueError, match=message):
        load_oslo_file(lens_file(surface=command), strict=True)


def test_coordinate_model_rejects_invalid_transform_order():
    # Models can be supplied independently of the text parser.
    with pytest.raises(ValueError, match="OSLO DT must be"):
        surface_coordinates({0: {"TH": 100}, 1: {"DT": 0}}, 1)


@pytest.mark.parametrize("radius", [0, 40])
def test_rdx_maps_toric_x_radius(lens_file, set_test_backend, radius):
    optic = load_oslo_file(lens_file(surface=f"RDX {radius}"), strict=True)
    sag = optic.surfaces[1].geometry.sag(be.array([2.0]), be.array([0.0]))
    expected = 0 if radius == 0 else 40 - math.sqrt(40**2 - 2**2)
    assert_allclose(sag, [expected])


def test_default_aperture_uses_lens_units(lens_file, set_test_backend):
    optic = load_oslo_file(lens_file(aperture="", system="UNI 10"), strict=True)
    assert optic.aperture.ap_type == "EPD"
    assert_allclose(optic.aperture.value, 20)


def test_image_na_rejects_afocal_system(lens_file):
    with pytest.raises(ValueError, match="afocal system"):
        load_oslo_file(
            lens_file(system="NAP .1", surface="RD 0", second="RD 0"), strict=True
        )


@pytest.mark.parametrize("scale", [1, 10])
def test_finite_gaussian_image_height_has_expected_magnification(
    lens_file, set_test_backend, scale
):
    optic = load_oslo_file(
        lens_file(system=f"UNI {scale}\nGIH 2", distance="100"), strict=True
    )
    # Unit-height marginal ray: u0=.01, u1=-.01, y2=.98, u2=-.0395.
    # The Gaussian magnification is u0/u2=-20/79, hence |OBH|=7.9.
    assert optic.fields.field_definition.__class__.__name__ == "ObjectHeightField"
    assert_allclose(optic.fields[-1].y, 7.9 * scale)


def test_model_glass_approximation_is_explicit(lens_file, set_test_backend):
    path = lens_file(surface="GLA MOD 1.6 50")
    with pytest.warns(UserWarning, match="Buchdahl model"):
        optic = load_oslo_file(path)
    material = optic.surfaces[1].material_post
    assert isinstance(material, AbbeMaterial)
    assert_allclose(material.n(0.5875618), 1.6, atol=1e-6)
    with pytest.raises(ValueError, match="dispersion is approximate"):
        load_oslo_file(path, strict=True)


def test_prefixed_historical_glass_warns(lens_file, monkeypatch):
    import optiland.fileio.oslo.reader.converter as converter

    def missing_glass(name, **kwargs):
        raise ValueError(name)

    monkeypatch.setattr(converter, "Material", missing_glass)
    path = lens_file(surface="GLA H_BAF13")
    with pytest.warns(UserWarning, match="historical Abbe dispersion"):
        optic = load_oslo_file(path)
    assert isinstance(optic.surfaces[1].material_post, AbbeMaterial)
    assert_allclose(optic.surfaces[1].material_post.n(0.5875618), 1.667, atol=1e-6)
    with pytest.raises(ValueError, match="material bindings: H_BAF13"):
        load_oslo_file(path, strict=True)


def test_perfect_lens_warns_and_scales_focal_length(lens_file, set_test_backend):
    path = lens_file(system="UNI 10", surface="PFL 15")
    with pytest.warns(UserWarning, match="paraxial thin lens"):
        optic = load_oslo_file(path)
    assert optic.surfaces[1].interaction_model.interaction_type == "thin_lens"
    assert_allclose(optic.surfaces[1].interaction_model.f, 150)
    with pytest.raises(ValueError, match="perfect imagery"):
        load_oslo_file(path, strict=True)


@pytest.mark.parametrize("pickup,message", [("PK LN 1 1", "invalid length range")])
def test_invalid_pickup_references(lens_file, pickup, message):
    with pytest.raises(ValueError, match=message):
        load_oslo_file(lens_file(second=pickup), strict=True)


@pytest.mark.parametrize(
    "source,pickup,message",
    [
        ("RFL", "PK GLA 1", "cannot pick up a reflector"),
        ("RCO", "PK TD 1", "global/return/bend"),
        ("TLA 10\nBEN", "PK TDM 1", "global/return/bend"),
    ],
)
def test_unsupported_pickup_sources(lens_file, source, pickup, message):
    with pytest.raises(ValueError, match=message):
        load_oslo_file(lens_file(surface=source, second=pickup), strict=True)


def test_curvature_pickup_replaces_profile_without_mutating_source():
    surfaces = {
        0: {},
        1: {"RD": 10, "CC": -1, "AD": 0.001},
        2: {
            "ASP": "ASR",
            "AS1": 1,
            "CVX": 2,
            "AE": 3,
            "pickups": [["CVM", "1", ".01"]],
        },
    }
    original = deepcopy(surfaces)
    resolved = resolve_pickups(surfaces)
    assert surfaces == original
    assert resolved[2]["RD"] == pytest.approx(-1 / 0.09)
    assert resolved[2]["CC"] == -1
    assert resolved[2]["AD"] == -0.001
    assert not {"ASP", "AS1", "CVX", "AE"}.intersection(resolved[2])


def test_model_pickup_rejects_unknown_type():
    with pytest.raises(ValueError, match="PK UNSUPPORTED is unsupported"):
        resolve_pickups({0: {}, 1: {"pickups": [["UNSUPPORTED", "0"]]}})


@pytest.mark.parametrize(
    "surface,image,message",
    [
        ("", "PY 1", "interior optical surface"),
        ("EC -1", "", "nonnegative edge height"),
        ("EC 21", "", "outside the surface's sag domain"),
        ("AIR\nPY 1", "", "could not be reached"),
    ],
)
def test_failed_solve_restores_saved_prescription(
    lens_file, set_test_backend, surface, image, message
):
    path = lens_file(surface=surface, image=image)
    model = OsloDataParser(path).parse()
    original = model.to_dict()
    with pytest.warns(UserWarning, match=message + ".*retained saved prescription"):
        optic = OsloToOpticConverter(model).convert()
    assert model.to_dict() == original
    assert_allclose(optic.surfaces[1].thickness, 2)
    assert_allclose(optic.surfaces[2].geometry.cs.z, 2)
    assert_allclose(optic.surfaces[2].geometry.radius, -20)
    with pytest.raises(ValueError, match=message):
        OsloToOpticConverter(model, strict=True).convert()


def test_failed_chief_slope_refinement_restores_curvature(
    lens_file, set_test_backend, monkeypatch
):
    import optiland.fileio.oslo.reader.solves as solves

    calls = []

    def fail_refinement(residual, **kwargs):
        calls.append(residual(0.2))  # Mimic a solver changing the optic before failure.
        return SimpleNamespace(converged=False)

    monkeypatch.setattr(solves, "root_scalar", fail_refinement)
    path = lens_file(system="ANG 5", second="PUC .03")
    with pytest.warns(UserWarning, match="refinement did not converge.*retained"):
        optic = load_oslo_file(path)
    assert len(calls) == 1
    assert_allclose(optic.surfaces[2].geometry.radius, -20)


@pytest.mark.parametrize("command,power", [("AS1 .001", 2), ("AS6 .001", 12)])
def test_general_even_asphere_export_preserves_sag(
    lens_file, tmp_path, set_test_backend, command, power
):
    optic = load_oslo_file(
        lens_file(surface=f"RD 0\nASP ASR 1\n{command}"), strict=power > 2
    )
    path = tmp_path / "general-asphere.len"
    save_oslo_file(optic, path)
    assert "ASP ASR" in path.read_text()
    restored = load_oslo_file(path, strict=power > 2)
    assert_allclose(
        restored.surfaces[1].geometry.sag(be.array([2.0]), be.array([0.0])),
        [0.001 * 2**power],
    )


@pytest.mark.parametrize(
    "field_type,y,message",
    [
        ("paraxial_image_height", 1, "field definition"),
        ("real_image_height", 1, "field definition"),
        ("angle", 90, "wide-angle field"),
        ("angle", -91, "wide-angle field"),
    ],
)
def test_unsupported_export_fields_preserve_destination(
    lens_file, tmp_path, field_type, y, message
):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.fields.set_type(field_type)
    optic.fields.fields.clear()
    optic.fields.add(y=y, x=0)
    target = tmp_path / "existing.len"
    target.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match=message):
        save_oslo_file(optic, target)
    assert target.read_text() == "saved design"


def test_formatter_preserves_escaped_notes(lens_file):
    model = OsloDataParser(lens_file()).parse()
    note = 'A "quoted" note; // with a \\ path'
    model.notes["SNO1"] = note
    path = lens_file()
    path.write_text(OsloDataFormatter(model).format(), encoding="utf-8")
    assert OsloDataParser(path, strict=True).parse().notes["SNO1"] == note


@pytest.mark.parametrize("value,expected", [(math.inf, 1e10), (-math.inf, -1e10)])
def test_formatter_signed_infinity_sentinels(value, expected):
    assert float(OsloDataFormatter(OsloDataModel())._fmt(value)) == expected


@pytest.mark.parametrize("name", ['lens "A"', '"', '"quoted"', 'path\\"'])
def test_quoted_name_and_note_preserve_trailing_quote(lens_file, name):
    model = OsloDataParser(lens_file()).parse()
    model.name = name
    model.notes["SNO1"] = name
    path = lens_file()
    path.write_text(OsloDataFormatter(model).format(), encoding="utf-8")
    restored = OsloDataParser(path, strict=True).parse()
    assert restored.name == name
    assert restored.notes["SNO1"] == name


def test_quoted_direct_glass_name(lens_file, set_test_backend):
    optic = load_oslo_file(
        lens_file(surface='GLA "test glass" 1.6 1.62 1.58'), strict=True
    )
    material = optic.surfaces[1].material_post
    assert material.name == "test glass"
    assert_allclose(material.n(0.48613), 1.62)


@pytest.mark.parametrize(
    "command",
    [
        "AP 1 ignored",
        "AP CHK 1 ignored",
        "BEN OFF",
        "RCO 0 1",
        "AIR ignored",
        "AST ignored",
        "ATD ignored",
        "ASP ASR 1 ignored",
        "ASP ASR -1",
        "GLA MOD G1",
    ],
)
def test_optical_commands_do_not_silently_discard_arguments(lens_file, command):
    with pytest.raises(ValueError):
        load_oslo_file(lens_file(surface=command), strict=True)


@pytest.mark.parametrize("next_block", ['LEN NEW "second" 1 1'])
def test_additional_configuration_cannot_replace_first_field_table(
    lens_file, next_block
):
    first = "RST NEW\nF 1 .5 0 0 0 0 -1 1 -1 1 1\nEND\n"
    second = "RST NEW\nF 1 1 0 0 0 0 -1 1 -1 1 1\nEND\n"
    path = lens_file(
        system="OBH 4", distance="100", footer=f"{first}{next_block}\nEND\n{second}"
    )
    with pytest.warns(UserWarning, match="additional configurations"):
        optic = load_oslo_file(path)
    assert_allclose(optic.fields.y_fields, [2])


def test_failed_pickup_rebuild_restores_converter_data(lens_file, set_test_backend):
    path = lens_file(system="NAP .1", surface="PU 0", second="PK CV 1")
    converter = OsloToOpticConverter(OsloDataParser(path).parse())
    original = converter.data.to_dict()
    with pytest.warns(UserWarning, match="afocal.*retained saved prescription"):
        optic = converter.convert()
    # The saved pickup produces equal radii; solving both to infinity makes
    # NAP undefined. Keep the original prescription available for another import.
    for index in (1, 2):
        assert_allclose(optic.surfaces[index].geometry.radius, 20)
    assert converter.data.to_dict() == original


def test_coupled_solve_cannot_silently_change_image_na(lens_file, set_test_backend):
    path = lens_file(system="NAP .1", surface="PU -.05")
    with pytest.warns(UserWarning, match="retained saved prescription"):
        optic = load_oslo_file(path)
    assert_allclose(abs(optic.paraxial.marginal_ray()[1][-2]), 0.1)
    assert_allclose(optic.surfaces[1].geometry.radius, 20)
    with pytest.raises(ValueError, match="target"):
        load_oslo_file(path, strict=True)


def test_tilt_and_bend_requires_a_reflector(lens_file):
    with pytest.raises(ValueError, match="BEN.*reflect"):
        load_oslo_file(lens_file(surface="TLA 20\nBEN"), strict=True)


@pytest.mark.parametrize("coordinate", ["GC 1", "RCO", "BEN"])
def test_tilt_pickup_rejects_target_reference_flags(lens_file, coordinate):
    with pytest.raises(ValueError, match="global/return/bend"):
        load_oslo_file(lens_file(second=f"{coordinate}\nPK TD 1"), strict=True)


def test_telecentric_export_preserves_chief_ray(lens_file, tmp_path, set_test_backend):
    optic = load_oslo_file(
        lens_file(system="OBH 2\nTELE ON", distance="100"), strict=True
    )
    expected = optic.paraxial.chief_ray()
    target = tmp_path / "telecentric.len"
    save_oslo_file(optic, target)
    restored = load_oslo_file(target, strict=True)
    assert restored.obj_space_telecentric
    for actual, original in zip(restored.paraxial.chief_ray(), expected, strict=True):
        assert_allclose(actual, original)


def test_custom_export_field_definition_preserves_destination(lens_file, tmp_path):
    optic = load_oslo_file(lens_file(), strict=True)

    class CustomField(type(optic.fields.field_definition)):
        pass

    optic.fields.field_definition = CustomField()
    target = tmp_path / "custom.len"
    target.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="field definition"):
        save_oslo_file(optic, target)
    assert target.read_text() == "saved design"


def test_encoder_reuse_does_not_retain_old_prescription(lens_file):
    optic = load_oslo_file(lens_file(), strict=True)
    encoder = OpticToOsloEncoder(optic)
    original = encoder.encode()
    optic.set_aperture("objectNA", 0.1)
    optic.updater.set_thickness(100, 0)
    optic.surfaces.remove(2)
    updated = encoder.encode()
    assert updated.aperture == {"NAO": 0.1}
    assert set(updated.surfaces) == {0, 1, 2}
    assert original.aperture == {"EPD": 4}
    assert len(original.surfaces) == 4


@pytest.mark.parametrize("name", ["first\nsecond", "first\rsecond", "trailing\n"])
def test_multiline_export_name_preserves_destination(lens_file, tmp_path, name):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.name = name
    target = tmp_path / "multiline.len"
    target.write_text("saved design", encoding="utf-8")
    with pytest.raises(ValueError, match="single line"):
        save_oslo_file(optic, target)
    assert target.read_text() == "saved design"


def test_large_checked_aperture_still_clips_after_unit_conversion(
    lens_file, set_test_backend
):
    optic = load_oslo_file(
        lens_file(system="UNI .000001", surface="AP CHK 2000000"), strict=True
    )
    aperture = optic.surfaces[1].aperture
    assert aperture is not None
    assert_allclose(aperture.r_max, 2)
    x, y = be.array([1.0, 3.0]), be.zeros(2)
    rays = RealRays(x, y, be.zeros(2), be.zeros(2), be.zeros(2), be.ones(2), 1, 0.55)
    aperture.clip(rays)
    assert_allclose(rays.i, [1, 0])


def test_later_solve_preserves_an_earlier_edge_contact(lens_file, set_test_backend):
    path = lens_file(surface="EC 2", second="PU 0")
    with pytest.warns(
        UserWarning, match="EC at surface 1.*retained saved prescription"
    ):
        optic = load_oslo_file(path)
    x, y = be.zeros(1), be.array([2.0])
    edge_thickness = (
        optic.surfaces[1].thickness
        + optic.surfaces[2].geometry.sag(x, y)
        - optic.surfaces[1].geometry.sag(x, y)
    )
    assert_allclose(edge_thickness, [0], atol=1e-12)
    assert_allclose(optic.surfaces[2].geometry.radius, -20)
    with pytest.raises(ValueError, match="EC at surface 1"):
        load_oslo_file(path, strict=True)


def test_deferred_asphere_diagnostic_points_to_coefficient(lens_file):
    path = lens_file(surface="AS1 .01")
    source_line = path.read_text().splitlines().index("AS1 .01") + 1
    with pytest.warns(UserWarning, match="general coefficients require"):
        model = OsloDataParser(path).parse()
    diagnostic = model.diagnostics[0]
    assert diagnostic.line == source_line
    assert diagnostic.surface == 1
    with pytest.raises(ValueError) as error:
        OsloDataParser(path, strict=True).parse()
    assert f"{path}:{source_line}:" in str(error.value)


@pytest.mark.parametrize("aperture, radius", [("EBR 2", 2), ("", 1)])
def test_finite_entrance_beam_radius_is_measured_at_surface_one(
    lens_file, set_test_backend, aperture, radius
):
    optic = load_oslo_file(
        lens_file(distance="100", second="AST", aperture=aperture), strict=True
    )
    # OSLO EBR=2 specifies the beam at the first surface, even when its
    # entrance pupil is displaced by refraction ahead of an internal stop.
    assert_allclose(optic.paraxial.marginal_ray()[0][1], [radius])


def test_diverging_fno_uses_a_positive_entrance_pupil(lens_file, set_test_backend):
    optic = load_oslo_file(
        lens_file(aperture="FNO 5", surface="RD -20", second="RD 20"), strict=True
    )
    assert float(optic.paraxial.EPD()) > 0
    assert_allclose(abs(optic.paraxial.marginal_ray()[1][-2]), [0.1])


def test_custom_material_cannot_export_as_its_catalog_like_name(lens_file, tmp_path):
    from optiland.materials import IdealMaterial

    # A custom subtype is unsupported even with a familiar base and glass name.
    class CustomMaterial(IdealMaterial):
        name = "BK7"

    optic = load_oslo_file(lens_file(), strict=True)
    optic.surfaces[1].material_post = CustomMaterial(1.8)
    path = tmp_path / "custom-material.len"
    path.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="material"):
        save_oslo_file(optic, path)
    assert path.read_text() == "saved design"


@pytest.mark.parametrize("index", [1.0, 1.5])
def test_absorbing_ideal_material_export_preserves_destination(
    lens_file, tmp_path, index
):
    from optiland.materials import IdealMaterial

    optic = load_oslo_file(lens_file(), strict=True)
    optic.surfaces[1].material_post = IdealMaterial(index, 0.01)
    path = tmp_path / "absorbing.len"
    path.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="absorption"):
        save_oslo_file(optic, path)
    assert path.read_text() == "saved design"


def test_custom_propagation_export_preserves_destination(lens_file, tmp_path):
    from optiland.propagation.homogeneous import HomogeneousPropagation

    class CustomPropagation(HomogeneousPropagation):
        pass

    optic = load_oslo_file(lens_file(), strict=True)
    material = optic.surfaces[1].material_post
    material.propagation_model = CustomPropagation(material)
    path = tmp_path / "custom-propagation.len"
    path.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="propagation"):
        save_oslo_file(optic, path)
    assert path.read_text() == "saved design"


@pytest.mark.parametrize("property_name", ["coating", "bsdf"])
def test_surface_loss_and_scatter_export_preserves_destination(
    lens_file, tmp_path, property_name
):
    from optiland.coatings import SimpleCoating
    from optiland.scatter import LambertianBSDF

    optic = load_oslo_file(lens_file(), strict=True)
    effect = SimpleCoating(0.5) if property_name == "coating" else LambertianBSDF()
    setattr(optic.surfaces[1].interaction_model, property_name, effect)
    path = tmp_path / "surface-effect.len"
    path.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="coatings or scattering"):
        save_oslo_file(optic, path)
    assert path.read_text() == "saved design"


def test_telecentric_import_survives_native_serialization(lens_file, set_test_backend):
    from optiland.optic import Optic

    optic = load_oslo_file(
        lens_file(system="OBH 2\nTELE ON", aperture="NAO .1", distance="100"),
        strict=True,
    )
    assert Optic.from_dict(optic.to_dict()).obj_space_telecentric


def test_telecentric_entrance_beam_can_launch_real_rays(lens_file, set_test_backend):
    from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer

    optic = load_oslo_file(
        lens_file(system="OBH 2\nTELE ON", distance="100"), strict=True
    )
    _, y, _, direction_x, direction_y, direction_z = ParaxialRayAimer(optic).aim_rays(
        (0, 1), optic.primary_wavelength, (be.zeros(2), be.array([0.0, 1.0]))
    )
    # The chief ray is parallel to z; the marginal ray advances 2 mm in y
    # across the 100 mm object distance. Both originate at the field point.
    assert_allclose(y, [2, 2])
    assert_allclose(direction_y / direction_z, [0, 0.02])
    assert_allclose(direction_x, [0, 0])


def test_failed_solve_retains_telecentricity(lens_file, set_test_backend):
    with pytest.warns(UserWarning, match="retained saved prescription"):
        optic = load_oslo_file(
            lens_file(
                system="OBH 2\nTELE ON",
                aperture="NAO .1",
                distance="100",
                surface="EC 40",
            )
        )
    assert optic.obj_space_telecentric


@pytest.mark.parametrize(
    "distance,system",
    [
        ("1e20", "TELE ON"),
        ("100", "OBH 2\nGLA 1.5\nTELE ON"),
        ("100", "ANG 2\nTELE ON"),
    ],
)
def test_unsupported_telecentric_launch_is_diagnosed(lens_file, distance, system):
    path = lens_file(distance=distance, system=system)
    with pytest.warns(UserWarning, match="TELE.*launch"):
        load_oslo_file(path)
    with pytest.raises(ValueError, match="TELE.*launch"):
        load_oslo_file(path, strict=True)


@pytest.mark.parametrize("command", ["PYC 1", "PUC .05"])
def test_telecentric_chief_solves_do_not_use_a_nontelecentric_ray(lens_file, command):
    path = lens_file(
        system="OBH 2\nTELE ON", distance="100", aperture="NAO .1", surface=command
    )
    with pytest.raises(ValueError, match="telecentric chief-ray"):
        load_oslo_file(path, strict=True)


@pytest.mark.parametrize("aperture", ["NAO 1", "NAO 1.1", "PUK 0"])
def test_telecentric_launch_rejects_invalid_cone(lens_file, aperture):
    with pytest.raises(ValueError, match="NAO|TELE.*NA"):
        load_oslo_file(
            lens_file(system="OBH 2\nTELE ON", distance="100", aperture=aperture),
            strict=True,
        )


@pytest.mark.parametrize(
    "distance,infinite", [(99999999, False), (1e8, True), (-1e8, True)]
)
@pytest.mark.parametrize("transformed", [False, True])
def test_oslo_object_infinity_boundary(
    lens_file, set_test_backend, distance, infinite, transformed
):
    # The OSLO cutoff is in lens units, before conversion to millimeters.
    optic = load_oslo_file(
        lens_file(
            system="UNI .001\nOBH 2",
            distance=str(distance),
            surface="DCY 0" if transformed else "",
        ),
        strict=True,
    )
    assert optic.object_surface.is_infinite == infinite
    if infinite:
        assert float(optic.object_surface.geometry.cs.z) == -math.copysign(
            math.inf, distance
        )
        assert isinstance(optic.fields.field_definition, AngleField)
    else:
        assert_allclose(optic.object_surface.geometry.cs.z, -99999.999)
        assert isinstance(optic.fields.field_definition, ObjectHeightField)


def test_export_rejects_finite_object_at_oslo_infinity_boundary(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(distance="100"), strict=True)
    optic.updater.set_thickness(1e8, 0)
    path = tmp_path / "finite-object.len"
    path.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="finite object distance"):
        save_oslo_file(optic, path)
    assert path.read_text() == "saved design"


def test_export_preserves_finite_object_just_below_infinity_boundary(
    lens_file, tmp_path, set_test_backend
):
    distance = math.nextafter(1e8, 0)
    optic = load_oslo_file(lens_file(distance=str(distance)), strict=True)
    path = tmp_path / "large-finite-object.len"
    save_oslo_file(optic, path)
    restored = load_oslo_file(path, strict=True)
    assert not restored.object_surface.is_infinite
    assert float(restored.object_surface.geometry.cs.z) == -distance


@pytest.mark.parametrize("slot", [1, 2])
def test_indexed_wavelength_edit_preserves_other_defaults(lens_file, slot):
    model = OsloDataParser(lens_file(system=f"WV{slot} .55\nWW2 3")).parse()
    expected = [0.58756, 0.48613, 0.65627]
    expected[slot - 1] = 0.55
    assert model.wavelengths["values"] == expected
    assert model.wavelengths["weights"] == [1, 3, 1]
    assert model.surfaces[1]["glass_wavelengths"] == expected


def test_indexed_wavelength_cannot_restore_removed_slots_implicitly(lens_file):
    with pytest.raises(ValueError, match="undefined wavelength slots"):
        OsloDataParser(lens_file(system="WV .55\nWV3 .7")).parse()


@pytest.mark.parametrize(
    "distance,medium,na",
    [("1e20", "", 0.1), ("100", "", 1), ("100", "", 1.1), ("100", "GLA 1.5", 1.6)],
)
def test_object_na_rejects_invalid_launch(
    lens_file, set_test_backend, distance, medium, na
):
    with pytest.raises(ValueError, match="NAO"):
        load_oslo_file(
            lens_file(distance=distance, system=medium, aperture=f"NAO {na}")
        )


def test_object_na_respects_object_medium_index(lens_file, set_test_backend):
    optic = load_oslo_file(
        lens_file(distance="100", system="GLA 1.5", aperture="NAO 1.2"), strict=True
    )
    # NA/n = 0.8; tan(arcsin(0.8)) = 4/3 in the object medium.
    assert_allclose(optic.paraxial.marginal_ray()[1][0], 4 / 3)


@pytest.mark.parametrize("distance,na", [("1e20", 0.1), ("100", 1.1)])
def test_invalid_object_na_export_preserves_destination(
    lens_file, tmp_path, distance, na
):
    optic = load_oslo_file(lens_file(distance=distance), strict=True)
    optic.set_aperture("objectNA", na)
    path = tmp_path / "invalid-na.len"
    path.write_text("saved design", encoding="utf-8")
    with pytest.raises(ValueError, match="NAO"):
        save_oslo_file(optic, path)
    assert path.read_text() == "saved design"


@pytest.mark.parametrize(
    "kind,coefficient",
    [("ASR", 1), ("ARA", 1), ("ARA", 2)] + [("ASX", i) for i in range(6)],
)
def test_low_order_asphere_paraxial_limit_is_diagnosed(lens_file, kind, coefficient):
    path = lens_file(surface=f"ASP {kind} 6\nAS{coefficient} .01")
    with pytest.warns(UserWarning, match="asphere.*paraxial"):
        load_oslo_file(path)
    with pytest.raises(ValueError, match="asphere.*paraxial"):
        load_oslo_file(path, strict=True)


def test_quadratic_sag_preserves_real_surface_power(lens_file, set_test_backend):
    path = lens_file(surface="RD 0\nASP ASR 1\nAS1 .01", second="RD 0")
    with pytest.warns(UserWarning, match="asphere.*paraxial"):
        optic = load_oslo_file(path)
    x, y = be.zeros(1), be.array([1e-5])
    surface = optic.surfaces[1]
    rays = RealRays(x, y, surface.geometry.sag(x, y), x, x, be.ones(1), 1, 0.55)
    surface.interaction_model.interact_real_rays(rays)
    # z=.01*y^2 gives curvature .02; Snell's small-height limit is
    # u/y = -(1.5-1)*.02/1.5 at the air-to-glass boundary.
    assert_allclose(rays.M / rays.N / y, [-1 / 150], rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("kind,coefficient", [("ASR", 2), ("ARA", 3), ("ASX", 6)])
def test_higher_order_aspheres_remain_strictly_supported(lens_file, kind, coefficient):
    load_oslo_file(
        lens_file(surface=f"ASP {kind} 6\nAS{coefficient} .001"), strict=True
    )


@pytest.mark.parametrize("index", [1.0000004, 1.573123456789])
def test_ideal_material_export_preserves_optical_path(
    lens_file, tmp_path, set_test_backend, index
):
    from optiland.materials import IdealMaterial

    optic = load_oslo_file(lens_file(), strict=True)
    optic.surfaces[1].material_post = IdealMaterial(index)
    optic.updater.set_thickness(1000, 1)
    path = tmp_path / "index-precision.len"
    save_oslo_file(optic, path)
    restored = load_oslo_file(path, strict=True)
    exported_index = float(restored.surfaces[1].material_post.n(0.55).item())
    # One meter through the medium must retain its optical-path excess over
    # vacuum; near-unity indices are still refracting media, not exactly air.
    assert (exported_index - 1) * 1000 == pytest.approx((index - 1) * 1000, abs=1e-10)


@pytest.mark.parametrize("command", ["WW 0 1 1", "WW1 0"])
def test_primary_wavelength_weight_must_be_positive(lens_file, command):
    with pytest.raises(ValueError, match="primary wavelength weight"):
        OsloDataParser(lens_file(system=command)).parse()


@pytest.mark.parametrize("primary_index", [0, 1])
def test_zero_primary_weight_export_preserves_destination(
    lens_file, tmp_path, primary_index
):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.wavelengths.primary_index = primary_index
    optic.wavelengths[primary_index].weight = 0
    path = tmp_path / "zero-primary-weight.len"
    path.write_text("saved design", encoding="utf-8")
    with pytest.raises(ValueError, match="primary wavelength weight"):
        save_oslo_file(optic, path)
    assert path.read_text() == "saved design"
