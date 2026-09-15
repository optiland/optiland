"""Independent OSLO command and physical-behavior regression tests."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.fileio.oslo.reader.parser import OsloDataParser
from tests.utils import assert_allclose


def write_lens(tmp_path, text):
    path = tmp_path / "commands.len"
    path.write_text(text, encoding="utf-8")
    return path


def test_statements_quotes_comments_and_empty_image(tmp_path):
    path = write_lens(
        tmp_path,
        '\ufefflen new "a; b // c" 1 2\n'
        "ebr 2; ang 1 // inline\n"
        "th 1e20; nxt // surface 1\n"
        "rd 30; th 5; nxt; end 2\n",
    )
    data = OsloDataParser(path).parse()
    assert data.name == "a; b // c"
    assert data.aperture["EPD"] == 4
    assert list(data.surfaces) == [0, 1, 2]
    assert data.surfaces[1]["RD"] == 30


def test_gto_updates_selected_surface_and_parser_reuse(tmp_path):
    path = write_lens(
        tmp_path,
        'LEN NEW "navigation" 1 2\nTH 1e10\nNXT\nRD 10\nNXT\nAIR\nGTO 1\nTH 3\nEND 2\n',
    )
    parser = OsloDataParser(path)
    for _ in range(2):
        data = parser.parse()
        assert data.surfaces[1]["TH"] == 3
        assert data.surfaces[1]["RD"] == 10
        assert len(data.surfaces) == 3


def test_unknown_optical_command_is_reported_and_strict_rejects(tmp_path):
    path = write_lens(
        tmp_path, 'LEN NEW "unsupported" 1 1\nTH 1e10\nNXT\nMAGIC 8\nEND 1\n'
    )
    with pytest.warns(UserWarning, match="MAGIC"):
        data = OsloDataParser(path).parse()
    diagnostic = data.diagnostics[0]
    assert diagnostic.command == "MAGIC"
    assert diagnostic.line == 4
    assert diagnostic.surface == 1
    with pytest.raises(ValueError, match="MAGIC"):
        OsloDataParser(path, strict=True).parse()


@pytest.mark.parametrize("command", ["RD", "TH nope", "AP nan", 'DES "unterminated'])
def test_malformed_command_has_source_context(tmp_path, command):
    path = write_lens(tmp_path, f'LEN NEW "bad" 1 1\n{command}\nNXT\nEND 1\n')
    with pytest.raises(ValueError, match=r"commands.len:2"):
        OsloDataParser(path).parse()


def test_legacy_encoding_and_trailing_analysis_are_not_executed(tmp_path):
    path = tmp_path / "legacy.len"
    path.write_bytes(
        'LEN NEW "Caf\xe9" 1 1\nTH 1e10\nNXT\nEND 1\nDLNR 0 11\nRD 999\n'.encode(
            "cp1252"
        )
    )
    data = OsloDataParser(path).parse()
    assert data.name == "Caf\xe9"
    assert "RD" not in data.surfaces[1]


def test_public_loader_exposes_strict_mode(tmp_path):
    path = write_lens(tmp_path, 'LEN NEW "strict" 1 1\nNXT\nUNKNOWN 1\nEND 1\n')
    with pytest.raises(ValueError, match="UNKNOWN"):
        load_oslo_file(path, strict=True)


def simple_lens(tmp_path, system="", surface="", footer=""):
    return write_lens(
        tmp_path,
        'LEN NEW "commands" 50 3\nEBR 2\nANG 0\n'
        f"{system}\nTH 1e20\nNXT\nGLA 1.5 1.5 1.5\n"
        f"RD 20\nTH 2\nAP 3\n{surface}\nNXT\n"
        f"AIR\nRD -20\nTH 20\nNXT\nAIR\n{footer}\nEND 3\n",
    )


def test_wavelength_slots_replace_preserve_order_weights_and_defaults(tmp_path):
    path = simple_lens(
        tmp_path, footer="WV .6 .5 .7\nWW 1 2 3\nWV2 .45\nWV4 .8\nWW2 4\nWW4 5"
    )
    data = OsloDataParser(path).parse()
    assert data.wavelengths["values"] == [0.6, 0.45, 0.7, 0.8]
    assert data.wavelengths["weights"] == [1, 4, 3, 5]
    path = simple_lens(tmp_path, system="WV .6 .5 .7", footer="WV .55\nWW 2")
    assert OsloDataParser(path).parse().wavelengths["values"] == [0.55]
    assert OsloDataParser(simple_lens(tmp_path)).parse().wavelengths["values"] == [
        0.58756,
        0.48613,
        0.65627,
    ]


@pytest.mark.parametrize("command", ["WV 0", "WV -1", "WW -1", "UNI 0", "UNI -2"])
def test_invalid_system_values_rejected(tmp_path, command):
    with pytest.raises(ValueError):
        OsloDataParser(simple_lens(tmp_path, footer=command)).parse()


def test_units_scale_prescription_aperture_and_object_height(
    tmp_path, set_test_backend
):
    path = simple_lens(tmp_path, system="UNI 25.4\nOBH 1")
    # Finite object: the field is a length in the same units as the lens.
    path.write_text(path.read_text().replace("TH 1e20", "TH 10"))
    optic = load_oslo_file(path)
    assert_allclose(optic.surfaces[1].geometry.radius, 20 * 25.4)
    assert_allclose(optic.surfaces[1].thickness, 2 * 25.4)
    assert optic.surfaces[1].aperture is None
    assert_allclose(optic.aperture.value, 4 * 25.4)
    assert_allclose(optic.fields.y_fields[-1], 25.4)
    assert_allclose(optic.wavelengths.primary_wavelength.value, 0.58756)


def test_infinite_object_field_uses_actual_oslo_distance(tmp_path, set_test_backend):
    path = simple_lens(tmp_path, system="OBH -1e20")
    optic = load_oslo_file(path)
    assert_allclose(optic.fields.y_fields[-1], 45)
    assert be.isinf(optic.surfaces[0].thickness)


def test_system_aperture_last_command_wins(tmp_path, set_test_backend):
    optic = load_oslo_file(simple_lens(tmp_path, system="FNO 8\nEBR 3"))
    assert optic.aperture.ap_type == "EPD"
    assert_allclose(optic.aperture.value, 6)


def test_image_na_and_telecentric(tmp_path, set_test_backend):
    optic = load_oslo_file(simple_lens(tmp_path, system="NAP .1\nTELE ON"))
    # Thick symmetric lens power = (n-1)(1/R1-1/R2+(n-1)t/(n R1 R2)).
    power = 0.5 * (0.1 - 0.5 * 2 / (1.5 * 400))
    assert optic.aperture.ap_type == "EPD"
    assert_allclose(optic.aperture.value, 0.2 / power)
    assert optic.obj_space_telecentric is True


@pytest.mark.parametrize(
    "command,radius",
    [("CV .05", 20), ("CVF -.1", -10), ("RDF 0", be.inf), ("RD 0", be.inf)],
)
def test_curvature_and_fixed_aliases(tmp_path, set_test_backend, command, radius):
    optic = load_oslo_file(simple_lens(tmp_path, surface=command), strict=True)
    assert_allclose(optic.surfaces[1].geometry.radius, radius)


def test_oslo_standard_asphere_power_and_scaled_sag(tmp_path, set_test_backend):
    optic = load_oslo_file(
        simple_lens(tmp_path, system="UNI 10", surface="RD 0\nAD .001\nAE .00001"),
        strict=True,
    )
    sag = optic.surfaces[1].geometry.sag(be.array([20.0]), be.array([0.0]))
    assert_allclose(sag, [10 * (0.001 * 2**4 + 0.00001 * 2**6)])
    out = tmp_path / "asphere.len"
    save_oslo_file(optic, out)
    restored = load_oslo_file(out, strict=True)
    assert_allclose(restored.surfaces[1].geometry.sag(be.array([20.0]), 0), sag)


@pytest.mark.parametrize(
    "command,expected",
    [
        ("ASP ASR 6\nAS1 .01\nAS6 1e-9", 0.01 * 4 + 1e-9 * 2**12),
        ("ASP ARA 3\nAS1 .01\nAS3 .001", 0.01 * 2 + 0.001 * 2**3),
        ("ASP ASX 2\nAS0 .02\nAS1 .01\nAS4 .003", 0.02 + 0.01 * 2),
    ],
)
def test_general_asphere_equations(tmp_path, set_test_backend, command, expected):
    with pytest.warns(UserWarning, match="asphere.*paraxial"):
        optic = load_oslo_file(simple_lens(tmp_path, surface="RD 0\n" + command))
    assert_allclose(
        optic.surfaces[1].geometry.sag(be.array([2.0]), be.array([0.0])), [expected]
    )


@pytest.mark.parametrize("cvx", [0, 0.025])
def test_toric_profile_matches_independent_equation(tmp_path, set_test_backend, cvx):
    optic = load_oslo_file(
        simple_lens(tmp_path, surface=f"CVX {cvx}\nAD .0001"), strict=True
    )
    x, y = be.array([1.0]), be.array([2.0])
    zy = 20 - (20**2 - 4) ** 0.5 + 0.0001 * 2**4
    expected = zy if cvx == 0 else 40 - ((40 - zy) ** 2 - 1) ** 0.5
    assert_allclose(optic.surfaces[1].geometry.sag(x, y), [expected])


def test_hatched_reflector_and_fixed_air(tmp_path, set_test_backend):
    optic = load_oslo_file(simple_lens(tmp_path, surface="RFH"), strict=True)
    assert optic.surfaces[1].interaction_model.is_reflective
    optic = load_oslo_file(simple_lens(tmp_path, surface="AIF"), strict=True)
    assert_allclose(optic.surfaces[1].material_post.n(0.55), 1)


@pytest.mark.parametrize(
    "definition",
    ["GLA CUSTOM 1.7 1.72 1.68", "GLA 1.7 1.72 1.68", "GLA MOD G1 1.7 1.72 1.68"],
)
def test_embedded_indices_use_definition_wavelengths(
    tmp_path, set_test_backend, definition
):
    path = simple_lens(
        tmp_path, surface="WV .6 .4 .8\n" + definition, footer="WV .6\nWW 1"
    )
    optic = load_oslo_file(path, strict=True)
    material = optic.surfaces[1].material_post
    assert_allclose(material.n(be.array([0.4, 0.6, 0.8])), [1.72, 1.7, 1.68])
    out = tmp_path / "material.len"
    save_oslo_file(optic, out)
    assert_allclose(load_oslo_file(out).surfaces[1].material_post.n(0.6), 1.7)


def test_single_direct_index_and_unknown_glass_strict(tmp_path, set_test_backend):
    optic = load_oslo_file(simple_lens(tmp_path, surface="GLA 1.65"), strict=True)
    assert_allclose(optic.surfaces[1].material_post.n(0.55), 1.65)
    with pytest.raises(ValueError, match="MISSING_GLASS"):
        load_oslo_file(simple_lens(tmp_path, surface="GLA MISSING_GLASS"), strict=True)


@pytest.mark.parametrize("order", [1, -1])
def test_oslo_tilt_sign_order_and_decenter(tmp_path, set_test_backend, order):
    import numpy as np

    a, b, c = np.deg2rad([-10, -20, 30])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    expected = rx @ ry @ rz if order == 1 else rz @ ry @ rx
    commands = f"DT {order}\nDCX 1\nDCY 2\nDCZ 3\nTLA 10\nTLB 20\nTLC 30"
    optic = load_oslo_file(simple_lens(tmp_path, surface=commands), strict=True)
    position, rotation = optic.surfaces[1].geometry.cs.get_effective_transform()
    assert_allclose(rotation, expected)
    assert_allclose(position, [1, 2, 3] if order == 1 else expected @ [1, 2, 3])
    assert be.isinf(optic.surfaces[0].geometry.cs.z)


def test_coordinate_return_global_reference_and_pivot(tmp_path, set_test_backend):
    commands = "DCY 4\nTLA 30\nRCO"
    optic = load_oslo_file(simple_lens(tmp_path, surface=commands), strict=True)
    assert_allclose(
        optic.surfaces[2].geometry.cs.get_effective_transform()[0], [0, 0, 2]
    )
    path = simple_lens(tmp_path, surface="DCY 4")
    path.write_text(path.read_text().replace("RD -20", "RD -20\nGC -1\nDCY 2\nDCZ 5"))
    optic = load_oslo_file(path, strict=True)
    assert_allclose(
        optic.surfaces[2].geometry.cs.get_effective_transform()[0], [0, 6, 5]
    )
    optic = load_oslo_file(
        simple_lens(tmp_path, surface="TLA 90\nTOZ 1\nRCO"), strict=True
    )
    assert_allclose(
        optic.surfaces[1].geometry.cs.get_effective_transform()[0], [0, -1, 1]
    )


def test_single_axis_mirror_bend_has_expected_physical_path(tmp_path, set_test_backend):
    path = write_lens(
        tmp_path,
        'LEN NEW "fold" 1 2\nEBR 1\nANG 0\nTH 1e20\n'
        "NXT\nRFH\nTLA 45\nBEN\nTH -10\nNXT\nAIR\nEND 2\n",
    )
    optic = load_oslo_file(path, strict=True)
    assert_allclose(
        optic.surfaces[2].geometry.cs.get_effective_transform()[0],
        [0, -10, 0],
        atol=1e-12,
    )
    from optiland.rays import RealRays

    rays = RealRays(
        be.array([0.0]),
        be.array([0.0]),
        be.array([-1.0]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        1,
        0.55,
    )
    optic.surfaces.trace(rays, skip=1)
    assert_allclose(rays.y, [-10], atol=1e-12)
    assert_allclose(rays.i, [1])


def test_multiple_pickups_relative_indices_curvature_and_length(
    tmp_path, set_test_backend
):
    path = write_lens(
        tmp_path,
        'LEN NEW "pickups" 1 4\nEBR 1\nANG 0\nTH 1e20\n'
        "NXT\nGLA 1.5\nRD 10\nCC -1\nAD .001\nTH 2\nAP 3\n"
        "NXT\nPK CVM -1 .01\nPK TH -1 1\nPK AP -1\nPK GLA -1\n"
        "NXT\nAIR\nPK CV -1 0\nPK LNM -2 0 10\nNXT\nAIR\nEND 4\n",
    )
    optic = load_oslo_file(path, strict=True)
    assert_allclose(optic.surfaces[2].geometry.radius, 1 / (-0.1 + 0.01))
    assert_allclose(optic.surfaces[2].geometry.k, -1)
    assert_allclose(optic.surfaces[2].geometry.coefficients[1], -0.001)
    assert_allclose(optic.surfaces[2].thickness, 3)
    assert_allclose(optic.surfaces[3].thickness, 5)
    assert optic.surfaces[2].aperture is None
    assert_allclose(optic.surfaces[2].material_post.n(0.55), 1.5)


@pytest.mark.parametrize("pickup", ["PK TH 2", "PK TH -5", "PK UNKNOWN -1"])
def test_invalid_or_unsupported_pickups_rejected(tmp_path, pickup):
    path = simple_lens(tmp_path, surface=pickup)
    with pytest.raises(ValueError, match="PK"):
        load_oslo_file(path, strict=True)


@pytest.mark.parametrize(
    "command,target,component,index",
    [
        ("PY 1", 1, "marginal_height", 2),
        ("PYC .1", 0.1, "chief_height", 2),
        ("PU -.05", -0.05, "marginal_slope", 1),
    ],
)
def test_paraxial_solves_use_requested_surface_and_value(
    tmp_path, set_test_backend, command, target, component, index
):
    optic = load_oslo_file(
        simple_lens(tmp_path, system="ANG 5", surface=command), strict=True
    )
    heights, slopes = (
        optic.paraxial.chief_ray()
        if component.startswith("chief")
        else optic.paraxial.marginal_ray()
    )
    assert_allclose(
        (heights if component.endswith("height") else slopes)[index], target, atol=1e-9
    )
    if command.startswith("PY"):
        # Moving an interior surface must retain the subsequent nominal spacing.
        assert_allclose(
            optic.surfaces[3].geometry.cs.z - optic.surfaces[2].geometry.cs.z, 20
        )


def test_edge_contact_solve_and_unsatisfiable_solve(tmp_path, set_test_backend):
    optic = load_oslo_file(simple_lens(tmp_path, surface="EC 2"), strict=True)
    assert_allclose(optic.surfaces[2].geometry.cs.z, 2 * (20 - (400 - 4) ** 0.5))
    path = simple_lens(tmp_path, surface="AIF\nPY 1")
    with pytest.raises(ValueError, match="PY"):
        load_oslo_file(path, strict=True)


def test_chief_angle_solve(tmp_path, set_test_backend):
    path = simple_lens(tmp_path, system="ANG 5")
    path.write_text(path.read_text().replace("RD -20", "RD -20\nPUC .03"))
    optic = load_oslo_file(path, strict=True)
    assert_allclose(optic.paraxial.chief_ray()[1][2], 0.03, atol=1e-9)


def test_explicit_field_table_signed_xy_weights_and_vignetting(
    tmp_path, set_test_backend
):
    path = simple_lens(tmp_path, system="ANG 10")
    path.write_text(
        path.read_text()
        + "RST NEW\nF 1 -.5 .25 0 0 0 -.8 .8 -.9 .9 2\nF 2 0 0 0 0 0 -1 1 -1 1 0\nEND\n"
    )
    optic = load_oslo_file(path, strict=True)
    import math

    assert len(optic.fields) == 2
    assert_allclose(
        optic.fields[0].y, math.degrees(math.atan(-0.5 * math.tan(math.radians(10))))
    )
    assert_allclose(
        optic.fields[0].x, math.degrees(math.atan(0.25 * math.tan(math.radians(10))))
    )
    assert_allclose([optic.fields[0].vy, optic.fields[0].vx], [0.2, 0.1])
    assert [f.weight for f in optic.fields] == [2, 0]


@pytest.mark.parametrize(
    "footer", ["CFG NEW\nTEM 2 30\nEND\n", "RST NEW\nF 1 0 0 1 0 0 -1 1 -1 1 1\nEND\n"]
)
def test_unsupported_configuration_and_field_aiming_are_diagnosed(tmp_path, footer):
    path = simple_lens(tmp_path)
    path.write_text(path.read_text() + footer)
    with pytest.raises(ValueError, match="configuration|field"):
        load_oslo_file(path, strict=True)


@pytest.mark.parametrize("command,target", [("FNO 5", 0.1), ("PUK .1", 0.1)])
def test_image_aperture_definitions_at_finite_conjugates(
    tmp_path, set_test_backend, command, target
):
    path = simple_lens(tmp_path, system=command)
    path.write_text(path.read_text().replace("TH 1e20", "TH 100"))
    optic = load_oslo_file(path, strict=True)
    assert_allclose(abs(optic.paraxial.marginal_ray()[1][-2]), target)


def test_gaussian_image_height_uses_focal_plane(tmp_path, set_test_backend):
    optic = load_oslo_file(simple_lens(tmp_path, system="GIH 2"), strict=True)
    import math

    expected = math.degrees(math.atan(2 / float(optic.paraxial.f2())))
    assert_allclose(optic.fields[-1].y, abs(expected))


@pytest.mark.parametrize("material,order", [("AIR", 1), ("RFL", -1)])
def test_linear_grating_diffraction_direction_and_lens_units(
    tmp_path, set_test_backend, material, order
):
    from optiland.rays import RealRays

    path = write_lens(
        tmp_path,
        'LEN NEW "grating" 1 2\nUNI 10\nEBR .1\nWV .5\nTH 1e20\nNXT\n'
        f"{material}\nGSP .0002\nGOR {order}\nTH 1\nNXT\nAIR\nEND 2\n",
    )
    optic = load_oslo_file(path, strict=True)
    rays = RealRays(
        be.array([0.0]),
        be.array([0.0]),
        be.array([-1.0]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        1,
        0.5,
    )
    optic.surfaces[1].trace(rays)
    assert_allclose(rays.L, [0], atol=1e-12)
    assert_allclose(rays.M, [order * 0.5 / 2])
    assert_allclose(rays.N, [(0.9375) ** 0.5 * (1 if material == "AIR" else -1)])


def test_perfect_lens_reports_nonparaxial_approximation(tmp_path):
    with pytest.raises(ValueError, match="PFL.*perfect"):
        load_oslo_file(simple_lens(tmp_path, surface="PFL 20"), strict=True)


def test_solved_thickness_feeds_pickups_and_export(tmp_path, set_test_backend):
    path = simple_lens(tmp_path, surface="PY 1")
    path.write_text(path.read_text().replace("TH 20", "TH 20\nPK TH -1 1"))
    optic = load_oslo_file(path, strict=True)
    assert_allclose(optic.surfaces[1].thickness, 30)
    assert_allclose(optic.surfaces[2].thickness, 31)
    assert_allclose(optic.surfaces[3].geometry.cs.z, 61)
    target = tmp_path / "solved.len"
    save_oslo_file(optic, target)
    assert_allclose(load_oslo_file(target, strict=True).surfaces[2].geometry.cs.z, 30)


@pytest.mark.parametrize(
    "commands", ["PY 1\nTH 2", "PK TH 0 1\nTHF 2", "PU -.05\nRD 20"]
)
def test_literal_parameters_replace_constraints(tmp_path, set_test_backend, commands):
    optic = load_oslo_file(simple_lens(tmp_path, surface=commands), strict=True)
    assert_allclose(optic.surfaces[1].geometry.radius, 20)
    assert_allclose(optic.surfaces[2].geometry.cs.z, 2)


def test_historical_glass_approximation_is_explicit(tmp_path, monkeypatch):
    import optiland.fileio.oslo.reader.converter as module

    def unavailable(*args, **kwargs):
        raise ValueError("no glass")

    monkeypatch.setattr(module, "Material", unavailable)
    path = simple_lens(tmp_path, surface="GLA BAF13")
    with pytest.raises(ValueError, match="material bindings: BAF13"):
        load_oslo_file(path, strict=True)
    with pytest.warns(UserWarning, match="approximate"):
        load_oslo_file(path)


@pytest.mark.parametrize(
    "text", ["", 'LEN NEW "bad" 1 3\nNXT\nEND 3\n', 'LEN WRONG "bad" 1 1\nNXT\nEND 1\n']
)
def test_invalid_lens_structure(tmp_path, text):
    with pytest.raises(ValueError, match="LEN|surface|prescription"):
        load_oslo_file(write_lens(tmp_path, text), strict=True)


@pytest.mark.parametrize(
    "commands",
    [
        "RD 10 extra",
        "DT 2",
        "GC .5",
        "EBR 0",
        "FNO 0",
        "NAP -1",
        "GLA",
        "GLA MOD",
        "PK TH",
        "AS1 .1",
    ],
)
def test_invalid_or_incomplete_commands(tmp_path, commands):
    with pytest.raises(ValueError):
        load_oslo_file(simple_lens(tmp_path, surface=commands), strict=True)


def test_last_stop_and_diagnostic_model_serialization(tmp_path):
    path = simple_lens(tmp_path, surface="AST")
    path.write_text(path.read_text().replace("RD -20", "RD -20\nAST"))
    optic = load_oslo_file(path, strict=True)
    assert not optic.surfaces[1].is_stop
    assert optic.surfaces[2].is_stop
    with pytest.warns(UserWarning, match="MAGIC"):
        model = OsloDataParser(
            simple_lens(tmp_path, system="TELE ON", surface="MAGIC")
        ).parse()
    assert model.to_dict()["settings"]["telecentric"]
    assert model.to_dict()["diagnostics"][0]["command"] == "MAGIC"


def test_tilt_pickup_retains_pivot(tmp_path, set_test_backend):
    path = write_lens(
        tmp_path,
        'LEN NEW "pivot inverse" 1 3\nEBR 1\nANG 0\nTH 1e20\nNXT\nAIR\n'
        "TLA 20\nDCY 2\nTOZ 4\nNXT\nAIR\nPK TDM -1\nTH 5\nNXT\nAIR\nEND 3\n",
    )
    optic = load_oslo_file(path, strict=True)
    position, rotation = optic.surfaces[2].geometry.cs.get_effective_transform()
    assert_allclose(position, [0, 0, 0], atol=1e-12)
    assert_allclose(rotation, be.eye(3), atol=1e-12)


@pytest.mark.parametrize("commands", ["GSP .002\nGOR 1", "DCY 2"])
def test_writer_rejects_unsupported_data_before_overwriting(tmp_path, commands):
    optic = load_oslo_file(simple_lens(tmp_path, surface=commands), strict=True)
    output = tmp_path / "protected.len"
    output.write_text("existing file")
    with pytest.raises((ValueError, NotImplementedError), match="export|writer"):
        save_oslo_file(optic, output)
    assert output.read_text() == "existing file"


def test_mixed_material_spectra_roundtrip(tmp_path, set_test_backend):
    from optiland.materials import AbbeMaterial, DataMaterial

    optic = load_oslo_file(simple_lens(tmp_path, system="WV .4 .5 .6 .7"))
    optic.surfaces[1].material_post = DataMaterial.from_samples(
        [0.4, 0.5, 0.6, 0.7], [1.6, 1.55, 1.5, 1.45]
    )
    optic.surfaces[2].material_post = AbbeMaterial(1.6, 50, model="buchdahl")
    path = tmp_path / "mixed.len"
    save_oslo_file(optic, path)
    restored = load_oslo_file(path, strict=True)
    wave = be.array([0.4, 0.5, 0.6, 0.7])
    assert_allclose(
        restored.surfaces[2].material_post.n(wave),
        optic.surfaces[2].material_post.n(wave),
        atol=1e-6,
    )


def test_names_fields_and_signed_infinity_roundtrip(tmp_path, set_test_backend):
    optic = load_oslo_file(simple_lens(tmp_path, system="ANG 10"))
    optic.name = 'a "quoted" lens; // design'
    optic.fields.fields.clear()
    optic.fields.add(y=-3, x=2, weight=2, vx=0.1, vy=0.2)
    optic.fields.add(y=5, x=-1, weight=0)
    optic.updater.set_thickness(-float("inf"), 0)
    path = tmp_path / "metadata.len"
    save_oslo_file(optic, path)
    restored = load_oslo_file(path, strict=True)
    assert restored.name == optic.name
    assert_allclose(restored.fields.x_fields, [2, -1])
    assert_allclose(restored.fields.y_fields, [-3, 5])
    assert [f.weight for f in restored.fields] == [2, 0]
    assert_allclose(restored.fields[0].vx, 0.1)
    assert restored.surfaces[0].thickness < 0


def test_nonfinite_coordinate_reference_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="finite|infinite"):
        load_oslo_file(simple_lens(tmp_path, surface="GC 0"), strict=True)


def test_legacy_zero_coordinate_return_undoes_local_decenter(
    tmp_path, set_test_backend
):
    optic = load_oslo_file(
        simple_lens(tmp_path, surface="DCY 2\nTLA 15\nRCO 0"), strict=True
    )
    assert_allclose(
        optic.surfaces[2].geometry.cs.get_effective_transform()[0], [0, 0, 2]
    )


def test_infinite_object_field_survives_absolute_pose_mapping(
    tmp_path, set_test_backend
):
    optic = load_oslo_file(
        simple_lens(tmp_path, system="OBH -1e18", surface="DCY .1\nRCO"), strict=True
    )
    import math

    assert optic.fields.field_definition.__class__.__name__ == "AngleField"
    assert_allclose(optic.fields[-1].y, math.degrees(math.atan(0.01)))
    rays = optic.trace(
        0, 0, optic.primary_wavelength, num_rays=3, distribution="line_y"
    )
    assert be.any(be.isfinite(rays.y) & (rays.i > 0))
