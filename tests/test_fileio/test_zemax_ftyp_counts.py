"""FTYP active-count regressions for file parsing and operand dispatch."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import optiland.backend as be
from optiland.fields import AngleField, ObjectHeightField
from optiland.fileio.zemax.model import ZemaxDataModel
from optiland.fileio.zemax.reader.converter import ZemaxToOpticConverter
from optiland.fileio.zemax.reader.parser import ZemaxDataParser
from optiland.fileio.zemax.writer.encoder import ZemaxFileEncoder
from optiland.optic import Optic
from tests.utils import assert_allclose

_FIXTURE = Path(__file__).parents[1] / "zemax_files" / "inactive_wavelengths.zmx"
_VALUES = (0.7800000, 0.4350000, 0.6438469, 0.5460740, 0.6500000)
_WEIGHTS = (1.0, 0.1, 0.5, 1.0, 0.7)
_FIELDS = "XFLN 3 1 2 9\nYFLN 30 10 20 90"
_FIELD_COLUMNS = {
    "x": (3.0, 1.0, 2.0, 9.0),
    "y": (30.0, 10.0, 20.0, 90.0),
    "weights": (3.0, 1.0, 2.0, 9.0),
    "vignette_compress_x": (0.3, 0.1, 0.2, 0.9),
    "vignette_compress_y": (0.03, 0.01, 0.02, 0.09),
    "vignette_decenter_x": (0.6, 0.4, 0.5, 0.8),
    "vignette_decenter_y": (0.06, 0.04, 0.05, 0.08),
    "vignette_tangent_angle": (6.0, 4.0, 5.0, 8.0),
}
_FIELD_OPERANDS = "\n".join(
    f"{operand} {' '.join(map(str, values))}"
    for operand, values in zip(
        ("XFLN", "YFLN", "FWGN", "VCXN", "VCYN", "VDXN", "VDYN", "VANN"),
        _FIELD_COLUMNS.values(),
        strict=True,
    )
)


def _prescription(
    ftyp: str = "FTYP 1 0 1 1 0 0 0 1",
    fields: str = "XFLN 0\nYFLN 0\nFWGN 1",
    waves: str | None = None,
    *,
    weighted: bool = True,
    primary: int = 1,
    primary_first: bool = False,
) -> str:
    source = _FIXTURE.read_text(encoding="utf-8")
    header = source.split("\nFTYP", 1)[0]
    surfaces = "SURF" + source.split("SURF", 1)[1]
    if waves is None:
        waves = "\n".join(
            line for line in source.splitlines() if line.startswith("WAVM ")
        )
    if not weighted:
        waves = "\n".join(" ".join(line.split()[:3]) for line in waves.splitlines())
    primary_line = f"PWAV {primary}"
    wave_section = (primary_line, waves) if primary_first else (waves, primary_line)
    return "\n".join((header, ftyp, fields, *wave_section, surfaces))


@pytest.fixture(params=("parse", "operands"))
def read_model(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[[str], ZemaxDataModel]:
    def read(text: str) -> ZemaxDataModel:
        path = tmp_path / "synthetic_ftyp.zmx"
        parser = ZemaxDataParser(str(path))
        if request.param == "parse":
            path.write_text(text, encoding="utf-8")
            return parser.parse()

        for line in text.splitlines():
            tokens = line.split()
            if tokens and tokens[0] in parser._operand_table:
                parser._operand_table[tokens[0]](tokens)
        # Keep the existing dispatch contract: no parse-only wavelength finalizer.
        parser._finalize_fields()
        parser._finalize_surface()
        return parser.data_model

    return read


def _assert_wavelengths(
    model: ZemaxDataModel,
    values: Sequence[float],
    weights: Sequence[float],
    primary: int = 0,
    optic: Optic | None = None,
) -> None:
    assert len(model.wavelengths["data"]) == len(values)
    assert len(model.wavelengths["weights"]) == len(weights)
    assert_allclose(model.wavelengths["data"], values)
    assert_allclose(model.wavelengths["weights"], weights)
    assert model.wavelengths["primary_index"] == primary
    assert_allclose(model.wavelengths["data"][primary], values[primary])
    if optic is not None:
        assert len(optic.wavelengths) == len(values)
        assert_allclose(optic.wavelengths.get_wavelengths(), values)
        assert_allclose(optic.wavelengths.weights, weights)
        assert optic.wavelengths.primary_index == primary
        assert_allclose(optic.primary_wavelength, values[primary])
        assert [wave.is_primary for wave in optic.wavelengths] == [
            index == primary for index in range(len(values))
        ]


def _assert_fields(
    model: ZemaxDataModel,
    columns: dict[str, Sequence[float]],
    field_type: str = "angle",
    optic: Optic | None = None,
) -> None:
    assert model.fields["type"] == field_type
    for key, values in columns.items():
        assert len(model.fields[key]) == len(values), key
        assert_allclose(model.fields[key], values)
    if optic is not None:
        definition = AngleField if field_type == "angle" else ObjectHeightField
        assert isinstance(optic.fields.field_definition, definition)
        assert len(optic.fields.fields) == len(columns["x"])
        for key, attribute in (
            ("x", "x"),
            ("y", "y"),
            ("weights", "weight"),
            ("vignette_compress_x", "vx"),
            ("vignette_compress_y", "vy"),
        ):
            default = 1.0 if key == "weights" else 0.0
            expected = columns.get(key, [default] * len(columns["x"]))
            assert_allclose(
                [getattr(field, attribute) for field in optic.fields.fields], expected
            )


@pytest.mark.parametrize("active_count", (1, 4, 5))
@pytest.mark.parametrize("weighted", (True, False), ids=("weighted", "unweighted"))
def test_active_wavelength_count(
    read_model: Callable[[str], ZemaxDataModel],
    set_test_backend: None,
    active_count: int,
    weighted: bool,
) -> None:
    model = read_model(
        _prescription(ftyp=f"FTYP 1 0 1 {active_count} 0 0 0 1", weighted=weighted)
    )
    optic = ZemaxToOpticConverter(model).convert()
    assert model.wavelengths["num_wavelengths"] == active_count
    weights = _WEIGHTS if weighted else (1.0,) * 5
    _assert_wavelengths(
        model, _VALUES[:active_count], weights[:active_count], optic=optic
    )
    _assert_fields(
        model, {"x": (0.0,), "y": (0.0,), "weights": (1.0,)}, "object_height", optic
    )


def test_padded_fields_without_fwgn(
    read_model: Callable[[str], ZemaxDataModel], set_test_backend: None
) -> None:
    # No FWGN: a short weights array must not hide ignored FTYP field counts.
    model = read_model(_prescription(ftyp="FTYP 0 0 2 1 0 0 0 1", fields=_FIELDS))
    optic = ZemaxToOpticConverter(model).convert()
    assert model.fields["num_fields"] == 2
    _assert_fields(model, {"x": (1.0, 3.0), "y": (10.0, 30.0)}, optic=optic)
    _assert_wavelengths(model, _VALUES[:1], _WEIGHTS[:1], optic=optic)


def test_short_ftyp_two_wavelengths_and_multiple_fields(
    read_model: Callable[[str], ZemaxDataModel], set_test_backend: None
) -> None:
    model = read_model(
        _prescription(
            ftyp="FTYP 0", fields=_FIELDS, waves="WAVM 1 .78 1\nWAVM 2 .435 .1"
        )
    )
    optic = ZemaxToOpticConverter(model).convert()
    _assert_fields(
        model,
        {"x": (1.0, 2.0, 3.0, 9.0), "y": (10.0, 20.0, 30.0, 90.0)},
        optic=optic,
    )
    _assert_wavelengths(model, _VALUES[:2], _WEIGHTS[:2], optic=optic)


@pytest.mark.parametrize("ftyp", ("", "FTYP 0"), ids=("absent", "short"))
@pytest.mark.parametrize("weighted", (True, False), ids=("weighted", "unweighted"))
def test_unspecified_counts_do_not_cap_wavm_by_token_length(
    read_model: Callable[[str], ZemaxDataModel],
    set_test_backend: None,
    ftyp: str,
    weighted: bool,
) -> None:
    model = read_model(
        _prescription(ftyp=ftyp, fields=_FIELDS, weighted=weighted, primary=5)
    )
    optic = ZemaxToOpticConverter(model).convert()
    _assert_wavelengths(model, _VALUES, _WEIGHTS if weighted else (1.0,) * 5, 4, optic)
    _assert_fields(
        model,
        {"x": (1.0, 2.0, 3.0, 9.0), "y": (10.0, 20.0, 30.0, 90.0)},
        optic=optic,
    )


@pytest.mark.parametrize(
    ("ftyp", "field_count", "wave_count"),
    [
        pytest.param("FTYP 0 0 2 1", 2, 1, id="both-valid"),
        pytest.param("FTYP 0 0 2", 2, 5, id="wave-missing"),
        pytest.param("FTYP 0 0 2 bad", 2, 5, id="wave-malformed"),
        pytest.param("FTYP 0 0 2 -1", 2, 5, id="wave-negative"),
        pytest.param("FTYP 0 0 2 0", 2, 0, id="wave-zero"),
        pytest.param("FTYP 0 0 bad 1", 4, 1, id="field-malformed"),
        pytest.param("FTYP 0 0 -1 1", 4, 1, id="field-negative"),
        pytest.param("FTYP 0 0 0 1", 0, 1, id="field-zero"),
        pytest.param("FTYP 0 0", 4, 5, id="counts-missing"),
        pytest.param("FTYP 0 0 bad bad", 4, 5, id="counts-malformed"),
        pytest.param("FTYP 0 0 -1 -1", 4, 5, id="counts-negative"),
        pytest.param("FTYP 0 0 0 0", 0, 0, id="counts-zero"),
    ],
)
def test_counts_are_independently_optional(
    read_model: Callable[[str], ZemaxDataModel],
    ftyp: str,
    field_count: int,
    wave_count: int,
) -> None:
    model = read_model(_prescription(ftyp=ftyp, fields=_FIELD_OPERANDS))
    order = {0: (), 2: (1, 0), 4: (1, 2, 0, 3)}[field_count]
    expected = {
        key: tuple(values[index] for index in order)
        for key, values in _FIELD_COLUMNS.items()
    }
    _assert_fields(model, expected)
    if wave_count:
        _assert_wavelengths(model, _VALUES[:wave_count], _WEIGHTS[:wave_count])
    else:
        # An explicit zero is not a missing count; no empty-set primary policy here.
        assert model.wavelengths["data"] == []
        assert model.wavelengths["weights"] == []


@pytest.mark.parametrize(
    ("ftyp", "order"),
    [
        pytest.param("FTYP 0 0 2 1 0 0 0 1", (1, 0), id="modern-padding"),
        pytest.param("FTYP 0", (1, 2, 0, 3), id="legacy-default-not-a-limit"),
        pytest.param("", (1, 2, 0, 3), id="absent-ftyp"),
    ],
)
def test_parallel_field_columns_remain_aligned(
    read_model: Callable[[str], ZemaxDataModel],
    set_test_backend: None,
    ftyp: str,
    order: tuple[int, ...],
) -> None:
    model = read_model(
        _prescription(ftyp=ftyp, fields=_FIELD_OPERANDS, waves="WAVM 1 .78 1")
    )
    optic = ZemaxToOpticConverter(model).convert()
    expected = {
        key: tuple(values[index] for index in order)
        for key, values in _FIELD_COLUMNS.items()
    }
    _assert_fields(model, expected, optic=optic)
    _assert_wavelengths(model, _VALUES[:1], _WEIGHTS[:1], optic=optic)


@pytest.mark.parametrize(
    ("ftyp", "count"),
    [("FTYP 0 0 2 1 0 0 0 1", 2), ("FTYP 0", 4), ("", 4)],
    ids=("modern", "short", "absent"),
)
def test_field_operand_counts_before_final_zip(ftyp: str, count: int) -> None:
    parser = ZemaxDataParser("unused.zmx")
    if ftyp:
        parser._operand_table["FTYP"](ftyp.split())
    for line, (key, values) in zip(
        _FIELD_OPERANDS.splitlines(), _FIELD_COLUMNS.items(), strict=True
    ):
        tokens = line.split()
        parser._operand_table[tokens[0]](tokens)
        # Another shorter column must not mask this operand's ignored count.
        assert len(parser.data_model.fields[key]) == count, key
        assert_allclose(parser.data_model.fields[key], values[:count])


@pytest.mark.parametrize(
    ("field_token", "wave_token", "order", "wave_count"),
    [("", "2", (1, 2, 0, 3), 2), ("2", "", (1, 0), 5)],
    ids=("field-empty-wave-valid", "field-valid-wave-empty"),
)
def test_empty_count_token_does_not_disable_other_count(
    field_token: str, wave_token: str, order: tuple[int, ...], wave_count: int
) -> None:
    parser = ZemaxDataParser("unused.zmx")
    # Empty positional tokens are a direct-handler input, not whitespace syntax.
    parser._operand_table["FTYP"](["FTYP", "0", "0", field_token, wave_token])
    for line in _prescription(ftyp="", fields=_FIELD_OPERANDS).splitlines():
        tokens = line.split()
        if tokens and tokens[0] in parser._operand_table:
            parser._operand_table[tokens[0]](tokens)
    parser._finalize_fields()
    parser._finalize_surface()
    expected = {
        key: tuple(values[index] for index in order)
        for key, values in _FIELD_COLUMNS.items()
    }
    _assert_fields(parser.data_model, expected)
    _assert_wavelengths(parser.data_model, _VALUES[:wave_count], _WEIGHTS[:wave_count])


@pytest.mark.parametrize(
    "tokens",
    [
        ["FTYP", "0"],
        ["FTYP"],
        ["FTYP", "", "", "", "", "", "", "", ""],
        ["FTYP", "bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad"],
    ],
    ids=("short", "missing", "empty", "malformed"),
)
def test_legacy_on_axis_defaults_before_field_input(tokens: list[str]) -> None:
    parser = ZemaxDataParser("unused.zmx")
    parser._operand_table["FTYP"](tokens)
    _assert_fields(parser.data_model, {"x": (0.0,), "y": (0.0,)})
    assert parser.data_model.fields["num_fields"] == 1
    assert parser.data_model.wavelengths["num_wavelengths"] == 0
    assert parser.data_model.fields["object_space_telecentric"] is False
    assert parser.data_model.fields["afocal_image_space"] is False


def test_active_zero_weight_is_not_an_inactive_slot(
    read_model: Callable[[str], ZemaxDataModel], set_test_backend: None
) -> None:
    model = read_model(
        _prescription(
            ftyp="FTYP 1 0 1 2 0 0 0 1",
            waves="WAVM 1 .78 1\nWAVM 2 .435 0\nWAVM 3 .6438469 .5",
            primary=2,
        )
    )
    optic = ZemaxToOpticConverter(model).convert()
    _assert_wavelengths(model, _VALUES[:2], (1.0, 0.0), 1, optic)


@pytest.mark.parametrize(
    "primary_first", (True, False), ids=("pwav-first", "pwav-last")
)
@pytest.mark.parametrize(
    ("count", "waves", "values", "weights", "primary"),
    [
        pytest.param(
            3,
            "WAVM 5 .65 .7\nWAVM 4 .546074 1\nWAVM 3 .6438469 .5\n"
            "WAVM 1 .78 1\nWAVM 2 .435 .1",
            _VALUES[:3],
            _WEIGHTS[:3],
            3,
            id="inactive-before-active",
        ),
        pytest.param(
            5,
            "WAVM 5 .435 .7\nWAVM 3 .6438469 .5\nWAVM 1 .78 1\n"
            "WAVM 4 .546074 1\nWAVM 2 .435 .1",
            (*_VALUES[:4], 0.435),
            _WEIGHTS,
            5,
            id="equal-values-distinct-slots",
        ),
    ],
)
def test_source_slots_determine_order_and_primary(
    read_model: Callable[[str], ZemaxDataModel],
    set_test_backend: None,
    primary_first: bool,
    count: int,
    waves: str,
    values: tuple[float, ...],
    weights: tuple[float, ...],
    primary: int,
) -> None:
    model = read_model(
        _prescription(
            ftyp=f"FTYP 1 0 1 {count} 0 0 0 1",
            waves=waves,
            primary=primary,
            primary_first=primary_first,
        )
    )
    optic = ZemaxToOpticConverter(model).convert()
    _assert_wavelengths(model, values, weights, primary - 1, optic)


@pytest.mark.parametrize(
    "primary_first", (True, False), ids=("pwav-first", "pwav-last")
)
def test_direct_handlers_expose_active_wavelengths_without_parse_finalization(
    primary_first: bool,
) -> None:
    parser = ZemaxDataParser("unused.zmx")
    parser._operand_table["FTYP"](["FTYP", "1", "0", "1", "2", "0", "0", "0", "1"])
    if primary_first:
        parser._operand_table["PWAV"](["PWAV", "2"])
    for line in ("WAVM 3 .6438469 .5", "WAVM 2 .435 .1", "WAVM 1 .78 1"):
        parser._operand_table["WAVM"](line.split())
    if not primary_first:
        parser._operand_table["PWAV"](["PWAV", "2"])
    _assert_wavelengths(parser.data_model, _VALUES[:2], _WEIGHTS[:2], 1)


def test_underfilled_declared_wavelength_count_is_not_an_equality_constraint(
    read_model: Callable[[str], ZemaxDataModel], set_test_backend: None
) -> None:
    model = read_model(_prescription(ftyp="FTYP 1 0 1 3 0 0 0 1", waves="WAVM 1 .78 1"))
    assert model.wavelengths["num_wavelengths"] == 3
    optic = ZemaxToOpticConverter(model).convert()
    _assert_wavelengths(model, _VALUES[:1], _WEIGHTS[:1], optic=optic)


def test_one_active_fixture_preserves_geometry_aperture_and_primary_trace(
    read_model: Callable[[str], ZemaxDataModel], set_test_backend: None
) -> None:
    model = read_model(_FIXTURE.read_text(encoding="utf-8"))
    imported = ZemaxToOpticConverter(model).convert()
    reference = Optic("SyntheticFTYP")
    reference.surfaces.add(index=0, radius=be.inf, thickness=10)
    reference.surfaces.add(index=1, radius=be.inf, thickness=20, is_stop=True)
    reference.surfaces.add(index=2, radius=be.inf, thickness=0)
    reference.set_aperture(aperture_type="EPD", value=4)
    reference.fields.set_type(field_type="object_height")
    reference.fields.add(x=0, y=0, weight=1)
    reference.wavelengths.add(value=0.78, is_primary=True, weight=1)

    assert model.name == "SyntheticFTYP"
    assert tuple(model.surfaces) == (0, 1, 2)
    assert model.surfaces[1]["semi_diameter"] == 2
    assert [surface["thickness"] for surface in model.surfaces.values()] == [10, 20, 0]
    for optic in (imported, reference):
        assert len(optic.surfaces) == 3
        assert [surface.is_stop for surface in optic.surfaces] == [False, True, False]
        assert optic.aperture.ap_type == "EPD"
        assert_allclose(optic.aperture.value, 4.0)
        # Optiland anchors the first optical surface, not the object plane, at z=0.
        assert_allclose(optic.surfaces.global_z_positions.ravel(), (-10.0, 0.0, 20.0))
        assert all(be.isinf(surface.geometry.radius) for surface in optic.surfaces)
        matrix = optic.paraxial.ray_transfer_matrix(1, 2)
        assert_allclose(matrix, be.array([[1.0, 20.0], [0.0, 1.0]]))
        marginal_y, marginal_u = optic.paraxial.marginal_ray()
        assert_allclose(marginal_y.ravel(), (0.0, 2.0, 6.0))
        assert_allclose(marginal_u.ravel(), (0.2, 0.2, 0.2))

    imported_rays = imported.trace_generic(0, 0, 0, 0.5, imported.primary_wavelength)
    reference_rays = reference.trace_generic(0, 0, 0, 0.5, reference.primary_wavelength)
    for attribute in ("x", "y", "z", "L", "M", "N", "i", "w", "opd"):
        assert_allclose(
            getattr(imported_rays, attribute), getattr(reference_rays, attribute)
        )
    assert_allclose(imported_rays.y, (3.0,))
    assert_allclose(imported_rays.z, (20.0,))
    assert_allclose(imported.paraxial.EPD() / 2, 2.0)
    _assert_wavelengths(model, _VALUES[:1], _WEIGHTS[:1], optic=imported)


@pytest.mark.parametrize(
    ("final_ftyp", "order", "wave_count"),
    [
        pytest.param("FTYP 0 0 3 3", (1, 2, 0), 3, id="shrink-grow"),
        pytest.param("FTYP 0", (1, 2, 0, 3), 5, id="remove-limits"),
    ],
)
def test_repeated_ftyp_restores_latest_raw_field_columns(
    read_model: Callable[[str], ZemaxDataModel],
    set_test_backend: None,
    final_ftyp: str,
    order: tuple[int, ...],
    wave_count: int,
) -> None:
    updated_columns = "\n".join(
        " ".join((tokens[0], *(str(float(value) / 2) for value in tokens[1:])))
        for tokens in (line.split() for line in _FIELD_OPERANDS.splitlines())
    )
    text = _prescription(
        ftyp="FTYP 0 0 4 5", fields=_FIELD_OPERANDS, primary=wave_count
    )
    text = text.replace(
        "\nSURF 0",
        f"\nFTYP 0 0 1 1\n{updated_columns}\n{final_ftyp}\nSURF 0",
        1,
    )
    model = read_model(text)
    optic = ZemaxToOpticConverter(model).convert()
    # Wavelength slots already retain raw records and provide a control here.
    _assert_wavelengths(
        model, _VALUES[:wave_count], _WEIGHTS[:wave_count], wave_count - 1, optic
    )
    expected = {
        key: tuple(values[index] / 2 for index in order)
        for key, values in _FIELD_COLUMNS.items()
    }
    _assert_fields(model, expected, optic=optic)
    assert model.fields["num_fields"] == len(order)
    assert model.wavelengths["num_wavelengths"] == wave_count


def test_zero_counts_then_legacy_restores_implicit_on_axis_field(
    read_model: Callable[[str], ZemaxDataModel], set_test_backend: None
) -> None:
    model = read_model(
        _prescription(
            ftyp="FTYP 0 0 0 0",
            fields="",
            waves="WAVM 1 .55 .2\nWAVM 2 .65 .8\nFTYP 0",
            primary=2,
        )
    )
    optic = ZemaxToOpticConverter(model).convert()
    _assert_fields(model, {"x": (0.0,), "y": (0.0,)}, optic=optic)
    _assert_wavelengths(model, (0.55, 0.65), (0.2, 0.8), 1, optic)
    assert model.fields["num_fields"] == 1
    assert model.wavelengths["num_wavelengths"] == 2


def test_legacy_wavl_preserves_existing_single_record_interpretation(
    read_model: Callable[[str], ZemaxDataModel],
) -> None:
    # Compatibility with the old incomplete reader, not certified WAVL vector syntax.
    model = read_model(_prescription(ftyp="FTYP 0", waves="WAVL .55 .55 .55"))
    _assert_wavelengths(model, (0.55,), (0.55,))


@pytest.mark.parametrize("scalar_primary", (False, True), ids=("int", "backend-int"))
def test_converter_accepts_array_backed_wavelength_dictionary(
    set_test_backend: None, scalar_primary: bool
) -> None:
    model = ZemaxDataParser(str(_FIXTURE)).parse()
    model.wavelengths = {
        "data": be.array([0.55, 0.65]),
        "weights": be.array([0.2, 0.8]),
        "primary_index": be.arange_indices(2)[1] if scalar_primary else 1,
    }
    optic = ZemaxToOpticConverter(model.to_dict()).convert()
    _assert_wavelengths(model, (0.55, 0.65), (0.2, 0.8), 1, optic)


def test_converter_rejects_invalid_primary_in_array_backed_dictionary(
    set_test_backend: None,
) -> None:
    model = ZemaxDataParser(str(_FIXTURE)).parse()
    model.wavelengths = {
        "data": be.array([0.55, 0.65]),
        "weights": be.array([0.2, 0.8]),
    }
    for primary in (None, -1, 2, "1", 0.5, float("nan"), float("inf")):
        model.wavelengths["primary_index"] = primary
        with pytest.raises(ValueError, match="(?i)primary.*wavelength"):
            ZemaxToOpticConverter(model.to_dict()).convert()


@pytest.mark.parametrize("ftyp", ("FTYP 0", ""), ids=("short", "absent"))
def test_inferred_public_counts_survive_model_header_round_trip(
    read_model: Callable[[str], ZemaxDataModel], ftyp: str
) -> None:
    # A duplicate field makes the published count differ from the raw column length.
    fields = "\n".join(
        f"{line} {line.split()[1]}" for line in _FIELD_OPERANDS.splitlines()
    )
    model = read_model(
        _prescription(ftyp=ftyp, fields=fields, weighted=False, primary=5)
    )
    expected = {
        key: tuple(values[index] for index in (1, 2, 0, 3))
        for key, values in _FIELD_COLUMNS.items()
    }
    assert model.fields["num_fields"] == 4
    assert model.wavelengths["num_wavelengths"] == 5
    _assert_fields(model, expected)
    _assert_wavelengths(model, _VALUES, (1.0,) * 5, 4)

    # Bypass the Optic formatter; test header counts and field weights, not surfaces
    # or the separate writer limitation that emits unit wavelength weights.
    lines = ZemaxFileEncoder(model).encode()
    ftyp_tokens = next(line.split() for line in lines if line.startswith("FTYP "))
    assert [int(value) for value in ftyp_tokens[3:5]] == [4, 5]
    restored = read_model("\n".join(lines))
    assert restored.fields["num_fields"] == 4
    assert restored.wavelengths["num_wavelengths"] == 5
    _assert_fields(restored, expected)
    _assert_wavelengths(restored, _VALUES, (1.0,) * 5, 4)


@pytest.mark.parametrize("has_pwav", (False, True), ids=("no-pwav", "inactive-pwav"))
def test_explicit_zero_wavelengths_encode_without_primary_record(
    read_model: Callable[[str], ZemaxDataModel], has_pwav: bool
) -> None:
    text = _prescription(ftyp="FTYP 0 0 1 0")
    if not has_pwav:
        text = text.replace("PWAV 1\n", "")
    model = read_model(text)
    assert model.wavelengths["num_wavelengths"] == 0
    assert model.wavelengths["data"] == []
    assert model.wavelengths["weights"] == []
    lines = ZemaxFileEncoder(model).encode()
    assert not any(line.startswith(("PWAV ", "WAVM ")) for line in lines)
    restored = read_model("\n".join(lines))
    assert restored.wavelengths["num_wavelengths"] == 0
    assert restored.wavelengths["data"] == []
    assert restored.wavelengths["weights"] == []


@pytest.mark.parametrize(
    ("count", "waves", "primary"),
    [
        pytest.param(3, "WAVM 1 .55 .2\nWAVM 3 .65 .8", 2, id="missing-slot"),
        pytest.param(1, "WAVM 1 .55 .2\nWAVM 2 .65 .8", 2, id="inactive-slot"),
        pytest.param(2, "WAVM 1 .55 .2\nWAVM 2 .65 .8", 0, id="zero-pwav"),
        pytest.param(2, "WAVM 1 .55 .2\nWAVM 2 .65 .8", -1, id="negative-pwav"),
    ],
)
def test_nonempty_model_with_missing_primary_slot_is_rejected_by_consumers(
    read_model: Callable[[str], ZemaxDataModel],
    set_test_backend: None,
    count: int,
    waves: str,
    primary: int,
) -> None:
    model = read_model(
        _prescription(ftyp=f"FTYP 0 0 1 {count}", waves=waves, primary=primary)
    )
    expected_count = 1 if count == 1 else 2
    assert len(model.wavelengths["data"]) == expected_count
    assert_allclose(model.wavelengths["data"], (0.55, 0.65)[:expected_count])
    assert_allclose(model.wavelengths["weights"], (0.2, 0.8)[:expected_count])
    with pytest.raises(ValueError, match="(?i)primary.*wavelength"):
        ZemaxToOpticConverter(model).convert()
    with pytest.raises(ValueError, match="(?i)primary.*wavelength"):
        ZemaxFileEncoder(model).encode()


def test_sparse_present_primary_becomes_valid_when_its_slot_arrives(
    read_model: Callable[[str], ZemaxDataModel], set_test_backend: None
) -> None:
    model = read_model(
        _prescription(
            ftyp="FTYP 0 0 1 3",
            waves="WAVM 1 .55 .2\nWAVM 3 .65 .8",
            primary=3,
            primary_first=True,
        )
    )
    optic = ZemaxToOpticConverter(model).convert()
    assert model.wavelengths["num_wavelengths"] == 3
    _assert_wavelengths(model, (0.55, 0.65), (0.2, 0.8), 1, optic)


@pytest.mark.parametrize(
    ("replacement", "weight"),
    [("WAVM 2 .61 .4", 0.4), ("WAVM 2 .61", 1.0)],
    ids=("replacement-weight", "replacement-default-weight"),
)
def test_repeated_wavm_slot_replaces_value_and_weight(
    read_model: Callable[[str], ZemaxDataModel], replacement: str, weight: float
) -> None:
    model = read_model(
        _prescription(
            ftyp="FTYP 0 0 1 2",
            waves=f"WAVM 1 .55 .2\nWAVM 2 .65 .8\n{replacement}",
            primary=2,
        )
    )
    _assert_wavelengths(model, (0.55, 0.61), (0.2, weight), 1)


@pytest.mark.parametrize("omitted_axis", ("x", "y"))
def test_omitted_field_axis_is_zero_for_every_supplied_field(
    read_model: Callable[[str], ZemaxDataModel],
    set_test_backend: None,
    omitted_axis: str,
) -> None:
    omitted_operand = "XFLN" if omitted_axis == "x" else "YFLN"
    fields = "\n".join(
        line
        for line in _FIELD_OPERANDS.splitlines()
        if not line.startswith(omitted_operand)
    )
    model = read_model(_prescription(ftyp="FTYP 0", fields=fields, weighted=False))
    # Sorting by y is stable when y is entirely omitted and therefore all zero.
    order = (1, 2, 0, 3) if omitted_axis == "x" else (0, 1, 2, 3)
    expected = {
        key: (0.0,) * 4 if key == omitted_axis else tuple(values[i] for i in order)
        for key, values in _FIELD_COLUMNS.items()
    }
    optic = ZemaxToOpticConverter(model).convert()
    _assert_fields(model, expected, optic=optic)
    assert model.fields["num_fields"] == 4
    _assert_wavelengths(model, _VALUES, (1.0,) * 5, optic=optic)


def test_encoder_defaults_missing_primary_key_but_rejects_explicit_none() -> None:
    model = ZemaxDataParser(str(_FIXTURE)).parse()
    model.wavelengths = {"data": [0.55], "num_wavelengths": 1}
    assert "PWAV 1" in ZemaxFileEncoder(model).encode()

    model.wavelengths["primary_index"] = None
    with pytest.raises(ValueError, match="(?i)primary.*wavelength"):
        ZemaxFileEncoder(model).encode()


def test_converter_preserves_missing_primary_key_error() -> None:
    model = ZemaxDataParser(str(_FIXTURE)).parse()
    model.wavelengths = {"data": [0.55], "num_wavelengths": 1}
    # Unlike the encoder, the converter never defaulted a missing primary to zero.
    with pytest.raises(KeyError, match="primary_index"):
        ZemaxToOpticConverter(model.to_dict()).convert()


@pytest.mark.parametrize("primary", (0, 1))
def test_encoder_infers_missing_wavelength_count_for_list_and_array(
    set_test_backend: None, tmp_path: Path, primary: int
) -> None:
    model = ZemaxDataParser(str(_FIXTURE)).parse()
    path = tmp_path / "inferred_wavelength_count.zmx"
    for data in ([0.55, 0.65], be.array([0.55, 0.65])):
        model.wavelengths = {"data": data, "primary_index": primary}
        lines = ZemaxFileEncoder(model).encode()
        ftyp = next(line.split() for line in lines if line.startswith("FTYP "))
        assert int(ftyp[4]) == 2
        assert f"PWAV {primary + 1}" in lines
        path.write_text("\n".join(lines), encoding="utf-8")
        restored = ZemaxDataParser(str(path)).parse()
        assert restored.wavelengths["num_wavelengths"] == 2
        _assert_wavelengths(restored, (0.55, 0.65), (1.0, 1.0), primary)


def test_encoder_infers_zero_count_for_empty_data_without_count_key(
    tmp_path: Path,
) -> None:
    model = ZemaxDataParser(str(_FIXTURE)).parse()
    model.wavelengths = {"data": []}
    lines = ZemaxFileEncoder(model).encode()
    ftyp = next(line.split() for line in lines if line.startswith("FTYP "))
    assert int(ftyp[4]) == 0
    assert not any(line.startswith(("PWAV ", "WAVM ")) for line in lines)
    path = tmp_path / "empty_wavelength_count.zmx"
    path.write_text("\n".join(lines), encoding="utf-8")
    restored = ZemaxDataParser(str(path)).parse()
    assert restored.wavelengths["num_wavelengths"] == 0
    assert restored.wavelengths["data"] == []
    assert restored.wavelengths["weights"] == []


def test_encoder_explicit_wavelength_count_overrides_data_length() -> None:
    model = ZemaxDataParser(str(_FIXTURE)).parse()
    for count in (0, 3):
        model.wavelengths = {
            "data": [0.55],
            "primary_index": 0,
            "num_wavelengths": count,
        }
        # Header precedence only: do not impose validation of conflicting records.
        lines: list[str] = []
        ZemaxFileEncoder(model)._encode_fields_header(lines)
        assert int(lines[0].split()[4]) == count
