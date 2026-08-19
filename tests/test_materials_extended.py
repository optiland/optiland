from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from optiland.materials import abbe, material_utils
from optiland.materials.abbe import AbbeMaterial, AbbeMaterialE

# -----------------------------------------------------------------------------
# AbbeMaterial Tests (Buchdahl & E-line)
# -----------------------------------------------------------------------------


def test_abbe_buchdahl_init(set_test_backend):
    # Test initialization with 'buchdahl' model
    mat = AbbeMaterial(1.5, 64.0, model="buchdahl")
    assert mat.model_name == "buchdahl"
    assert isinstance(mat.model, abbe.BuchdahlDModel)

    # Test calculations
    n = mat.n(0.5876)
    assert abs(n - 1.5) < 1e-3


def test_abbe_invalid_model(set_test_backend):
    with pytest.raises(ValueError, match="Unknown model"):
        AbbeMaterial(1.5, 64.0, model="invalid")


def test_abbe_e_init(set_test_backend):
    # Test AbbeMaterialE
    mat = AbbeMaterialE(1.5, 64.0)
    assert isinstance(mat.model, abbe.BuchdahlEModel)

    # Test calculations (check near reference wavelength)
    n = mat.n(0.5461)
    assert abs(n - 1.5) < 1e-3


def test_abbe_to_from_dict(set_test_backend):
    mat = AbbeMaterial(1.5, 64.0, model="buchdahl")
    d = mat.to_dict()
    assert d["model"] == "buchdahl"
    assert d["index"] == 1.5

    mat2 = AbbeMaterial.from_dict(d)
    assert mat2.model_name == "buchdahl"
    assert mat2.index.item() == 1.5


def test_abbe_e_to_from_dict(set_test_backend):
    mat = AbbeMaterialE(1.5, 64.0)
    d = mat.to_dict()
    assert d["index"] == 1.5

    mat2 = AbbeMaterialE.from_dict(d)
    assert mat2.index.item() == 1.5


# -----------------------------------------------------------------------------
# Material Utils Tests
# -----------------------------------------------------------------------------


@patch("optiland.materials.material_utils.resources.files")
def test_glasses_selection(mock_files, set_test_backend):
    # Mock CSV content
    csv_content = (
        "group,filename,filename_no_ext,min_wavelength,max_wavelength\n"
        "glass,catalog/glass1,glass1,0.3,1.0\n"
        "glass,catalog/glass2,glass2,0.4,0.8\n"
        "plastic,catalog/plastic1,plastic1,0.3,1.0\n"
    )

    mock_path = MagicMock()
    # Ensure the path behaves like a string or path-like object
    mock_file_path = MagicMock()
    mock_file_path.__str__.return_value = "dummy.csv"
    mock_path.joinpath.return_value = mock_file_path
    mock_files.return_value = mock_path

    with patch("builtins.open", mock_open(read_data=csv_content)):
        # Case 1: Broad range, only glass1 fits
        selection = material_utils.glasses_selection(0.35, 0.95)
        assert "glass1" in selection
        assert "glass2" not in selection  # 0.8 < 0.95

        # Case 2: Narrow range
        selection = material_utils.glasses_selection(0.5, 0.6)
        assert "glass1" in selection
        assert "glass2" in selection


@patch("optiland.materials.material_utils.Material")
def test_get_nd_vd(mock_material_cls, set_test_backend):
    # Mock Material instance and file loading
    mock_mat_instance = MagicMock()
    mock_material_cls.return_value = mock_mat_instance
    mock_mat_instance._retrieve_file.return_value = ("dummy.yml", None)

    yaml_content = """
    SPECS:
        nd: 1.5
        Vd: 64.0
    """

    with patch("pathlib.Path.open", mock_open(read_data=yaml_content)):
        with patch("yaml.safe_load", return_value={"SPECS": {"nd": 1.5, "Vd": 64.0}}):
            nd, vd = material_utils.get_nd_vd("MyGlass")
            assert nd == 1.5
            assert vd == 64.0


def test_downsample_glass_map(set_test_backend):
    # Create dummy glass dict
    glass_dict = {
        "G1": (1.5, 60.0),
        "G2": (1.51, 61.0),
        "G3": (1.8, 30.0),
        "G4": (1.81, 31.0),
    }

    # We expect 2 clusters essentially (low index/high abbe, high index/low abbe)
    selected = material_utils.downsample_glass_map(glass_dict, 2)
    assert len(selected) == 2

    # Should likely pick one from G1/G2 and one from G3/G4
    names = list(selected.keys())
    # Check if we covered the range roughly
    vals = list(selected.values())
    nd_vals = [v[0] for v in vals]
    assert min(nd_vals) < 1.6
    assert max(nd_vals) > 1.7


def test_get_neighbour_glasses(set_test_backend):
    glass_dict = {"Ref": (1.5, 60.0), "Near": (1.501, 60.1), "Far": (1.8, 30.0)}

    neighbors = material_utils.get_neighbour_glasses(
        "Ref", glass_dict=glass_dict, num_neighbours=1
    )
    assert neighbors == ["Near"]


@patch("matplotlib.pyplot.subplots")
def test_plot_glass_map(mock_subplots, set_test_backend):
    # Mock figure and axis
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    # Patch get_nd_vd to avoid file I/O
    with patch("optiland.materials.material_utils.get_nd_vd", return_value=(1.5, 60.0)):
        material_utils.plot_glass_map(["G1", "G2"], ["G1"])

        # Verify plotting calls
        assert mock_ax.scatter.call_count >= 1


@patch("optiland.materials.material_utils.get_nd_vd")
def test_find_closest_glass(mock_get_nd_vd, set_test_backend):
    # Setup mock return values side effect
    def side_effect(name):
        d = {"G1": (1.5, 60.0), "G2": (1.8, 30.0)}
        return d[name]

    mock_get_nd_vd.side_effect = side_effect

    closest = material_utils.find_closest_glass((1.51, 60.1), ["G1", "G2"])
    assert closest == "G1"


@patch("matplotlib.pyplot.subplots")
def test_plot_nk(mock_subplots, set_test_backend):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    # Mock get_legend_handles_labels
    mock_ax.get_legend_handles_labels.return_value = ([], [])

    # Mock twinx and its legend handles
    mock_ax_k = MagicMock()
    mock_ax.twinx.return_value = mock_ax_k
    mock_ax_k.get_legend_handles_labels.return_value = ([], [])

    # Create a dummy material
    mat = MagicMock()
    mat.material_data = {
        "min_wavelength": 0.4,
        "max_wavelength": 0.7,
        "category_name_full": "Test Glass",
        "reference": "TEST",
    }
    mat.n.return_value = np.array([1.5] * 10)
    mat.k.return_value = np.array([0.0] * 10)

    material_utils.plot_nk(mat, wavelength_range=(0.4, 0.7))
    assert mock_ax.plot.call_count >= 1
