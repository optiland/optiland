from __future__ import annotations

import tempfile

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.backend as be
from optiland.physical_apertures import (
    BaseAperture,
    DifferenceAperture,
    EllipticalAperture,
    FileAperture,
    IntersectionAperture,
    OffsetRadialAperture,
    PolygonAperture,
    RadialAperture,
    RectangularAperture,
    UnionAperture,
)
from optiland.physical_apertures.radial import configure_aperture
from optiland.rays import RealRays

from .utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


class TestRadialAperture:
    def test_clip(self, set_test_backend):
        aperture = RadialAperture(r_max=5, r_min=2)
        rays = RealRays(
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        aperture.clip(rays)
        assert be.all(rays.i == be.array([0, 0, 1, 1, 0, 0]))

    def test_scale(self, set_test_backend):
        aperture = RadialAperture(r_max=5, r_min=2)
        aperture.scale(2)
        assert aperture.r_max == 10
        assert aperture.r_min == 4

        aperture = RadialAperture(r_max=5, r_min=2)
        aperture.scale(0.5)
        assert aperture.r_max == 2.5
        assert aperture.r_min == 1

    def test_to_dict(self, set_test_backend):
        aperture = RadialAperture(r_max=5, r_min=2)
        assert aperture.to_dict() == {"type": "RadialAperture", "r_max": 5, "r_min": 2}

    def test_from_dict(self, set_test_backend):
        data = {"type": "RadialAperture", "r_max": 5, "r_min": 2}
        aperture = RadialAperture.from_dict(data)
        assert aperture.r_max == 5
        assert aperture.r_min == 2
        assert isinstance(aperture, RadialAperture)

    def test_view(self):
        aperture = RadialAperture(r_max=5, r_min=2)
        fig, ax = aperture.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_extent(self, set_test_backend):
        aperture = RadialAperture(r_max=5, r_min=2)
        assert aperture.extent == (-5, 5, -5, 5)


class TestOffsetRadialAperture:
    def test_clip(self, set_test_backend):
        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        rays = RealRays(
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        aperture.clip(rays)
        assert be.all(rays.i == be.array([0, 0, 0, 1, 1, 0]))

    def test_scale(self, set_test_backend):
        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        aperture.scale(2)
        assert aperture.r_max == 10
        assert aperture.r_min == 4
        assert aperture.offset_x == 2
        assert aperture.offset_y == 2

        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        aperture.scale(0.5)
        assert aperture.r_max == 2.5
        assert aperture.r_min == 1
        assert aperture.offset_x == 0.5
        assert aperture.offset_y == 0.5

    def test_to_dict(self, set_test_backend):
        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        assert aperture.to_dict() == {
            "type": "OffsetRadialAperture",
            "r_max": 5,
            "r_min": 2,
            "offset_x": 1,
            "offset_y": 1,
        }

    def test_from_dict(self, set_test_backend):
        data = {
            "type": "OffsetRadialAperture",
            "r_max": 5,
            "r_min": 2,
            "offset_x": 1,
            "offset_y": 1,
        }
        aperture = OffsetRadialAperture.from_dict(data)
        assert aperture.r_max == 5
        assert aperture.r_min == 2
        assert aperture.offset_x == 1
        assert aperture.offset_y == 1
        assert isinstance(aperture, OffsetRadialAperture)

    def test_extent(self, set_test_backend):
        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        assert aperture.extent == (-4, 6, -4, 6)

    def test_view(self):
        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        fig, ax = aperture.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestBooleanApertures:
    def setup_method(self, set_test_backend):
        self.aperture1 = RadialAperture(r_max=1)
        self.aperture2 = RadialAperture(r_max=0.5)

    def test_union_aperture(self, set_test_backend):
        union_aperture = UnionAperture(self.aperture1, self.aperture2)
        x = be.array([0, 0.0, 0.7, 1.5])
        y = be.array([0, 0.5, 0.0, 1.5])
        result = union_aperture.contains(x, y)
        expected = be.array([True, True, True, False])
        assert be.all(result == expected)

    def test_intersection_aperture(self, set_test_backend):
        intersection_aperture = IntersectionAperture(self.aperture1, self.aperture2)
        x = be.array([0, 0.0, 0.7, 1.5])
        y = be.array([0, 0.5, 0.0, 1.5])
        result = intersection_aperture.contains(x, y)
        expected = be.array([True, True, False, False])
        assert be.all(result == expected)

    def test_difference_aperture(self, set_test_backend):
        difference_aperture = DifferenceAperture(self.aperture1, self.aperture2)
        x = be.array([0, 0.0, 0.7, 1.5])
        y = be.array([0, 0.5, 0.0, 1.5])
        result = difference_aperture.contains(x, y)
        expected = be.array([False, False, True, False])
        assert be.all(result == expected)

    def test_union_type(self, set_test_backend):
        a = self.aperture1 | self.aperture2
        assert isinstance(a, UnionAperture)

        a = self.aperture1 + self.aperture2
        assert isinstance(a, UnionAperture)

    def test_intersection_type(self, set_test_backend):
        a = self.aperture1 & self.aperture2
        assert isinstance(a, IntersectionAperture)

    def test_difference_type(self, set_test_backend):
        a = self.aperture1 - self.aperture2
        assert isinstance(a, DifferenceAperture)

    def test_extent(self, set_test_backend):
        union_aperture = UnionAperture(self.aperture1, self.aperture2)
        assert union_aperture.extent == (-1, 1, -1, 1)

        intersection_aperture = IntersectionAperture(self.aperture1, self.aperture2)
        assert intersection_aperture.extent == (-1, 1, -1, 1)

        difference_aperture = DifferenceAperture(self.aperture1, self.aperture2)
        assert difference_aperture.extent == (-1, 1, -1, 1)

    def test_view(self, set_test_backend):
        union_aperture = UnionAperture(self.aperture1, self.aperture2)
        fig, ax = union_aperture.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

        intersection_aperture = IntersectionAperture(self.aperture1, self.aperture2)
        fig, ax = intersection_aperture.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

        difference_aperture = DifferenceAperture(self.aperture1, self.aperture2)
        fig, ax = difference_aperture.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestRectangularAperture:
    def setup_method(self, set_test_backend):
        self.aperture = RectangularAperture(x_min=-1, x_max=1, y_min=-0.5, y_max=0.5)

    def test_clip(self, set_test_backend):
        rays = RealRays(
            [0, 1, 2, 3, 4, 5],
            [0, 0.5, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        self.aperture.clip(rays)
        assert be.all(rays.i == be.array([1, 1, 0, 0, 0, 0]))

    def test_scale(self, set_test_backend):
        self.aperture.scale(2)
        assert self.aperture.x_min == -2
        assert self.aperture.x_max == 2
        assert self.aperture.y_min == -1
        assert self.aperture.y_max == 1

        self.aperture.scale(0.5)
        assert self.aperture.x_min == -1
        assert self.aperture.x_max == 1
        assert self.aperture.y_min == -0.5
        assert self.aperture.y_max == 0.5

    def test_to_dict(self, set_test_backend):
        assert self.aperture.to_dict() == {
            "type": "RectangularAperture",
            "x_min": -1,
            "x_max": 1,
            "y_min": -0.5,
            "y_max": 0.5,
        }

    def test_from_dict(self, set_test_backend):
        data = {
            "type": "RectangularAperture",
            "x_min": -1,
            "x_max": 1,
            "y_min": -0.5,
            "y_max": 86,
        }
        aperture = RectangularAperture.from_dict(data)
        assert aperture.x_min == -1
        assert aperture.x_max == 1
        assert aperture.y_min == -0.5
        assert aperture.y_max == 86
        assert isinstance(aperture, RectangularAperture)

    def test_extent(self, set_test_backend):
        assert self.aperture.extent == (-1, 1, -0.5, 0.5)

    def test_view(self, set_test_backend):
        fig, ax = self.aperture.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestEllipticalAperture:
    def setup_method(self, set_test_backend):
        self.aperture = EllipticalAperture(a=1, b=0.5)

    def test_clip(self, set_test_backend):
        rays = RealRays(
            [0, 1, 0, 3, 4, 5],
            [0, 0, 0.5, 3, 4, 5],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        self.aperture.clip(rays)
        assert be.all(rays.i == be.array([1, 1, 1, 0, 0, 0]))

    def test_scale(self, set_test_backend):
        self.aperture.scale(2)
        assert self.aperture.a == 2
        assert self.aperture.b == 1

        self.aperture.scale(0.5)
        assert self.aperture.a == 1
        assert self.aperture.b == 0.5

    def test_to_dict(self, set_test_backend):
        assert self.aperture.to_dict() == {
            "type": "EllipticalAperture",
            "a": 1,
            "b": 0.5,
            "offset_x": 0,
            "offset_y": 0,
        }

    def test_from_dict(self, set_test_backend):
        data = {
            "type": "EllipticalAperture",
            "a": 1,
            "b": 0.5,
            "offset_x": 0,
            "offset_y": 0.123,
        }
        aperture = EllipticalAperture.from_dict(data)
        assert aperture.a == 1
        assert aperture.b == 0.5
        assert aperture.offset_x == 0
        assert aperture.offset_y == 0.123
        assert isinstance(aperture, EllipticalAperture)

    def test_extent(self, set_test_backend):
        assert self.aperture.extent == (-1, 1, -0.5, 0.5)

    def test_view(self, set_test_backend):
        fig, ax = self.aperture.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestPolygonAperture:
    @pytest.fixture(autouse=True)
    def setup_method(self, set_test_backend):
        self.aperture = PolygonAperture(x=[-10, 10, 10, -10], y=[-15, -15, 15, 15])

    def test_clip(self):
        rays = RealRays(
            [0, 5, 0, 9.999, 20, 20],
            [0, 0, 6, 14.999, 0, 21],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        self.aperture.clip(rays)
        assert_allclose(rays.i, [1, 1, 1, 1, 0, 0])

    def test_scale(self):
        self.aperture.scale(2)
        assert be.all(
            self.aperture.vertices
            == be.array([[-20, -30], [20, -30], [20, 30], [-20, 30]]),
        )

        self.aperture.scale(0.5)
        assert be.all(
            self.aperture.vertices
            == be.array([[-10, -15], [10, -15], [10, 15], [-10, 15]]),
        )

    def test_to_dict(self):
        data = self.aperture.to_dict()
        assert data["type"] == "PolygonAperture"
        assert be.all(data["x"] == be.array([-10, 10, 10, -10]))
        assert be.all(data["y"] == be.array([-15, -15, 15, 15]))

    def test_from_dict(self):
        data = {"type": "PolygonAperture", "x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}
        aperture = PolygonAperture.from_dict(data)
        assert_allclose(aperture.x, [0, 1, 1, 0])
        assert_allclose(aperture.y, [0, 0, 1, 1])
        assert isinstance(aperture, PolygonAperture)

    def test_extent(self):
        assert self.aperture.extent == (-10, 10, -15, 15)

    def test_view(self, set_test_backend):
        fig, ax = self.aperture.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestFileAperture:
    def setup_method(self, temp_aperture_file):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("0, 0\n1, 0\n1, 1\n0, 1")
            temp_path = f.name
        self.aperture = FileAperture(temp_path, delimiter=",")

    def test_clip(self, set_test_backend):
        # manually write the file to avoid backend issues with file creation
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("0, 0\n1, 0\n1, 1\n0, 1")
            temp_path = f.name
        aperture = FileAperture(temp_path, delimiter=",")
        rays = RealRays(
            [0.5, 0, 0.9, 10, 20, 20],
            [0.5, 0.9, 0.999, 15, 0, 21],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        aperture.clip(rays)
        assert_allclose(rays.i, [1, 1, 1, 0, 0, 0])

    def test_scale(self, set_test_backend):
        self.aperture.scale(2)
        assert be.all(
            self.aperture.vertices == be.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
        )

    def test_to_dict(self, set_test_backend):
        data = self.aperture.to_dict()
        assert data["type"] == "FileAperture"
        assert data["filepath"] == self.aperture.filepath
        assert data["delimiter"] == ","
        assert data["skip_header"] == 0
        assert be.all(data["x"] == be.array([0, 1, 1, 0]))
        assert be.all(data["y"] == be.array([0, 0, 1, 1]))

    def test_from_dict(self):
        data = {
            "type": "FileAperture",
            "filepath": self.aperture.filepath,
            "delimiter": ",",
            "skip_header": 0,
        }
        aperture = FileAperture.from_dict(data)
        assert aperture.filepath == self.aperture.filepath
        assert isinstance(aperture, FileAperture)

    def test_extent(self, set_test_backend):
        assert self.aperture.extent == (0, 1, 0, 1)


class TestConfigureAperture:
    def test_none_input(self, set_test_backend):
        assert configure_aperture(None) is None

    @pytest.mark.parametrize(
        "scalar_input, expected_r_max",
        [
            (2, 1.0),
            (0.0, 0.0),
            (3.5, 1.75),
        ],
    )
    def test_scalar_input(self, set_test_backend, scalar_input, expected_r_max):
        result = configure_aperture(scalar_input)
        assert isinstance(result, RadialAperture)
        assert_allclose(result.r_max, expected_r_max)

    def test_valid_base_aperture_instance(self, set_test_backend):
        class DummyAperture(BaseAperture):
            def contains(self, x, y):
                pass

            def extent(self):
                pass

            def scale(self, scale_factor):
                pass

        ap = DummyAperture()
        assert configure_aperture(ap) is ap

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "circle",
            [1, 2],
            {"r": 1},
            object(),
            set([1.0]),
        ],
    )
    def test_invalid_input_raises_value_error(self, set_test_backend, invalid_input):
        with pytest.raises(ValueError, match="Invalid `aperture` provided"):
            configure_aperture(invalid_input)

    def test_custom_base_aperture_subclass(self, set_test_backend):
        class CustomAperture(BaseAperture):
            def __init__(self):
                self.special = True

            def contains(self, x, y):
                pass

            def extent(self):
                pass

            def scale(self, scale_factor):
                pass

        ap = CustomAperture()
        result = configure_aperture(ap)
        assert result is ap
        assert hasattr(result, "special")
