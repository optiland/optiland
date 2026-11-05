"""Tests for the pluggable component and ray coloring."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import optiland.backend as be
from optiland.samples.objectives import CookeTriplet
from optiland.samples.simple import CementedAchromat
from optiland.samples.telescopes import HubbleTelescope
from optiland.visualization.coloring import (
    ColorScale,
    ComponentColorizer,
    ComponentRef,
    PropertyColorizer,
    RayColorizer,
    ThemeColorizer,
    UniformColorizer,
    UniformRayColorizer,
)
from optiland.visualization.system import OpticViewer3D

matplotlib.use("Agg")

VIRIDIS = matplotlib.colormaps["viridis"]


def _draw(optic, **kwargs):
    kwargs.setdefault("show_apertures", False)
    fig, ax = optic.draw(show=False, **kwargs)
    return fig, ax


def _patch_colors(ax):
    return [(p.get_facecolor(), p.get_edgecolor()) for p in ax.patches]


def _ray_lines(ax):
    return [line for line in ax.lines if not line.get_label().startswith("Surface")]


def _line_colors(ax):
    return [to_rgba(line.get_color()) for line in ax.lines]


class TestColorScale:
    def test_resolved_fills_limits(self):
        scale = ColorScale().resolved([1.0, np.nan, 3.0, 2.0])
        assert (scale.vmin, scale.vmax) == (1.0, 3.0)

    def test_resolved_keeps_user_limits(self):
        scale = ColorScale(vmin=0.0).resolved([1.0, 3.0])
        assert (scale.vmin, scale.vmax) == (0.0, 3.0)

    def test_resolved_without_finite_values_raises(self):
        with pytest.raises(ValueError, match="no finite values"):
            ColorScale().resolved([np.nan, np.inf])

    def test_degenerate_range_maps_to_midpoint(self):
        rgba = ColorScale().resolved([2.0, 2.0]).to_rgba([2.0])
        np.testing.assert_allclose(rgba[0], VIRIDIS(0.5))

    def test_log_with_non_positive_values_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ColorScale(log=True).resolved([0.0, 1.0])
        with pytest.raises(ValueError, match="positive"):
            ColorScale(log=True, vmin=-1.0).resolved([1.0, 2.0])

    def test_log_maps_geometric_midpoint_to_midpoint(self):
        rgba = ColorScale(log=True).resolved([1.0, 100.0]).to_rgba([10.0])
        np.testing.assert_allclose(rgba[0], VIRIDIS(0.5))

    def test_non_finite_values_give_nan_rows(self):
        rgba = ColorScale().resolved([0.0, 1.0]).to_rgba([0.0, np.nan, 1.0])
        assert rgba.shape == (3, 4)
        assert np.isnan(rgba[1]).all()
        assert np.isfinite(rgba[[0, 2]]).all()

    def test_values_outside_limits_are_clipped(self):
        rgba = ColorScale(vmin=0.0, vmax=1.0).to_rgba([-5.0, 5.0])
        np.testing.assert_allclose(rgba, [VIRIDIS(0.0), VIRIDIS(1.0)])

    def test_unresolved_to_rgba_raises(self):
        with pytest.raises(ValueError, match="resolved"):
            ColorScale().to_rgba([1.0])

    def test_add_colorbar_sets_label(self):
        fig, ax = plt.subplots()
        colorbar = ColorScale(label="value").resolved([1.0, 2.0]).add_colorbar(fig, ax)
        assert colorbar.ax.get_ylabel() == "value"
        plt.close(fig)

    def test_scalar_bar_actor(self):
        bar = ColorScale(label="value", log=True).resolved([1.0, 10.0])
        actor = bar.scalar_bar_actor()
        assert actor.GetTitle() == "value"
        assert actor.GetLookupTable().GetTableRange() == (1.0, 10.0)


class TestComponentColoring:
    def test_theme_colorizer_matches_default(self, set_test_backend):
        fig_a, ax_a = _draw(CookeTriplet(), reference="chief")
        fig_b, ax_b = _draw(
            CookeTriplet(), reference="chief", component_colorizer=ThemeColorizer()
        )
        assert _patch_colors(ax_a) == _patch_colors(ax_b)
        assert _line_colors(ax_a) == _line_colors(ax_b)
        plt.close("all")

    def test_uniform_colorizer_changes_faces_only(self, set_test_backend):
        _, base = _draw(CookeTriplet())
        _, ax = _draw(CookeTriplet(), component_colorizer=UniformColorizer("red"))
        assert len(ax.patches) == len(base.patches) == 3
        for patch_, (_, edge) in zip(ax.patches, _patch_colors(base), strict=True):
            assert patch_.get_facecolor() == to_rgba("red")
            assert patch_.get_edgecolor() == edge
        assert _line_colors(ax)[-1] != _line_colors(base)[-1]  # image plane line
        assert [to_rgba(line.get_color()) for line in _ray_lines(ax)] == [
            to_rgba(line.get_color()) for line in _ray_lines(base)
        ]
        plt.close("all")

    def test_property_colorizer_orders_colors_and_adds_colorbar(self, set_test_backend):
        colorizer = PropertyColorizer(lambda optic, ref: ref.surface_indices[0])
        fig, ax = _draw(CookeTriplet(), component_colorizer=colorizer)
        faces = [p.get_facecolor() for p in ax.patches]
        np.testing.assert_allclose(faces, [VIRIDIS(0.0), VIRIDIS(0.5), VIRIDIS(1.0)])
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_colorbar_can_be_hidden(self, set_test_backend):
        colorizer = PropertyColorizer(lambda optic, ref: ref.surface_indices[0])
        fig, _ = _draw(
            CookeTriplet(), component_colorizer=colorizer, show_colorbar=False
        )
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_none_value_keeps_default_color(self, set_test_backend):
        _, base = _draw(CookeTriplet())

        def value_fn(optic, ref):
            return None if ref.surface_indices[0] == 3 else ref.surface_indices[0]

        _, ax = _draw(CookeTriplet(), component_colorizer=PropertyColorizer(value_fn))
        faces = [p.get_facecolor() for p in ax.patches]
        assert faces[1] == base.patches[1].get_facecolor()
        assert faces[0] != base.patches[0].get_facecolor()
        np.testing.assert_allclose(faces[0], VIRIDIS(0.0))
        np.testing.assert_allclose(faces[2], VIRIDIS(1.0))
        plt.close("all")

    def test_cemented_doublet_elements_get_distinct_colors(self, set_test_backend):
        seen = []

        def value_fn(optic, ref):
            seen.append(ref)
            return ref.surface_indices[0]

        _, ax = _draw(
            CementedAchromat(), component_colorizer=PropertyColorizer(value_fn)
        )
        assert [r.surface_indices for r in seen] == [(1, 2), (2, 3)]
        faces = [p.get_facecolor() for p in ax.patches]
        assert len(faces) == 2
        assert faces[0] != faces[1]
        plt.close("all")

    def test_mirror_kind_colors_mirrors_only(self, set_test_backend):
        optic = HubbleTelescope()
        reflective = [
            k for k, s in enumerate(optic.surfaces) if s.interaction_model.is_reflective
        ]
        colorizer = PropertyColorizer(lambda optic, ref: 1.0, kinds=("mirror",))
        _, ax = _draw(optic, component_colorizer=colorizer)
        colored = [c for c in _line_colors(ax) if c == to_rgba(VIRIDIS(0.5))]
        assert len(reflective) > 0
        assert len(colored) == len(reflective)
        plt.close("all")

    def test_no_matching_kind_shows_no_colorbar(self, set_test_backend):
        colorizer = PropertyColorizer(lambda optic, ref: 1.0, kinds=("mirror",))
        fig, ax = _draw(CookeTriplet(), component_colorizer=colorizer)
        assert colorizer.scale is None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_value_fn_errors_propagate(self, set_test_backend):
        def boom(optic, ref):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _draw(CookeTriplet(), component_colorizer=PropertyColorizer(boom))
        plt.close("all")

    def test_tensor_values_leave_no_graph_references(self, set_test_backend):
        def value_fn(optic, ref):
            value = be.array(float(ref.surface_indices[0]))
            if be.get_backend() == "torch":
                value.requires_grad_(True)
            return value * 2.0

        colorizer = PropertyColorizer(value_fn)
        _, ax = _draw(CookeTriplet(), component_colorizer=colorizer)
        assert len(ax.patches) == 3
        for color in colorizer._colors.values():
            assert all(type(channel) is float for channel in color)
        assert type(colorizer.scale.vmin) is float
        plt.close("all")

    def test_custom_colorizer_receives_component_refs(self, set_test_backend):
        class Recorder(ComponentColorizer):
            def prepare(self, optic, refs):
                self.refs = list(refs)

            def color_for(self, ref):
                return None

        recorder = Recorder()
        _draw(CookeTriplet(), component_colorizer=recorder)
        kinds = {ref.kind for ref in recorder.refs}
        assert ComponentRef("lens_element", (1, 2)) in recorder.refs
        assert kinds == {"lens_element", "surface"}
        plt.close("all")


class TestRayColoring:
    def test_uniform_ray_colorizer_colors_all_rays(self, set_test_backend):
        _, ax = _draw(
            CookeTriplet(), reference="chief", ray_colorizer=UniformRayColorizer("k")
        )
        rays = _ray_lines(ax)
        assert len(rays) > 3
        assert all(to_rgba(line.get_color()) == to_rgba("k") for line in rays)
        plt.close("all")

    def test_bundle_reference_describes_each_bundle(self, set_test_backend):
        seen = []

        class Recorder(RayColorizer):
            def ray_colors(self, bundle, rays):
                seen.append((bundle, rays))
                return None

        _, ax = _draw(CookeTriplet(), reference="chief", ray_colorizer=Recorder())
        bundles = [b for b, _ in seen]
        assert [b.is_reference for b in bundles[:2]] == [False, True]
        assert {b.num_fields for b in bundles} == {3}
        assert {b.num_wavelengths for b in bundles} == {1}
        rays = seen[0][1]
        assert rays.x.shape == rays.z.shape == rays.intensity.shape
        assert isinstance(rays.z, np.ndarray)
        plt.close("all")

    def test_nan_rows_keep_default_color(self, set_test_backend):
        class FirstRayRed(RayColorizer):
            def ray_colors(self, bundle, rays):
                colors = np.full((rays.num_rays, 4), np.nan)
                colors[0] = to_rgba("red")
                return colors

        _, base = _draw(CookeTriplet())
        _, ax = _draw(CookeTriplet(), ray_colorizer=FirstRayRed())
        base_colors = [to_rgba(line.get_color()) for line in _ray_lines(base)]
        colors = [to_rgba(line.get_color()) for line in _ray_lines(ax)]
        assert colors.count(to_rgba("red")) == 3  # one per field
        assert [c for c in colors if c != to_rgba("red")] == [
            c for c, b in zip(base_colors, colors, strict=True) if b != to_rgba("red")
        ]
        plt.close("all")

    def test_wrong_shape_raises(self, set_test_backend):
        class Bad(RayColorizer):
            def ray_colors(self, bundle, rays):
                return np.zeros((1, 4))

        with pytest.raises(ValueError, match="expected"):
            _draw(CookeTriplet(), ray_colorizer=Bad())
        plt.close("all")


class TestColoring3D:
    @staticmethod
    def _view(optic, **kwargs):
        viewer = OpticViewer3D(optic)
        with (
            patch.object(viewer.iren, "Start"),
            patch.object(viewer.ren_win, "Render"),
        ):
            viewer.view(num_rays=6, **kwargs)
        return viewer.ren_win.GetRenderers().GetFirstRenderer()

    @staticmethod
    def _actors(renderer, collection):
        actors = getattr(renderer, collection)()
        actors.InitTraversal()
        return [actors.GetNextItemAsObject() for _ in range(actors.GetNumberOfItems())]

    def test_component_colors_and_scalar_bar(self, set_test_backend):
        colorizer = PropertyColorizer(lambda optic, ref: ref.surface_indices[0])
        renderer = self._view(CookeTriplet(), component_colorizer=colorizer)
        glass = {
            tuple(round(c, 4) for c in actor.GetProperty().GetColor())
            for actor in self._actors(renderer, "GetActors")
            if actor.GetProperty().GetOpacity() == 0.5
        }
        expected = {tuple(round(c, 4) for c in VIRIDIS(f)[:3]) for f in (0.0, 0.5, 1.0)}
        assert expected <= glass
        assert len(self._actors(renderer, "GetActors2D")) == 1

    def test_scalar_bar_can_be_hidden(self, set_test_backend):
        colorizer = PropertyColorizer(lambda optic, ref: ref.surface_indices[0])
        renderer = self._view(
            CookeTriplet(), component_colorizer=colorizer, show_colorbar=False
        )
        assert self._actors(renderer, "GetActors2D") == []

    def test_ray_colors(self, set_test_backend):
        renderer = self._view(CookeTriplet(), ray_colorizer=UniformRayColorizer("red"))
        lines = [
            a
            for a in self._actors(renderer, "GetActors")
            if a.GetProperty().GetOpacity() == 1.0
            and a.GetProperty().GetColor() == (1.0, 0.0, 0.0)
        ]
        assert len(lines) > 0
