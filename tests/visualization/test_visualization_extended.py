from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest

from optiland.visualization.info import providers
from optiland.visualization.system import interaction, lens, ray_bundle, surface


@pytest.fixture
def mock_fig_ax():
    fig, ax = plt.subplots()
    return fig, ax


@pytest.fixture
def mock_optic():
    optic = MagicMock()
    optic.surfaces = MagicMock()
    optic.surfaces = []
    return optic


class TestInfoProviders:
    def test_surface_info_provider(self, mock_optic):
        # Mock surface
        surf_2d = MagicMock(spec=surface.Surface2D)
        real_surf = MagicMock()
        real_surf.is_stop = True
        real_surf.comment = "Test Surface"
        real_surf.geometry.radius = 100.0
        real_surf.geometry.conic = 0.0
        real_surf.thickness = 5.0
        real_surf.material_post.name = "Glass"
        surf_2d.surf = real_surf

        # Add to group so index can be found
        mock_optic.surfaces = [real_surf]
        provider = providers.SurfaceInfoProvider(mock_optic.surfaces)

        info = provider.get_info(surf_2d)
        assert "Surface: 0 (STOP)" in info
        assert "Comment: Test Surface" in info
        assert "Radius: 100.000" in info
        assert "Material: Glass" in info

    def test_lens_info_provider(self, mock_optic):
        # Mock Lens2D
        lens_2d = MagicMock(spec=lens.Lens2D)
        s1, s2 = MagicMock(), MagicMock()
        s1.surf = MagicMock()
        s2.surf = MagicMock()
        lens_2d.surfaces = [s1, s2]

        mock_optic.surfaces = [s1.surf, s2.surf]
        provider = providers.LensInfoProvider(mock_optic.surfaces)

        # Mock material info
        s1.surf.material_post.n.return_value.item.return_value = 1.5
        s1.surf.material_post.abbe.return_value.item.return_value = 60.0
        s1.surf.thickness = 10.0

        info = provider.get_info(lens_2d)
        assert "Lens (Surfaces: 0-1)" in info
        assert "Material: n_d=1.5000, Vd=60.0" in info
        assert "Center Thickness: 10.000" in info

    def test_ray_bundle_info_provider(self):
        provider = providers.RayBundleInfoProvider()
        bundle = MagicMock(spec=ray_bundle.RayBundle)
        bundle.field = (0.0, 1.0)
        bundle.wavelength = 0.55

        info = provider.get_info(bundle)
        assert "Ray Bundle" in info
        assert "Field: (0.00, 1.00)" in info
        assert "Wavelength: 0.6 nm" in info  # 0.55 rounds to 0.6 with .1f


class TestInteractionManager:
    def test_init(self, mock_fig_ax, mock_optic):
        fig, ax = mock_fig_ax
        manager = interaction.InteractionManager(fig, ax, mock_optic)
        assert manager.tooltip.get_visible() == False

    def test_register_artist(self, mock_fig_ax, mock_optic):
        fig, ax = mock_fig_ax
        manager = interaction.InteractionManager(fig, ax, mock_optic)
        artist = MagicMock()
        obj = MagicMock()
        manager.register_artist(artist, obj)
        assert manager.artist_registry[artist] == obj

    def test_on_hover_no_match(self, mock_fig_ax, mock_optic):
        fig, ax = mock_fig_ax
        manager = interaction.InteractionManager(fig, ax, mock_optic)

        event = MagicMock()
        event.inaxes = ax
        # No artists registered
        manager.on_hover(event)
        assert manager.active_artist is None

    def test_on_hover_match(self, mock_fig_ax, mock_optic):
        fig, ax = mock_fig_ax
        manager = interaction.InteractionManager(fig, ax, mock_optic)

        artist = MagicMock()
        artist.contains.return_value = (True, {})
        obj = MagicMock()
        manager.register_artist(artist, obj)

        event = MagicMock()
        event.inaxes = ax
        event.xdata = 0
        event.ydata = 0

        # Should trigger timer
        with patch("optiland.visualization.system.interaction.Timer") as MockTimer:
            manager.on_hover(event)
            assert manager.active_artist == artist
            MockTimer.assert_called()

    def test_show_tooltip(self, mock_fig_ax, mock_optic):
        fig, ax = mock_fig_ax
        manager = interaction.InteractionManager(fig, ax, mock_optic)

        artist = MagicMock()
        # Mock get_linewidth to avoid errors in highlight
        artist.get_linewidth.return_value = 1.0

        # Register a surface
        surf = MagicMock(spec=surface.Surface2D)
        surf.surf = MagicMock()
        manager.register_artist(artist, surf)

        event = MagicMock()
        event.xdata, event.ydata = 0, 0

        # Need to ensure SurfaceInfoProvider can run
        with patch(
            "optiland.visualization.info.providers.SurfaceInfoProvider.get_info",
            return_value="Tooltip Info",
        ):
            manager.show_tooltip(artist, event)
            assert manager.tooltip.get_text() == "Tooltip Info"
            assert manager.tooltip.get_visible() == True

    def test_info_panel(self, mock_fig_ax, mock_optic):
        fig, ax = mock_fig_ax
        manager = interaction.InteractionManager(fig, ax, mock_optic)

        # Test showing panel
        with patch.object(manager, "get_info_text", return_value="Panel Info"):
            manager.show_info_panel(MagicMock())
            assert manager.info_panel is not None

            # Test closing panel via method
            manager.close_info_panel()
            assert manager.info_panel is None
