"""Camera presets and off-screen PNG export for 3D viewers.

Covers issue #1047: default camera regression, named presets, and
headless save_image output (size, scale, transparency).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from optiland.nonsequential.visualization import NSQViewer3D
from optiland.samples.objectives import ReverseTelephoto
from optiland.visualization.base import BaseViewer3D
from optiland.visualization.system import OpticViewer3D

# Camera recorded on upstream master with ReverseTelephoto.view() defaults
# after a real render (interactor Start patched). Used to lock the default
# preset so view() with no new arguments looks as before.
_DEFAULT_POSITION = (-10.493988086870143, 0.014922022819519043, -3.6040260761915963)
_DEFAULT_FOCAL = (0.0, 0.014922022819519043, 2.454680770635605)
_DEFAULT_VIEW_UP = (0.0, 1.0, 0.0)

_PRESETS = {
    "side": {"looks_from": (1.0, 0.0, 0.0), "view_up": (0.0, 1.0, 0.0)},
    "top": {"looks_from": (0.0, 1.0, 0.0), "view_up": (-1.0, 0.0, 0.0)},
    "front": {"looks_from": (0.0, 0.0, -1.0), "view_up": (0.0, 1.0, 0.0)},
    "iso": {"looks_from": (1.0, 1.0, -1.0), "view_up": (0.0, 1.0, 0.0)},
}


def _camera(viewer):
    renderer = viewer.ren_win.GetRenderers().GetFirstRenderer()
    return renderer.GetActiveCamera()


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    assert norm > 0
    return vector / norm


def _assert_parallel(actual, expected, atol=1e-6):
    assert np.allclose(_unit(actual), _unit(expected), atol=atol), (
        actual,
        expected,
    )


def _headless_view(viewer, **kwargs):
    with patch.object(viewer.iren, "Start") as mock_start:
        viewer.view(**kwargs)
        mock_start.assert_called_once()


def _minimal_nsq_scene():
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential import (
        CollimatedSourceConfig,
        IrradianceDetectorConfig,
        LensConfig,
        NSQScene,
        Spectrum,
    )

    scene = NSQScene()
    spectrum = Spectrum.monochromatic(0.55)
    scene.add_source(
        "S1",
        CoordinateSystem(z=-15.0),
        CollimatedSourceConfig(spectrum=spectrum, total_flux=1.0, aperture_radius=5.0),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=0.0),
        LensConfig(
            r1=50.0,
            r2=-50.0,
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=10.0,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=60.0),
        IrradianceDetectorConfig(
            width=5.0, height=5.0, num_pixels_x=64, num_pixels_y=64
        ),
    )
    return scene


class TestApplyCamera:
    def test_unknown_preset_lists_valid_names(self):
        viewer = OpticViewer3D(ReverseTelephoto())
        renderer = viewer._make_renderer()
        with pytest.raises(ValueError, match="Unknown camera preset") as info:
            viewer._apply_camera(renderer, "diagonal")
        message = str(info.value)
        for name in BaseViewer3D._CAMERA_PRESETS:
            assert repr(name) in message


class TestOpticViewer3DCamera:
    def test_default_camera_matches_legacy_look(self, set_test_backend):
        viewer = OpticViewer3D(ReverseTelephoto())
        _headless_view(viewer)
        cam = _camera(viewer)
        assert np.allclose(cam.GetPosition(), _DEFAULT_POSITION, atol=1e-6)
        assert np.allclose(cam.GetFocalPoint(), _DEFAULT_FOCAL, atol=1e-6)
        assert np.allclose(cam.GetViewUp(), _DEFAULT_VIEW_UP, atol=1e-6)

    def test_view_without_camera_arg_is_default(self, set_test_backend):
        a = OpticViewer3D(ReverseTelephoto())
        b = OpticViewer3D(ReverseTelephoto())
        _headless_view(a)
        _headless_view(b, camera="default")
        cam_a, cam_b = _camera(a), _camera(b)
        assert np.allclose(cam_a.GetPosition(), cam_b.GetPosition(), atol=1e-6)
        assert np.allclose(cam_a.GetFocalPoint(), cam_b.GetFocalPoint(), atol=1e-6)
        assert np.allclose(cam_a.GetViewUp(), cam_b.GetViewUp(), atol=1e-6)

    @pytest.mark.parametrize("preset", list(_PRESETS))
    def test_preset_direction_and_view_up(self, set_test_backend, preset):
        viewer = OpticViewer3D(ReverseTelephoto())
        _headless_view(viewer, camera=preset)
        cam = _camera(viewer)
        looks_from = np.asarray(cam.GetPosition()) - np.asarray(cam.GetFocalPoint())
        _assert_parallel(looks_from, _PRESETS[preset]["looks_from"])
        _assert_parallel(cam.GetViewUp(), _PRESETS[preset]["view_up"])

    def test_view_unknown_camera_raises(self, set_test_backend):
        viewer = OpticViewer3D(ReverseTelephoto())
        with pytest.raises(ValueError, match="Unknown camera preset"):
            _headless_view(viewer, camera="birdseye")


class TestOpticViewer3DSaveImage:
    def test_writes_png_of_requested_size(self, set_test_backend, tmp_path):
        viewer = OpticViewer3D(ReverseTelephoto())
        path = viewer.save_image(
            tmp_path / "system.png", figsize=(200, 100), num_rays=6
        )
        assert path == tmp_path / "system.png"
        assert path.is_file()
        with Image.open(path) as image:
            assert image.size == (200, 100)
        # Never starts the interactor; attributes point at the used window.
        assert viewer.ren_win is not None
        assert viewer.iren is not None
        assert viewer.ren_win.GetOffScreenRendering() == 1

    def test_scale_multiplies_output_size(self, set_test_backend, tmp_path):
        viewer = OpticViewer3D(ReverseTelephoto())
        path = viewer.save_image(
            tmp_path / "scaled.png", figsize=(200, 100), scale=2, num_rays=6
        )
        with Image.open(path) as image:
            assert image.size == (400, 200)

    def test_transparent_background_is_rgba(self, set_test_backend, tmp_path):
        viewer = OpticViewer3D(ReverseTelephoto())
        path = viewer.save_image(
            tmp_path / "clear.png",
            figsize=(200, 100),
            transparent=True,
            num_rays=6,
        )
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            pixels = np.array(image)
        # A corner pixel should be the clear background, not a lens.
        assert pixels[0, 0, 3] == 0

    def test_save_image_applies_camera_preset(self, set_test_backend, tmp_path):
        viewer = OpticViewer3D(ReverseTelephoto())
        viewer.save_image(
            tmp_path / "side.png", camera="side", figsize=(200, 100), num_rays=6
        )
        cam = _camera(viewer)
        looks_from = np.asarray(cam.GetPosition()) - np.asarray(cam.GetFocalPoint())
        _assert_parallel(looks_from, _PRESETS["side"]["looks_from"])
        _assert_parallel(cam.GetViewUp(), _PRESETS["side"]["view_up"])

    def test_view_after_save_image_renders_onscreen(self, set_test_backend, tmp_path):
        """view() following save_image() must not stay off-screen.

        save_image() marks the reused ``ren_win`` for off-screen rendering;
        a later interactive ``view()`` must turn that back off.
        """
        viewer = OpticViewer3D(ReverseTelephoto())
        viewer.save_image(tmp_path / "a.png", figsize=(200, 100), num_rays=6)
        assert viewer.ren_win.GetOffScreenRendering() == 1
        _headless_view(viewer, num_rays=6)
        assert viewer.ren_win.GetOffScreenRendering() == 0

    def test_save_image_missing_parent_dir_raises(self, set_test_backend, tmp_path):
        """A missing destination directory must raise OSError.

        vtkPNGWriter silently does nothing when the parent directory does
        not exist; save_image() must report the failure instead of
        returning a path to a file that was never written.
        """
        viewer = OpticViewer3D(ReverseTelephoto())
        target = tmp_path / "missing" / "x.png"
        with pytest.raises(OSError, match="failed to write PNG"):
            viewer.save_image(target, figsize=(200, 100), num_rays=6)
        assert not target.exists()
        assert not target.parent.exists()


class TestNSQViewer3DCameraAndExport:
    def test_default_camera_keeps_auto_view(self, set_test_backend, tmp_path):
        """NSQ default camera must keep the pre-preset auto view.

        Before camera presets existed, NSQViewer3D set no camera at all
        and let VTK's automatic first-render ResetCamera frame the scene.
        """
        viewer = NSQViewer3D(_minimal_nsq_scene())
        viewer.save_image(tmp_path / "auto.png", figsize=(160, 80), num_rays=0)
        cam = _camera(viewer)
        assert np.allclose(cam.GetDirectionOfProjection(), (0.0, 0.0, -1.0), atol=1e-6)
        assert np.allclose(cam.GetViewUp(), (0.0, 1.0, 0.0), atol=1e-6)

    def test_preset_direction_and_view_up(self, set_test_backend):
        viewer = NSQViewer3D(_minimal_nsq_scene())
        window, interactor = None, None

        def _capture(renderer, figsize=(1200, 800), title="NSQ Scene -- 3D View"):
            nonlocal window, interactor
            window, interactor = BaseViewer3D._make_window(
                viewer, renderer, figsize, title
            )
            interactor.Start = lambda: None
            return window, interactor

        viewer._make_window = _capture
        viewer.view(num_rays=0, camera="front")
        cam = window.GetRenderers().GetFirstRenderer().GetActiveCamera()
        looks_from = np.asarray(cam.GetPosition()) - np.asarray(cam.GetFocalPoint())
        _assert_parallel(looks_from, _PRESETS["front"]["looks_from"])
        _assert_parallel(cam.GetViewUp(), _PRESETS["front"]["view_up"])

    def test_save_image_size_and_no_interactor(self, set_test_backend, tmp_path):
        viewer = NSQViewer3D(_minimal_nsq_scene())
        started = []

        def _wrap(renderer, figsize=(1200, 800), title="NSQ Scene -- 3D View"):
            window, interactor = BaseViewer3D._make_window(
                viewer, renderer, figsize, title
            )
            original_start = interactor.Start

            def _start():
                started.append(True)
                return original_start()

            interactor.Start = _start
            return window, interactor

        viewer._make_window = _wrap
        path = viewer.save_image(tmp_path / "nsq.png", figsize=(160, 80), num_rays=0)
        assert path.is_file()
        assert started == []
        with Image.open(path) as image:
            assert image.size == (160, 80)
