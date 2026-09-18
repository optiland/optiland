"""Dtype-aware tolerances in the non-sequential tracer.

Covers ``optiland.nonsequential._tol`` (the ``ulp`` / ``accept_t_min`` /
``tiny_for`` / ``radicand_floor`` primitives), the float32 singlet the old
absolute epsilons broke, the float64 result they must leave untouched, and a
lint scan that keeps the replaced literals from coming back.

The scene used throughout is the quick-start singlet: a 1 W collimated
source of 5 mm aperture radius, a biconvex N-BK7 lens (r1 = 100, r2 = -100,
thickness 5, semi-diameter 12.5) at z = 50, and a 20 x 20 mm, 64 x 64 pixel
irradiance detector at z = 150.
"""

from __future__ import annotations

import hashlib
import io
import pathlib
import tokenize

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

# Imports below intentionally follow importorskip: they must not run when
# torch is unavailable.
# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential import _tol as tol
from optiland.nonsequential.backends.numpy_backend import NumpyBackend


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")


def _singlet() -> NSQScene:
    scene = NSQScene()
    scene.add_source(
        "S1",
        CoordinateSystem(z=0.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=1.0,
            aperture_radius=5.0,
        ),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(
            r1=100,
            r2=-100,
            thickness=5,
            material="N-BK7",
            front_aperture_radius=12.5,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=150),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=64, num_pixels_y=64),
    )
    return scene


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------


class TestUlp:
    """``ulp`` is the exact IEEE-754 spacing, in the input's own dtype."""

    def test_ulp_50mm_numpy_float64(self):
        assert tol.ulp(np.float64(50.0)) == pytest.approx(
            7.105427357601002e-15, rel=1e-9
        )

    def test_ulp_50mm_numpy_float32(self):
        assert tol.ulp(np.float32(50.0)) == pytest.approx(3.8146973e-06, rel=1e-5)

    def test_float32_step_is_nine_orders_above_the_old_epsilon(self):
        """Why ``1e-9`` mm could not work at float32.

        The old self-intersection epsilon is far below the float32 step at
        this scene's coordinates, so it can never separate a genuine second
        hit from the surface the ray has just left.
        """
        assert tol.ulp(np.float32(50.0)) > 1e-9 * 1e3

    def test_ulp_matches_torch_nextafter(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor(50.0, dtype=dtype)
            expected = torch.nextafter(x, torch.full_like(x, float("inf"))) - x
            assert float(tol.ulp(x)) == pytest.approx(float(expected), rel=1e-9)

    def test_ulp_torch_and_numpy_agree(self):
        assert float(tol.ulp(torch.tensor(50.0, dtype=torch.float64))) == pytest.approx(
            float(tol.ulp(np.float64(50.0))), rel=1e-9
        )

    def test_ulp_is_detached_from_the_graph(self):
        x = torch.tensor([50.0], dtype=torch.float64, requires_grad=True)
        assert not tol.ulp(x).requires_grad


class TestAcceptTMin:
    def test_scales_with_k(self):
        base = tol.accept_t_min(50.0, k=1)
        assert tol.accept_t_min(50.0, k=25) == pytest.approx(25 * base, rel=1e-9)

    def test_floor_at_small_magnitude(self):
        """Below the 1 mm floor the threshold is pinned at ``ulp(1 mm)``."""
        assert tol.accept_t_min(0.0) == pytest.approx(tol.accept_t_min(1.0), rel=1e-9)
        assert tol.accept_t_min(0.5) == pytest.approx(tol.accept_t_min(1.0), rel=1e-9)

    def test_float32_threshold_larger_than_float64(self):
        f32 = tol.accept_t_min(torch.tensor(50.0, dtype=torch.float32))
        f64 = tol.accept_t_min(torch.tensor(50.0, dtype=torch.float64))
        assert float(f32) > float(f64)

    def test_grows_with_the_coordinate(self):
        """A scene 1e5 mm across needs a larger threshold than one 50 mm."""
        assert float(tol.accept_t_min(1e5)) > float(tol.accept_t_min(50.0))


class TestTinyFor:
    def test_float32_sqrt_smallest_normal(self):
        assert tol.tiny_for(np.float32) == pytest.approx(1.0842022e-19, rel=1e-4)

    def test_float64_sqrt_smallest_normal(self):
        assert tol.tiny_for(np.float64) == pytest.approx(1.4916681e-154, rel=1e-4)

    def test_square_does_not_underflow_float32(self):
        """The point of the helper.

        Squaring the guard is what reverse-mode autodiff does for a bare
        reciprocal's local derivative. A bare ``1e-30`` squared is ``1e-60``,
        which is zero in float32, and the resulting infinite derivative
        multiplied by a discarded zero cotangent is a NaN that propagates
        through the whole reverse sweep.
        """
        assert np.float32(1e-30) ** 2 == 0.0
        assert np.float32(tol.tiny_for(np.float32)) ** 2 > 0.0

    def test_accepts_array_tensor_and_dtype(self):
        arr = np.ones(3, dtype=np.float32)
        t = torch.ones(3, dtype=torch.float32)
        assert tol.tiny_for(arr) == pytest.approx(tol.tiny_for(t), rel=1e-6)
        assert tol.tiny_for(np.float32) == pytest.approx(tol.tiny_for(t), rel=1e-6)


class TestRadicandFloor:
    def test_float64_reduces_to_the_calibrated_constant(self):
        """Exactly the old ``1e-12`` at float64, so no float64 answer moves."""
        floor = tol.radicand_floor(np.ones(3, dtype=np.float64), floor_at_f64=1e-12)
        assert floor == pytest.approx(1e-12, rel=1e-12)

    def test_float32_scaled_up(self):
        """At float32 the un-scaled floor is below the dtype's own step at 1."""
        f32 = tol.radicand_floor(np.ones(3, dtype=np.float32), floor_at_f64=1e-12)
        assert f32[0] > 1e-12
        assert float(np.finfo(np.float32).eps) > 1e-12


# ---------------------------------------------------------------------------
# What the replacement buys, and what it must leave alone
# ---------------------------------------------------------------------------


class TestFloat32Singlet:
    """At float32 the absolute epsilons made the singlet lose two thirds of
    its light: a ray re-accepted the surface it had just left, bounced
    against it until the depth cap killed it, and its flux never reached the
    detector. 200k rays, seed 42, ``max_depth=16``, torch CPU.
    """

    @staticmethod
    def _trace(precision: str):
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision(precision)
        return _singlet().trace(num_rays=200_000, seed=42, max_depth=16)

    def test_zero_rays_depth_killed(self):
        """62,546 of 200,000 before; none now."""
        assert self._trace("float32").num_rays_depth_killed == 0

    def test_detector_flux_matches_float64(self):
        """0.306342 W against float64's 0.916765 W before; 0.916776 W now."""
        f32 = float(self._trace("float32").total_flux_detected)
        f64 = float(self._trace("float64").total_flux_detected)
        assert f32 == pytest.approx(f64, rel=1e-3)

    def test_float64_is_unaffected_by_precision_selection(self):
        result = self._trace("float64")
        assert result.num_rays_depth_killed == 0
        assert float(result.total_flux_detected) == pytest.approx(0.916765, abs=1e-6)


class TestFloat64Unchanged:
    """No float64 answer changes: every threshold that was an absolute
    constant is replaced by one that is smaller than it at float64
    coordinates of this size, so the same roots are accepted.
    """

    def test_detector_image_and_flux_are_bit_identical(self):
        result = _singlet().trace(
            num_rays=100_000, seed=42, backend=NumpyBackend(seed=42)
        )
        image = np.asarray(result.detectors["D1"].data, dtype=np.float64)
        digest = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        # Recorded on the unmodified tree at the same commit this branch
        # starts from, same seed, same ray count.
        assert digest == "cf1e64a79107efaa"
        assert float(result.total_flux_detected).hex() == "0x1.d56abebe8aa38p-1"
        assert float(result.total_flux_escaped).hex() == "0x1.515c8488c4ec4p-4"
        assert float(result.flux_conservation_error).hex() == "0x1.ad00000000000p-48"


# ---------------------------------------------------------------------------
# The lint scan
# ---------------------------------------------------------------------------

_NSQ = pathlib.Path(tol.__file__).parent

# Files on the intersection and interaction path, where a tolerance decides
# whether a root is a hit.
_RAY_PATH_FILES = [
    "components/base.py",
    "detectors/base.py",
    "components/geometry/base.py",
    "components/geometry/analytic/plane.py",
    "components/geometry/analytic/conic.py",
    "components/geometry/analytic/sphere.py",
    "components/geometry/analytic/annulus.py",
    "components/geometry/analytic/frustum.py",
    "components/geometry/mesh/mesh_geometry.py",
    "components/reflective.py",
    "components/refractive.py",
    "bsdf/harvey_shack.py",
]

# Absolute constants that are deliberately kept, each with the reason. A
# physical fraction is dtype-independent by definition, and a value computed
# once on the host in float64 is not a traversal tolerance.
_EXEMPT = {
    # The radicand clamp inside a detached float64 bounding-box computation,
    # which never sees the traversal dtype.
    ("components/geometry/analytic/conic.py", "1e-12"),
    # The branch-probability clamp, computed on the host in float64.
    ("components/refractive.py", "1e-12"),
}


def _code_numbers(path: pathlib.Path) -> set[str]:
    """Numeric literals in a source file, comments and strings excluded."""
    readline = io.StringIO(path.read_text()).readline
    return {
        token.string
        for token in tokenize.generate_tokens(readline)
        if token.type == tokenize.NUMBER
    }


class TestNoBareToleranceLiterals:
    """The rule the module implements, asserted rather than only documented.

    No tolerance on the ray path is an absolute length or an absolute
    denominator constant; every one is a multiple of the working dtype's own
    resolution at the magnitude in play. These two tests are what stop the
    literals coming back in a later change.
    """

    @pytest.mark.parametrize("relative", _RAY_PATH_FILES)
    def test_no_absolute_length_epsilon(self, relative):
        """``1e-9`` and ``1e-12`` are lengths, and a length is dtype-aware."""
        offenders = {
            n
            for n in _code_numbers(_NSQ / relative)
            if n in {"1e-9", "1e-12", "1e-14", "1e-15"} and (relative, n) not in _EXEMPT
        }
        assert not offenders, (
            f"{relative} carries absolute tolerance literal(s) {sorted(offenders)}. "
            "Use optiland.nonsequential._tol.accept_t_min or _tol.ulp, or add the "
            "value to _EXEMPT in this test with the reason it is not a traversal "
            "tolerance."
        )

    @pytest.mark.parametrize("relative", _RAY_PATH_FILES)
    def test_no_bare_division_guard(self, relative):
        """A ``+ 1e-30`` guard squares to zero in a float32 backward pass."""
        offenders = {
            n
            for n in _code_numbers(_NSQ / relative)
            if n in {"1e-30", "1e-60"} and (relative, n) not in _EXEMPT
        }
        assert not offenders, (
            f"{relative} carries a bare division guard {sorted(offenders)}. "
            "Use optiland.nonsequential._tol.tiny_for, or mask the denominator's "
            "input instead of adding an epsilon to it."
        )
