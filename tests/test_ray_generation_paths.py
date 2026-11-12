"""Tests for the field runs of angle-field ray generation.

The field runs (``optiland.fields.field_types._runs``) are a fast path of
``AngleField.get_ray_origins``. They exist only on the NumPy backend, so
these tests run on NumPy. A fast path must give the bits of the generic
code, so the tests compare the rays with and without runs bit for bit.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
import optiland.fields.field_types._runs as runs
import optiland.fields.field_types.angle as angle
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.paraxial_path import UnsupportedParaxialGeometryError
from optiland.samples.objectives import CookeTriplet
from tests.test_entrance_pupil_behind_object import _pupil_behind_object
from tests.test_folded_paraxial import straight
from tests.test_folded_paraxial_hardening import (
    entered_along_neg_z,
    entered_along_x_finite,
    straight_finite,
)
from tests.utils import assert_bits_equal

RAY_ATTRS = ("x", "y", "z", "L", "M", "N", "i", "w")


def _telecentric():
    """A singlet with a telecentric object space and an object-height field."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=32.0, material=IdealMaterial(n=1.0))
    optic.surfaces.add(
        index=1, radius=16.0, thickness=16.0, material=IdealMaterial(n=2.0)
    )
    optic.surfaces.add(
        index=2, thickness=48.0, is_stop=True, material=IdealMaterial(n=2.0)
    )
    optic.surfaces.add(index=3, material=IdealMaterial(n=2.0))
    optic.fields.set_type("object_height")
    optic.fields.add(y=0.25)
    optic.set_aperture("objectNA", 0.05)
    optic.wavelengths.add(0.55, is_primary=True)
    optic.obj_space_telecentric = True
    return optic


SYSTEMS = {
    "infinite": straight,
    "infinite_entry_frame": entered_along_neg_z,
    "finite_angle": straight_finite,
    "finite_entry_frame": entered_along_x_finite,
    "pupil_behind_object": _pupil_behind_object,
    "cooke": CookeTriplet,
    "telecentric": _telecentric,
}

# Only AngleField.get_ray_origins uses field runs. These systems have an
# object-height field, so their rays never use runs.
OBJECT_HEIGHT_SYSTEMS = ("pupil_behind_object", "telecentric")

# Field points, each for all pupil points. The x fields +0.0 and -0.0 are
# different runs.
FIELDS = [(0.0, 0.0), (0.0, 0.5), (0.3, 1.0), (-0.0, 0.0), (0.0, 0.0)]


@pytest.fixture
def paths(monkeypatch):
    """Select field runs or one value per ray, and count the uses of runs.

    ``paths(True)`` lets every bundle use runs; ``paths(False)`` makes
    ``field_runs`` return None, so the per-ray expressions run.
    """
    used = {"runs": 0}
    field_runs = runs.field_runs

    def counted_runs(*args):
        result = field_runs(*args)
        used["runs"] += result is not None
        return result

    def none(*args):
        return None

    def select(fast):
        if fast:
            monkeypatch.setattr(runs, "_MIN_RAYS", 1)
            monkeypatch.setattr(runs, "_MIN_RUN", 1)
        # angle.py holds field_runs under its own name.
        monkeypatch.setattr(angle, "field_runs", counted_runs if fast else none)

    select.used = used
    return select


def _pupil(n=60):
    """Random points in the unit pupil; the first two are chief rays."""
    rng = np.random.default_rng(1)
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    t = rng.uniform(0.0, 2 * np.pi, n)
    px, py = r * np.cos(t), r * np.sin(t)
    # With -0.0, the sign of a zero field term shows in the ray origin:
    # -0.0 + 0.0 is 0.0, but -0.0 + -0.0 is -0.0.
    px[0] = py[0] = 0.0
    px[1] = py[1] = -0.0
    return px, py


def _generate(optic, fields=FIELDS):
    """Generate the rays of all pupil points for each field point."""
    px, py = _pupil()
    hx = np.repeat([f[0] for f in fields], px.size)
    hy = np.repeat([f[1] for f in fields], px.size)
    rays = optic.ray_tracer.ray_generator.generate_rays(
        hx, hy, np.tile(px, len(fields)), np.tile(py, len(fields)), 0.55
    )
    return {a: np.array(getattr(rays, a)) for a in RAY_ATTRS}


@pytest.mark.filterwarnings("ignore:paraxial ray tracing")
@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_generated_rays_are_identical(paths, name):
    """Rays generated with field runs have the bits of the per-ray rays."""
    paths(False)
    generic = _generate(SYSTEMS[name]())
    paths(True)
    fast = _generate(SYSTEMS[name]())
    for attr in RAY_ATTRS:
        assert_bits_equal(fast[attr], generic[attr])
    assert paths.used["runs"] == (0 if name in OBJECT_HEIGHT_SYSTEMS else 1)


class TestFieldRuns:
    """The runs found in the field coordinates, and when none are found."""

    n = 8192

    def test_one_field_is_one_run(self):
        """One field point is one run, which expand does not repeat."""
        h = np.full(self.n, 0.5)
        p = np.zeros(self.n)
        found = runs.field_runs(h, h, p, p)
        assert_bits_equal(found.hx, np.array([0.5]))
        values = np.array([2.0])
        assert found.expand(values) is values
        assert_bits_equal(found.per_ray(values), np.full(self.n, 2.0))

    def test_runs_compare_bits(self):
        """+0.0 and -0.0 are different runs, as their tangents differ."""
        hx = np.repeat([0.0, -0.0, 0.0, 1.0], self.n // 4)
        hy = np.zeros(self.n)
        p = np.zeros(self.n)
        found = runs.field_runs(hx, hy, p, p)
        assert_bits_equal(found.hx, np.array([0.0, -0.0, 0.0, 1.0]))
        expanded = found.expand(np.arange(4.0))
        assert_bits_equal(expanded, np.repeat(np.arange(4.0), self.n // 4))

    def test_short_runs_and_small_bundles_use_one_value_per_ray(self):
        """Runs shorter than _MIN_RUN or fewer than _MIN_RAYS rays give None."""
        p = np.zeros(self.n)
        alternating = np.tile([0.0, 1.0], self.n // 2)
        assert runs.field_runs(alternating, p, p, p) is None
        small = np.zeros(runs._MIN_RAYS - 1)
        assert runs.field_runs(small, small, small, small) is None

    @pytest.mark.parametrize("change", ["float32", "list", "length", "2d"])
    def test_other_inputs_use_one_value_per_ray(self, change):
        """Only 1-D float64 NumPy arrays of one length give runs."""
        h = np.zeros(self.n)
        other = {
            "float32": h.astype(np.float32),
            "list": h.tolist(),
            "length": h[:-1],
            "2d": h.reshape(2, -1),
        }[change]
        assert runs.field_runs(h, h, other, h) is None
        assert runs.field_runs(other, h, h, h) is None

    def test_float32_precision_uses_one_value_per_ray(self):
        """The float32 precision of the backend gives no runs."""
        h = np.zeros(self.n)
        be.set_precision("float32")
        try:
            assert runs.field_runs(h, h, h, h) is None
        finally:
            be.set_precision("float64")

    def test_torch_uses_one_value_per_ray(self):
        """The Torch backend gives no runs."""
        pytest.importorskip("torch")
        h = np.zeros(self.n)
        be.set_backend("torch")
        try:
            assert runs.field_runs(h, h, h, h) is None
        finally:
            be.set_backend("numpy")


@pytest.mark.parametrize(
    ("max_field", "field"), [(80.0, (0.9, 0.9)), (90.0, (0.0, 1.0))]
)
def test_field_errors_are_the_same(paths, max_field, field):
    """A wide two-dimensional field and a singular tangent raise as before."""
    optic = straight()
    optic.fields.add(x=0.0, y=max_field)
    messages = []
    for fast in (False, True):
        paths(fast)
        with pytest.raises(UnsupportedParaxialGeometryError) as error:
            _generate(optic, fields=[(0.0, 0.0), field])
        messages.append(str(error.value))
    assert messages[0] == messages[1]
