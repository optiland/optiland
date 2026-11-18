"""A zero Forbes coefficient still gets its derivative.

The Forbes geometries remove the trailing zero coefficients of each family
once per coefficient change: the radial terms of the Q (slope-orthogonal)
surface, and the m = 0 terms and the cosine and sine terms of each m >= 1 of
the Q-2D surface. The Clenshaw sums then evaluate only the coefficients that
stay. The values do not change, but the derivative with respect to a removed
coefficient was zero, or missing when all coefficients were zero. Without
it, a gradient-based optimizer cannot move a coefficient that starts at zero
(issue #1169).

In reverse mode, ``ZeroTermDerivative`` attaches the derivative of the
removed terms, and only the backward pass computes it. In forward mode and
inside ``torch.func`` transforms, the geometry evaluates the coefficients
with their trailing zeros. Without a derivative to request (NumPy,
``torch.no_grad``, tensors without requires_grad), the geometry takes the
path of before, with the same operations and values.

The internals of the fix are imported inside the tests, so that this file
also runs on a tree without the fix.
"""

from __future__ import annotations

import contextlib
import warnings

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.forbes import (
    ForbesQ2dGeometry,
    ForbesQNormalSlopeGeometry,
    ForbesSurfaceConfig,
)
from optiland.geometries.forbes import geometry as forbes_geometry
from optiland.rays import RealRays

from .nr_implicit_test_utils import backend_state

torch = pytest.importorskip("torch")

NORM = 8.0
# The trailing terms 2 and 3 are zero.
QBFS = {0: 1.0e-3, 1: -5.0e-4, 2: 0.0, 3: 0.0}
QBFS_ZERO = {0: 0.0, 1: 0.0, 2: 0.0}
# Zeros in each kind of family. Trailing zeros: m = 0 (n = 1), cosine m = 1
# (n = 1 and 3, with an absent n = 2), sine m = 1 (n = 2, after the interior
# zero n = 0), the whole sine family m = 2 (n = 1), sine m = 3 (n = 0) and
# cosine m = 3 (n = 2).
Q2D = {
    ("a", 0, 0): 1.0e-3,
    ("a", 0, 1): 0.0,
    ("a", 1, 0): 2.0e-4,
    ("a", 1, 1): 0.0,
    ("a", 1, 3): 0.0,
    ("b", 1, 0): 0.0,
    ("b", 1, 1): 1.0e-4,
    ("b", 1, 2): 0.0,
    ("a", 2, 0): -1.5e-4,
    ("b", 2, 1): 0.0,
    ("b", 3, 0): 0.0,
    ("a", 3, 2): 0.0,
}
Q2D_ZERO_KEYS = [key for key, value in Q2D.items() if value == 0.0]
# Only zeros, in the m = 0, cosine and sine families.
Q2D_ZERO = {("a", 0, 0): 0.0, ("a", 1, 1): 0.0, ("b", 1, 0): 0.0, ("b", 2, 2): 0.0}
# The vertex, points inside the normalization radius, and one point outside.
X = [3.0, -2.0, 0.5, 0.0, 6.5, 9.0]
Y = [1.0, 4.0, -0.2, 0.0, -3.0, 1.0]
OUTPUTS = ["sag", "normal"]
Q2D_OUTPUTS = ["sag", "normal", "normal_polar"]


def _geometry(terms, polar=False):
    """A Q or Q-2D geometry (from the type of the keys) with these terms."""
    if isinstance(next(iter(terms)), tuple):
        cls = ForbesQ2dGeometry
    else:
        cls = ForbesQNormalSlopeGeometry
    config = ForbesSurfaceConfig(
        radius=-40.0, conic=-0.5, norm_radius=NORM, terms=dict(terms)
    )
    geometry = cls(CoordinateSystem(), config)
    if polar:
        geometry._use_cartesian_normal = False
    return geometry


def _terms(base, dtype=torch.float64, requires_grad=False):
    """The coefficients of ``base`` as new tensors."""
    return {
        key: torch.tensor(value, dtype=dtype, requires_grad=requires_grad)
        for key, value in base.items()
    }


def _shifted(base, key, value, dtype=torch.float64):
    """The coefficients of ``base`` as tensors, with ``value`` added to ``key``."""
    terms = _terms(base, dtype)
    terms[key] = terms[key] + value
    return terms


def _points(dtype=torch.float64, device="cpu"):
    return (torch.tensor(v, dtype=dtype, device=device) for v in (X, Y))


def _value(geometry, output, x, y):
    """The sag, or a fixed mix of the normal components."""
    if output == "sag":
        return geometry.sag(x, y)
    nx, ny, nz = geometry._surface_normal(x, y)
    return 0.7 * nx - 1.3 * ny + 0.4 * nz


def _ad_and_fd(base, key, output, dtype=torch.float64):
    """The derivative of the summed output with respect to ``key``: AD and FD.

    Returns:
        tuple[float, list[float]]: The reverse-mode derivative, and central
        differences for the steps 2e-6 and 1e-6.
    """
    polar = output == "normal_polar"
    output = "normal" if polar else output
    x, y = _points(dtype)
    leaf = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    geometry = _geometry(_shifted(base, key, leaf, dtype), polar)
    (ad,) = torch.autograd.grad(_value(geometry, output, x, y).sum(), leaf)
    fd = []
    for h in (2e-6, 1e-6):
        with torch.no_grad():
            plus = _geometry(_shifted(base, key, h, dtype), polar)
            minus = _geometry(_shifted(base, key, -h, dtype), polar)
            difference = (
                _value(plus, output, x, y).sum() - _value(minus, output, x, y).sum()
            )
        fd.append(float(difference) / (2 * h))
    return float(ad), fd


def _assert_matches_fd(ad, fd):
    assert fd[0] == pytest.approx(fd[1], rel=1e-6, abs=1e-12)
    assert ad != 0.0
    assert ad == pytest.approx(fd[1], rel=1e-6, abs=1e-12)


@contextlib.contextmanager
def _grad_mode(enabled):
    """Set Optiland's Torch grad mode, and restore it; other tests can change it."""
    previous = be.grad_mode.requires_grad
    if enabled:
        be.grad_mode.enable()
    else:
        be.grad_mode.disable()
    try:
        yield
    finally:
        if previous:
            be.grad_mode.enable()
        else:
            be.grad_mode.disable()


def _grad_mode_off():
    """Switch Optiland's Torch grad mode off until the end of the block."""
    return _grad_mode(False)


def _bits(tensor):
    """The bits of a float64 tensor, so that -0.0 and 0.0 differ."""
    return tensor.detach().contiguous().view(torch.int64)


def _family_lengths(families):
    """(len(radial),) or (len(cm0), len(ams[0]), ..., len(bms[0]), ...)."""
    if len(families) == 1:
        return (len(families[0]),)
    cm0, ams, bms = families
    return (len(cm0), *(len(f) for f in ams), *(len(f) for f in bms))


def _graph_has(tensor, name):
    """Return True if a node of the graph of ``tensor`` has ``name`` in its name."""
    seen, stack = set(), [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if name in type(node).__name__:
            return True
        stack.extend(next_node for next_node, _ in node.next_functions)
    return False


class TestPortedFromFirstFix:
    """The tests of the first fix (local commit 4ab8c7ee), for this design."""

    @staticmethod
    def _fd(build, h=1e-6):
        x, y = be.array([3.0, -2.0]), be.array([1.0, 4.0])
        with torch.no_grad():
            plus = float(build(h).sag(x, y).sum())
            minus = float(build(-h).sag(x, y).sum())
        return (plus - minus) / (2 * h)

    @staticmethod
    def _ad(geometry, leaf):
        x, y = be.array([3.0, -2.0]), be.array([1.0, 4.0])
        (grad,) = torch.autograd.grad(geometry.sag(x, y).sum(), leaf)
        return float(grad)

    @pytest.mark.parametrize("zero_term", [0, 2])
    def test_qbfs_zero_term_gets_gradient(self, zero_term):
        """Term 2 of {0: 1e-3, 1: -5e-4, 2: 0}: trailing; term 0 of all zero."""
        with backend_state("torch", "float64"):
            base = {0: 1e-3, 1: -5e-4, 2: 0.0} if zero_term == 2 else {0: 0.0}

            def build(value):
                return _geometry(_shifted(base, zero_term, value))

            leaf = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            ad = self._ad(build(leaf), leaf)
            fd = self._fd(build)
            assert ad != 0.0
            assert ad == pytest.approx(fd, rel=1e-6, abs=1e-12)

    @pytest.mark.parametrize("zero_key", [("a", 1, 1), ("a", 0, 0)])
    def test_q2d_zero_coefficient_gets_gradient(self, zero_key):
        """("a", 1, 1) trails a non-zero ("a", 1, 0); ("a", 0, 0) is alone."""
        with backend_state("torch", "float64"):
            if zero_key == ("a", 1, 1):
                base = {("a", 0, 0): 1e-3, ("a", 1, 0): 2e-4, ("a", 1, 1): 0.0}
            else:
                base = {("a", 0, 0): 0.0}

            def build(value):
                return _geometry(_shifted(base, zero_key, value))

            leaf = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            ad = self._ad(build(leaf), leaf)
            fd = self._fd(build)
            assert ad != 0.0
            assert ad == pytest.approx(fd, rel=1e-6, abs=1e-12)

    def test_trimmed_zeros_that_can_carry_a_derivative_are_collected(self):
        """Only the removed zeros that can carry a derivative are collected."""
        with backend_state("torch", "float64"), _grad_mode_off():
            terms = _terms(QBFS)
            terms[3] = terms[3].clone().requires_grad_()
            geometry = _geometry(terms)
            geometry._ensure_coeffs()
            assert len(geometry._prepared_radial_coeffs) == 2  # trimmed as before
            assert geometry._zero_terms.sources == (terms[3],)  # not terms[2]
            assert geometry._zero_terms.families[0].degrees == (3,)

            plain = _geometry(_terms(QBFS))
            plain._ensure_coeffs()
            assert plain._zero_terms is None

            # A trailing absent coefficient of Q-2D is a float: nothing to carry.
            sparse = _geometry({("a", 1, 3): torch.tensor(0.0, requires_grad=True)})
            assert [f.degrees for f in sparse._zero_terms.families] == [(3,)]
        with backend_state("numpy", "float64"):
            geometry = _geometry(dict(QBFS))
            geometry._ensure_coeffs()
            assert geometry._zero_terms is None

    def test_evaluation_selects_the_path_for_its_context(self):
        """Reverse mode: the trimmed coefficients; forward mode: the zeros too."""
        from optiland.geometries.forbes._zero_terms import KEEP, REVERSE

        forward_ad = torch.autograd.forward_ad
        with backend_state("torch", "float64"), _grad_mode_off():
            geometry = _geometry(_terms(QBFS, requires_grad=True))
            geometry._ensure_coeffs()
            assert geometry._zero_term_mode() == REVERSE
            assert len(geometry._radial_coeffs(REVERSE)) == 2
            with torch.no_grad():
                assert geometry._zero_term_mode() is None
            with forward_ad.dual_level():
                tangent = torch.tensor(1.0, dtype=torch.float64)
                terms = _terms(QBFS)
                terms[2] = forward_ad.make_dual(terms[2], tangent)
                dual = _geometry(terms)
                dual._ensure_coeffs()
                assert dual._zero_term_mode() == KEEP
                assert len(dual._radial_coeffs(KEEP)) == 4


class TestReverseMode:
    @pytest.mark.parametrize("output", OUTPUTS)
    @pytest.mark.parametrize("key", [2, 3])
    def test_qbfs_trailing_zero_matches_fd(self, key, output):
        """The trailing zeros of a Q surface, for the sag and the normal."""
        with backend_state("torch", "float64"):
            _assert_matches_fd(*_ad_and_fd(QBFS, key, output))

    @pytest.mark.parametrize("output", OUTPUTS)
    @pytest.mark.parametrize("key", [0, 1, 2])
    def test_qbfs_all_zero_matches_fd(self, key, output):
        """No term stays: the normal has no departure to attach to."""
        with backend_state("torch", "float64"):
            _assert_matches_fd(*_ad_and_fd(QBFS_ZERO, key, output))

    @pytest.mark.parametrize("output", Q2D_OUTPUTS)
    @pytest.mark.parametrize("key", Q2D_ZERO_KEYS, ids=str)
    def test_q2d_cosine_sine_and_m0_zeros_match_fd(self, key, output):
        """Each zero of Q2D, for the sag, the normal and the polar normal."""
        with backend_state("torch", "float64"):
            _assert_matches_fd(*_ad_and_fd(Q2D, key, output))

    @pytest.mark.parametrize("output", Q2D_OUTPUTS)
    @pytest.mark.parametrize("key", list(Q2D_ZERO), ids=str)
    def test_q2d_all_zero_matches_fd(self, key, output):
        """A Q-2D set with only zeros, in each kind of family."""
        with backend_state("torch", "float64"):
            _assert_matches_fd(*_ad_and_fd(Q2D_ZERO, key, output))

    @pytest.mark.parametrize(
        "base, key", [(QBFS, 3), (QBFS_ZERO, 1), (Q2D, ("b", 2, 1))]
    )
    def test_distance_gradient_matches_fd(self, base, key):
        """The implicit distance gradient at a zero coefficient matches FD."""
        with backend_state("torch", "float64"):
            L, M = 0.08, -0.05
            rays = RealRays(
                x=2.5,
                y=-1.6,
                z=-5.0,
                L=L,
                M=M,
                N=(1.0 - L * L - M * M) ** 0.5,
                intensity=1.0,
                wavelength=0.587,
            )

            def distance(value):
                geometry = _geometry(_shifted(base, key, value))
                geometry.tol = 1e-13
                return geometry.distance(rays)[0]

            leaf = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            (ad,) = torch.autograd.grad(distance(leaf), leaf)
            fd = []
            for h in (2e-5, 1e-5):
                with torch.no_grad():
                    plus = float(distance(torch.tensor(h, dtype=torch.float64)))
                    minus = float(distance(torch.tensor(-h, dtype=torch.float64)))
                fd.append((plus - minus) / (2 * h))
            _assert_matches_fd(float(ad), fd)

    @pytest.mark.parametrize(
        "base, key, output",
        [(QBFS, 2, "normal"), (QBFS_ZERO, 0, "normal"), (Q2D, ("a", 1, 1), "normal")],
    )
    def test_float32_matches_float64(self, base, key, output):
        """In float32, the derivative agrees with float64 to float32 precision."""
        results = {}
        for precision, dtype in (
            ("float64", torch.float64),
            ("float32", torch.float32),
        ):
            with backend_state("torch", precision):
                x, y = _points(dtype)
                leaf = torch.tensor(0.0, dtype=dtype, requires_grad=True)
                geometry = _geometry(_shifted(base, key, leaf, dtype))
                value = _value(geometry, output, x, y)
                (grad,) = torch.autograd.grad(value.sum(), leaf)
                assert grad.dtype == dtype
                results[precision] = float(grad)
        assert results["float32"] != 0.0
        assert results["float32"] == pytest.approx(results["float64"], rel=1e-4)

    def test_backward_alone_computes_the_zero_term_derivative(self, monkeypatch):
        """The forward pass evaluates only the trimmed coefficients."""
        from optiland.geometries.forbes import _zero_terms

        lengths, tables = [], []
        original_basis = _zero_terms._basis_table

        def counted_basis(*args):
            tables.append(args)
            return original_basis(*args)

        monkeypatch.setattr(_zero_terms, "_basis_table", counted_basis)
        # The Clenshaw sums of the forward pass, the position of their first
        # coefficient argument, and the number of coefficient arguments.
        for name, first, count in (
            ("clenshaw_qbfs", 0, 1),
            ("compute_z_zprime_qbfs", 0, 1),
            ("compute_z_q2d", 0, 3),
            ("_q2d_cartesian_eval", 2, 3),
        ):
            original = getattr(forbes_geometry, name)

            def counted(*args, original=original, first=first, count=count, **kw):
                lengths.append(_family_lengths(args[first : first + count]))
                return original(*args, **kw)

            monkeypatch.setattr(forbes_geometry, name, counted)
        with backend_state("torch", "float64"):
            x, y = _points()
            for base in (QBFS, Q2D):
                lengths.clear()
                tables.clear()
                geometry = _geometry(_terms(base, requires_grad=True))
                sag = geometry.sag(x, y)
                normal = geometry._surface_normal(x, y)[0]
                if base is QBFS:
                    trimmed = (geometry._prepared_radial_coeffs,)
                else:
                    trimmed = (
                        geometry._prepared_cm0_coeffs,
                        geometry._prepared_ams_coeffs,
                        geometry._prepared_bms_coeffs,
                    )
                assert lengths == [_family_lengths(trimmed)] * 2
                assert lengths[0] != _family_lengths(
                    (list(geometry.coeffs_c),)
                    if base is QBFS
                    else (geometry.cm0_coeffs, geometry.ams_coeffs, geometry.bms_coeffs)
                )
                assert _graph_has(sag, "ZeroTermDerivative")
                assert _graph_has(normal, "ZeroTermDerivative")
                assert not tables  # nothing of the zero terms in the forward pass
                grads = torch.autograd.grad(
                    sag.sum() + normal.sum(), geometry._zero_terms.sources
                )
                assert tables
                assert all(float(g) != 0.0 for g in grads)

    @pytest.mark.parametrize(
        "base, polar",
        [
            (QBFS, False),
            (QBFS_ZERO, False),
            (Q2D, False),
            (Q2D, True),
            (Q2D_ZERO, True),
        ],
    )
    def test_other_gradients_keep_their_bits(self, base, polar):
        """Non-zero coefficients, base, positions: the same gradient bits."""

        def gradients(track_zeros):
            terms = {
                key: torch.tensor(
                    value, dtype=torch.float64, requires_grad=value != 0 or track_zeros
                )
                for key, value in base.items()
            }
            geometry = _geometry(terms, polar)
            params = {
                "radius": torch.tensor(-40.0, dtype=torch.float64, requires_grad=True),
                "k": torch.tensor(-0.5, dtype=torch.float64, requires_grad=True),
                "norm_radius": torch.tensor(NORM, dtype=torch.float64).requires_grad_(),
            }
            for name, value in params.items():
                setattr(geometry, name, value)
            x, y = (p.clone().requires_grad_() for p in _points())
            weights = torch.linspace(0.3, 1.7, len(X), dtype=torch.float64)
            nx, ny, nz = geometry._surface_normal(x, y)
            loss = (weights * (geometry.sag(x, y) + 0.7 * nx - 1.3 * ny + nz)).sum()
            named = {k: v for k, v in terms.items() if base[k] != 0}
            named.update(params, x=x, y=y)
            grads = torch.autograd.grad(loss, list(named.values()))
            return dict(zip(named, grads, strict=True))

        with backend_state("torch", "float64"):
            plain, tracked = gradients(False), gradients(True)
        assert plain.keys() == tracked.keys()
        for name, value in plain.items():
            assert torch.equal(_bits(value), _bits(tracked[name])), name


class TestValues:
    @pytest.mark.parametrize("mode", ["reverse", "forward"])
    @pytest.mark.parametrize("polar", [False, True], ids=["cartesian", "polar"])
    def test_values_keep_their_bits(self, mode, polar):
        """Sag and normal keep their bits with a derivative, also a -0.0."""
        forward_ad = torch.autograd.forward_ad
        x = torch.cat([torch.zeros(1), torch.linspace(-9.0, 9.0, 37)]).double()
        y = torch.cat([torch.zeros(1), torch.linspace(7.5, -8.5, 37)]).double()
        # With R < 0 and a negative sum at the vertex, the sag there is -0.0.
        negative = {0: -1.0e-3, 1: 0.0, 2: 0.0}
        with backend_state("torch", "float64"), _grad_mode_off():
            for base in (QBFS, QBFS_ZERO, negative, Q2D, Q2D_ZERO):
                if polar and not isinstance(next(iter(base)), tuple):
                    continue
                plain = _geometry(_terms(base), polar)
                with torch.no_grad():
                    expected = (plain.sag(x, y), *plain._surface_normal(x, y))
                if mode == "reverse":
                    tracked = _geometry(_terms(base, requires_grad=True), polar)
                    values = (tracked.sag(x, y), *tracked._surface_normal(x, y))
                else:
                    with forward_ad.dual_level():
                        terms = {
                            key: forward_ad.make_dual(value, torch.ones_like(value))
                            for key, value in _terms(base).items()
                        }
                        tracked = _geometry(terms, polar)
                        values = tuple(
                            forward_ad.unpack_dual(value).primal
                            for value in (
                                tracked.sag(x, y),
                                *tracked._surface_normal(x, y),
                            )
                        )
                for value, reference in zip(values, expected, strict=True):
                    assert torch.equal(_bits(value), _bits(reference))
                if base is negative:
                    assert bool(torch.signbit(expected[0][0]))  # -0.0 was checked

    @pytest.mark.parametrize("context", ["numpy", "untracked", "no_grad"])
    @pytest.mark.parametrize("base", [QBFS, Q2D], ids=["qbfs", "q2d"])
    def test_without_derivative_the_path_of_before(self, context, base):
        """No derivative to request: nothing collected or no path selected."""
        backend = "numpy" if context == "numpy" else "torch"
        grad_mode = contextlib.nullcontext if context == "numpy" else _grad_mode_off
        with backend_state(backend, "float64"), grad_mode():
            if context == "numpy":
                terms = dict(base)
            else:
                terms = _terms(base, requires_grad=context == "no_grad")
            geometry = _geometry(terms)
            x, y = be.array(X), be.array(Y)
            with torch.no_grad():
                geometry.sag(x, y)
                geometry._surface_normal(x, y)
                assert geometry._zero_term_mode() is None
            if context != "no_grad":
                assert geometry._zero_terms is None


class TestForwardModeAndTransforms:
    @staticmethod
    def _reverse_jacobian(base, key, output):
        """d(output at each point)/d(key) from reverse mode."""
        x, y = _points()
        leaf = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        value = _value(_geometry(_shifted(base, key, leaf)), output, x, y)
        rows = [
            torch.autograd.grad(value[i], leaf, retain_graph=True)[0]
            for i in range(len(X))
        ]
        return torch.stack(rows)

    @pytest.mark.parametrize(
        "base, key", [(QBFS, 3), (QBFS_ZERO, 0), (Q2D, ("b", 2, 1)), (Q2D, ("a", 1, 3))]
    )
    @pytest.mark.parametrize("output", OUTPUTS)
    def test_forward_ad_tangent_matches_reverse_mode(self, base, key, output):
        """A forward_ad tangent along a zero coefficient is the derivative."""
        forward_ad = torch.autograd.forward_ad
        with backend_state("torch", "float64"), _grad_mode_off():
            expected = self._reverse_jacobian(base, key, output)
            x, y = _points()
            with forward_ad.dual_level():
                tangent = torch.tensor(1.0, dtype=torch.float64)
                terms = _terms(base)
                terms[key] = forward_ad.make_dual(terms[key], tangent)
                value = _value(_geometry(terms), output, x, y)
                result = forward_ad.unpack_dual(value).tangent
            torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-15)
            assert float(result.abs().max()) > 0.0

    @pytest.mark.parametrize(
        "base, key", [(QBFS, 2), (QBFS_ZERO, 1), (Q2D, ("a", 0, 1)), (Q2D, ("b", 3, 0))]
    )
    @pytest.mark.parametrize("output", OUTPUTS)
    def test_func_jvp_tangent_matches_reverse_mode(self, base, key, output):
        """A torch.func.jvp tangent along a zero coefficient is the derivative."""
        with backend_state("torch", "float64"), _grad_mode_off():
            expected = self._reverse_jacobian(base, key, output)
            x, y = _points()

            def fn(value):
                return _value(_geometry(_shifted(base, key, value)), output, x, y)

            zero = torch.tensor(0.0, dtype=torch.float64)
            _, result = torch.func.jvp(fn, (zero,), (torch.ones_like(zero),))
            torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-15)
            assert float(result.abs().max()) > 0.0

    @pytest.mark.parametrize("base", [QBFS, Q2D], ids=["qbfs", "q2d"])
    def test_vmap_over_coefficient_sets(self, base):
        """torch.func.vmap over coefficient sets gives the per-set values."""
        with backend_state("torch", "float64"), _grad_mode_off():
            keys = list(base)
            first = torch.tensor([base[k] for k in keys], dtype=torch.float64)
            batch = torch.stack([first, 2.0 * first + 1e-4 * (first == 0)])
            x, y = _points()

            def fn(values):
                geometry = _geometry(dict(zip(keys, values, strict=True)))
                return torch.stack(
                    [geometry.sag(x, y), *geometry._surface_normal(x, y)]
                )

            result = torch.func.vmap(fn)(batch)
            with torch.no_grad():
                for k in range(2):
                    expected = fn(list(batch[k]))
                    torch.testing.assert_close(result[k], expected, rtol=1e-14, atol=0)

    @pytest.mark.parametrize("output", OUTPUTS)
    @pytest.mark.parametrize(
        "base", [QBFS, QBFS_ZERO, Q2D], ids=["qbfs", "zero", "q2d"]
    )
    def test_jacrev_over_positions_with_tracked_coefficients(self, base, output):
        """torch.func.jacrev over x, with and without tracked coefficients."""
        with backend_state("torch", "float64"), _grad_mode_off():
            x, y = _points()
            plain = _geometry(_terms(base))
            tracked = _geometry(_terms(base, requires_grad=True))
            expected = torch.func.jacrev(lambda v: _value(plain, output, v, y))(x)
            result = torch.func.jacrev(lambda v: _value(tracked, output, v, y))(x)
            assert torch.equal(result, expected)


class TestCreateGraph:
    @pytest.mark.parametrize(
        "base", [QBFS, QBFS_ZERO, Q2D], ids=["qbfs", "zero", "q2d"]
    )
    def test_second_derivative_in_x_is_that_of_untracked_coefficients(self, base):
        """create_graph=True works; d2/dx2 does not change."""
        with backend_state("torch", "float64"):

            def second_derivative(terms):
                x = torch.tensor(X, dtype=torch.float64, requires_grad=True)
                y = torch.tensor(Y, dtype=torch.float64)
                geometry = _geometry(terms)
                total = geometry.sag(x, y).sum()
                total = total + geometry._surface_normal(x, y)[0].sum()
                (first,) = torch.autograd.grad(total, x, create_graph=True)
                (second,) = torch.autograd.grad(first.sum(), x)
                return second

            expected = second_derivative(_terms(base))
            result = second_derivative(_terms(base, requires_grad=True))
            # Q-2D has no second derivative at the vertex (NaN, also before).
            torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)

    @pytest.mark.parametrize(
        "base, zero_key, other_key",
        [(QBFS, 2, 0), (QBFS_ZERO, 1, 0), (Q2D, ("a", 1, 1), ("a", 2, 0))],
    )
    def test_mixed_second_derivative_matches_fd(self, base, zero_key, other_key):
        """d2L/(dc_zero dc_other) through create_graph=True matches FD."""
        with backend_state("torch", "float64"):
            x, y = _points()

            def first_derivative(other, create_graph):
                leaf = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
                terms = _shifted(base, zero_key, leaf)
                terms[other_key] = terms[other_key] + other
                value = _value(_geometry(terms), "normal", x, y).sum()
                (grad,) = torch.autograd.grad(value, leaf, create_graph=create_graph)
                return grad

            other = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            (ad,) = torch.autograd.grad(first_derivative(other, True), other)
            h = 1e-6
            plus = first_derivative(torch.tensor(h, dtype=torch.float64), False)
            minus = first_derivative(torch.tensor(-h, dtype=torch.float64), False)
            fd = float(plus - minus) / (2 * h)
            assert float(ad) != 0.0
            assert float(ad) == pytest.approx(fd, rel=1e-5, abs=1e-10)


class TestBasis:
    @pytest.mark.parametrize(
        "m0, m", [(True, 0), (False, 1), (False, 2), (False, 3), (False, 5)]
    )
    def test_basis_gives_the_clenshaw_sums(self, m0, m):
        """D^T psi(x) is Q_n(x) of qpoly, and D^T A psi(x) is dQ_n/dx."""
        from optiland.geometries.forbes import qpoly
        from optiland.geometries.forbes._zero_terms import _basis_table, _rows

        size = 9
        degrees = tuple(range(size))
        with backend_state("torch", "float64"), _grad_mode_off():
            x = torch.linspace(0.0, 1.2, 13, dtype=torch.float64)
            table = _basis_table(m0, m, size, x)
            rows = _rows(m0, m, degrees, False, torch.float64, x.device)
            slope_rows = _rows(m0, m, degrees, True, torch.float64, x.device)
            q, dq = rows @ table, slope_rows @ table
            for n in range(size):
                coefficients = [0.0] * n + [1.0]
                if m0:
                    alphas = qpoly.clenshaw_qbfs_der(coefficients, x, j=1)
                    first = 2 * (alphas[0, 0] + alphas[0, 1]) if n else 2 * alphas[0, 0]
                    slope = 2 * (alphas[1, 0] + alphas[1, 1]) if n else 2 * alphas[1, 0]
                else:
                    alphas = qpoly.clenshaw_q2d_der(coefficients, m, x, j=1)
                    first = qpoly.q2d_sum_from_alphas(alphas[0], m, n + 1)
                    slope = qpoly.q2d_sum_from_alphas(alphas[1], m, n + 1)
                torch.testing.assert_close(q[n], first, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(dq[n], slope, rtol=1e-12, atol=1e-12)


def _singlet(terms):
    """A singlet in N-BK7 whose back surface is a Forbes surface with ``terms``."""
    from optiland.optic import Optic

    if isinstance(next(iter(terms)), tuple):
        back = {"surface_type": "forbes_q2d", "freeform_coeffs": terms}
    else:
        back = {"surface_type": "forbes_qbfs", "radial_terms": terms}
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=30.0, thickness=5.0, is_stop=True, material="N-BK7"
    )
    optic.surfaces.add(
        index=2, thickness=45.0, radius=-40.0, conic=-0.5, norm_radius=9.0, **back
    )
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.fields.add(y=4.0)
    optic.wavelengths.add(value=0.5876, is_primary=True)
    return optic


class TestTrace:
    @pytest.mark.parametrize(
        "base, key", [(QBFS, 3), (QBFS_ZERO, 1), (Q2D, ("b", 2, 1)), (Q2D, ("a", 1, 1))]
    )
    def test_traced_merit_gradient_matches_fd(self, base, key):
        """d(merit at the image)/d(zero coefficient) of a full trace matches FD."""

        def merit(value):
            optic = _singlet(_shifted(base, key, value))
            optic.surfaces[2].geometry.tol = 1e-14
            rays = optic.trace(0.5, 0.7, 0.5876, num_rays=6, distribution="hexapolar")
            # An off-axis field in x and y and a merit without symmetry, so
            # that no derivative is zero by symmetry.
            return torch.nansum((rays.x - 0.3) ** 2 + 1.3 * rays.y**2 + rays.x * rays.y)

        with (
            backend_state("torch", "float64"),
            _grad_mode(True),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            leaf = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            (ad,) = torch.autograd.grad(merit(leaf), leaf)
            fd = []
            with torch.no_grad():
                for h in (2e-6, 1e-6):
                    plus = merit(torch.tensor(h, dtype=torch.float64))
                    minus = merit(torch.tensor(-h, dtype=torch.float64))
                    fd.append(float(plus - minus) / (2 * h))
        _assert_matches_fd(float(ad), fd)


class TestOptimization:
    @pytest.mark.parametrize("kind", ["forbes_qbfs", "forbes_q2d"])
    def test_torch_optimizer_moves_a_zero_coefficient(self, kind):
        """One Adam step moves a coefficient that starts at zero."""
        from optiland.optimization import OptimizationProblem, TorchAdamOptimizer

        with (
            backend_state("torch", "float64"),
            _grad_mode(True),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            if kind == "forbes_qbfs":
                optic = _singlet({0: 1e-3, 1: 0.0, 2: 0.0})
                variable = {"variable_type": "forbes_qbfs_coeff", "coeff_number": 2}
            else:
                optic = _singlet({("a", 0, 0): 1e-3, ("a", 2, 1): 0.0})
                variable = {
                    "variable_type": "forbes_q2d_coeff",
                    "coeff_tuple": ("a", 2, 1),
                }

            problem = OptimizationProblem()
            problem.add_variable(optic, surface_number=2, **variable)
            problem.add_operand(
                operand_type="rms_spot_size",
                target=0.0,
                weight=1.0,
                input_data={
                    "optic": optic,
                    "surface_number": -1,
                    "Hx": 0.0,
                    "Hy": 1.0,
                    "num_rays": 4,
                    "wavelength": 0.5876,
                    "distribution": "hexapolar",
                },
            )
            optimizer = TorchAdamOptimizer(problem)
            optimizer.optimize(n_steps=1, lr=1e-5, disp=False)
            assert float(optimizer.params[0].detach()) != 0.0


CUDA_CASES = [
    (base, key, output)
    for base, key in ((QBFS, 3), (QBFS_ZERO, 0))
    for output in OUTPUTS
] + [(Q2D, key, output) for key in (("a", 1, 1), ("b", 2, 1)) for output in Q2D_OUTPUTS]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestCuda:
    @pytest.mark.parametrize("base, key, output", CUDA_CASES)
    def test_cuda_matches_cpu(self, base, key, output):
        """On CUDA: the same values as without a derivative, the CPU gradient."""
        polar = output == "normal_polar"
        output = "normal" if polar else output
        results = {}
        for device in ("cpu", "cuda"):
            with backend_state("torch", "float64"):
                be.set_device(device)
                x, y = _points(device=device)
                terms = _terms(base)
                terms = {k: v.to(device) for k, v in terms.items()}
                plain = _geometry(terms, polar)
                with torch.no_grad():
                    expected = _value(plain, output, x, y)
                leaf = torch.tensor(0.0, dtype=torch.float64, device=device)
                leaf.requires_grad_()
                shifted = dict(terms)
                shifted[key] = shifted[key] + leaf
                value = _value(_geometry(shifted, polar), output, x, y)
                assert torch.equal(_bits(value), _bits(expected))
                (grad,) = torch.autograd.grad(value.sum(), leaf)
                results[device] = float(grad)
        assert results["cuda"] != 0.0
        assert results["cuda"] == pytest.approx(results["cpu"], rel=1e-12)
