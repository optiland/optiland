"""WP0: the mirrored-source fingerprint table (plan 3.7, design 8.3).

These tests are the alarm that tells a future upstream merge it has moved a
function the Metal kernel copies expression-for-expression.  They are
fork-local by design: upstream CI cannot run Metal.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import ast  # noqa: E402
import functools  # noqa: E402
import inspect  # noqa: E402
import textwrap  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from optiland.backend.torch_backend.metal import trace_mirror as TM  # noqa: E402

_MIRRORED = [fp for fp in TM.FINGERPRINTS if fp.klass == TM.MIRRORED]
_CONTRACT = [fp for fp in TM.FINGERPRINTS if fp.klass == TM.CONTRACT]

_REPO = Path(__file__).resolve().parents[2]
_VALUE_TEST_FILES = (
    _REPO / "tests" / "metal" / "test_trace_adapters.py",
    _REPO / "tests" / "metal" / "test_trace_batch.py",
    # round 1 iteration 3: the `be.sign` value row of finding R1-V2-04
    _REPO / "tests" / "metal" / "test_trace_adversarial_round1.py",
)


def test_table_is_non_empty_and_split_into_two_classes():
    # 63 MIRRORED at I0; the round-3 fix lane added the five rows of finding
    # R3-V2-02 and then the three conic-solver rows of R3-V2-04 (the coverage
    # censuses below are what keep the count honest).
    assert len(_MIRRORED) == 71
    assert len(_CONTRACT) == 17
    assert len(TM.FINGERPRINTS) == 88
    assert {fp.klass for fp in TM.FINGERPRINTS} == {TM.MIRRORED, TM.CONTRACT}


def test_qualnames_are_unique():
    names = [fp.qualname for fp in TM.FINGERPRINTS]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("fp", _MIRRORED, ids=lambda fp: fp.qualname)
def test_fingerprints_match(fp):
    """The live Python source still hashes to the recorded digest."""
    live = TM.source_digest(fp.qualname)
    assert live == fp.sha256, (
        f"Python `{fp.qualname}` changed; re-verify MSL `{fp.msl_function}` in "
        "kernels/trace.metal, then run `python -m "
        "optiland.backend.torch_backend.metal.trace_mirror --update --verified "
        f'"{fp.qualname}=<why it is still mirrored>"`'
    )


@pytest.mark.parametrize(
    "identity", TM.STRUCTURAL_IDENTITIES, ids=lambda fn: fn.__name__
)
def test_structural_identities(identity):
    assert identity() is True, (
        f"structural identity {identity.__name__} no longer holds"
    )


def test_fingerprint_ignores_docstrings_and_formatting():
    """Reformatting and docstring edits must not flap the digest."""
    qualname = "optiland.geometries.newton_raphson:_nz_threshold"
    original = inspect.getsource(TM._resolve(qualname))
    baseline = TM.source_digest(qualname)

    def digest_of(src: str) -> str:
        tree = TM._strip_docstring(ast.parse(textwrap.dedent(src)))
        import hashlib

        return hashlib.sha256(("ast:" + ast.dump(tree)).encode("utf-8")).hexdigest()

    assert digest_of(original) == baseline

    # (a) a different docstring
    tree = ast.parse(textwrap.dedent(original))
    fn = tree.body[0]
    assert isinstance(fn.body[0], ast.Expr), (
        "probe assumes the function has a docstring"
    )
    fn.body[0] = ast.Expr(value=ast.Constant(value="completely different docstring"))
    assert digest_of(ast.unparse(tree)) == baseline

    # (b) added comments and blank lines
    commented = original.replace("\n", "\n    # noise\n", 1)
    assert digest_of(commented) == baseline

    # (c) a real expression change must NOT match
    changed = original.replace("return", "return 1.0 *", 1)
    assert digest_of(changed) != baseline


def test_contract_rows_are_not_hashed():
    """CONTRACT rows carry no digest and name the value test that guards them."""
    for fp in _CONTRACT:
        assert fp.sha256 is None, f"{fp.qualname} is CONTRACT but carries a digest"
        assert fp.msl_function.startswith("test_"), fp.msl_function

    present = [p for p in _VALUE_TEST_FILES if p.exists()]
    if not present:
        pytest.skip(
            "WP2/WP7 have not landed tests/metal/test_trace_adapters.py or "
            "test_trace_batch.py yet; the required value-test names are frozen in "
            "NOTES/fused-trace-research/status.md (WP0 section)"
        )
    text = "\n".join(p.read_text(encoding="utf-8") for p in present)
    missing = sorted(
        {fp.msl_function for fp in _CONTRACT if fp.msl_function not in text}
    )
    assert not missing, (
        "CONTRACT rows name value tests that do not exist in "
        f"{[p.name for p in present]}: {missing}"
    )


def test_mirrored_rows_name_an_msl_function_or_host_site():
    for fp in _MIRRORED:
        assert fp.msl_function, fp.qualname
        assert not fp.msl_function.startswith("test_"), fp.qualname


def test_check_all_detects_injected_change(monkeypatch):
    """A one-expression edit to one mirrored source is reported, and only it."""
    target = "optiland.geometries.newton_raphson:_nz_threshold"
    victim = TM._resolve(target)
    original = inspect.getsource(victim)
    tampered = original.replace("return", "return 1.0 *", 1)
    assert tampered != original

    real_getsource = inspect.getsource

    def fake_getsource(obj):
        if obj is victim:
            return tampered
        return real_getsource(obj)

    monkeypatch.setattr(TM.inspect, "getsource", fake_getsource)
    problems = TM.check_all()
    assert len(problems) == 1, problems
    assert target in problems[0]
    assert "nz_threshold" in problems[0]


def test_check_all_reports_a_renamed_source(monkeypatch):
    """A rename (qualname gone) is drift, not a crash."""
    real_resolve = TM._resolve
    target = _MIRRORED[0].qualname

    def fake_resolve(qualname):
        if qualname == target:
            raise AttributeError("renamed upstream")
        return real_resolve(qualname)

    monkeypatch.setattr(TM, "_resolve", fake_resolve)
    problems = TM.check_all()
    assert len(problems) == 1, problems
    assert "cannot be resolved" in problems[0]


def test_check_all_detects_a_broken_structural_identity(monkeypatch):
    def broken():
        return False

    broken.__name__ = "broken_identity"
    monkeypatch.setattr(TM, "STRUCTURAL_IDENTITIES", (broken,))
    problems = TM.check_all()
    assert problems == ["structural identity broken_identity no longer holds"]


def test_check_all_is_clean_on_this_checkout():
    assert TM.check_all() == []


def test_table_hash_is_stable_and_sensitive():
    first = TM.table_hash()
    assert first == TM.table_hash()
    assert len(first) == 64


# ---------------------------------------------------------------------------
# Completeness: the census that keeps the table's COVERAGE honest (R3-V2-02)
#
# Every test above checks a row that EXISTS.  None of them could see a physics
# function the MSL reproduces that nobody wrote a row for -- and five such
# functions shipped: ``StandardGeometry.surface_normal``,
# ``NewtonRaphsonGeometry.surface_normal``, ``ObjectSurface._trace_real``,
# ``BaseGeometry.localize`` and ``BaseGeometry.globalize``.  An injected edit
# to any of them left ``check_all()`` empty, ``fused_trace_skip:mirror_drift``
# at zero and the kernel serving the physics the Python path no longer
# computed (10 of 10 checks, both modes; see
# ``tests/metal/test_trace_adversarial_round3.py::test_r3v202_*``).
#
# The alarm against a SIXTH omission is mechanical: profile a per-op trace,
# collect every ``optiland.*`` function that really runs inside
# ``SurfaceGroup.trace``, and require each one to be either fingerprinted or
# named in an exemption list with the reason it carries no physics.  An
# upstream merge that adds a function to the trace path fails this test until
# somebody decides which of the two it is.
# ---------------------------------------------------------------------------

#: Fixtures the census traces.  Between them they cover every geometry code
#: (plane, infinite-radius standard, conic, even and odd asphere), all four
#: aperture kinds, a mirror, a TIR chain, a three-angle pose and an
#: ``ObjectSurface`` at infinity -- i.e. every branch of the kernel's v1 scope.
_CENSUS_FIXTURES: tuple[str, ...] = (
    "cooke",
    "hubble",
    "aspheric_singlet",
    "odd_asphere_singlet",
    "tilted_triplet_rxryrz",
    "rect_aperture",
    "ellipse_aperture",
    "offset_radial_aperture",
    "tir_singlet",
    "fold_mirror",
)

#: Modules whose code is SHARED with the per-op path rather than mirrored by
#: MSL: both paths run the same ``df64_core.h`` / ``sf64_core.h`` arithmetic,
#: so an edit under here moves the two answers together and cannot make the
#: kernel serve stale physics.  ``optiland.backend`` itself (the dispatch
#: module, no trailing dot) is exempted by name below.
#:
#: The drop is NOT unconditional: :data:`_CONIC_PREFIXES` below names the part
#: of that tree where the reason is false (round-3 finding R3-V2-04).  What
#: the drop still covers after the carve-out is the backend's elementwise
#: arithmetic (``metal.ops_*``, ``metal.tensor``, ``metal.encode``, the
#: kernels) and the dispatch wrappers around it (``torch_backend.creation``,
#: ``.indexing``, ``.passthrough``, ``.reductions``, ...).  Four rows still
#: live there -- ``ops_elementwise:_binary``, ``_host_scalar_op``,
#: ``_pow_tensor_scalar`` and ``_sign`` -- and all four are CONTRACT rows,
#: which plan 3.7 deliberately guards with a VALUE test rather than a digest;
#: :func:`test_no_executed_mirrored_row_is_hidden_from_the_census` pins that
#: exactly, so a MIRRORED row can never again be dropped unseen.
_SHARED_PREFIXES: tuple[str, ...] = ("optiland.backend.",)

#: The carve-out from :data:`_SHARED_PREFIXES` (round-3 finding R3-V2-04).
#:
#: These three modules are the per-op path's CONIC SOLVER -- the tier-A
#: reference plan 7.1 measures the kernel against, on the GPU, for every
#: spherical and conic surface.  They are physics, not shared arithmetic: an
#: edit to ``metal.conic:conic_candidates`` (the root pair and the five flag
#: bits) moves the per-op answer alone, which is exactly the shape R3-V2-02
#: was filed for, and the blanket ``optiland.backend.`` drop meant the alarm
#: could not look.  The verifier demonstrated it: a root-order edit gave
#: ``check_all() == []``, ``fused_trace:traces == 1`` and fused != per-op.
_CONIC_PREFIXES: tuple[str, ...] = (
    "optiland.backend._conic:",
    "optiland.backend.torch_backend.conic:",
    "optiland.backend.torch_backend.metal.conic:",
)

_ACCESSOR = "accessor: returns a stored attribute; no expression the MSL copies"
_MATERIAL_HOST_READ = (
    "material host read: the record's SR_NPRE/SR_NPOST/SR_U/SR_ALPHA come "
    "through this same call (trace_record._index_at), so an edit moves the "
    "fused and per-op answers together"
)
_BOTH_PATHS = "runs on both paths, outside the mirrored loop"
_NESTED = "comprehension inside a fingerprinted function; hashed with it"

#: Executed physics-layer functions that deliberately carry no fingerprint,
#: each with the reason.  A function may leave this dict only by gaining a row
#: and may enter it only with a reason a reviewer can check.
_CENSUS_EXEMPT: dict[str, str] = {
    "optiland.backend:__getattr__": "backend dispatch; shared by both paths",
    "optiland.backend:get_backend": "backend dispatch; shared by both paths",
    "optiland.coordinate_system:CoordinateSystem.rx": _ACCESSOR,
    "optiland.coordinate_system:CoordinateSystem.ry": _ACCESSOR,
    "optiland.coordinate_system:CoordinateSystem.rz": _ACCESSOR,
    "optiland.coordinate_system:CoordinateSystem.x": _ACCESSOR,
    "optiland.coordinate_system:CoordinateSystem.y": _ACCESSOR,
    "optiland.coordinate_system:CoordinateSystem.z": _ACCESSOR,
    "optiland.geometries.newton_raphson:__create_fn__.<locals>.__init__": (
        "dataclass-generated __init__; carries no expression"
    ),
    "optiland.interactions.base:BaseInteractionModel.geometry": _ACCESSOR,
    "optiland.interactions.base:BaseInteractionModel.material_post": _ACCESSOR,
    "optiland.interactions.base:BaseInteractionModel.material_pre": _ACCESSOR,
    "optiland.materials.base:BaseMaterial._array_metadata_key": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial._array_size": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial._as_backend_array": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial._backend_context": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial._broadcast_like": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial._compute_grad_aware": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial._detach_if_tensor": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial._evaluate_property.<locals>.<genexpr>": (
        _NESTED
    ),
    "optiland.materials.base:BaseMaterial._is_uniform_key": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial._requires_grad": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial._state_key": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial.k": _MATERIAL_HOST_READ,
    "optiland.materials.base:BaseMaterial.n": _MATERIAL_HOST_READ,
    "optiland.materials.base:_array_content_key": _MATERIAL_HOST_READ,
    "optiland.materials.base:_array_uniform_key": _MATERIAL_HOST_READ,
    "optiland.materials.base:_uniform_scalar": _MATERIAL_HOST_READ,
    "optiland.materials.ideal:IdealMaterial._cache_state": _MATERIAL_HOST_READ,
    "optiland.materials.ideal:IdealMaterial._calculate_k": _MATERIAL_HOST_READ,
    "optiland.materials.ideal:IdealMaterial._calculate_n": _MATERIAL_HOST_READ,
    "optiland.materials.material:Material._cache_state": _MATERIAL_HOST_READ,
    "optiland.materials.material_file:MaterialFile._cache_state": _MATERIAL_HOST_READ,
    "optiland.materials.material_file:MaterialFile._calculate_k": _MATERIAL_HOST_READ,
    "optiland.materials.material_file:MaterialFile._calculate_n": _MATERIAL_HOST_READ,
    "optiland.materials.material_file:MaterialFile._formula_2": _MATERIAL_HOST_READ,
    "optiland.materials.material_file:MaterialFile._formula_3": _MATERIAL_HOST_READ,
    "optiland.surfaces.standard_surface:Surface._coordinator": _ACCESSOR,
    "optiland.surfaces.standard_surface:Surface.material_post": _ACCESSOR,
    "optiland.surfaces.standard_surface:Surface.material_pre": _ACCESSOR,
    "optiland.surfaces.standard_surface:Surface.previous_surface": _ACCESSOR,
    "optiland.surfaces.standard_surface:_aperture_aware_distance.<locals>.<genexpr>": (
        _NESTED
    ),
    "optiland.surfaces.surface_group:SurfaceGroup.reset": (
        "the hook runs after SurfaceGroup.trace's own reset(), so both paths "
        "execute it unchanged"
    ),
    "optiland.surfaces.surface_group:SurfaceGroup.surfaces": _ACCESSOR,
    "optiland.surfaces.surface_group:SurfaceGroup.surfaces.<locals>.<genexpr>": _NESTED,
    "optiland.surfaces.surface_group:_fused_metal_declined": _BOTH_PATHS,
    "optiland.surfaces.surface_group:_fused_metal_trace": (
        "the hook itself; SurfaceGroup.trace, which contains it, is a MIRRORED row"
    ),
}

_NESTED_IN_EXEMPT = (
    "comprehension inside a function this list already exempts, with that "
    "function's reason"
)

_CONIC_ROUTING = (
    "routing: it chooses between implementations of the same "
    "`_conic_candidates` arithmetic and computes no distance itself.  That "
    "they agree is a value test "
    "(tests/metal/test_conic_kernel.py::test_masks_match_numpy_and_roots_agree), "
    "and the implementation it picks on mps -- conic_intersection_metal -- is "
    "a MIRRORED row"
)

#: The conic-solver counterpart of :data:`_CENSUS_EXEMPT`, for the functions
#: the GPU census reaches under :data:`_CONIC_PREFIXES`.  Same rule: a
#: function may leave this dict only by gaining a row, and may enter it only
#: with a reason a reviewer can check.
_CONIC_EXEMPT: dict[str, str] = {
    "optiland.backend.torch_backend.conic:ConicMixin.conic_intersection": (
        _CONIC_ROUTING
    ),
    "optiland.backend.torch_backend.conic:_can_fuse_metal": _CONIC_ROUTING,
    "optiland.backend.torch_backend.conic:can_fuse_cpu": _CONIC_ROUTING,
    "optiland.backend.torch_backend.conic:can_fuse_cpu.<locals>.<genexpr>": (
        _NESTED_IN_EXEMPT
    ),
    "optiland.backend.torch_backend.metal.conic:_ConicMetal.setup_context": (
        "autograd plumbing: it saves tensors for the backward pass.  The "
        "fused path runs with grad off by construction (the gate refuses "
        "requires_grad), so it computes nothing the kernel mirrors"
    ),
    "optiland.backend.torch_backend.metal.conic:_kernel_library": (
        "compiles and caches kernels/conic.metal; no expression the MSL copies"
    ),
    "optiland.backend.torch_backend.metal.conic:can_fuse_metal": _CONIC_ROUTING,
    "optiland.backend.torch_backend.metal.conic:can_fuse_metal.<locals>.<genexpr>": (
        _NESTED_IN_EXEMPT
    ),
    "optiland.backend.torch_backend.metal.conic:conic_candidates.<locals>.<lambda>": (
        _NESTED
    ),
}


def _census_executed(backend: str = "numpy", mode: str = "df64") -> set[str]:
    """Every ``optiland.*`` function that runs inside a per-op trace.

    ``backend="numpy"`` (the default) needs no GPU: the physics layer is
    backend-independent and the Metal arithmetic lives under
    ``optiland.backend.``, which :data:`_SHARED_PREFIXES` drops.  Measured
    equal to the GPU census, which reaches the same 104 physics-layer
    functions out of 339 executed
    (``NOTES/fused-trace-research/probes/round3/r3v2i2_11_mirror_coverage.py``).

    ``backend="metal"`` traces the same fixtures on ``mps`` with the fused
    hook OFF.  It is the only way to reach the conic solver of
    :data:`_CONIC_PREFIXES`: that code is the per-op GPU path, so the NumPy
    census never executes it and -- until round-3 finding R3-V2-04 -- the
    shared-prefix drop meant nothing else did either.  The hook must be off:
    with it on, an eligible fixture never reaches the per-op conic code at all
    and the census silently shrinks to the refused fixtures (measured: 2 of
    34).
    """
    import sys

    import optiland.backend as be

    repo_scripts = str(_REPO / "scripts")
    if repo_scripts not in sys.path:
        sys.path.insert(0, repo_scripts)
    import trace_fixtures as fx

    seen: set[str] = set()

    def profile(frame, event, arg):  # pragma: no cover - the profiler itself
        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        if module.startswith("optiland."):
            seen.add(f"{module}:{frame.f_code.co_qualname}")

    previous = be.get_backend()
    previous_switch = os.environ.get("OPTILAND_METAL_FUSED_TRACE")
    if backend == "numpy":
        be.set_backend("numpy")
        restore = None
    else:
        # Imported HERE, outside the profiled region, so that the census never
        # sees `metal.conic:<module>` or the `_ConicMetal` class body: those
        # frames run once per process, at import, and whether they land inside
        # a profiled trace depends only on what imported first.
        import optiland.backend.torch_backend.metal.conic  # noqa: F401
        from optiland.backend.torch_backend import metal

        restore = metal.get_mode()
        os.environ["OPTILAND_METAL_FUSED_TRACE"] = "0"
        be.set_backend("torch")
        be.set_device("mps")
        be.set_precision("float64")
        be.grad_mode.disable()
        metal.set_mode(mode)
    try:
        for name in _CENSUS_FIXTURES:
            made = fx.FIXTURES[name]()
            if isinstance(made, tuple):
                optic, rays = made[0], made[1](made[0])
            else:
                optic, rays = made, fx.pupil_bundle(made)
            sys.setprofile(profile)
            try:
                optic.surfaces.trace(rays, record=True)
            finally:
                sys.setprofile(None)
    finally:
        if restore is not None:
            from optiland.backend.torch_backend import metal

            metal.set_mode(restore)
        if previous_switch is None:
            os.environ.pop("OPTILAND_METAL_FUSED_TRACE", None)
        else:
            os.environ["OPTILAND_METAL_FUSED_TRACE"] = previous_switch
        be.set_backend(previous)
    return seen


def _is_shared(qualname: str) -> bool:
    """True when the census drops ``qualname`` as shared backend arithmetic."""
    return qualname.startswith(_SHARED_PREFIXES) and not qualname.startswith(
        _CONIC_PREFIXES
    )


def _census_physics(backend: str = "numpy", mode: str = "df64") -> set[str]:
    """:func:`_census_executed` minus the shared backend arithmetic."""
    return {q for q in _census_executed(backend, mode) if not _is_shared(q)}


def test_every_executed_physics_function_is_fingerprinted_or_exempt():
    """No physics function reaches the kernel's scope without a decision.

    This is the coverage half of plan 3.7's "every mirrored Python function is
    fingerprinted" (round-3 finding R3-V2-02).  The five functions that were
    missing are rows now; this test is what makes the sixth one fail loudly
    instead of silently serving stale physics after an upstream merge.
    """
    physics = _census_physics()
    table = {fp.qualname for fp in TM.FINGERPRINTS}
    gap = sorted(physics - table - set(_CENSUS_EXEMPT))
    assert gap == [], (
        "these optiland functions execute inside SurfaceGroup.trace but are "
        "neither in trace_mirror.FINGERPRINTS nor in _CENSUS_EXEMPT: "
        f"{gap}.  Decide for each one: if kernels/trace.metal reproduces its "
        "expressions, add a MIRRORED row (the kernel would otherwise keep "
        "serving the old physics after an upstream edit -- R3-V2-02); if it "
        "carries no expression the MSL copies, add it to _CENSUS_EXEMPT with "
        "the reason."
    )


def test_the_five_r3v202_rows_are_in_the_census_and_in_the_table():
    """The five functions finding R3-V2-02 named are covered, not just listed.

    Guards the fix against a row that names a qualname nothing executes: each
    one must appear in the census AND carry a MIRRORED fingerprint.
    """
    five = {
        "optiland.geometries.standard:StandardGeometry.surface_normal",
        "optiland.geometries.newton_raphson:NewtonRaphsonGeometry.surface_normal",
        "optiland.surfaces.object_surface:ObjectSurface._trace_real",
        "optiland.geometries.base:BaseGeometry.localize",
        "optiland.geometries.base:BaseGeometry.globalize",
    }
    assert five <= _census_physics(), sorted(five - _census_physics())
    mirrored = {fp.qualname for fp in _MIRRORED}
    assert five <= mirrored, sorted(five - mirrored)


def test_the_census_exemption_list_has_no_dead_entries():
    """Every exemption still names a function the census actually reaches.

    An exemption nobody executes is an assertion nobody checks: it would let a
    qualname that upstream renamed keep vouching for a function that no longer
    exists.  Removing a dead entry is the whole fix.
    """
    executed = _census_executed()
    dead = sorted(set(_CENSUS_EXEMPT) - executed)
    assert dead == [], (
        f"_CENSUS_EXEMPT names functions the census no longer executes: {dead}"
    )


def test_no_exempt_function_is_also_fingerprinted():
    """The two lists are disjoint, so neither can hide a row from review."""
    table = {fp.qualname for fp in TM.FINGERPRINTS}
    both = sorted(table & set(_CENSUS_EXEMPT))
    assert both == [], both


# ---------------------------------------------------------------------------
# R3-V2-04 -- the census has to be able to look at the conic solver
#
# `_SHARED_PREFIXES` dropped every `optiland.backend.*` qualname with the
# reason "both paths run the same df64_core.h / sf64_core.h arithmetic, so an
# edit under here moves the two answers together".  For the elementwise
# arithmetic that is true.  For `optiland/backend/torch_backend/metal/conic.py`
# it is not: that module is the per-op GPU path's own conic solver -- the
# tier-A reference plan 7.1 measures the kernel against -- and the fused
# kernel does not run a line of it.  Round-3 verifier 2 injected a root-order
# edit into `conic_candidates` and got the full R3-V2-02 shape back:
# `check_all()` clean, `fused_trace:traces = 1`, fused != per-op at tier A
# while fused == the pre-edit answer.
#
# The carve-out below is what lets the alarm see that module tree; the three
# MIRRORED rows are what make the edit refuse.  The regression test for the
# refusal itself is
# `tests/metal/test_trace_adversarial_round3.py::test_r3v204_*`.
# ---------------------------------------------------------------------------


def _mps_or_skip():
    """The GPU census needs a Metal device; skip cleanly where there is none."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
        pytest.skip("Metal GPU required")
    return torch


@functools.cache
def _metal_census() -> frozenset[str]:
    """The GPU per-op census, traced once per process (it costs ~10 traces)."""
    return frozenset(_census_executed("metal"))


def test_the_conic_carve_out_changes_what_the_census_can_see():
    """The carve-out is not decoration: it moves qualnames across the drop.

    Without this, a later edit could set ``_CONIC_PREFIXES = ()`` and every
    test below would still pass -- vacuously, because the GPU census would go
    back to dropping the whole tree.
    """
    conic = "optiland.backend.torch_backend.metal.conic:conic_candidates"
    arithmetic = "optiland.backend.torch_backend.metal.ops_elementwise:_binary"
    assert not _is_shared(conic), (
        "the conic solver is back under the shared-arithmetic drop, so the "
        "completeness census can no longer see it (R3-V2-04)"
    )
    assert _is_shared(arithmetic), (
        "the drop must still cover the elementwise arithmetic both paths run"
    )
    # ... and on the NumPy census it really adds a name, which is a row.
    assert "optiland.backend._conic:_select_distance" in _census_physics()


def test_every_executed_conic_function_is_fingerprinted_or_exempt():
    """No conic-solver function reaches the per-op GPU path without a decision.

    The GPU half of plan 3.7's "every mirrored Python function is
    fingerprinted".  ``conic_candidates``, ``_ConicMetal.forward`` and
    ``_scalar_float`` are rows now; everything else this census reaches under
    :data:`_CONIC_PREFIXES` must name itself in :data:`_CONIC_EXEMPT` with the
    reason it carries no physics.
    """
    _mps_or_skip()
    executed = _metal_census()
    reached = {q for q in executed if q.startswith(_CONIC_PREFIXES)}
    assert "optiland.backend.torch_backend.metal.conic:conic_candidates" in reached, (
        "the GPU census no longer reaches the fused conic solver, so this "
        "test would pass without checking anything; is the fused-trace hook "
        "on, or has the per-op path stopped routing conic distances through "
        "be.conic_intersection?"
    )
    table = {fp.qualname for fp in TM.FINGERPRINTS}
    gap = sorted(reached - table - set(_CONIC_EXEMPT))
    assert gap == [], (
        "these conic-solver functions execute inside a per-op GPU trace but "
        "are neither in trace_mirror.FINGERPRINTS nor in _CONIC_EXEMPT: "
        f"{gap}.  Decide for each one: if it computes physics the fused "
        "kernel reproduces, add a MIRRORED row (an edit to it would otherwise "
        "move the per-op answer while the kernel keeps serving the physics it "
        "was verified against -- R3-V2-04); if it is routing or plumbing, add "
        "it to _CONIC_EXEMPT with the reason."
    )


def test_the_conic_exemption_list_has_no_dead_entries():
    """Every conic exemption still names a function the GPU census reaches."""
    _mps_or_skip()
    dead = sorted(set(_CONIC_EXEMPT) - set(_metal_census()))
    assert dead == [], (
        f"_CONIC_EXEMPT names functions the GPU census no longer executes: {dead}"
    )


def test_no_executed_mirrored_row_is_hidden_from_the_census():
    """A MIRRORED row that runs in a trace is always visible to a census.

    This is the exact statement the R3-V2-04 evidence disproved before the
    carve-out: six fingerprint rows executed inside a per-op GPU trace and all
    six were dropped, so the alarm's stated reason ("shared arithmetic") was
    false by the table's own contents.  Two of the six were MIRRORED
    (``_conic:_select_distance``, ``metal.conic:conic_intersection_metal``)
    and are carved in now.  The other four are ``ops_elementwise`` rows, and
    plan 3.7 makes those CONTRACT *on purpose* -- "not hashed, so they cannot
    cause alarm fatigue"; their guard is a value test, not a digest.  So the
    invariant that must hold is the one below.
    """
    _mps_or_skip()
    executed = _metal_census()
    mirrored = {fp.qualname for fp in _MIRRORED}
    hidden = sorted(q for q in executed & mirrored if _is_shared(q))
    assert hidden == [], (
        "these MIRRORED rows run inside a per-op GPU trace and the census "
        f"drops them as shared arithmetic: {hidden}.  A digest that nothing "
        "re-checks against what actually executes is how R3-V2-04 happened; "
        "either carve the module out in _CONIC_PREFIXES or make the row "
        "CONTRACT with a named value test."
    )
    contract = {fp.qualname for fp in _CONTRACT}
    dropped_rows = sorted(q for q in executed & (mirrored | contract) if _is_shared(q))
    assert dropped_rows == [
        "optiland.backend.torch_backend.metal.ops_elementwise:_binary",
        "optiland.backend.torch_backend.metal.ops_elementwise:_host_scalar_op",
        "optiland.backend.torch_backend.metal.ops_elementwise:_pow_tensor_scalar",
        "optiland.backend.torch_backend.metal.ops_elementwise:_sign",
    ], dropped_rows
