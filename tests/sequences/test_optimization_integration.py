"""Phase 4 payoff test: one merit function mixing operands over an Optic and
a SequencedOptic derived from it (SPEC_multi_sequence_20260731.md phase 4).

This is deliberately test-only: SequencedOptic already exposes optic.trace()
and optic.surfaces.x/y[...] the same way Optic does, so RayOperand.rms_spot_size
works against a SequencedOptic without any operand-side changes.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optimization import OptimizationProblem
from optiland.samples.objectives import CookeTriplet

from ..utils import assert_allclose


def _build_ghost_problem(batching: bool):
    optic = CookeTriplet()
    seq = optic.add_sequence(
        "ghost_2_3", steps=[0, 1, 2, (3, "reflect"), (2, "reflect"), 3, 4, 5, 6, 7]
    )

    problem = OptimizationProblem(batching=batching)
    problem.add_operand(
        "rms_spot_size",
        target=0.0,
        weight=1.0,
        input_data={
            "optic": optic,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 20,
            "wavelength": 0.55,
        },
    )
    problem.add_operand(
        "rms_spot_size",
        target=0.0,
        weight=0.1,
        input_data={
            "optic": seq,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 20,
            "wavelength": 0.55,
            "nan_policy": "omit",
        },
    )
    return optic, seq, problem


class TestMixedOperandMeritFunction:
    @pytest.mark.parametrize("batching", [True, False])
    def test_merit_function_evaluates_both_operands(self, batching):
        optic, seq, problem = _build_ghost_problem(batching)

        values = problem.fun_array()
        assert len(values) == 2
        assert all(v == v for v in values)  # no NaNs

    def test_batched_and_unbatched_evaluation_agree(self):
        _, _, batched = _build_ghost_problem(batching=True)
        _, _, unbatched = _build_ghost_problem(batching=False)

        assert batched.sum_squared() == pytest.approx(unbatched.sum_squared())

    def test_editing_base_surface_moves_both_operands(self):
        """The linkage requirement from the SPEC (§3): geometry is shared by
        reference, so a change to a base surface is visible to every
        operand referencing either the base optic or a sequence over it.
        """
        optic, seq, problem = _build_ghost_problem(batching=True)

        before = problem.fun_array()

        radius = optic.surfaces.surfaces[1].geometry.radius
        optic.surfaces.surfaces[1].geometry.radius = radius * 0.9

        after = problem.fun_array()

        assert before[0] != pytest.approx(after[0])
        assert before[1] != pytest.approx(after[1])

    def test_sequence_traces_independently_of_base(self):
        """The ghost sequence and the base optic each own their own record
        buffers, even though both wrap the same Surface objects.

        Up to surface 1 (before the ghost bounce), both traces follow an
        identical physical path, so their recorded ray data should match.
        After the bounce the paths diverge, so the final spot (image plane)
        should not.
        """
        optic, seq, problem = _build_ghost_problem(batching=True)
        problem.fun_array()

        assert optic.surfaces[1] is not seq.surfaces[1]
        assert_allclose(seq.surfaces[1].x, optic.surfaces[1].x)
        assert_allclose(seq.surfaces[1].y, optic.surfaces[1].y)

        assert not be.all(seq.surfaces[-1].y == optic.surfaces[-1].y)
