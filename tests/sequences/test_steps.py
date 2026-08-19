from __future__ import annotations

import pytest

from optiland.sequences.steps import SequenceStep, parse_steps


def test_plain_int_steps_are_forward_nominal():
    steps = parse_steps([0, 1, 2, 3])
    assert steps == [
        SequenceStep(0, reverse=False, interaction_override=None),
        SequenceStep(1, reverse=False, interaction_override=None),
        SequenceStep(2, reverse=False, interaction_override=None),
        SequenceStep(3, reverse=False, interaction_override=None),
    ]


def test_direction_flips_after_each_reflection():
    # The motivating ghost example from SPEC_multi_sequence_20260731.md.
    steps = parse_steps([0, 1, 2, (3, "reflect"), (2, "reflect"), 3, 4])
    assert steps == [
        SequenceStep(0, reverse=False, interaction_override=None),
        SequenceStep(1, reverse=False, interaction_override=None),
        SequenceStep(2, reverse=False, interaction_override=None),
        SequenceStep(3, reverse=False, interaction_override="reflect"),
        SequenceStep(2, reverse=True, interaction_override="reflect"),
        SequenceStep(3, reverse=False, interaction_override=None),
        SequenceStep(4, reverse=False, interaction_override=None),
    ]


def test_list_pairs_are_parsed_identically_to_tuples():
    steps = parse_steps([0, 1, [2, "reflect"], 3])
    assert steps == [
        SequenceStep(0, reverse=False, interaction_override=None),
        SequenceStep(1, reverse=False, interaction_override=None),
        SequenceStep(2, reverse=False, interaction_override="reflect"),
        SequenceStep(3, reverse=True, interaction_override=None),
    ]


def test_refract_override_does_not_flip_direction():
    steps = parse_steps([0, (1, "refract"), 2])
    assert [s.reverse for s in steps] == [False, False, False]
    assert steps[1].interaction_override == "refract"


def test_empty_sequence_raises():
    with pytest.raises(ValueError, match="at least one step"):
        parse_steps([])


def test_invalid_override_raises():
    with pytest.raises(ValueError, match="Unknown interaction override"):
        parse_steps([0, (1, "bounce")])


def test_malformed_tuple_raises():
    with pytest.raises(ValueError, match="must be"):
        parse_steps([(1, "reflect", "extra")])


def test_non_int_surface_index_raises():
    with pytest.raises(ValueError, match="must be an int"):
        parse_steps([0, ("1", "reflect")])


def test_invalid_step_type_raises():
    with pytest.raises(ValueError, match="Invalid raw step"):
        parse_steps([0, "invalid"])
