"""Scene IR -- the backend-portable, data-only description of an NSQ scene.

``optiland.nonsequential.ir`` is the single change that is hardest to
retrofit later: a scene becomes describable as plain data
(:class:`~.scene_ir.SceneIR`) rather than as live Python objects with
methods. ``lower()`` converts a live
:class:`~optiland.nonsequential.scene.NSQScene` into that data form.

This is what makes a third-party backend (Mitsuba 3, OptiX) possible without
reworking the NumPy/Torch reference engines again: any backend that can
interpret a :class:`~.scene_ir.SceneIR` can trace the scene.
:mod:`optiland.nonsequential.backends` builds a fresh ``SceneIR`` at the
start of every ``trace()`` call and drives its per-bounce interaction
dispatch from it (see :mod:`.interpreter`) rather than branching on the
Python class of each ``scene.surfaces`` entry -- the reference
*interpreter* of this data, not the scene's source of truth.

Translatability checklist
--------------------------
Every physics feature added to NSQ from here on must satisfy these rules:

1. No Python-side per-hit state. All interaction behaviour is expressible as
   ``(BsdfIR, MediumIR)`` data, not as a callback.
2. Every parameter is a backend array or a plain scalar -- never a closure,
   never an object with behaviour.
3. No unbounded loops in the interaction path; iteration counts are
   compile-time constants or scene-level parameters.
4. Transforms are ``(4, 4)`` matrices, not ``CoordinateSystem`` objects
   (which carry parent-chain behaviour).
5. Everything in the IR is serializable to JSON without loss.

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland.nonsequential.ir.bsdf_ir import BsdfIR
from optiland.nonsequential.ir.interpreter import (
    apply_primitive_interactions,
    assert_bsdf_matches,
    assert_component_kind_matches,
)
from optiland.nonsequential.ir.lower import lower
from optiland.nonsequential.ir.medium_ir import MediumIR
from optiland.nonsequential.ir.scene_ir import (
    EmitterIR,
    PrimitiveIR,
    RngContract,
    SamplingPolicy,
    SceneIR,
    SensorIR,
    VolumeIR,
    scene_ir_from_dict,
    scene_ir_to_dict,
)

__all__ = [
    "BsdfIR",
    "EmitterIR",
    "MediumIR",
    "PrimitiveIR",
    "RngContract",
    "SamplingPolicy",
    "SceneIR",
    "SensorIR",
    "VolumeIR",
    "apply_primitive_interactions",
    "assert_bsdf_matches",
    "assert_component_kind_matches",
    "lower",
    "scene_ir_from_dict",
    "scene_ir_to_dict",
]
