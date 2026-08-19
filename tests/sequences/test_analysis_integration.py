"""Phase 4 validation matrix: existing analyses over a SequencedOptic
(SPEC_multi_sequence_20260731.md phase 4).

Per the SPEC, this phase is expected to be test-only: SpotDiagram and MTF
read `optic.trace`/`optic.trace_generic`/`optic.surfaces`/`optic.image_surface`,
all of which SequencedOptic already exposes, so they should work unmodified.
"""

from __future__ import annotations

import math

from optiland.analysis.spot_diagram.core import SpotDiagram
from optiland.mtf.geometric import GeometricMTF
from optiland.samples.objectives import CookeTriplet

from ..utils import assert_allclose

FORWARD_STEPS = [0, 1, 2, 3, 4, 5, 6, 7]
GHOST_STEPS = [0, 1, 2, (3, "reflect"), (2, "reflect"), 3, 4, 5, 6, 7]


class TestSpotDiagramOverSequence:
    def test_forward_sequence_matches_base_optic(self):
        optic = CookeTriplet()
        seq = optic.add_sequence("fwd", steps=FORWARD_STEPS)

        base_sd = SpotDiagram(optic, fields="all", wavelengths="all", num_rings=3)
        seq_sd = SpotDiagram(seq, fields="all", wavelengths="all", num_rings=3)

        base_rms = base_sd.rms_spot_radius()
        seq_rms = seq_sd.rms_spot_radius()

        for base_field, seq_field in zip(base_rms, seq_rms, strict=True):
            for base_val, seq_val in zip(base_field, seq_field, strict=True):
                assert_allclose(base_val, seq_val)

    def test_ghost_sequence_produces_finite_spot_for_on_axis_field(self):
        optic = CookeTriplet()
        seq = optic.add_sequence("ghost_2_3", steps=GHOST_STEPS)

        sd = SpotDiagram(seq, fields=[(0.0, 0.0)], wavelengths="primary", num_rings=3)
        rms = sd.rms_spot_radius()

        assert len(rms) == 1
        assert math.isfinite(float(rms[0][0]))


class TestGeometricMTFOverSequence:
    def test_forward_sequence_runs_without_error(self):
        optic = CookeTriplet()
        seq = optic.add_sequence("fwd", steps=FORWARD_STEPS)

        mtf = GeometricMTF(seq, fields=[(0.0, 0.0)], wavelength="primary", num_rays=32)
        tangential, sagittal = mtf.mtf[0]
        assert len(mtf.freq) > 0
        assert all(0.0 <= v <= 1.0 + 1e-9 for v in tangential)
        assert all(0.0 <= v <= 1.0 + 1e-9 for v in sagittal)
