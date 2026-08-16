from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.sequences.resolver import SequenceValidationError
from optiland.sequences.sequenced_optic import SequencedOptic

from ..utils import assert_allclose


def _build_optic():
    optic = Optic()
    optic.surfaces.add(index=0, thickness=100)
    optic.surfaces.add(
        index=1, thickness=10, material=IdealMaterial(n=1.5), is_stop=True
    )
    optic.surfaces.add(index=2, thickness=50)
    optic.surfaces.add(index=3)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.set_aperture("EPD", 10.0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


class TestSequencedOpticConstruction:
    def test_add_sequence_returns_and_stores(self, set_test_backend):
        optic = _build_optic()
        seq = optic.add_sequence("fwd", steps=[0, 1, 2, 3])

        assert isinstance(seq, SequencedOptic)
        assert optic.sequences["fwd"] is seq
        assert len(seq.surfaces) == 4

    def test_invalid_sequence_raises_and_is_not_stored(self, set_test_backend):
        optic = _build_optic()
        with pytest.raises(SequenceValidationError):
            optic.add_sequence("bad", steps=[0, 2])

        assert "bad" not in optic.sequences

    def test_direct_construction(self, set_test_backend):
        optic = _build_optic()
        seq = SequencedOptic(optic, "direct", [0, 1, 2, 3])
        assert seq.base_optic is optic
        assert seq.name == "direct"


class TestSequencedOpticDelegation:
    """SequencedOptic delegates (not copies) aperture/fields/wavelengths/etc."""

    def test_delegated_properties_are_identical_objects(self, set_test_backend):
        optic = _build_optic()
        seq = optic.add_sequence("fwd", steps=[0, 1, 2, 3])

        assert seq.aperture is optic.aperture
        assert seq.fields is optic.fields
        assert seq.wavelengths is optic.wavelengths
        assert seq.apodization is optic.apodization
        assert seq.paraxial is optic.paraxial
        assert seq.polarization == optic.polarization
        assert seq.primary_wavelength == optic.primary_wavelength

    def test_editing_base_optic_field_is_visible_through_sequence(
        self, set_test_backend
    ):
        optic = _build_optic()
        seq = optic.add_sequence("fwd", steps=[0, 1, 2, 3])

        optic.fields.add(y=5.0)
        assert len(seq.fields.fields) == len(optic.fields.fields)

    def test_n_delegates_to_sequenced_surfaces(self, set_test_backend):
        optic = _build_optic()
        seq = optic.add_sequence("fwd", steps=[0, 1, 2, 3])
        assert_allclose(seq.n(0.55), seq.surfaces.n(0.55))
        assert_allclose(seq.n("primary"), seq.surfaces.n(0.55))


class TestSequencedOpticTrace:
    def test_forward_sequence_matches_base_optic_trace(self, set_test_backend):
        optic = _build_optic()
        optic.add_sequence("fwd", steps=[0, 1, 2, 3])

        base_rays = optic.trace(0, 0, 0.55, num_rays=6)
        # base_optic.trace() mutates optic.surfaces' record buffers, so trace
        # the sequence from a freshly built optic to keep the comparison clean.
        optic2 = _build_optic()
        seq2 = optic2.add_sequence("fwd", steps=[0, 1, 2, 3])
        seq_rays = seq2.trace(0, 0, 0.55, num_rays=6)

        assert_allclose(seq_rays.x, base_rays.x)
        assert_allclose(seq_rays.y, base_rays.y)
        assert_allclose(seq_rays.L, base_rays.L)
        assert_allclose(seq_rays.opd, base_rays.opd)

    def test_ghost_sequence_traces_without_raising(self, set_test_backend):
        optic = _build_optic()
        seq = optic.add_sequence(
            "ghost", steps=[0, 1, (2, "reflect"), (1, "reflect"), 2, 3]
        )
        rays = seq.trace(0, 0, 0.55, num_rays=6)

        assert be.size(rays.x) > 0
        assert seq.surfaces.x.shape[0] == 6

    def test_trace_does_not_mutate_base_optic_surfaces(self, set_test_backend):
        optic = _build_optic()
        seq = optic.add_sequence("fwd", steps=[0, 1, 2, 3])

        assert be.size(optic.surfaces[1].x) == 0
        seq.trace(0, 0, 0.55, num_rays=6)
        assert be.size(optic.surfaces[1].x) == 0
        assert be.size(seq.surfaces[1].x) > 0


class TestSequencedOpticSerialization:
    def test_serialization_round_trip_preserves_sequences(self):
        optic = _build_optic()
        optic.add_sequence("ghost", steps=[0, 1, (2, "reflect"), (1, "reflect"), 2, 3])

        data = optic.to_dict()
        assert "sequences" in data
        assert "ghost" in data["sequences"]

        restored = Optic.from_dict(data)
        assert "ghost" in restored.sequences
        assert len(restored.sequences["ghost"].surfaces) == 6
