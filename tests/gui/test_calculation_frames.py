"""Framed worker decoding preserves ownership without redundant payload copies."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QByteArray

from optiland_gui.services.calculation_jobs import CalculationJobs
from optiland_gui.services.calculation_worker import MAX_MESSAGE_BYTES, encode_message


def receiver():
    process = SimpleNamespace(chunk=b"")
    process.readAllStandardOutput = lambda: QByteArray(process.chunk)
    messages, failures = [], []
    service = SimpleNamespace(
        sender=lambda: process,
        _process=process,
        _buffer=bytearray(),
        _message=messages.append,
        _kill_worker=lambda: failures.append(True),
        _stderr="",
    )
    return service, process, messages, failures


def feed(service, process, chunk):
    process.chunk = chunk
    CalculationJobs._read_stdout(service)


def test_fragmented_header_payload_and_multiple_frames():
    service, process, messages, failures = receiver()
    first = {"event": "ready"}
    second = {"event": "result", "data": np.arange(10000, dtype=np.float32)}
    third = {"event": "progress", "stage": "next"}
    wire = encode_message(first) + encode_message(second) + encode_message(third)
    for start, stop in ((0, 2), (2, 19), (19, 700), (700, len(wire) - 1)):
        feed(service, process, wire[start:stop])
    assert messages[0] == first
    np.testing.assert_array_equal(messages[1]["data"], second["data"])
    assert len(messages) == 2
    feed(service, process, wire[-1:])
    assert messages[2] == third
    assert not failures and not service._buffer


def test_decoded_arrays_own_data_after_receive_storage_reuse():
    service, process, messages, failures = receiver()
    expected = np.arange(99999, dtype=np.float32).reshape(-1, 3)
    shared = {"points": expected, "also_points": expected}
    feed(service, process, encode_message({"data": shared}))
    actual = messages[0]["data"]
    assert actual["points"] is actual["also_points"]
    service._buffer.extend(b"overwrite reused receive storage" * 50000)
    service._buffer[:] = b"\x00" * len(service._buffer)
    np.testing.assert_array_equal(actual["points"], expected)
    actual["points"][0, 0] = -12
    assert expected[0, 0] == 0
    assert not failures


def test_oversized_or_invalid_frames_keep_protocol_failure_behavior():
    for wire in (
        struct.pack("!I", MAX_MESSAGE_BYTES + 1),
        struct.pack("!I", 3) + b"bad",
    ):
        service, process, messages, failures = receiver()
        feed(service, process, wire)
        assert failures == [True]
        assert service._stderr
        assert not messages
        # Any decode memoryviews are released even on an exception.
        service._buffer.clear()
