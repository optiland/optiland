"""Subprocess fixture for queue, crash and noninterruptible cancellation tests."""

from __future__ import annotations

import os
import pickle
import struct
import sys
import time

from optiland_gui.services.calculation_worker import encode_message


def send(message):
    sys.stdout.buffer.write(encode_message(message))
    sys.stdout.buffer.flush()


send({"event": "ready"})
while header := sys.stdin.buffer.read(4):
    size = struct.unpack("!I", header)[0]
    message = pickle.loads(sys.stdin.buffer.read(size))
    if message["command"] == "shutdown":
        break
    if message["command"] != "run":
        continue
    job_id = message["job_id"]
    parameters = message["parameters"]
    send({"event": "progress", "job_id": job_id, "stage": "fixture"})
    if parameters.get("crash"):
        if parameters.get("stderr"):
            print(parameters["stderr"], file=sys.stderr, flush=True)
        os._exit(7)
    if parameters.get("malformed"):
        sys.stdout.buffer.write(struct.pack("!I", 2**32 - 1))
        sys.stdout.buffer.flush()
        time.sleep(30)
    time.sleep(parameters.get("delay", 0.01))
    send(
        {
            "event": "result",
            "job_id": job_id,
            "status": "succeeded",
            "data": parameters.get("value", 42),
        }
    )
