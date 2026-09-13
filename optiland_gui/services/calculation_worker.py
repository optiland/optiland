"""Private persistent worker; stdout contains only framed job messages."""

from __future__ import annotations

import importlib
import pickle
import queue
import struct
import sys
import threading
import traceback
from contextlib import redirect_stdout

from optiland_gui.services.job_records import CalculationCancelled, check_cancelled

MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def encode_message(message: dict) -> bytes:
    """Encode one internally generated message; never an external file format."""
    payload = pickle.dumps(message, protocol=5)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Calculation message exceeds the 256 MiB transfer limit.")
    return struct.pack("!I", len(payload)) + payload


def _progress_reporter(job_id, cancelled, send):
    """Create a cancellation-aware reporter for counts and plain job metadata."""

    def progress(
        stage: str,
        completed: int | None = None,
        total: int | None = None,
        *,
        details: dict | None = None,
    ) -> None:
        check_cancelled(cancelled)
        send(
            {
                "event": "progress",
                "job_id": job_id,
                "stage": stage,
                "completed": completed,
                "total": total,
                "details": details,
            }
        )

    return progress


def main() -> None:
    """Keep cancellation readable while numerical work occupies the main thread."""
    incoming: queue.Queue = queue.Queue()
    cancellations: dict[int, threading.Event] = {}
    state_lock = threading.Lock()
    output = sys.stdout.buffer

    def send(message: dict) -> None:
        output.write(encode_message(message))
        output.flush()

    def read_exact(count: int) -> bytes:
        parts = bytearray()
        while len(parts) < count:
            part = sys.stdin.buffer.read(count - len(parts))
            if not part:
                raise EOFError()
            parts.extend(part)
        return bytes(parts)

    def read_commands() -> None:
        try:
            while True:
                size = struct.unpack("!I", read_exact(4))[0]
                if size > MAX_MESSAGE_BYTES:
                    raise ValueError("Calculation request is too large.")
                message = pickle.loads(read_exact(size))
                if message["command"] == "cancel":
                    with state_lock:
                        event = cancellations.get(message["job_id"])
                        if event is not None:
                            event.set()
                elif message["command"] == "run":
                    with state_lock:
                        cancellations[message["job_id"]] = threading.Event()
                    incoming.put(message)
                elif message["command"] == "shutdown":
                    break
        finally:
            with state_lock:
                for event in cancellations.values():
                    event.set()
            incoming.put(None)

    threading.Thread(target=read_commands, daemon=True).start()
    send({"event": "ready"})
    while (message := incoming.get()) is not None:
        job_id = message["job_id"]
        with state_lock:
            cancelled = cancellations[job_id]

        progress = _progress_reporter(job_id, cancelled, send)

        try:
            check_cancelled(cancelled)
            module_name, function_name = message["handler"].split(":", 1)
            if not module_name.startswith("optiland_gui."):
                raise ValueError("Calculation handlers must belong to optiland_gui.")
            with redirect_stdout(sys.stderr):
                handler = getattr(importlib.import_module(module_name), function_name)
                result = handler(
                    message["snapshot"], message["parameters"], progress, cancelled
                )
            check_cancelled(cancelled)
            send(
                {
                    "event": "result",
                    "job_id": job_id,
                    "status": "succeeded",
                    "data": result,
                }
            )
        except CalculationCancelled:
            send({"event": "result", "job_id": job_id, "status": "cancelled"})
        except Exception:
            send(
                {
                    "event": "result",
                    "job_id": job_id,
                    "status": "failed",
                    "error": traceback.format_exc(),
                }
            )
        finally:
            with state_lock:
                cancellations.pop(job_id, None)


if __name__ == "__main__":
    main()
