"""The project's websocket policy server — a thin transport shell.

Model-agnostic: it holds any object satisfying the :class:`apxinf.Policy` contract
(``obs dict -> result dict``) and does only protocol translation, so the same
server serves :class:`~apxinf.policies.impls.pi05.Pi05Policy` today and a future
``GrootPolicy`` unchanged. Its wire protocol is compatible with the unmodified
``openpi_client.WebsocketClientPolicy`` (metadata-on-connect, msgpack-numpy
frames, ``/healthz``), so existing robot clients connect without changes.

The policy is called **in-process** (no subprocess / stdio hop). The library's
rich result keys (``normalized_actions`` / ``token_ids`` / ``noise`` /
``metadata``) stay in-process; only ``actions`` + ``policy_timing`` are put on the
wire — see :func:`wire_response`.
"""

from __future__ import annotations

import asyncio
import http
import itertools
import json
import logging
import os
import queue
import time
import sys
import threading
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from .msgpack_numpy import packer, unpackb

__all__ = ["AsyncJsonLogger", "WebsocketPolicyServer", "wire_response", "health_check"]

logger = logging.getLogger(__name__)

_DIAGNOSTIC_LOG_QUEUE_SIZE = 256
_DIAGNOSTIC_REQUEST_ID = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _tensor_descriptor(value)
    if isinstance(value, Path):
        return str(value)
    return {"type": type(value).__name__}


def _tensor_descriptor(value: Any) -> dict[str, Any]:
    """Describe a tensor without scanning or copying its payload."""
    array = np.asarray(value)
    return {
        "type": "tensor",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "ndim": int(array.ndim),
        "nbytes": int(array.nbytes),
        "strides": list(array.strides),
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
    }


def describe_value(value: Any) -> Any:
    """Build an O(1)-per-field JSON-safe description of a request/result tree."""
    if isinstance(value, np.ndarray):
        return _tensor_descriptor(value)
    if isinstance(value, np.generic):
        return {"type": type(value).__name__, "value": value.item()}
    if isinstance(value, Mapping):
        return {str(key): describe_value(item) for key, item in value.items()}
    if isinstance(value, str):
        return {"type": "str", "chars": len(value), "preview": value[:160]}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "nbytes": len(value)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return {"type": type(value).__name__}


class AsyncJsonLogger:
    """Non-blocking JSONL printer used only by explicit ``--log`` mode.

    The request thread builds small O(1) descriptors and calls ``put_nowait``.
    JSON encoding, stream writes, and flushing happen on a daemon thread. If the
    bounded queue is full, records are dropped rather than delaying inference.
    """

    _STOP = object()

    def __init__(
        self,
        enabled: bool,
        *,
        max_queue: int = 256,
        stream: TextIO | None = None,
        log_path: Path | None = None,
    ) -> None:
        if max_queue <= 0:
            raise ValueError("log queue size must be positive")
        self.enabled = bool(enabled)
        if stream is not None and log_path is not None:
            raise ValueError("pass either log_stream or log_path, not both")
        self._owns_stream = False
        if self.enabled and stream is None:
            if log_path is None:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                log_path = Path.cwd() / "apxinf_logs" / f"apxinf_{stamp}_{os.getpid()}.jsonl"
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("a", encoding="utf-8", buffering=1)
            self._owns_stream = True
        self.log_path = str(log_path) if log_path is not None else None
        self._stream = stream if stream is not None else sys.stdout
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._worker,
                name="apxinf-json-log",
                daemon=True,
            )
            self._thread.start()

    @property
    def dropped(self) -> int:
        return self._dropped

    def emit(self, record: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(dict(record))
        except queue.Full:
            self._dropped += 1

    def close(self) -> None:
        if not self.enabled or self._thread is None:
            return
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            # Shutdown is outside inference; waiting here prevents truncating
            # already accepted records and cannot affect request latency.
            self._queue.put(self._STOP)
        self._thread.join(timeout=10)
        self._stream.flush()
        if self._owns_stream:
            self._stream.close()
        self._thread = None

    def _worker(self) -> None:
        while True:
            record = self._queue.get()
            if record is self._STOP:
                break
            rendered = json.dumps(
                record,
                default=_json_default,
                separators=(",", ":"),
                sort_keys=True,
            )
            self._stream.write(f"APXINF_LOG {rendered}\n")
            self._stream.flush()


def _summarize_timings(rows: list[Mapping[str, float]]) -> dict[str, dict[str, float | int]]:
    """Summarize in-memory timings after a connection closes."""
    if not rows:
        return {}
    summary: dict[str, dict[str, float | int]] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "samples": int(values.size),
            "min": float(values.min()),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
    return summary


def wire_response(result: dict) -> dict:
    """Project a ``apxinf`` policy result onto the wire response.

    Clients read ``actions`` and ``policy_timing``; every other key the library
    returns is in-process detail and is deliberately *not* serialized. The shape
    matches what ``openpi_client`` expects.
    """
    actions = np.ascontiguousarray(result["actions"], dtype=np.float32)
    timing = result.get("timing", {}) or {}
    policy_timing = {"infer_ms": float(timing.get("model_ms", 0.0))}
    if "total_ms" in timing:
        policy_timing["policy_ms"] = float(timing["total_ms"])
    for key in ("preprocess_ms", "postprocess_ms"):
        if key in timing:
            policy_timing[key] = float(timing[key])
    return {"actions": actions, "policy_timing": policy_timing}


class WebsocketPolicyServer:
    """Serve any ``apxinf.Policy`` over the websocket protocol (openpi-compatible wire)."""

    def __init__(
        self,
        policy: Any,
        host: str,
        port: int,
        *,
        metadata: dict | None = None,
        log: bool = False,
        log_stream: TextIO | None = None,
        log_path: Path | None = None,
        log_context: Mapping[str, Any] | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = (
            dict(metadata)
            if metadata is not None
            else dict(getattr(policy, "metadata", {}))
        )
        self._request_ids = itertools.count(1)
        self._diagnostic_log = AsyncJsonLogger(
            log,
            max_queue=_DIAGNOSTIC_LOG_QUEUE_SIZE,
            stream=log_stream,
            log_path=log_path,
        )
        self._log_context = dict(log_context or {})
        configure = getattr(policy, "set_diagnostics", None)
        if callable(configure):
            configure(log)
        self._diagnostic_log.emit(
            {
                "event": "server_config",
                "timestamp_ns": time.time_ns(),
                "host": host,
                "port": port,
                "metadata": self._metadata,
                "logging": {
                    "mode": "async_jsonl",
                    "request_sampling": "first_request",
                    "queue_capacity": _DIAGNOSTIC_LOG_QUEUE_SIZE,
                    "tensor_values_scanned": False,
                    "log_file": self._diagnostic_log.log_path,
                },
                "startup": self._log_context,
            }
        )

    async def handler(self, websocket: websocket_server.ServerConnection) -> None:
        logger.info("connection from %s opened", websocket.remote_address)
        pack = packer()
        await websocket.send(pack.pack(self._metadata))
        previous_total_ms = None
        connection_timings: list[dict[str, float]] = []
        while True:
            try:
                payload = await websocket.recv()
                if isinstance(payload, str):
                    raise TypeError(
                        "inference requests must be binary MessagePack frames"
                    )
                request_id = next(self._request_ids)
                request_started = time.perf_counter_ns()
                unpack_started = request_started
                observation = unpackb(payload)
                unpack_finished = time.perf_counter_ns()
                infer_started = unpack_finished
                # Call the policy directly on the event-loop thread rather than
                # offloading to a worker (``asyncio.to_thread``): the L1 handle
                # ``apxinf_py.Model`` is *unsendable* — its CUDA context is bound
                # to the thread that created it (the main thread, where the
                # policy was constructed), and touching it from a thread-pool
                # thread panics. Inference is one-at-a-time per GPU anyway, so
                # briefly blocking the loop here is the correct, simplest shape.
                result = self._policy.infer(observation)
                infer_finished = time.perf_counter_ns()
                response = wire_response(result)
                response["server_timing"] = {
                    "request_id": request_id,
                    "unpack_ms": (unpack_finished - unpack_started) / 1_000_000.0,
                    "infer_ms": (infer_finished - infer_started) / 1_000_000.0,
                    "payload_bytes": len(payload),
                }
                if previous_total_ms is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total_ms

                pack_started = time.perf_counter_ns()
                packed_response = pack.pack(response)
                pack_finished = time.perf_counter_ns()
                send_started = pack_finished
                await websocket.send(packed_response)
                send_finished = time.perf_counter_ns()
                previous_total_ms = (send_finished - request_started) / 1_000_000.0

                timing_record = None
                if self._diagnostic_log.enabled:
                    # Appending a handful of floats is the only per-request log
                    # work after the sampled records. Sorting/JSON output waits
                    # until the client disconnects, outside the inference path.
                    timing = result.get("timing", {}) or {}
                    timing_record = {
                        "unpack": (unpack_finished - unpack_started) / 1_000_000.0,
                        "preprocess": float(timing.get("preprocess_ms", 0.0)),
                        "model": float(timing.get("model_ms", 0.0)),
                        "postprocess": float(timing.get("postprocess_ms", 0.0)),
                        "policy": float(timing.get("total_ms", 0.0)),
                        "wire_response_and_pack": (pack_finished - infer_finished)
                        / 1_000_000.0,
                        "send": (send_finished - send_started) / 1_000_000.0,
                        "server_after_receive": previous_total_ms,
                    }
                    layer_timings = timing.get("layer_timings_ms")
                    if isinstance(layer_timings, Mapping):
                        timing_record["model_layers"] = {
                            str(name): float(value) for name, value in layer_timings.items()
                        }
                    for key in ("input_steps_ms", "output_steps_ms"):
                        steps = timing.get(key)
                        if isinstance(steps, Mapping):
                            timing_record[key] = {
                                str(name): float(value) for name, value in steps.items()
                            }
                    connection_timings.append(timing_record)

                should_log = request_id == _DIAGNOSTIC_REQUEST_ID
                if self._diagnostic_log.enabled and should_log:
                    assert timing_record is not None
                    self._diagnostic_log.emit(
                        {
                            "event": "request",
                            "timestamp_ns": time.time_ns(),
                            "request_id": request_id,
                            "cold_start_candidate": request_id == 1,
                            "remote": str(websocket.remote_address),
                            "payload_bytes": len(payload),
                            "response_bytes": len(packed_response),
                            "timing_ms": timing_record,
                            "observation": describe_value(observation),
                            "tensors": describe_value(result.get("diagnostic_tensors", {})),
                            "log_queue_dropped": self._diagnostic_log.dropped,
                        }
                    )
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                if self._diagnostic_log.enabled and connection_timings:
                    self._diagnostic_log.emit(
                        {
                            "event": "connection_summary",
                            "timestamp_ns": time.time_ns(),
                            "remote": str(websocket.remote_address),
                            "requests": len(connection_timings),
                            "timing_ms": _summarize_timings(connection_timings),
                            "log_queue_dropped": self._diagnostic_log.dropped,
                        }
                    )
                break
            except Exception:
                logger.exception("websocket inference failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                break

    async def run(self) -> None:
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        async with websocket_server.serve(
            self.handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=health_check,
        ) as server:
            logger.info("websocket policy server listening on %s", server.sockets)
            await server.serve_forever()

    @property
    def log_path(self) -> str | None:
        """Path of the automatically created JSONL file, if logging is enabled."""
        return self._diagnostic_log.log_path

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        finally:
            self.close()

    def close(self) -> None:
        """Flush and stop the optional asynchronous diagnostic logger."""
        self._diagnostic_log.close()


def health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
