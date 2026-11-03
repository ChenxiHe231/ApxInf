#!/usr/bin/env python3
"""OpenAI-compatible AIOS experiment proxy with explicit opt-in transforms.

Short SSE streams are buffered by upstream.read(65536); first_body_ms therefore
does not measure model TTFT. Use the inference server's request profile instead.
Captures can contain private prompts/tool output: keep them under devlocal and
publish only sanitized timing/usage evidence, never credentials or raw payloads.
"""

import argparse
import datetime as dt
import http.client
import json
import pathlib
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


LOG_LOCK = threading.Lock()
AMAP_WEATHER_FIELDS = (
    "date",
    "week",
    "dayweather",
    "nightweather",
    "daytemp",
    "nighttemp",
    "daywind",
    "nightwind",
    "daypower",
    "nightpower",
)


def write_log(path, record):
    record["time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with LOG_LOCK:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")


SHANGHAI_TZ = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")
TIME_CONTEXT = re.compile(
    r"(Time sampled while preparing turn \d+, step \d+: )"
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))"
    r"(?:\[[^\]\r\n]+\])?"
)


def normalize_time_context(messages):
    """Canonicalize the weather benchmark's clock to its Shanghai date."""
    changed = 0
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            continue
        def replace(match):
            value = dt.datetime.fromisoformat(match[2].replace("Z", "+00:00"))
            date = value.astimezone(SHANGHAI_TZ).date().isoformat()
            return match[1] + date + "T00:00:00+08:00[Asia/Shanghai]"
        content, count = TIME_CONTEXT.subn(replace, message["content"])
        if content != message["content"]:
            message["content"] = content
            changed += count
    return changed


def compress_amap_weather(messages):
    """Keep only the requested tomorrow forecast in AMap tool results."""
    sampled_date = None
    wants_tomorrow = False
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            continue
        content = message["content"]
        if message.get("role") == "user" and "明天" in content:
            wants_tomorrow = True
        matches = list(TIME_CONTEXT.finditer(content))
        if matches:
            sampled_date = dt.datetime.fromisoformat(matches[-1][2].replace("Z", "+00:00")).astimezone(SHANGHAI_TZ).date()
    if not wants_tomorrow or sampled_date is None:
        return None

    target_date = (sampled_date + dt.timedelta(days=1)).isoformat()
    changes = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            result = json.loads(content)
        except (TypeError, ValueError):
            continue
        if not isinstance(result, dict) or not isinstance(result.get("forecasts"), list):
            continue
        forecast = next(
            (
                item
                for item in result["forecasts"]
                if isinstance(item, dict) and item.get("date") == target_date
            ),
            None,
        )
        if forecast is None:
            continue
        if not result.get("city") or not all(forecast.get(key) is not None for key in ("daytemp", "nighttemp", "dayweather", "nightweather")):
            continue
        compact = {
            "city": result.get("city"),
            "forecasts": [{
                field: forecast[field]
                for field in AMAP_WEATHER_FIELDS
                if field in forecast
            }],
        }
        replacement = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if replacement == content:
            continue
        message["content"] = replacement
        changes.append({
            "original_chars": len(content),
            "compressed_chars": len(replacement),
        })
    if not changes:
        return None
    return {"target_date": target_date, "tool_results": len(changes),
            "original_chars": sum(item["original_chars"] for item in changes),
            "compressed_chars": sum(item["compressed_chars"] for item in changes)}


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        return

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def _forward(self):
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        request_meta = {
            "event": "request",
            "method": self.command,
            "path": self.path,
            "bytes": len(body),
        }
        outbound = body
        if body and "json" in self.headers.get("Content-Type", ""):
            try:
                payload = json.loads(body)
                messages = payload.get("messages") or []
                request_meta.update(
                    model=payload.get("model"),
                    temperature=payload.get("temperature"),
                    top_p=payload.get("top_p"),
                    top_k=payload.get("top_k"),
                    max_tokens=payload.get("max_tokens"),
                    stream=payload.get("stream"),
                    messages=len(messages),
                    message_chars=sum(
                        len(str(message.get("content") or ""))
                        for message in messages
                        if isinstance(message, dict)
                    ),
                    tools=len(payload.get("tools") or []),
                    tool_names=[
                        str((tool.get("function") or {}).get("name") or tool.get("name") or "")
                        for tool in (payload.get("tools") or [])
                        if isinstance(tool, dict)
                    ],
                    presence_penalty=payload.get("presence_penalty"),
                    frequency_penalty=payload.get("frequency_penalty"),
                )
                if self.server.force_temperature is not None:
                    request_meta["original_temperature"] = payload.get("temperature")
                    payload["temperature"] = self.server.force_temperature
                    request_meta["temperature"] = self.server.force_temperature
                    outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.sampling_defaults:
                    request_meta["original_sampling"] = {
                        "temperature": payload.get("temperature"),
                        "top_k": payload.get("top_k"),
                        "top_p": payload.get("top_p"),
                        "seed": payload.get("seed"),
                    }
                    payload.setdefault("temperature", 1.0)
                    payload.setdefault("top_k", 20)
                    payload.setdefault("top_p", 0.95)
                    if self.server.seed is not None:
                        payload.setdefault("seed", self.server.seed)
                    request_meta.update(
                        temperature=payload["temperature"],
                        top_k=payload["top_k"],
                        top_p=payload["top_p"],
                        seed=payload.get("seed"),
                    )
                    outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.disable_spec_decode:
                    payload["disable_spec_decode"] = True
                    request_meta["disable_spec_decode"] = True
                    outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.single_tool_call:
                    payload["parallel_tool_calls"] = False
                    request_meta["parallel_tool_calls"] = False
                    outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.reuse_context:
                    payload["reuse_context"] = True
                    request_meta["reuse_context"] = True
                    outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.no_reuse_context:
                    payload["reuse_context"] = False
                    request_meta["reuse_context"] = False
                    outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.no_cache_generated_tokens:
                    payload["cache_generated_tokens"] = False
                    request_meta["cache_generated_tokens"] = False
                    outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.normalize_time_context:
                    normalized = normalize_time_context(messages)
                    if normalized:
                        request_meta["normalized_time_contexts"] = normalized
                        outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.compress_amap_weather:
                    compression = compress_amap_weather(messages)
                    if compression:
                        request_meta["amap_weather_compression"] = compression
                        outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.require_initial_tool and len(messages) <= 4:
                    request_meta["original_tool_choice"] = payload.get("tool_choice")
                    payload["tool_choice"] = "required"
                    request_meta["tool_choice"] = "required"
                    outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if self.server.force_initial_tool and len(messages) <= 4:
                    name = self.server.force_initial_tool
                    matching = [
                        tool
                        for tool in (payload.get("tools") or [])
                        if isinstance(tool, dict)
                        and str((tool.get("function") or {}).get("name") or "") == name
                    ]
                    if len(matching) != 1:
                        raise ValueError(
                            f"expected exactly one initial tool named {name!r}, "
                            f"found {len(matching)}"
                        )
                    request_meta["original_tool_choice"] = payload.get("tool_choice")
                    if self.server.restrict_initial_tools:
                        request_meta["original_tools"] = len(payload.get("tools") or [])
                        payload["tools"] = matching
                        request_meta["tools"] = 1
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": name},
                    }
                    request_meta["tool_choice"] = name
                    outbound = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            except (ValueError, TypeError):
                request_meta["json_parse_failed"] = True
        if self.server.capture_dir:
            capture_dir = pathlib.Path(self.server.capture_dir)
            capture_dir.mkdir(parents=True, exist_ok=True)
            (capture_dir / f"{request_id}.request").write_bytes(outbound)
            request_meta["capture_id"] = request_id
        write_log(self.server.log_path, request_meta)

        connection = http.client.HTTPConnection(
            self.server.upstream_host, self.server.upstream_port, timeout=900
        )
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        headers["Host"] = f"{self.server.upstream_host}:{self.server.upstream_port}"
        headers["Content-Length"] = str(len(outbound))
        headers["Connection"] = "close"
        captured = bytearray()
        total_bytes = 0
        first_body_at = None
        try:
            connection.request(self.command, self.path, body=outbound, headers=headers)
            upstream = connection.getresponse()
            self.send_response(upstream.status, upstream.reason)
            for key, value in upstream.getheaders():
                if key.lower() not in {"content-length", "transfer-encoding", "connection"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = upstream.read(65536)
                if not chunk:
                    break
                if first_body_at is None:
                    first_body_at = time.monotonic()
                total_bytes += len(chunk)
                if len(captured) < 65536:
                    captured.extend(chunk[: 65536 - len(captured)])
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
            ended = time.monotonic()
            response_meta = {
                "event": "response",
                "method": self.command,
                "path": self.path,
                "status": upstream.status,
                "duration_ms": round((ended - started) * 1000, 2),
                "first_body_ms": (
                    round((first_body_at - started) * 1000, 2) if first_body_at else None
                ),
                "bytes": total_bytes,
                "content_type": upstream.getheader("Content-Type"),
            }
            if self.server.capture_dir:
                capture_dir = pathlib.Path(self.server.capture_dir)
                (capture_dir / f"{request_id}.response").write_bytes(captured)
                response_meta["capture_id"] = request_id
            text = bytes(captured).decode("utf-8", errors="replace")
            if upstream.status >= 400:
                response_meta["error_preview"] = text[:2048]
            elif "application/json" in (upstream.getheader("Content-Type") or ""):
                try:
                    result = json.loads(text)
                    choice = (result.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    response_meta.update(
                        finish_reason=choice.get("finish_reason"),
                        output_chars=len(str(message.get("content") or "")),
                        usage=result.get("usage"),
                    )
                except (ValueError, TypeError):
                    response_meta["response_json_parse_failed"] = True
            else:
                data_lines = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
                response_meta["sse_data_lines_captured"] = len(data_lines)
                response_meta["sse_done_captured"] = "data: [DONE]" in text
                tool_names = []
                finish_reasons = []
                output_chars = 0
                for data_line in data_lines:
                    if data_line == "[DONE]":
                        continue
                    try:
                        event = json.loads(data_line)
                    except ValueError:
                        continue
                    for choice in event.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reasons.append(choice["finish_reason"])
                        delta = choice.get("delta") or {}
                        output_chars += len(str(delta.get("content") or ""))
                        for tool_call in delta.get("tool_calls") or []:
                            name = ((tool_call.get("function") or {}).get("name"))
                            if name and name not in tool_names:
                                tool_names.append(name)
                response_meta["output_chars"] = output_chars
                response_meta["tool_call_names"] = tool_names
                response_meta["finish_reasons"] = finish_reasons
            write_log(self.server.log_path, response_meta)
        except Exception as error:  # noqa: BLE001 - proxy must report upstream failures
            write_log(
                self.server.log_path,
                {
                    "event": "proxy_error",
                    "method": self.command,
                    "path": self.path,
                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                    "error": repr(error),
                },
            )
            if not self.wfile.closed:
                try:
                    payload = json.dumps({"error": {"message": str(error)}}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            self.close_connection = True
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=18001)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=18000)
    parser.add_argument("--log", required=True)
    parser.add_argument("--force-temperature", type=float)
    parser.add_argument("--sampling-defaults", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--disable-spec-decode", action="store_true")
    parser.add_argument("--single-tool-call", action="store_true", help="Request one tool call per model response; enables native tool-terminator stopping on supporting models")
    parser.add_argument("--reuse-context", action="store_true")
    parser.add_argument("--no-reuse-context", action="store_true")
    parser.add_argument("--no-cache-generated-tokens", action="store_true")
    parser.add_argument("--normalize-time-context", action="store_true")
    parser.add_argument("--compress-amap-weather", action="store_true")
    parser.add_argument("--require-initial-tool", action="store_true")
    parser.add_argument("--force-initial-tool")
    parser.add_argument("--restrict-initial-tools", action="store_true")
    parser.add_argument("--capture-dir")
    args = parser.parse_args()
    if args.reuse_context and args.no_reuse_context:
        parser.error("--reuse-context and --no-reuse-context are mutually exclusive")

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    server.daemon_threads = True
    server.upstream_host = args.upstream_host
    server.upstream_port = args.upstream_port
    server.log_path = args.log
    server.force_temperature = args.force_temperature
    server.sampling_defaults = args.sampling_defaults
    server.seed = args.seed
    server.disable_spec_decode = args.disable_spec_decode
    server.single_tool_call = args.single_tool_call
    server.reuse_context = args.reuse_context
    server.no_reuse_context = args.no_reuse_context
    server.no_cache_generated_tokens = args.no_cache_generated_tokens
    server.normalize_time_context = args.normalize_time_context
    server.compress_amap_weather = args.compress_amap_weather
    server.require_initial_tool = args.require_initial_tool
    server.force_initial_tool = args.force_initial_tool
    server.restrict_initial_tools = args.restrict_initial_tools
    server.capture_dir = args.capture_dir
    print(
        f"proxy http://{args.listen_host}:{args.listen_port} -> "
        f"http://{args.upstream_host}:{args.upstream_port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
