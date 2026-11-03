#!/usr/bin/env python3
"""Record a sequential AIOS weather run against an already running server.

Does not stop/start services, autotune, change clocks or compile CUDA. The caller
must reserve the GPU and start qwen35_aios_proxy.py with log/capture enabled.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time
import urllib.request
import uuid


CASE = "明天上海的天气怎么样"


def server_log(args):
    command = ["tail", "-c", "2097152", args.server_log]
    if args.ssh_host:
        command = ["ssh", args.ssh_host, shlex.join(command)]
    return subprocess.check_output(command).decode("utf-8", errors="replace")


def parse_profile(line):
    fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
    result = {}
    for key in ("prompt_tokens", "reused_prompt_tokens", "completion_tokens"):
        result[key] = int(fields[key])
    for key in ("ttft_ms", "total_latency_ms"):
        match = re.fullmatch(r"Some\(([^)]+)\)", fields[key])
        if not match:
            raise ValueError("missing measured " + key)
        result[key] = float(match[1])
    result["decode_ms"] = result["total_latency_ms"] - result["ttft_ms"]
    result["computed_prompt_tokens"] = result["prompt_tokens"] - result["reused_prompt_tokens"]
    result["decode_intervals"] = max(0, result["completion_tokens"] - 1)
    if min(result["decode_ms"], result["computed_prompt_tokens"]) < 0:
        raise ValueError("invalid profile counters")
    return result


def phases(e2e_ms, profiles, requests, responses, tool_execution_ms=None):
    if len(profiles) < 2 or len(requests) != len(profiles) or len(responses) != len(profiles):
        raise ValueError("model profiles and HTTP calls must match, with at least two calls")
    timestamp = lambda x: dt.datetime.fromisoformat(x["time"])
    gaps = [(timestamp(requests[i + 1]) - timestamp(responses[i])).total_seconds() * 1000 for i in range(len(profiles) - 1)]
    if isinstance(tool_execution_ms, (int, float)):
        tool_execution_ms = [tool_execution_ms]
    tools = gaps if tool_execution_ms is None else tool_execution_ms
    if len(tools) != len(gaps):
        raise ValueError("each inter-model gap must have exactly one weather-tool measurement")
    tool_label = "AMap 查询及工具编排间隔" if tool_execution_ms is None else "AMap天气查询（AIOS工具计时）"
    rows = []
    for index, profile in enumerate(profiles):
        rows.extend([
            ("模型调用%d Prefill/TTFT：新增 %d tokens" % (index + 1, profile["computed_prompt_tokens"]), profile["ttft_ms"]),
            ("模型调用%d Decode：%d 个间隔" % (index + 1, profile["decode_intervals"]), profile["decode_ms"]),
        ])
        if index < len(tools):
            tool_ms = tools[index]
            if not math.isfinite(tool_ms) or tool_ms < 0 or tool_ms > gaps[index] + 2.0:
                raise ValueError("tool measurement is outside the inter-model gap")
            rows.append((tool_label + (" %d" % (index + 1) if len(tools) > 1 else ""), tool_ms))
    other = e2e_ms - sum(value for _, value in rows)
    if any(gap < 0 for gap in gaps) or other < 0:
        raise ValueError("non-additive timeline; check clock or profile matching")
    rows.append(("AIOS、模型服务、HTTP及其他余项", other))
    return [{"stage": name, "time_ms": value, "percent": 100 * value / e2e_ms} for name, value in rows]


def weather_tool_snapshot(opener, aios_url):
    with opener.open(aios_url.rstrip("/") + "/api/admin/dashboard", timeout=10) as response:
        dashboard = json.load(response)
    name = "mcp__amap__maps_weather"
    return {
        "calls": dashboard.get("tool_calls", {}).get(name, 0),
        "recent": [item for item in dashboard.get("recent_tools", []) if item.get("tool") == name],
    }


def sse_events(path):
    for line in path.read_text().splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[6:])


def validate_weather(args, requests, responses, reply):
    if len(requests) < 2 or len(responses) != len(requests):
        return False, "expected matching model calls and responses"
    for request in requests:
        if any(event.get("error") for event in sse_events(Path(args.capture_dir) / (request["capture_id"] + ".response"))):
            return False, "model stream contains error, including after partial answer"
    if any(response.get("tool_call_names", ["mcp__amap__maps_weather"]) != ["mcp__amap__maps_weather"] for response in responses[:-1]):
        return False, "unexpected intermediate tool calls"
    if responses[-1].get("tool_call_names"):
        return False, "final model call did not finish with an answer"
    for request in requests[:-1]:
        fragments = []
        for event in sse_events(Path(args.capture_dir) / (request["capture_id"] + ".response")):
            if event.get("error"):
                return False, "model stream contains error"
            for choice in event.get("choices", []):
                fragments.extend((choice.get("delta") or {}).get("tool_calls") or [])
        names = [(item.get("function") or {}).get("name") for item in fragments]
        if "mcp__amap__maps_weather" not in names:
            return False, "missing structured AMap weather call"
        try:
            arguments = json.loads("".join((item.get("function") or {}).get("arguments", "") for item in fragments))
        except ValueError:
            return False, "malformed tool arguments"
        city = str(arguments.get("city", "")).strip().lower() if isinstance(arguments, dict) else ""
        if not city:
            return False, "empty city argument"
        if city not in ("上海", "上海市", "shanghai") and not re.fullmatch(r"31\d{4}", city):
            return False, "weather tool did not query Shanghai"
    payload = json.loads((Path(args.capture_dir) / (requests[-1]["capture_id"] + ".request")).read_text())
    # The harness may inject UTC while the task's user timezone is Shanghai;
    # time normalization can also erase the original hour. Require the date
    # fixed in the case manifest rather than silently adding a day to UTC.
    expected_date = getattr(args, "expected_target_date", None)
    try:
        tomorrow = dt.date.fromisoformat(expected_date).isoformat()
    except (TypeError, ValueError):
        return False, "explicit case_target_date is required for tomorrow validation"
    for message in payload.get("messages", []):
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message.get("content", ""))
        except ValueError:
            continue
        if not isinstance(result, dict):
            continue
        forecasts = result.get("forecasts", [result.get("forecast")])
        if not isinstance(forecasts, list):
            continue
        for forecast in forecasts:
            if not isinstance(forecast, dict) or forecast.get("date") != tomorrow:
                continue
            temps = [str(forecast.get(key, "")) for key in ("daytemp", "nighttemp")]
            weather = [str(forecast.get(key, "")) for key in ("dayweather", "nightweather")]
            numbers = set(re.findall(r"-?\d+(?:\.\d+)?", reply))
            valid = all(value and value in numbers for value in temps) and all(value and value in reply for value in weather)
            valid = valid and "<tool_call>" not in reply
            return valid, "next-day temperatures/weather match" if valid else "reply does not match next-day tool data"
    return False, "no successful next-day forecast in tool result"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aios-url", default="http://127.0.0.1:8081")
    parser.add_argument("--proxy-log", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--ssh-host")
    parser.add_argument("--metadata", required=True, help="JSON containing source_commit, binary_sha256, configuration, recipes and hardware")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata).read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", metadata.get("source_commit", "")):
        parser.error("metadata.source_commit must be a full verified Git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", metadata.get("binary_sha256", "")):
        parser.error("metadata.binary_sha256 must be a verified binary SHA256")
    try:
        args.expected_target_date = dt.date.fromisoformat(metadata["case_target_date"]).isoformat()
    except (KeyError, TypeError, ValueError):
        parser.error("metadata.case_target_date must explicitly specify the forecast date in the user's timezone")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups >= 0 and repeats >= 1 are required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows = []
    for index in range(args.warmups + args.repeats):
        log_before = server_log(args)
        proxy_offset = Path(args.proxy_log).stat().st_size
        tools_before = weather_tool_snapshot(opener, args.aios_url)
        session = "weather-audit-" + uuid.uuid4().hex
        payload = {"asr_text": CASE, "user_id": "weather_commit_audit", "session_id": session, "new_session": True, "language": "zh-CN"}
        request = urllib.request.Request(args.aios_url.rstrip("/") + "/v1/process", data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
        row = {"index": index, "phase": "warmup" if index < args.warmups else "formal", "session_id": session, "correct": False}
        started = time.monotonic()
        try:
            with opener.open(request, timeout=180) as response:
                result = json.load(response)
            row.update(aios=result, wall_ms=(time.monotonic() - started) * 1000)
            tools_after = weather_tool_snapshot(opener, args.aios_url)
            row["aios_weather_tool"] = {"before": tools_before, "after": tools_after}
            after = server_log(args)
            if after.startswith(log_before):
                delta = after[len(log_before):]
            else:
                raise ValueError("server log rotated/truncated; refusing profile attribution")
            row["server_profile_lines"] = [line for line in delta.splitlines() if "request profile:" in line or "MTP profile:" in line]
            profiles = [parse_profile(line) for line in delta.splitlines() if "request profile:" in line]
            with Path(args.proxy_log).open("rb") as stream:
                stream.seek(proxy_offset)
                events = [json.loads(line) for line in stream.read().decode().splitlines() if line.strip()]
            events = [x for x in events if x.get("method") == "POST" and x.get("path") == "/v1/chat/completions"]
            requests = [x for x in events if x.get("event") == "request"]
            by_id = {x["capture_id"]: x for x in events if x.get("event") == "response"}
            responses = [by_id[x["capture_id"]] for x in requests]
            row["proxy_events"] = [{key: x[key] for key in ("event", "time", "duration_ms", "capture_id", "status", "tool_call_names", "finish_reasons") if key in x} for x in events]
            row["profiles"] = profiles
            row["stream_errors"] = [
                event["error"]
                for request_event in requests
                for event in sse_events(Path(args.capture_dir) / (request_event["capture_id"] + ".response"))
                if event.get("error")
            ]
            row["correct"], row["validation"] = validate_weather(args, requests, responses, result.get("reply_text", ""))
            row["answer_correct"] = row["correct"]
            tool_count = tools_after["calls"] - tools_before["calls"]
            if tool_count != len(requests) - 1 or tool_count < 1 or len(tools_after["recent"]) < tool_count:
                raise ValueError("cannot attribute AIOS weather-tool timings to each model gap")
            tools = list(reversed(tools_after["recent"][:tool_count]))
            if any(tool.get("status") != "done" for tool in tools):
                raise ValueError("AIOS weather tool failed")
            row["inter_model_gaps_ms"] = [
                (dt.datetime.fromisoformat(requests[i + 1]["time"]) - dt.datetime.fromisoformat(responses[i]["time"])).total_seconds() * 1000
                for i in range(len(requests) - 1)
            ]
            row["stages"] = phases(result["execution_time_ms"], profiles, requests, responses, [tool["duration_ms"] for tool in tools])
        except Exception as error:
            row["error"] = str(error)
            row["correct"] = False
        rows.append(row)
        (output / ("run-%02d.json" % index)).write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: row.get(key) for key in ("index", "phase", "answer_correct", "correct", "validation", "error")}, ensure_ascii=False), flush=True)
    formal = [row for row in rows if row["phase"] == "formal"]
    valid = [row for row in formal if row["correct"] and row.get("stages")]
    times = sorted(row["aios"]["execution_time_ms"] for row in valid)
    summary = {"metadata": metadata, "metadata_file_sha256": hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest(), "answer_correct": sum(bool(row.get("answer_correct")) for row in formal), "correct": len(valid), "total": len(formal), "all_runs_valid": len(valid) == len(formal)}
    if times:
        summary.update(successful_subset_median_ms=statistics.median(times), successful_subset_p90_ms=times[math.ceil(.9 * len(times)) - 1])
    if len(valid) == len(formal):
        median = statistics.median(times)
        representative = min(valid, key=lambda row: abs(row["aios"]["execution_time_ms"] - median))
        text = ["# Weather E2E", "", "正式 %d 轮；以下为最接近中位数的同一轮，非各阶段独立分位数。单轮不能作为统计中位数。" % len(formal), "", "| 阶段 | 时间 | 占比 |", "|---|---:|---:|"]
        for stage in representative["stages"]:
            text.append("| %s | %.6f s | %.2f%% |" % (stage["stage"], stage["time_ms"] / 1000, stage["percent"]))
        text.append("| **端到端总计** | **%.6f s** | **100%%** |" % (representative["aios"]["execution_time_ms"] / 1000))
        (output / "weather.md").write_text("\n".join(text) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["all_runs_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
