"""CPU-only regressions for the reusable AIOS weather evidence tools."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from bench_qwen35_aios_weather import parse_profile, phases, validate_weather
from qwen35_aios_proxy import compress_amap_weather


class WeatherEvidenceTests(unittest.TestCase):
    def messages(self):
        return [
            {"role": "user", "content": "明天上海的天气怎么样"},
            {"role": "system", "content": "Time sampled while preparing turn 1, step 2: 2026-11-03T00:00:00+00:00[UTC]"},
            {"role": "tool", "content": json.dumps({"city": "上海市", "forecasts": [
                {"date": "2026-11-03", "daytemp": "20", "nighttemp": "16", "dayweather": "小雨", "nightweather": "中雨"},
                {"date": "2026-11-04", "daytemp": "19", "nighttemp": "15", "dayweather": "小雨", "nightweather": "小雨", "unused": "metadata"},
            ]}, ensure_ascii=False)},
        ]

    def test_compression_selects_next_day_and_is_idempotent(self):
        messages = self.messages()
        stats = compress_amap_weather(messages)
        result = json.loads(messages[-1]["content"])
        self.assertEqual(stats["target_date"], "2026-11-04")
        self.assertEqual(result["forecast"]["daytemp"], "19")
        self.assertNotIn("unused", result["forecast"])
        self.assertIsNone(compress_amap_weather(messages))

    def test_compression_requires_date_and_tomorrow_intent(self):
        for index in (0, 1):
            messages = self.messages()
            messages.pop(index)
            before = copy.deepcopy(messages)
            self.assertIsNone(compress_amap_weather(messages))
            self.assertEqual(before, messages)

    def test_missing_forecast_is_not_replaced_with_today(self):
        messages = self.messages()
        messages[-1]["content"] = '{"forecasts":[{"date":"2026-11-03"}]}'
        before = copy.deepcopy(messages)
        self.assertIsNone(compress_amap_weather(messages))
        self.assertEqual(before, messages)

    def test_profile_uses_server_ttft_and_measured_new_tokens(self):
        profile = parse_profile("request profile: prompt_tokens=9058 reused_prompt_tokens=8994 completion_tokens=25 ttft_ms=Some(123.387574) total_latency_ms=Some(457.407242)")
        self.assertEqual(profile["computed_prompt_tokens"], 64)
        self.assertEqual(profile["decode_intervals"], 24)
        self.assertAlmostEqual(profile["decode_ms"], 334.019668)

    def test_same_run_phases_sum_to_e2e_and_retries_fail(self):
        profiles = [
            {"computed_prompt_tokens": 1, "decode_intervals": 37, "ttft_ms": 47.458740, "decode_ms": 421.493843},
            {"computed_prompt_tokens": 64, "decode_intervals": 24, "ttft_ms": 123.387574, "decode_ms": 334.019668},
        ]
        requests = [{"time": "2026-11-03T05:05:20.387974+00:00"}, {"time": "2026-11-03T05:05:21.296010+00:00"}]
        responses = [{"time": "2026-11-03T05:05:20.876107+00:00"}, {"time": "2026-11-03T05:05:21.835343+00:00"}]
        rows = phases(1526.96, profiles, requests, responses)
        self.assertAlmostEqual(sum(row["time_ms"] for row in rows), 1526.96)
        self.assertAlmostEqual(sum(row["percent"] for row in rows), 100)
        self.assertAlmostEqual(rows[2]["time_ms"], 419.903)
        with self.assertRaises(ValueError):
            phases(1526.96, profiles * 2, requests, responses)

    def test_weather_text_without_structured_tool_call_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            event = {"choices": [{"delta": {"content": "上海晴，15到22度"}}]}
            (path / "a.response").write_text("data: " + json.dumps(event) + "\n\ndata: [DONE]\n")
            valid, reason = validate_weather(SimpleNamespace(capture_dir=directory), [{"capture_id": "a"}, {"capture_id": "b"}], [{}, {}], "上海晴，15到22度")
            self.assertFalse(valid)
            self.assertIn("structured", reason)

    def test_tool_success_still_requires_tomorrow_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            event = {"choices": [{"delta": {"tool_calls": [{"function": {"name": "mcp__amap__maps_weather", "arguments": '{"city":"上海"}'}}]}}]}
            (path / "a.response").write_text("data: " + json.dumps(event) + "\n\ndata: [DONE]\n")
            (path / "b.request").write_text(json.dumps({"messages": self.messages()}, ensure_ascii=False))
            args = SimpleNamespace(capture_dir=directory)
            reqs = [{"capture_id": "a"}, {"capture_id": "b"}]
            self.assertTrue(validate_weather(args, reqs, [{}, {}], "明天上海小雨，15到19度")[0])
            self.assertFalse(validate_weather(args, reqs, [{}, {}], "明天上海小雨转中雨，16到20度")[0])


if __name__ == "__main__":
    unittest.main()
