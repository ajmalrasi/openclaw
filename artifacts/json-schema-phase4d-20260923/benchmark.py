#!/usr/bin/env python3
"""Short, paired HTTP calibration of the isolated full-binding candidate."""
import concurrent.futures
import json
import os
import statistics
import time

import httpx

URL = "http://127.0.0.1:11435/v1/chat/completions"
PROMPT = "Return a JSON object containing twelve single-digit integers in a numbers array. No explanation."
SCHEMA = {
    "type": "object",
    "properties": {"numbers": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 9}, "minItems": 12, "maxItems": 12}},
    "required": ["numbers"],
    "additionalProperties": False,
}
GUIDE = {"type": "json_schema", "json_schema": {"name": "numbers", "strict": True, "schema": SCHEMA}}
PID = int(os.environ["CANDIDATE_SERVER_PID"])


def memory():
    with open(f"/proc/{PID}/status", encoding="utf-8") as stream:
        status = stream.read().splitlines()
    with open("/proc/meminfo", encoding="utf-8") as stream:
        system = stream.read().splitlines()
    return {
        "process_kib": {key: int(value.split()[0]) for line in status if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")) for key, value in [line.split(":", 1)]},
        "system_kib": {key: int(value.split()[0]) for line in system if line.startswith(("MemAvailable:", "SwapFree:")) for key, value in [line.split(":", 1)]},
    }


def run(mode):
    body = {"model": "openclaw-json-candidate", "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0, "max_tokens": 96, "stream": True, "stream_options": {"include_usage": True}}
    if mode == "guided":
        body["response_format"] = GUIDE
    start = time.monotonic()
    stamps, chunks, usage, finish, done = [], [], None, None, False
    with httpx.Client(timeout=60) as client:
        with client.stream("POST", URL, json=body) as response:
            assert response.status_code == 200, (mode, response.status_code, response.read().decode())
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    done = True
                    break
                event = json.loads(line[6:])
                assert "error" not in event, event
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        chunks.append(delta)
                        stamps.append(time.monotonic())
                    finish = choice.get("finish_reason") or finish
    end = time.monotonic()
    assert done and usage and stamps and finish == "stop", (mode, done, usage, finish)
    output = "".join(chunks)
    if mode == "guided":
        parsed = json.loads(output)
        assert list(parsed) == ["numbers"] and len(parsed["numbers"]) == 12
        assert all(type(value) is int and 0 <= value <= 9 for value in parsed["numbers"])
    return {"mode": mode, "latency_s": end - start, "ttft_s": stamps[0] - start,
            "decode_s": end - stamps[0], "completion_tokens": usage["completion_tokens"],
            "decode_tokens_per_s": max(0, usage["completion_tokens"] - 1) / max(0.001, end - stamps[0]),
            "chunk_count": len(chunks), "finish_reason": finish, "output": output}


start = time.monotonic()
print(json.dumps({"event": "ready_memory", "memory": memory()}), flush=True)
run("plain")
run("guided")
print(json.dumps({"event": "post_warmup_memory", "memory": memory()}), flush=True)
samples = []
for mode in ("plain", "guided", "guided", "plain", "plain", "guided"):
    result = run(mode)
    samples.append(result)
    print(json.dumps({"event": "sample", **result}), flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    pair = list(pool.map(run, ("guided", "guided")))
print(json.dumps({"event": "guided_pair", "results": pair}), flush=True)
print(json.dumps({"event": "final_memory", "memory": memory()}), flush=True)
for mode in ("plain", "guided"):
    subset = [sample for sample in samples if sample["mode"] == mode]
    print(json.dumps({"event": "summary", "mode": mode, "n": len(subset),
                      "median_latency_s": statistics.median(x["latency_s"] for x in subset),
                      "median_ttft_s": statistics.median(x["ttft_s"] for x in subset),
                      "median_decode_tokens_per_s": statistics.median(x["decode_tokens_per_s"] for x in subset),
                      "median_completion_tokens": statistics.median(x["completion_tokens"] for x in subset)}), flush=True)
print(json.dumps({"event": "PHASE4D_CALIBRATION", "passed": True, "elapsed_s": time.monotonic() - start}), flush=True)
