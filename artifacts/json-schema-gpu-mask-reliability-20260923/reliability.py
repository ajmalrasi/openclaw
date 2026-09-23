#!/usr/bin/env python3
"""Bounded batch-two JSON Schema reliability check for the isolated candidate."""
import concurrent.futures
import json
import os
import time

import httpx

BASE = "http://127.0.0.1:11435"
PID = int(os.environ["CANDIDATE_SERVER_PID"])
DURATION = 165
OBJECT = {"type": "object", "properties": {"city": {"enum": ["Paris", "東京"]},
           "ok": {"type": "boolean"}}, "required": ["city", "ok"], "additionalProperties": False}
ARRAY = {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 9},
         "minItems": 3, "maxItems": 5}


def memory():
    with open(f"/proc/{PID}/status", encoding="utf-8") as stream:
        lines = stream.read().splitlines()
    with open("/proc/meminfo", encoding="utf-8") as stream:
        system = stream.read().splitlines()
    return {
        "process_kib": {key: int(value.split()[0]) for line in lines if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")) for key, value in [line.split(":", 1)]},
        "system_kib": {key: int(value.split()[0]) for line in system if line.startswith(("MemAvailable:", "SwapFree:")) for key, value in [line.split(":", 1)]},
    }


def run(schema, prompt):
    payload = {"model": "openclaw-json-candidate", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 128,
               "response_format": {"type": "json_schema", "json_schema":
                                   {"name": "reliability", "strict": True, "schema": schema}}}
    with httpx.Client(timeout=30) as client:
        response = client.post(BASE + "/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
    assert body["choices"][0]["finish_reason"] == "stop", body
    value = json.loads(body["choices"][0]["message"]["content"])
    if schema is OBJECT:
        assert set(value) == {"city", "ok"} and value["city"] in ("Paris", "東京") and type(value["ok"]) is bool
    else:
        assert isinstance(value, list) and 3 <= len(value) <= 5
        assert all(type(item) is int and 1 <= item <= 9 for item in value)
    return body["usage"]["completion_tokens"]


start = time.monotonic()
print(json.dumps({"event": "start", "memory": memory()}), flush=True)
pairs = 0
tokens = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    while time.monotonic() - start < DURATION:
        first = pool.submit(run, OBJECT, "Return a tiny city/status object.")
        second = pool.submit(run, ARRAY, "Return a short array of digits.")
        tokens += first.result() + second.result()
        pairs += 1
        if pairs % 5 == 0:
            with httpx.Client(timeout=5) as client:
                health = client.get(BASE + "/health").json()
            assert health["status"] == "healthy" and health["active_requests"] == 0, health
            print(json.dumps({"event": "checkpoint", "pairs": pairs, "tokens": tokens,
                              "elapsed_s": time.monotonic() - start, "memory": memory()}), flush=True)
with httpx.Client(timeout=5) as client:
    health = client.get(BASE + "/health").json()
assert health["status"] == "healthy" and health["active_requests"] == 0 and health["queued_requests"] == 0
print(json.dumps({"event": "GPU_MASK_RELIABILITY", "passed": True, "pairs": pairs,
                  "requests": pairs * 2, "tokens": tokens, "elapsed_s": time.monotonic() - start,
                  "memory": memory(), "health": health}), flush=True)
