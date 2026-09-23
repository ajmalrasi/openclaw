#!/usr/bin/env python3
"""Measure the existing deployed binding on the same production-sized engine."""
import json
import os
import time

import httpx

PID = int(os.environ["CANDIDATE_SERVER_PID"])
URL = "http://127.0.0.1:11435/v1/chat/completions"
PROMPT = "Return a JSON object containing twelve single-digit integers in a numbers array. No explanation."


def memory():
    with open(f"/proc/{PID}/status", encoding="utf-8") as stream:
        process = stream.read().splitlines()
    with open("/proc/meminfo", encoding="utf-8") as stream:
        system = stream.read().splitlines()
    return {
        "process_kib": {key: int(value.split()[0]) for line in process if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")) for key, value in [line.split(":", 1)]},
        "system_kib": {key: int(value.split()[0]) for line in system if line.startswith(("MemAvailable:", "SwapFree:")) for key, value in [line.split(":", 1)]},
    }


print(json.dumps({"event": "ready_memory", "memory": memory()}), flush=True)
body = {"model": "openclaw-json-candidate", "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "max_tokens": 96}
with httpx.Client(timeout=60) as client:
    for index in range(8):
        response = client.post(URL, json=body)
        response.raise_for_status()
        data = response.json()
        assert data["choices"][0]["finish_reason"] == "stop"
        print(json.dumps({"event": "request", "index": index, "memory": memory(),
                          "completion_tokens": data["usage"]["completion_tokens"]}), flush=True)
        time.sleep(0.1)
print(json.dumps({"event": "final_memory", "memory": memory()}), flush=True)
print("PHASE4H_BASELINE_MEMORY=passed", flush=True)
