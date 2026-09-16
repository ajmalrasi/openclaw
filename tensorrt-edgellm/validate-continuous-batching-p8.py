#!/usr/bin/env python3
"""Bounded live-service checks for the P8 continuous-batching rollout."""

import concurrent.futures
import json
import sys
import threading
import time

import httpx


URL = "http://127.0.0.1:11434"


def payload(label: str, tokens: int, *, stream: bool = False) -> dict:
    return {
        "model": "openclaw",
        "messages": [{"role": "user", "content": (
            "Write a numbered list of ordinary household objects. " + label)}],
        "temperature": 0,
        "max_tokens": tokens,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
    }


def request(body: dict) -> dict:
    with httpx.Client(timeout=80) as client:
        response = client.post(URL + "/v1/chat/completions", json=body)
    assert response.status_code == 200, (response.status_code, response.text)
    result = response.json()
    assert result["choices"][0]["message"]["content"]
    usage = result["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    return result


def stream(label: str, first: threading.Event | None = None) -> dict:
    began = time.monotonic()
    text, usage, done, stamps = "", None, False, []
    with httpx.Client(timeout=80) as client:
        with client.stream("POST", URL + "/v1/chat/completions",
                           json=payload(label, 96, stream=True)) as response:
            assert response.status_code == 200, (response.status_code, response.read().decode())
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                value = line[6:]
                if value == "[DONE]":
                    done = True
                    break
                chunk = json.loads(value)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    piece = choice.get("delta", {}).get("content")
                    if piece:
                        text += piece
                        stamps.append(time.monotonic())
                        if first:
                            first.set()
    assert done and text and usage and stamps, (done, text, usage)
    return {"text": text, "usage": usage, "ttft": stamps[0] - began,
            "last": stamps[-1]}


models = httpx.get(URL + "/v1/models", timeout=10)
assert models.status_code == 200, models.text
assert any(model["id"] == "openclaw" and model["owned_by"] == "tensorrt-edgellm"
           for model in models.json()["data"]), models.json()

health = httpx.get(URL + "/health", timeout=10).json()
assert health["status"] == "healthy", health
assert health["capabilities"]["max_num_seqs"] == 2, health
execution = health.get("execution", {})
assert execution.get("captures") == 3, health

single = request(payload("single request", 16))
first = threading.Event()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    a = pool.submit(stream, "first concurrent request", first)
    assert first.wait(30), "first stream did not begin"
    b = pool.submit(stream, "second concurrent request")
    a_result, b_result = a.result(), b.result()

assert b_result["ttft"] < a_result["last"], (a_result, b_result)
final_health = httpx.get(URL + "/health", timeout=10).json()
assert final_health["status"] == "healthy", final_health
assert final_health["active_requests"] == final_health["queued_requests"] == 0, final_health
assert final_health.get("execution", {}).get("replays", 0) > 0, final_health

print(json.dumps({
    "P8_LIVE_GATE": "passed",
    "single_usage": single["usage"],
    "first_ttft": a_result["ttft"],
    "second_ttft": b_result["ttft"],
    "execution": final_health.get("execution"),
}, sort_keys=True), flush=True)
