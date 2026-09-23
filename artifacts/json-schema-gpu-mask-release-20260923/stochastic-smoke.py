#!/usr/bin/env python3
"""Production non-greedy and SSE JSON-Schema smoke after GPU-mask release."""
import json

import httpx

BASE = "http://127.0.0.1:11434"
SCHEMA = {"type": "object", "properties": {
    "city": {"enum": ["Paris", "Tokyo", "Oslo"]},
    "count": {"type": "integer", "minimum": 1, "maximum": 9}},
    "required": ["city", "count"], "additionalProperties": False}
FORMAT = {"type": "json_schema", "json_schema": {"name": "stochastic", "strict": True, "schema": SCHEMA}}


def body(stream):
    return {"model": "openclaw", "messages": [{"role": "user", "content": "Return a city and count as JSON."}],
            "temperature": 0.7, "top_p": 0.9, "max_tokens": 128,
            "response_format": FORMAT, "stream": stream}


def validate(text):
    value = json.loads(text)
    assert set(value) == {"city", "count"}
    assert value["city"] in ("Paris", "Tokyo", "Oslo")
    assert type(value["count"]) is int and 1 <= value["count"] <= 9
    return value


with httpx.Client(timeout=30) as client:
    for index in range(8):
        response = client.post(BASE + "/v1/chat/completions", json=body(False))
        response.raise_for_status()
        result = response.json()
        assert result["choices"][0]["finish_reason"] == "stop"
        print("NON_GREEDY", index, validate(result["choices"][0]["message"]["content"]), flush=True)
    chunks = []
    finished = False
    with client.stream("POST", BASE + "/v1/chat/completions", json=body(True)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            result = json.loads(line[6:])
            for choice in result.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    chunks.append(content)
                finished = choice.get("finish_reason") == "stop" or finished
    assert finished
    print("NON_GREEDY_SSE", validate("".join(chunks)), flush=True)
    health = client.get(BASE + "/health").json()
    assert health["status"] == "healthy" and health["active_requests"] == 0, health
print("GPU_MASK_STOCHASTIC_GATE=passed", flush=True)
