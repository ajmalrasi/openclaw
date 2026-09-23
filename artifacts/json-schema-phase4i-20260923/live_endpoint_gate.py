# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded HTTP gate for JSON Schema phases 1-3."""

import concurrent.futures
import json
import sys
import time

import httpx


BASE = "http://127.0.0.1:11434"
CLIENT = httpx.Client(base_url=BASE, timeout=60)


def response_format(name, schema):
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"enum": ["Paris", "東京", "München"]},
        "temperature": {"type": "integer", "minimum": -20, "maximum": 50},
        "ok": {"type": "boolean"},
    },
    "required": ["city", "temperature", "ok"],
    "additionalProperties": False,
}

ARRAY_SCHEMA = {
    "type": "array",
    "items": {"type": "integer", "minimum": 1, "maximum": 9},
    "minItems": 3,
    "maxItems": 5,
}


def request(schema, prompt, *, stream=False, max_tokens=128):
    return {
        "model": "openclaw",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "response_format": response_format("gate", schema),
    }


def complete(schema, prompt):
    response = CLIENT.post("/v1/chat/completions", json=request(schema, prompt))
    response.raise_for_status()
    payload = response.json()
    return payload, json.loads(payload["choices"][0]["message"]["content"])


def check_object(value):
    assert set(value) == {"city", "temperature", "ok"}
    assert value["city"] in {"Paris", "東京", "München"}
    assert type(value["temperature"]) is int and -20 <= value["temperature"] <= 50
    assert type(value["ok"]) is bool


def check_array(value):
    assert isinstance(value, list) and 3 <= len(value) <= 5
    assert all(type(item) is int and 1 <= item <= 9 for item in value)


def stream_complete():
    text = ""
    terminal = None
    with CLIENT.stream("POST", "/v1/chat/completions",
                       json=request(OBJECT_SCHEMA, "Return weather JSON.", stream=True)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            payload = json.loads(data)
            if not payload.get("choices"):
                continue
            choice = payload["choices"][0]
            text += choice.get("delta", {}).get("content") or ""
            terminal = choice.get("finish_reason") or terminal
    value = json.loads(text)
    check_object(value)
    assert terminal == "stop"
    return value


def expect_400(payload, fragment):
    response = CLIENT.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400, response.text
    assert fragment in response.text, response.text


def active_requests():
    response = CLIENT.get("/health", timeout=5)
    response.raise_for_status()
    return response.json()["active_requests"]


def main():
    started = time.monotonic()
    payload, value = complete(OBJECT_SCHEMA, "Return one weather record as JSON.")
    check_object(value)
    assert payload["choices"][0]["finish_reason"] == "stop"
    print("SINGLE_OBJECT", json.dumps(value, ensure_ascii=False))

    stream_value = stream_complete()
    print("STREAM_OBJECT", json.dumps(stream_value, ensure_ascii=False))

    nested_schema = {
        "type": "object",
        "properties": {
            "place": {
                "type": "object",
                "properties": {"name": {"const": "東京😊"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        "required": ["place"],
        "additionalProperties": False,
    }
    _, nested = complete(nested_schema, "Return the required place JSON.")
    assert nested == {"place": {"name": "東京😊"}}, nested
    print("NESTED_UNICODE_GATE passed=1")

    bad_ref = request({"$ref": "https://example.com/schema.json"}, "x")
    expect_400(bad_ref, "only local references")
    unsupported = request({"type": "array", "uniqueItems": True}, "x")
    expect_400(unsupported, "unsupported keyword")
    conflict = request(OBJECT_SCHEMA, "x")
    conflict["tools"] = [{
        "type": "function",
        "function": {"name": "x", "parameters": {"type": "object"}},
    }]
    expect_400(conflict, "cannot be combined with tools")
    unresolved = request({"$ref": "#/$defs/missing"}, "x")
    expect_400(unresolved, "unresolved local reference")
    print("HTTP_400_GATE passed=1")

    incomplete = CLIENT.post("/v1/chat/completions",
                             json=request(ARRAY_SCHEMA, "Return an array.", max_tokens=1))
    assert incomplete.status_code == 500, incomplete.text
    assert "incomplete" in incomplete.text, incomplete.text
    print("INCOMPLETE_GATE passed=1")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        object_future = pool.submit(complete, OBJECT_SCHEMA, "Return weather JSON.")
        array_future = pool.submit(complete, ARRAY_SCHEMA, "Return a short integer array.")
        object_value = object_future.result()[1]
        array_value = array_future.result()[1]
    check_object(object_value)
    check_array(array_value)
    print("BATCH_TWO_GATE", json.dumps(object_value, ensure_ascii=False), json.dumps(array_value))

    long_schema = {
        "type": "array",
        "items": {"type": "string", "minLength": 64, "maxLength": 64},
        "minItems": 20,
        "maxItems": 20,
    }
    with CLIENT.stream("POST", "/v1/chat/completions",
                       json=request(long_schema, "Return twenty long strings.",
                                    stream=True, max_tokens=2048)) as response:
        response.raise_for_status()
        first = next(line for line in response.iter_lines() if line.startswith("data: "))
        assert first.startswith("data: {")
    for delay in (1, 2, 3):
        time.sleep(delay)
        if active_requests() == 0:
            break
    assert active_requests() == 0, "cancelled stream retained a native slot"
    _, reused = complete(ARRAY_SCHEMA, "Return a short integer array.")
    check_array(reused)
    print("CANCEL_REUSE_GATE passed=1")
    print(f"PHASE1_3_ENDPOINT_GATE passed=1 elapsed_seconds={time.monotonic() - started:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
