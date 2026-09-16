# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded real HTTP validation. Run under an external timeout with one model."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import httpx

ROOT = Path('/home/ajmalrasi/continuous-batching-p6-20260915')
URL = 'http://127.0.0.1:11435'
START = time.monotonic()


def record(name, **values):
    print(json.dumps(dict(event=name, seconds=time.monotonic()-START, **values)), flush=True)


def body(label, count, temperature=0, seed=42):
    return dict(model='openclaw', messages=[dict(role='user', content=
        'Continue a numbered list of ordinary household objects without commentary. ' + label)],
        temperature=temperature, top_p=.9, top_k=30 if temperature else 1,
        seed=seed, max_tokens=count)


def completion(payload):
    start = time.monotonic()
    with httpx.Client(timeout=65) as client:
        response = client.post(URL+'/v1/chat/completions', json=payload)
    assert response.status_code == 200, (response.status_code, response.text)
    data = response.json()
    assert data['usage']['total_tokens'] == sum(data['usage'][key] for key in ('prompt_tokens','completion_tokens'))
    record('completion', elapsed=time.monotonic()-start, result=data)
    return data


def stream(payload, first=None, disconnect=False):
    payload = dict(payload, stream=True, stream_options=dict(include_usage=True))
    chunks, text, times, usage, done, terminal = [], '', [], None, False, 0
    with httpx.Client(timeout=65) as client:
        with client.stream('POST', URL+'/v1/chat/completions', json=payload) as response:
            assert response.status_code == 200, (response.status_code, response.read().decode())
            for line in response.iter_lines():
                if not line.startswith('data: '):
                    continue
                value = line[6:]
                if value == '[DONE]':
                    done = True
                    break
                chunk = json.loads(value)
                assert 'error' not in chunk, chunk
                chunks.append(chunk)
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    if choice.get('finish_reason'):
                        terminal += 1
                    piece = choice.get('delta', {}).get('content')
                    if piece:
                        text += piece
                        times.append(time.monotonic()-START)
                        if first:
                            first.set()
                        if disconnect:
                            record('disconnect', text=text)
                            return None
    assert done and terminal == 1 and usage, (done, terminal, usage, chunks)
    record('stream', text=text, usage=usage, times=times, chunks=chunks)
    return dict(text=text, usage=usage, times=times, chunks=chunks)


def health():
    return httpx.get(URL+'/health', timeout=3).json()


log = (ROOT/'candidate.log').open('w')
server = subprocess.Popen([sys.executable, '-m', 'experimental.server',
    '/home/ajmalrasi/Qwen3.5-4B/quantized', '--engine-dir',
    '/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla',
    '--host','127.0.0.1','--port','11435','--served-model-name','openclaw',
    '--reasoning-parser','qwen3','--tool-call-parser','qwen3_xml',
    '--enable-auto-tool-choice','--max-queued-requests','8','--queue-timeout','30'],
    cwd=ROOT/'source', stdout=log, stderr=subprocess.STDOUT)
(ROOT/'candidate.pid').write_text(str(server.pid))
try:
    for _ in range(180):
        assert server.poll() is None, (ROOT/'candidate.log').read_text()[-8000:]
        try:
            state = health()
            if state['status'] == 'healthy':
                break
        except httpx.HTTPError:
            pass
        time.sleep(.25)
    else:
        raise AssertionError('candidate readiness timeout')
    assert state['capabilities']['max_num_seqs'] == 2, state
    record('ready', health=state)
    tool_payload = dict(model='openclaw', temperature=0, max_tokens=96,
        messages=[dict(role='user', content='Call get_weather for Paris. Do not answer in prose.')],
        tools=[dict(type='function', function=dict(name='get_weather', description='Get weather for a city',
            parameters=dict(type='object', properties=dict(city=dict(type='string')), required=['city'])))],
        tool_choice='required')
    result = stream(tool_payload)
    calls = [call for chunk in result['chunks'] for choice in chunk.get('choices', [])
             for call in (choice.get('delta', {}).get('tool_calls') or [])]
    assert any(call.get('function', {}).get('name') == 'get_weather' for call in calls), calls
    arguments = ''.join(call.get('function', {}).get('arguments', '') for call in calls)
    assert json.loads(arguments)['city'].lower() == 'paris', calls
    record('P6_TOOL_STREAM_GATE', passed=True)
    thinking = stream(dict(body('brief thought', 16), enable_thinking=True))
    assert any(choice.get('delta', {}).get('reasoning_content') for chunk in thinking['chunks']
               for choice in chunk.get('choices', [])), thinking
    record('P6_REASONING_GATE', passed=True)
    tokenizer = json.loads(Path('/home/ajmalrasi/Qwen3.5-4B/quantized/tokenizer.json').read_text())
    x_id = tokenizer['model']['vocab']['X']
    release_headers = threading.Event()
    statuses = []
    status_lock = threading.Lock()
    def overload_client():
        with httpx.Client(timeout=15) as client:
            with client.stream('POST', URL+'/v1/chat/completions', json=dict(
                    body('overload', 128), stream=True, logit_bias={str(x_id):100})) as response:
                with status_lock:
                    statuses.append(response.status_code)
                    if len(statuses) == 12:
                        release_headers.set()
                assert release_headers.wait(12)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda _: overload_client(), range(12)))
    assert 429 in statuses and 200 in statuses and set(statuses) <= {200,429}, statuses
    for _ in range(100):
        state = health()
        if state['active_requests'] == state['queued_requests'] == 0:
            break
        time.sleep(.05)
    assert state['active_requests'] == state['queued_requests'] == 0, state
    record('P6_OVERLOAD_GATE', passed=True, statuses=statuses)
    completion(body('after overload', 4))
    record('P6_HTTP_GATE', passed=True)
finally:
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    record('server_exit', code=server.returncode)
    log.close()
