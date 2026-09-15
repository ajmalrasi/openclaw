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
    return dict(text=text, usage=usage, times=times)


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
    a, b, c = body('A.', 16), body('B.', 96, .7, 17), body('C.', 16, 1.1, 29)
    refs = [completion(p) for p in (a,b,c)]
    first = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        fa = pool.submit(stream, a, first)
        assert first.wait(15)
        fb = pool.submit(stream, b)
        for _ in range(80):
            state = health()
            if state['active_requests'] == 2:
                break
            time.sleep(.01)
        assert state['active_requests'] == 2, state
        record('two_resident', health=state)
        fc = pool.submit(stream, c)
        results = [f.result() for f in (fa,fb,fc)]
    for ref, result in zip(refs, results):
        assert ref['choices'][0]['message']['content'] == result['text'], (ref,result)
        assert ref['usage'] == result['usage'], (ref,result)
    assert results[1]['times'][0] < results[0]['times'][-1], results
    assert results[2]['times'][0] < results[1]['times'][-1], results
    record('P6_STAGGERED_GATE', passed=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        cancelled = pool.submit(stream, body('cancel', 96), None, True)
        peer = pool.submit(completion, c)
        cancelled.result()
        assert peer.result()['choices'][0]['message']['content'] == refs[2]['choices'][0]['message']['content']
    record('P6_DISCONNECT_GATE', passed=True)
    prompt = dict(body('logprobs', 8, 1.1), logprobs=True, top_logprobs=3)
    result = completion(prompt)
    assert all(item['logprob'] is not None for item in result['choices'][0]['logprobs']['content'])
    result_stream = stream(prompt)
    assert result['usage'] == result_stream['usage']
    assert result['choices'][0]['message']['content'] == result_stream['text']
    record('P6_LOGPROBS_GATE', passed=True)
    bad = dict(body('invalid', 9000))
    response = httpx.post(URL+'/v1/chat/completions', json=bad, timeout=10)
    assert response.status_code == 400, (response.status_code, response.text)
    assert completion(c)['choices'][0]['message']['content'] == refs[2]['choices'][0]['message']['content']
    record('P6_INVALID_REUSE_GATE', passed=True)
    assert health()['active_requests'] == 0
    long_payload = body(' '.join(['A cupboard contains plates, bowls and cups.'] * 80), 24)
    long_ref = completion(long_payload)
    first = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(stream, body('long peer', 32), first)
        assert first.wait(15)
        newcomer = pool.submit(stream, long_payload)
        long_result = newcomer.result()
        active.result()
    assert long_ref['usage']['prompt_tokens'] > 128
    assert long_ref['usage'] == long_result['usage']
    assert long_ref['choices'][0]['message']['content'] == long_result['text']
    record('P6_CHUNKED_HTTP_GATE', passed=True)
    tokenizer = json.loads(Path('/home/ajmalrasi/Qwen3.5-4B/quantized/tokenizer.json').read_text())
    x_id = tokenizer['model']['vocab']['X']
    eos_id = next(entry['id'] for entry in tokenizer['added_tokens'] if entry['content'] == '<|im_end|>')
    forced = dict(body('forced stop', 8), logit_bias={str(x_id): 100}, stop=['XX'])
    result = completion(forced)
    assert result['choices'][0]['message']['content'] in ('', None), result
    assert result['usage']['completion_tokens'] == 2, result
    assert stream(forced)['usage'] == result['usage']
    result = completion(dict(body('forced eos', 8), logit_bias={str(eos_id): 100}))
    assert result['usage']['completion_tokens'] == 1 and result['choices'][0]['finish_reason'] == 'stop', result
    record('P6_STOP_EOS_GATE', passed=True)
    tool_payload = dict(model='openclaw', temperature=0, max_tokens=96,
        messages=[dict(role='user', content='Call get_weather for Paris. Do not answer in prose.')],
        tools=[dict(type='function', function=dict(name='get_weather', description='Get weather for a city',
            parameters=dict(type='object', properties=dict(city=dict(type='string')), required=['city'])))],
        tool_choice='required')
    result = completion(tool_payload)
    calls = result['choices'][0]['message'].get('tool_calls')
    assert calls and calls[0]['function']['name'] == 'get_weather', result
    assert json.loads(calls[0]['function']['arguments'])['city'].lower() == 'paris', result
    record('P6_TOOL_GATE', passed=True)
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
    assert completion(c)['choices'][0]['message']['content'] == refs[2]['choices'][0]['message']['content']
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
