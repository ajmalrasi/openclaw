# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One externally capped comparison session, including all model warmup."""
import atexit
import signal
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import httpx

ROOT = Path('/home/ajmalrasi/continuous-batching-p7-20260915')
REFERENCE = Path('/home/ajmalrasi/TensorRT-Edge-LLM')
ENGINE = '/home/ajmalrasi/Qwen3.5-4B/engines/llm-b2-input6144-kv8192-vanilla'
CHECKPOINT = '/home/ajmalrasi/Qwen3.5-4B/quantized'
URL = 'http://127.0.0.1:11435'
START = time.monotonic()


def record(event, **data):
    print(json.dumps(dict(event=event, seconds=time.monotonic()-START, **data)), flush=True)


def percentile(values, p):
    return sorted(values)[min(len(values)-1, int(p * (len(values)-1)))] if values else None


def payload(text, count):
    return dict(model='openclaw', messages=[dict(role='user', content=text)],
                temperature=0, max_tokens=count, stream=True,
                stream_options=dict(include_usage=True))


PROMPT = 'List the integers from 1 to 100, each on a separate line. Start immediately.'


def request(body, first=None):
    begin = time.monotonic()
    stamps, text, usage, done = [], '', None, False
    with httpx.Client(timeout=90) as client:
        with client.stream('POST', URL+'/v1/chat/completions', json=body) as response:
            assert response.status_code == 200, (response.status_code, response.read().decode())
            for line in response.iter_lines():
                if not line.startswith('data: '):
                    continue
                if line[6:] == '[DONE]':
                    done = True
                    break
                data = json.loads(line[6:])
                assert 'error' not in data, data
                usage = data.get('usage') or usage
                for choice in data.get('choices', []):
                    piece = choice.get('delta', {}).get('content')
                    if piece:
                        text += piece
                        stamps.append(time.monotonic())
                        if first:
                            first.set()
    end = time.monotonic()
    assert done and usage and stamps, (done, usage, text)
    return dict(text=text, usage=usage, start=begin-START, end=end-START,
                latency=end-begin, ttft=stamps[0]-begin,
                stamps=[stamp-START for stamp in stamps],
                gaps=[b-a for a,b in zip(stamps, stamps[1:])])


def group(label, stagger=False, long=False):
    started = time.monotonic()
    prompt_b = (('The shelf holds books. ' * 1000) + '\n' + PROMPT) if long else PROMPT
    first = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(request, payload(prompt_b if long else PROMPT, 48 if not long else 8), first)
        if stagger and not long:
            assert first.wait(20)
        b = pool.submit(request, payload(prompt_b, 48 if not long else 8))
        results = [a.result(), b.result()]
    elapsed = time.monotonic()-started
    data = dict(label=label, elapsed=elapsed, aggregate_tps=sum(r['usage']['completion_tokens'] for r in results)/elapsed,
                results=results)
    record('group', **data)
    return data


def memory(pid):
    status = Path(f'/proc/{pid}/status').read_text()
    return dict(process=[line for line in status.splitlines() if line.startswith(('VmRSS:', 'VmHWM:', 'VmSwap:'))],
                system=[line for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith(('MemAvailable:', 'SwapFree:'))])


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, interrupted)
telemetry_log = (ROOT/'tegrastats.log').open('w')
telemetry = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=telemetry_log,
                             stderr=subprocess.STDOUT)


def stop_telemetry():
    telemetry.terminate()
    try:
        telemetry.wait(timeout=3)
    except subprocess.TimeoutExpired:
        telemetry.kill()
        telemetry.wait()
    telemetry_log.close()


atexit.register(stop_telemetry)
subprocess.run([str(ROOT/'build/continuous_batching_probe'), ENGINE, CHECKPOINT, '--graphs'],
               stdout=(ROOT/'graphs.log').open('w'), stderr=subprocess.STDOUT, check=True, timeout=70)
record('graph_qualification', passed=True)
all_results = {}
for mode in ('legacy', 'candidate', 'eager'):
    assert time.monotonic()-START < 235, 'insufficient remaining session budget'
    env = dict(os.environ, EDGELLM_CONTINUOUS_BATCHING='0' if mode == 'legacy' else '1',
               EDGELLM_CONTINUOUS_GRAPHS='1' if mode == 'candidate' else '0')
    if mode == 'legacy':
        env.pop('EDGELLM_PYBIND_DIR', None)
    else:
        env['EDGELLM_PYBIND_DIR'] = str(ROOT/'build')
    log = (ROOT/f'{mode}.log').open('w')
    server = subprocess.Popen([sys.executable, '-m', 'experimental.server', CHECKPOINT, '--engine-dir', ENGINE,
        '--host','127.0.0.1','--port','11435','--served-model-name','openclaw',
        '--reasoning-parser','qwen3','--tool-call-parser','qwen3_xml','--max-queued-requests','8'],
        cwd=REFERENCE if mode == 'legacy' else ROOT/'source', env=env, stdout=log, stderr=subprocess.STDOUT)
    (ROOT/'candidate.pid').write_text(str(server.pid))
    try:
        for _ in range(180):
            assert server.poll() is None, (ROOT/f'{mode}.log').read_text()[-6000:]
            try:
                health = httpx.get(URL+'/health',timeout=2).json()
                if health['status'] == 'healthy':
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.25)
        else:
            raise AssertionError('readiness timeout')
        before = memory(server.pid)
        record('ready', mode=mode, health=health, memory=before)
        request(payload(PROMPT, 8))
        singles = [request(payload(PROMPT, 32)) for _ in range(3 if mode != 'eager' else 1)]
        pair = group(mode+'-simultaneous')
        stagger = group(mode+'-staggered', True) if mode != 'eager' else None
        near = group(mode+'-near-capacity', True, True) if mode != 'eager' else None
        reuse = [request(payload(PROMPT, 4)) for _ in range(8)] if mode != 'eager' else []
        after = memory(server.pid)
        health = httpx.get(URL+'/health',timeout=3).json()
        all_results[mode] = dict(singles=singles, pair=pair, stagger=stagger, near=near,
                                 reuse=reuse, before=before, after=after, health=health)
        record('mode_result', mode=mode, result=all_results[mode])
        (ROOT/'benchmark-results.json').write_text(json.dumps(all_results, indent=2))
        if mode == 'candidate':
            assert health['execution']['captures'] == 3 and health['execution']['replays'] > 0, health
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()
        record('server_exit', mode=mode, code=server.returncode)
legacy, candidate = all_results['legacy'], all_results['candidate']
for old, new in zip(legacy['singles'], candidate['singles']):
    assert old['text'] == new['text'] and old['usage'] == new['usage'], (old,new)
old_latency = statistics.median(r['latency'] for r in legacy['singles'])
new_latency = statistics.median(r['latency'] for r in candidate['singles'])
gaps = [gap for result in candidate['pair']['results'] for gap in result['gaps']]
summary = dict(singleton_latency_ratio=new_latency/old_latency,
    singleton_pass=new_latency <= old_latency*1.1,
    aggregate_ratio=candidate['pair']['aggregate_tps']/legacy['pair']['aggregate_tps'],
    aggregate_pass=candidate['pair']['aggregate_tps'] > legacy['pair']['aggregate_tps'],
    staggered_ttft_legacy=legacy['stagger']['results'][1]['ttft'],
    staggered_ttft_candidate=candidate['stagger']['results'][1]['ttft'],
    median_gap=statistics.median(gaps), p95_gap=percentile(gaps,.95),
    elapsed=time.monotonic()-START)
record('P7_PERFORMANCE_GATE', **summary)
(ROOT/'benchmark-summary.json').write_text(json.dumps(summary, indent=2))
