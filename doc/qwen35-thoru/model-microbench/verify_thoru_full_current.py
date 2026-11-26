"""Recompute the published Thor-U TRT/Final tables; optionally verify raw assets."""
import argparse
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('evidence', type=Path)
p.add_argument('--raw-root', type=Path)
a = p.parse_args()
e = json.loads(a.evidence.read_text())
assert e['schema'] == 'apxinf.thoru_current_full_benchmark.evidence.v1'
assert e['source_commit'] == 'ae909d2b0f992fbd3a2b33baa88b60057e96c479'
assert all(e['postflight'].values())
assert e['build']['completed'] and e['build']['source_commit']==e['source_commit']
assert e['resource_receipt']['policy_run']=={'governor':'performance','min_freq':'1530000000','max_freq':'1530000000'}
assert e['resource_receipt']['desktop_before']['monitor_off'] and e['resource_receipt']['desktop_after']['monitor_off']
assert e['resource_restored'] and e['scored_requests'] == 240 and e['warmup_requests'] == 48
expected = {f'decode-{n}k' for n in [4, 8, 16, 24, 32]}
expected |= {f'append-{n}k' for n in [4, 8, 16, 32]}
expected |= {f'text-decode-{n}k' for n in [4, 16, 32]}
assert {r['case']['id'] for r in e['rows']} == expected

def close(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11), (actual, expected)

summary = []
for row in e['rows']:
    case = row['case']
    assert case['kv'] <= 32768 and case['input_tokens'] < 49152
    for backend, result in row['backends'].items():
        data = result['rows']
        assert len(data) == 12 and sum(x['warmup'] for x in data) == 2
        assert [x['iteration'] for x in data] == list(range(12))
        assert all(x['output_tokens'] == case['outputs'] for x in data)
        measured = [x for x in data if not x['warmup']]
        assert len(measured) == 10
        if backend == 'apxinf':
            assert result['distinct_outputs'] == 1
            assert all(x['strict_recipe_required'] and x['performance_eligible']
                       and not x['autotune'] and not x['profiling'] and not x['repeat_diverged'] for x in data)
            if case['kind'] == 'decode':
                assert all(x['computed_prompt_tokens'] == case['input_tokens'] and x['reused_prompt_tokens'] == 0 for x in data)
                assert all(x['speculation']['draft_steps'] == 3 and x['speculation']['target_verified_rows'] == 4*x['speculation']['rounds'] for x in data)
            else:
                assert all(x['initial_kv_tokens'] == case['kv'] and x['computed_input_tokens'] == 4096 for x in data)
        else:
            reuse = case['kv'] if case['kind'] == 'append' else 0
            assert all(x['input_tokens'] == case['input_tokens'] and
                       ((case['input_tokens']-2048 <= x['delta']['reused_tokens'] < case['input_tokens']) if x.get('context_mode')=='cached-prefix' else x['delta']['reused_tokens'] == reuse)
                       and x['computed_input_tokens_derived'] == case['input_tokens']-x['delta']['reused_tokens']
                       and not x['native_profiling_enabled'] for x in data)
        metrics = result['metrics']
        if case['kind'] == 'decode':
            assert all(x['decode_intervals'] == 1023 for x in data)
            close(metrics['decode_tps'], 1023*10000/math.fsum(x['decode_interval_ms'] for x in measured))
            close(metrics['request_output_tps'] if result['request_scope']=='cached-prefix-generation' else metrics['e2e_tps'], 1024*10000/math.fsum(x['total_latency_ms'] for x in measured))
            close(metrics['ttft_ms'], math.fsum(x['ttft_ms'] for x in measured)/10)
        else:
            close(metrics['prefill_ms'], math.fsum(x['prefill_ms'] for x in measured)/10)
        assert result['minimum_ordinary_memory_gib'] >= 3
        assert result['frequency_samples'] > 0 and result['gpu_hw_samples'] > 0
        entry = result['entry']
        assert entry['desktop_before']['monitor_off'] and entry['desktop_after']['monitor_off']
        assert entry['returncode'] == 0 and not entry['faults'] and entry['warmups'] == 2 and entry['repeats'] == 10
    am = row['backends']['apxinf']['metrics']
    tm = row['backends']['trt']['metrics']
    for metric in ((['decode_tps', 'e2e_tps'] if row['cold_e2e_comparable'] else ['decode_tps']) if case['kind'] == 'decode' else ['prefill_ms']):
        gain = 100*(1-am[metric]/tm[metric]) if metric == 'prefill_ms' else 100*(am[metric]/tm[metric]-1)
        close(row['comparison'][metric], gain)
    summary.append({'case': case['id'], 'apxinf': am, 'trt': tm, 'comparison': row['comparison'],
                    'cold_e2e_comparable':row['cold_e2e_comparable'],'trt_distinct_outputs': row['backends']['trt']['distinct_measured_outputs']})
if a.raw_root:
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, digest in e['file_hashes'].items():
        assert sha(a.raw_root/name) == digest, name
    for name, binary in e['build']['binaries'].items():
        assert sha(a.raw_root/(name+'_current')) == binary['sha256'], name
    for name, digest in e['recipes'].items():
        assert sha(a.raw_root/'recipes-current-full'/name) == digest, name
    for row in e['rows']:
        for backend, record in row['backends'].items():
            directory = a.raw_root / Path(record['entry']['directory']).relative_to(Path(e['raw_root']))
            resources=[json.loads(line) for line in (directory/'resources.jsonl').read_text().splitlines()]
            assert resources and all(x['gpu_hz'] in [0,1530000000] and (x.get('desktop') is None or x['desktop']['monitor_off']) for x in resources)
            path = directory / ('stdout.jsonl' if backend=='apxinf' else 'samples.jsonl')
            actual = []
            for line in path.read_text().splitlines():
                try: value = json.loads(line)
                except ValueError: continue
                if backend=='trt' or value.get('schema') in ['apxinf.qwen35.benchmark.iteration.v2','apxinf.qwen35.context_benchmark.iteration.v1']:
                    actual.append({k:v for k,v in value.items() if k!='output_ids'})
            assert actual==record['rows'], (row['case']['id'],backend)
        case = row['case']
        input_path=Path(case['ids_path'])
        assert sha(input_path) == case['input_sha256']
        digest=0xcbf29ce484222325
        for byte in input_path.read_bytes():digest=((digest^byte)*0x100000001b3)&0xffffffffffffffff
        assert row['backends']['apxinf']['configs'][0]['input_digest_fnv1a64']==f'{digest:016x}'
print(json.dumps({'ok': True, 'scored_requests': 240, 'warmups': 48, 'raw_assets_verified': bool(a.raw_root),
                  'rows': summary}, indent=2, ensure_ascii=False))
