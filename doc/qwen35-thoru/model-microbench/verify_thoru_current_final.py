"""Verify the rebuilt latest-Final / latest-TRT Thor-U comparison without GPU runs."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('--raw-root',type=Path);p.add_argument('--reference',type=Path);a=p.parse_args();e=json.loads(a.evidence.read_text())
assert e['schema']=='apxinf.thoru_current_final.evidence.v1'
assert e['source_commit']=='ae909d2b0f992fbd3a2b33baa88b60057e96c479'
assert e['build']['completed'] and e['build']['source_restored'] and e['build']['source_commit']==e['source_commit']
assert e['operator_preparation']['completed'] and e['operator_preparation']['resources_restored']
assert e['operator_preparation']['native_archive_sha256']==e['build']['native_archive_sha256_after']
assert all(all(x.values()) for x in e['identity_audit'].values())
assert all(x['all_three_tokenizers_and_prefix_chunks_identical'] for x in e['input_validation'])
for phase in e['phases'].values():
 assert phase['complete'] and phase['source_commit']==e['source_commit']
 assert phase['policy_run']=={'governor':'performance','min_freq':'1530000000','max_freq':'1530000000'}
 assert phase['policy_before']==phase['policy_after'] and phase['pages_before']==phase['pages_after']==7380
 assert not phase['owners_after'] and phase['desktop_before']['monitor_off'] and phase['desktop_after']['monitor_off']
def close(x,y):
    assert math.isfinite(x) and math.isclose(x,y,rel_tol=1e-11,abs_tol=1e-11),(x,y)

def check(result,n):
    data=result['rows'];formal=[x for x in data if not x['warmup'] and not x['diagnostic']]
    assert len(formal)==10 and sum(x['warmup'] for x in data)==2
    assert [x['iteration'] for x in data]==list(range(len(data)))
    assert all(x['input_tokens']==n and x['reused_tokens']==n-1 and x['computed_tokens']==1
               and x['output_tokens']==1024 and x['decode_intervals']==1023
               and x['performance_eligible']==(not x['diagnostic']) for x in data)
    assert result['entry']['returncode']==0 and not result['entry']['faults']
    rs=result['resource_summary']
    assert rs['all_display_samples_off'] and rs['desktop_samples']>0
    assert rs['minimum_ordinary_memory_gib']>=3 and rs['frequency_samples']>0 and rs['hardware_clock_samples']>0
    m=result['metrics']
    close(m['decode_tps'],1023e3*len(formal)/math.fsum(x['decode_ms'] for x in formal))
    close(m['median_decode_tps'],statistics.median(1023e3/x['decode_ms'] for x in formal))
    assert result['distinct_scored_outputs']==len({x['output_sha256'] for x in formal})
    if result['entry']['backend']=='final':
        assert len({x['output_sha256'] for x in data})==1
        for x in data:
            s=x['backend_result']['speculation']
            assert s['emitted_tokens']==1023 and s['draft_steps']==3
            assert s['proposed_tokens']==3*s['rounds'] and s['target_verified_rows']==4*s['rounds']
    else:
        assert not result['entry']['ready']['profiling_formal']
    for d in result['diagnostics']:
        assert d['counters']['total_generated_tokens']==1024 and 256<=d['counters']['total_iterations']<=1023
        close(d['output_tokens_per_mtp_round'],1024/d['counters']['total_iterations'])
        assert d['matches_scored_iterations']==[x['iteration'] for x in formal if x['output_sha256']==d['output_sha256']]
    return formal


reference_path=a.reference or a.evidence.parent/'thoru-locked-repeat-20261126.json'
assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==e['previous_locked_reference_sha256']
reference=json.loads(reference_path.read_text())
assert e['postflight']['hugepages']==7380 and e['postflight']['shared_gpu_lock_free']
assert e['postflight']['source_commit']==e['source_commit'] and e['postflight']['binary_sha256']==e['build']['binary_sha256']
assert 'Monitor is Off' in e['postflight']['display']
assert e['postflight']['gpu_policy']==e['phases']['formal-11-current-repeat']['policy_before']
assert len(e['rows'])==2 and {r['case']['input_tokens'] for r in e['rows']}=={8192,16384}
scored=warm=diag=0
for row in e['rows']:
 assert set(row['backends'])=={'new-trt','final'}
 for result in row['backends'].values():
  f=check(result,row['case']['input_tokens']);scored+=len(f);warm+=2;diag+=len(result['diagnostics'])
for repeat in e['seed_repeats']:
 f=check(repeat['result'],repeat['input_tokens']);scored+=len(f);warm+=2;diag+=len(repeat['result']['diagnostics'])
assert (scored,warm,diag)==(60,12,12)==(e['scored_requests'],e['warmups'],e['diagnostic_requests'])
for row in e['rows']:
 first=row['backends']['new-trt'];second=next(x['result'] for x in e['seed_repeats'] if x['case']==row['case']['id'])
 formal=[x for r in [first,second] for x in r['rows'] if not x['warmup'] and not x['diagnostic']]
 v=row['two_prefix_new_trt'];close(v['decode_tps'],1023e3*20/math.fsum(x['decode_ms'] for x in formal))
 close(v['final_gain_percent'],100*(row['backends']['final']['metrics']['decode_tps']/v['decode_tps']-1))
 assert v['distinct_scored_outputs']==len({x['output_sha256'] for x in formal})
 assert v['mtp_rounds']==sorted({d['counters']['total_iterations'] for r in [first,second] for d in r['diagnostics']})
 prior=row['previous_final_comparison'];ref=next(x for x in reference['rows'] if x['case']['id']==row['case']['id'])
 close(prior['decode_tps'],ref['backends']['final']['metrics']['decode_tps'])
 assert prior['source_commit']==reference['source_commit']
 close(prior['change_percent'],100*(row['backends']['final']['metrics']['decode_tps']/prior['decode_tps']-1))
 if a.raw_root:
  for backend in ['new-trt','final']:
   result=row['backends'][backend];d=a.raw_root/'runs'/result['phase']/(row['case']['id']+'-'+backend)
   raw=[json.loads(l) for l in (d/'resources.jsonl').read_text().splitlines()]
   assert all(x['gpu_hz'] in [0,1530000000] and all(h in [0,1530000000] for h in x.get('gpc_hz',[])) for x in raw)
if a.raw_root:
 for name,digest in e['file_hashes'].items():assert hashlib.sha256((a.raw_root/name).read_bytes()).hexdigest()==digest,name
 assert hashlib.sha256((a.raw_root/'qwen35_current_decode_worker').read_bytes()).hexdigest()==e['build']['binary_sha256']
 for name,digest in e['recipes'].items():assert hashlib.sha256((a.raw_root/'recipes-current'/name).read_bytes()).hexdigest()==digest,name
print(json.dumps({'ok':True,'source_commit':e['source_commit'],'counts':[scored,warm,diag],'raw_assets_verified':bool(a.raw_root),'results':[{'case':r['case']['id'],'final_tps':r['backends']['final']['metrics']['decode_tps'],'trt':r['two_prefix_new_trt']} for r in e['rows']]},ensure_ascii=False,indent=2))
