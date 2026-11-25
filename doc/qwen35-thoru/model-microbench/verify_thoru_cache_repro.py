"""Recompute the exact N-1 cache protocol tables; optionally check retained raw assets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('evidence',type=Path)
p.add_argument('--raw-root',type=Path)
a=p.parse_args();e=json.loads(a.evidence.read_text())
assert e['schema']=='apxinf.thoru_cache_repro.evidence.v1'
assert e['source_commit']=='9f7aecd204073eafd25bfdb99c1c4ab586f95fad'
assert all(all(v.values()) for v in e['identity_audit'].values())
assert e['resources_restored'] and e['build']['completed'] and e['build']['source_restored']
assert e['build']['source_commit']==e['source_commit']
assert e['operator_preparation']['completed'] and e['operator_preparation']['resources_restored']
assert {r['case']['id'] for r in e['rows']}=={'decode-8k','decode-16k'} and len(e['rows'])==2
assert all(x['all_three_tokenizers_and_prefix_chunks_identical'] for x in e['input_validation'])

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

summary=[];scored=warmups=diagnostics=0
for row in e['rows']:
    assert set(row['backends'])=={'old-trt','new-trt','final'}
    for backend,result in row['backends'].items():
        f=check(result,row['case']['input_tokens']);scored+=len(f);warmups+=2;diagnostics+=len(result['diagnostics'])
    am=row['backends']['final']['metrics']['decode_tps']
    for be in ['old-trt','new-trt']:
        close(row['final_gain_percent'][be],100*(am/row['backends'][be]['metrics']['decode_tps']-1))
    summary.append({'case':row['case']['id'],'primary_tps':{k:v['metrics']['decode_tps'] for k,v in row['backends'].items()},'final_gain_percent':row['final_gain_percent']})
for repeat in e.get('seed_repeats',[]):
    f=check(repeat['result'],repeat['input_tokens']);scored+=len(f);warmups+=2;diagnostics+=len(repeat['result']['diagnostics'])
for row in e['rows']:
    first=row['backends']['new-trt']
    second=next(x['result'] for x in e['seed_repeats'] if x['case']==row['case']['id'])
    data=[x for result in [first,second] for x in result['rows'] if not x['warmup'] and not x['diagnostic']]
    combined=row['two_prefix_new_trt']
    assert len(data)==20
    close(combined['decode_tps'],1023e3*20/math.fsum(x['decode_ms'] for x in data))
    close(combined['final_gain_percent'],100*(row['backends']['final']['metrics']['decode_tps']/combined['decode_tps']-1))
    assert combined['distinct_scored_outputs']==len({x['output_sha256'] for x in data})
    assert combined['mtp_rounds']==sorted({d['counters']['total_iterations'] for result in [first,second] for d in result['diagnostics']})
assert (scored,warmups,diagnostics)==(e['scored_requests'],e['warmups'],e['diagnostic_requests'])
if a.raw_root:
    for name,digest in e['file_hashes'].items():
        assert hashlib.sha256((a.raw_root/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((a.raw_root/'qwen35_cached_decode_worker').read_bytes()).hexdigest()==e['build']['binary_sha256']
    for name,digest in e['recipes'].items():
        assert hashlib.sha256((a.raw_root/'recipes'/name).read_bytes()).hexdigest()==digest,name
print(json.dumps({'ok':True,'scored_requests':scored,'warmups':warmups,'diagnostic_requests':diagnostics,
                  'raw_assets_verified':bool(a.raw_root),'rows':summary},ensure_ascii=False,indent=2))
