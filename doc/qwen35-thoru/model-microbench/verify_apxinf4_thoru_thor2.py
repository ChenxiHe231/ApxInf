"""Recompute model metrics and reject invalid stage/case comparisons offline."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('task',type=Path);p.add_argument('--run',required=True);p.add_argument('--complete',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
plan=json.loads((a.task/'results/plan.json').read_text());folder=a.task/'runs'/a.run;summary=json.loads((folder/'summary.json').read_text());expected_repeats=10 if summary['phase']=='formal' else 3
assert summary.get('ok'),summary.get('error')
assert summary['commit']==plan['final_source']
expected_arms=set(summary['arms']);assert expected_arms and expected_arms<=set(plan['stages'])
rows={};file_hashes={};seen=set();recipe_reference=None;total=0;common_environment=None;case_arguments={}
for receipt in sorted(folder.glob('*/*/receipt.json')):
 r=json.loads(receipt.read_text());case=r['case']['id'];arm=r['arm'];key=(case,arm);assert key not in seen;seen.add(key)
 assert r['ok'] and r['returncode']==0,(key,r.get('error'))
 assert r['commit']==plan['stages'][arm]['commit']
 assert r['case']['kv']==int(case.split('-')[1][:-1])*1024
 assert r['argv'][1]==json.loads((a.task/'results/thor2-checkpoint-reference.json').read_text())['path']
 assert '--autotune' not in r['argv'] and '--profile-measured' not in r['argv']
 expected_options={'--mode':'mtp','--qwen35-moe-activation':'nvfp4','--qwen35-kv-cache':'fp8_dynamic','--warmups':'2','--repeats':str(expected_repeats),'--max-new':str(r['case']['outputs'])}
 expected_options['--prompt-tokens' if r['case']['kind']=='decode' else '--prefix-tokens']=str(r['case']['kv'])
 if r['case']['kind']=='append':expected_options['--input-tokens']='4096'
 for option,value in expected_options.items():assert r['argv'].count(option)==1 and r['argv'][r['argv'].index(option)+1]==value,(case,arm,option)
 if case not in case_arguments:case_arguments[case]=r['argv'][1:]
 assert case_arguments[case]==r['argv'][1:]
 variant_flags={'APXINF_MOE_SETUP_PREFIX','APXINF_QWEN35_FFN_FORK','APXINF_QWEN35_GDN_PREFIX_SNAPSHOTS','APXINF_QWEN35_GDN_SOURCE_STATE'}
 common={k:v for k,v in r['environment'].items() if k not in variant_flags}
 if common_environment is None:common_environment=common
 assert common_environment==common
 assert common['APXINF_QWEN35_MTP_FIXED_DEPTH']=='3' and common['APXINF_QWEN35_CUDA_GRAPHS']=='1' and common['APXINF_QWEN35_GRAPH_GDN']=='full'
 manifest=a.task/'stages'/r['commit']/'source.json';build=json.loads((manifest.parent/'build-receipt.json').read_text());assert build['ok']
 assert r['source_manifest_sha256']==sha(manifest)==build['source_manifest_sha256']
 assert r['controller_sha256']==sha(folder/'controller.py')
 binary='qwen35_mtp_bench' if r['case']['kind']=='decode' else 'qwen35_context_bench'
 assert r['ELF_sha256']==build['artifacts'][binary]['sha256']
 assert not r['frequency_bad_samples'];assert r['recipes_before']==r['recipes_after']
 if recipe_reference is None:recipe_reference=r['recipes_before']
 assert recipe_reference==r['recipes_before'] and len(recipe_reference)==736
 for flag,value in zip(('APXINF_MOE_SETUP_PREFIX','APXINF_QWEN35_FFN_FORK','APXINF_QWEN35_GDN_PREFIX_SNAPSHOTS','APXINF_QWEN35_GDN_SOURCE_STATE'),plan['stages'][arm]['flags']):assert r['environment'][flag]==str(value)
 log=receipt.with_name('stdout.jsonl');samples=[]
 for line in log.read_text().splitlines():
  try:x=json.loads(line)
  except json.JSONDecodeError:continue
  if x.get('schema') in ('apxinf.qwen35.benchmark.iteration.v2','apxinf.qwen35.context_benchmark.iteration.v1'):samples.append(x)
 assert len(samples)==2+expected_repeats
 assert sum(bool(x['warmup']) for x in samples)==2
 measured=[x for x in samples if not x['warmup']];assert len(measured)==expected_repeats;total+=len(measured)
 for x in samples:
  assert x['strict_recipe_required'] and x['performance_eligible'] and not x['repeat_diverged']
  assert x['output_tokens']==r['case']['outputs']
  assert x['token_digest_fnv1a64']==r['token_digest'] and x['speculation']==r['speculation']
 base=rows.setdefault(case,{});data={'commit':r['commit'],'digest':r['token_digest'],'speculation':r['speculation'],'measured':len(measured),'ELF_sha256':r['ELF_sha256']}
 if r['case']['kind']=='decode':
  assert all(x['decode_intervals']==1023 for x in samples)
  data.update(decode_tps=1023*len(measured)*1000/math.fsum(x['decode_interval_ms'] for x in measured),e2e_tps=1024*len(measured)*1000/math.fsum(x['total_latency_ms'] for x in measured),total_ms=statistics.median(x['total_latency_ms'] for x in measured),ttft_ms=statistics.median(x['ttft_ms'] for x in measured),decode_tps_range=[min(x['decode_tokens_per_second'] for x in measured),max(x['decode_tokens_per_second'] for x in measured)],total_ms_samples=[x['total_latency_ms'] for x in measured])
  data.update({('mean_'+k):math.fsum(x[k] for x in measured)/len(measured) for k in ('ttft_ms','decode_interval_ms','total_latency_ms')})
  assert math.isclose(data['decode_tps'],r['pooled_decode_tps'],rel_tol=1e-12)
  assert math.isclose(data['e2e_tps'],r['model_e2e_tps'],rel_tol=1e-12)
 else:
  assert all(x['initial_kv_tokens']==r['case']['kv'] and x['computed_input_tokens']==4096 for x in samples)
  data.update(prefill_ms=statistics.median(x['prefill_ms'] for x in measured),prefill_ms_samples=[x['prefill_ms'] for x in measured])
 base[arm]=data
 for path in (receipt,log,receipt.with_name('frequency.json'),receipt.with_name('stderr.log')):file_hashes[str(path.relative_to(a.task))]=sha(path)
 frequencies=json.loads(receipt.with_name('frequency.json').read_text());assert frequencies
 for x in frequencies:
  assert x['min_freq']==x['max_freq']=='1530000000' and x['cur_freq'] in ('0','1530000000')
  for v in x['cpu'].values():assert v['scaling_cur_freq']==v['scaling_min_freq']==v['scaling_max_freq']
  assert x['emc']['cur_freq']==x['emc']['min_freq']==x['emc']['max_freq']
for case,arms in rows.items():
 assert set(arms)==expected_arms,case
 assert len({x['digest'] for x in arms.values()})==1
 assert len({json.dumps(x['speculation'],sort_keys=True) for x in arms.values()})==1
 golden=json.loads((a.task.parent/'apxinf4-bench/runs/both-formal-01/summary.json').read_text())
 frozen=next(x for x in golden['cases'] if x['case']['id']==case)
 anchor=arms.get('final',next(iter(arms.values())))
 assert anchor['digest']==frozen['token_digest'] and anchor['speculation']==frozen['speculation']
 for parent,child in zip(('final','prefix','ffn'),('prefix','ffn','gdn')):
  if parent not in arms or child not in arms:continue
  x=arms[parent];y=arms[child]
  if case.startswith('decode'):
   y.update(parent_decode_gain_pct=(y['decode_tps']/x['decode_tps']-1)*100,parent_e2e_tps_gain_pct=(y['e2e_tps']/x['e2e_tps']-1)*100,parent_total_reduction_pct=(1-y['total_ms']/x['total_ms'])*100)
  else:y['parent_prefill_reduction_pct']=(1-y['prefill_ms']/x['prefill_ms'])*100
 if 'final' not in arms:continue
 x=arms['final']
 for name,y in arms.items():
  if name=='final':continue
  if case.startswith('decode'):y.update(final_decode_gain_pct=(y['decode_tps']/x['decode_tps']-1)*100,final_e2e_tps_gain_pct=(y['e2e_tps']/x['e2e_tps']-1)*100,final_total_reduction_pct=(1-y['total_ms']/x['total_ms'])*100)
  else:y['final_prefill_reduction_pct']=(1-y['prefill_ms']/x['prefill_ms'])*100
if a.complete:
 assert expected_arms==set(plan['stages'])
 assert summary['phase']=='formal' and len(rows)==11 and len(seen)==44 and total==440
 expected={f'decode-{n}k' for n in (4,8,16,24,32,48)}|{f'append-{n}k' for n in (4,8,16,32,48)};assert set(rows)==expected
 assert len(summary['cases'])==len(seen)
 assert summary['checkpoint_unchanged_after'] and summary['checkpoint_verified_original_files']==14
r={'ok':True,'run':a.run,'phase':summary['phase'],'stage_cases':len(seen),'measured_requests':total,'rows':rows,'raw_file_sha256':file_hashes,'driver_sha256':sha(Path(__file__))};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'ok':True,'stage_cases':len(seen),'measured_requests':total,'cases':{case:{arm:{k:v for k,v in x.items() if k in ('decode_tps','e2e_tps','prefill_ms','parent_decode_gain_pct','parent_total_reduction_pct','final_total_reduction_pct')} for arm,x in arms.items()} for case,arms in rows.items()}},indent=2))
