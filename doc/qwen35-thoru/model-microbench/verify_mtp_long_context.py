"""Offline exact-artifact/clock/output audit and per-commit regression screen."""
import argparse,hashlib,json,math,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('task',type=pathlib.Path);p.add_argument('--run',default='long-formal-01');a=p.parse_args();T=a.task
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((T/'results/plan.json').read_text());root=T/'runs'/a.run;s=json.loads((root/'summary.json').read_text());assert s['ok'] and s['phase']=='formal'
assert s['recipe_before']==s['recipe_after'] and s['model_before']==s['model_after']
assert len(s['cases'])==45
before=json.loads((root/'clock-before.json').read_text());after=json.loads((root/'clock-restored.json').read_text())
for k in ['governor','min_freq','max_freq']:assert before[k]==after[k]
for policy,values in before['cpu'].items():
 for k in ['scaling_governor','scaling_min_freq','scaling_max_freq']:assert values[k]==after['cpu'][policy][k]
for k in ['min_freq','max_freq']:assert before['emc'][k]==after['emc'][k]
freq=json.loads((root/'frequency.json').read_text());assert freq
for c in freq:
 assert c['min_freq']==c['max_freq']=='1530000000' and c['cur_freq'] in ['0','1530000000']
 for values in c['cpu'].values():assert values['scaling_min_freq']==values['scaling_max_freq']==values['scaling_cur_freq']=='2601000'
 assert c['emc']['min_freq']==c['emc']['max_freq']==c['emc']['cur_freq']=='4266000000'
records={};hashes={}
for r in s['cases']:
 cid=r['case']['id'];commit=r['commit'];assert r['ok'] and r['warmups']==2 and r['repeats']==10
 assert r['recipe_before']==r['recipe_after']==s['recipe_before']
 stage=T/'stages'/commit;b=stage/('qwen35_mtp_bench' if r['case']['kind']=='decode' else 'qwen35_context_bench');build=json.loads((stage/'build-receipt.json').read_text());assert build['ok']
 assert sha(b)==r['ELF_sha256']==build['artifacts'][b.name]['sha256'];assert sha(stage/'source.json')==r['source_manifest_sha256']
 d=T/r['attempt'];stdout=d/'stdout.jsonl';schema='apxinf.qwen35.benchmark.iteration.v2' if r['case']['kind']=='decode' else 'apxinf.qwen35.context_benchmark.iteration.v1';rows=[]
 for line in stdout.read_text().splitlines():
  try:x=json.loads(line)
  except ValueError:continue
  if x.get('schema')==schema:rows.append(x)
 assert rows==r['rows'] and len(rows)==12 and sum(x['warmup'] for x in rows)==2
 assert all(x['performance_eligible'] and x['strict_recipe_required'] and not x['autotune'] and not x['profiling'] and not x['repeat_diverged'] for x in rows)
 assert all(x['output_tokens']==r['case']['outputs'] for x in rows)
 assert len({x['token_digest_fnv1a64'] for x in rows})==1 and len({json.dumps(x['speculation'],sort_keys=True) for x in rows})==1
 measured=[x for x in rows if not x['warmup']]
 if r['case']['kind']=='decode':
  assert all(x['computed_prompt_tokens']==r['case']['kv'] and x['reused_prompt_tokens']==0 and x['decode_intervals']==1023 for x in rows)
  assert all(x['mode']=='mtp' and x['speculation']['draft_steps']==3 and x['speculation']['target_verified_rows']==4*x['speculation']['rounds'] for x in rows)
  metrics={'decode_tps':1023*10000/math.fsum(x['decode_interval_ms'] for x in measured),'e2e_tps':1024*10000/math.fsum(x['total_latency_ms'] for x in measured),'ttft_ms':statistics.mean(x['ttft_ms'] for x in measured)}
 else:
  assert all(x['initial_kv_tokens']==r['case']['kv'] and x['computed_input_tokens']==4096 for x in rows)
  metrics={'prefill_ms':statistics.mean(x['prefill_ms'] for x in measured)}
 for k,value in metrics.items():assert math.isclose(value,r['metrics'][k],rel_tol=1e-12)
 assert (cid,commit) not in records
 records[(cid,commit)]={'metrics':metrics,'rows':rows,'ELF_sha256':sha(b),'digest':rows[0]['token_digest_fnv1a64'],'speculation':rows[0]['speculation'],'stdout_sha256':sha(stdout),'stdout_path':str(stdout.relative_to(T)),'metadata':r['metadata']}
 hashes[str(stdout.relative_to(T))]=sha(stdout)
tables=[];flags=[]
for case in plan['cases']:
 row={'case':case,'stages':{c:records[(case['id'],c)] for c in plan['commits']},'comparisons':[]}
 assert len({x['digest'] for x in row['stages'].values()})==1
 assert row['stages'][plan['commits'][1]]['speculation']==row['stages'][plan['commits'][2]]['speculation']
 input_digests={x['input_digest_fnv1a64'] for stage in row['stages'].values() for x in stage['metadata']};assert len(input_digests)==1
 for i,j in [(0,1),(1,2),(0,2)]:
  base,candidate=plan['commits'][i],plan['commits'][j];metrics={}
  for metric in (['decode_tps','e2e_tps'] if case['kind']=='decode' else ['prefill_ms']):
   av,bv=row['stages'][base]['metrics'][metric],row['stages'][candidate]['metrics'][metric]
   gain=100*((1-bv/av) if metric=='prefill_ms' else (bv/av-1));metrics[metric]={'baseline':av,'candidate':bv,'gain_pct':gain}
   if gain <= -2:flags.append({'case':case['id'],'baseline':base,'candidate':candidate,'metric':metric,'gain_pct':gain,'action':'paired reverse2warm10 on this case before publication'})
  row['comparisons'].append({'baseline':base,'candidate':candidate,'metrics':metrics})
 tables.append(row)
activation=json.loads((T/'results/activation.json').read_text());assert activation['ok'] and len(activation['cases'])==4
assert all(x['observed_partition_head_kernel_calls']>0 for x in activation['cases'])
result={'ok':True,'formal_run':a.run,'cases':tables,'formal_scored_requests':450,'warmup_requests':90,'frequency_samples':len(freq),'clock_policies_restored':True,'output_all_versions_equal':True,'Graph_increment_MTP_work_equal':True,'activation_verified':True,'source_commits':plan['commits'],'recipes':s['recipe_before'],'model_files':s['model_before'],'file_hashes':hashes,'regression_screen_threshold_pct':-2,'needs_reverse_retest':flags,'publication_ready_without_followups':not flags,'note':'Flags are screening signals, not confirmed regressions. A suspected >=2% loss must receive a counterordered paired test. Previous real13 gains are separate evidence; do not add percentages.'}
(T/'results/long-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'ok':True,'scored':450,'regression_flags':flags,'rows':[{'case':r['case']['id'],'metrics':{c[:8]:x['metrics'] for c,x in r['stages'].items()}} for r in tables]},indent=2))
