import argparse,collections,hashlib,json,math,pathlib
P=argparse.ArgumentParser();P.add_argument('task',type=pathlib.Path);P.add_argument('--run',required=True);P.add_argument('--complete',action='store_true');P.add_argument('--fast-baseline',action='store_true');a=P.parse_args()
root=a.task/'runs'/a.run;summary=json.loads((root/'summary.json').read_text())
groups={'original5':{'92','100','109','112','154'},'heldout8':{'82','96','102','115','123','131','145','155'}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def aggregate(xs):
 n=sum(x['output_tokens'] for x in xs);d=sum(x['decode_tokens'] for x in xs)
 t=math.fsum(x['timing']['request_wall_ms'] for x in xs);dt=math.fsum(x['timing']['callback_decode_ms'] for x in xs)
 accepted=sum(x['speculation']['verified_accepted_tokens'] for x in xs);proposed=sum(x['speculation']['proposed_tokens'] for x in xs)
 return {'requests':len(xs),'input_tokens':sum(x['input_tokens'] for x in xs),'output_tokens':n,'decode_tokens':d,'request_ms':t,'decode_ms':dt,'e2e_tps':n*1000/t,'decode_tps':d*1000/dt,'acceptance':accepted/proposed,'rounds':sum(x['speculation']['rounds'] for x in xs)}
arms=[];identities=collections.defaultdict(set);file_hashes={}
for folder in sorted(root.iterdir()):
 if not folder.is_dir() or not (folder/'samples.jsonl').exists():continue
 xs=[json.loads(l) for l in (folder/'samples.jsonl').read_text().splitlines()]
 if not xs:continue
 rows=[x for x in xs if not x['warmup']]
 arm={'folder':folder.name,'warmups':sum(x['warmup'] for x in xs),'measured':len(rows),'groups':{}}
 for group,ids in groups.items():
  subset=[x for x in rows if str(x['case_id']) in ids]
  if subset:arm['groups'][group]=aggregate(subset)
 for x in xs:
  identities[str(x['case_id'])].add((x['output_sha256'],None if summary['arguments'].get('proposal_change') else json.dumps(x['speculation'],sort_keys=True)))
  assert x['decode_tokens']==x['output_tokens']-1 and x['callback_tokens']==x['output_tokens']
  assert x['speculation']['emitted_tokens']==x['decode_tokens']
  assert len(x['output_ids'])==x['output_tokens']
  assert hashlib.sha256(__import__('struct').pack('<'+'I'*x['output_tokens'],*x['output_ids'])).hexdigest()==x['output_sha256']
 arms.append(arm);file_hashes[str((folder/'samples.jsonl').relative_to(a.task))]=sha(folder/'samples.jsonl')
assert all(len(x)==1 for x in identities.values()),'output/work identity mismatch'
result={'ok':False,'run':a.run,'arms':arms,'file_hashes':file_hashes,'groups':{},'pair_groups':[]}
if a.complete:
 assert summary['ok'] and summary['performance_eligible']
 if a.fast_baseline:
  expected={'APXINF_QWEN35_MTP_SPLIT_KV':'1','APXINF_QWEN35_MTP_DRAFT_KV_SUFFIX_SYNC':'1','APXINF_QWEN35_MTP_KV_ONLY_MATERIALIZE':'1','APXINF_QWEN35_MTP_MATERIALIZE_SESSION':'0','APXINF_FA2_PACKED_GQA':'0'}
  assert summary['environment']==expected, 'baseline configuration mismatch'
  for arm in summary['arms']:
   ready=arm['ready']
   assert ready['draft_split_kv'] and ready['APXINF_QWEN35_MTP_DRAFT_KV_SUFFIX_SYNC'] and ready['APXINF_QWEN35_MTP_KV_ONLY_MATERIALIZE']
  result['fast_baseline_checked']=True
 assert summary['arguments']['warmups']==26 and summary['arguments']['repeats']==5 and summary['arguments']['budget']==512
 assert len(arms)==4
 assert all(x['warmups']==26 and x['measured']==65 for x in arms)
 assert summary['recipe_before']==summary['recipe_after']
 assert set(identities)==set.union(*groups.values())
 result['artifacts']={}
 for arm in summary['arms']:
  stage=a.task/'stages'/arm['stage'];receipt=json.loads((stage/'receipt.json').read_text());binary=stage/'mtp_real_request_worker'
  assert receipt['ok'] and receipt['base_commit']=='1f605865ad208f58a9264d35264aad551aea3319'
  assert sha(binary)==arm['binary_sha256']==receipt['artifacts'][binary.name]['sha256']
  result['artifacts'][arm['stage']]={'worker_sha256':sha(binary),'receipt_sha256':sha(stage/'receipt.json'),'base_commit':receipt['base_commit']}
  for warmup,count in [(True,2),(False,5)]:
   assert collections.Counter(str(x['case_id']) for x in arm['samples'] if x['warmup']==warmup)=={cid:count for cid in identities}
 result['verifier_sha256']=sha(pathlib.Path(__file__))
 before=json.loads((root/'clock-before.json').read_text());after=json.loads((root/'clock-restored.json').read_text())
 for k in ['governor','min_freq','max_freq']:assert before[k]==after[k]
 for policy,v in before['cpu'].items():
  for k in ['scaling_governor','scaling_min_freq','scaling_max_freq']:assert v[k]==after['cpu'][policy][k]
 for k in ['min_freq','max_freq']:assert before['emc'][k]==after['emc'][k]
 freq=json.loads((root/'frequency.json').read_text());assert freq
 for c in freq:
  assert c['min_freq']==c['max_freq']=='1530000000' and c['cur_freq'] in ['0','1530000000']
  for v in c['cpu'].values():assert v['scaling_min_freq']==v['scaling_max_freq']==v['scaling_cur_freq']=='2601000'
  assert c['emc']['min_freq']==c['emc']['max_freq']==c['emc']['cur_freq']=='4266000000'
 result['clock_policies_restored']=True
 result['valid_frequency_samples']=len(freq)
 all_rows=[]
 for folder in sorted(root.iterdir()):
  if folder.is_dir() and (folder/'samples.jsonl').exists():all_rows.append([json.loads(l) for l in (folder/'samples.jsonl').read_text().splitlines() if not json.loads(l)['warmup']])
 for group,ids in groups.items():
  side={k:aggregate([x for i in indices for x in all_rows[i] if str(x['case_id']) in ids]) for k,indices in [('baseline',[0,3]),('candidate',[1,2])]}
  result['groups'][group]={**side,'e2e_gain_pct':100*(side['candidate']['e2e_tps']/side['baseline']['e2e_tps']-1),'decode_gain_pct':100*(side['candidate']['decode_tps']/side['baseline']['decode_tps']-1),'acceptance_delta':side['candidate']['acceptance']-side['baseline']['acceptance']}
 for i,j in [(0,1),(3,2)]:
  result['pair_groups'].append({g:{'e2e_gain_pct':100*(arms[j]['groups'][g]['e2e_tps']/arms[i]['groups'][g]['e2e_tps']-1),'decode_gain_pct':100*(arms[j]['groups'][g]['decode_tps']/arms[i]['groups'][g]['decode_tps']-1)} for g in groups})
 result.update(ok=True,measured_requests=260,repeated_positive=all(p[g]['e2e_gain_pct']>0 for p in result['pair_groups'] for g in groups))
 out=a.task/'results'/(a.run+'-verified.json');out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
