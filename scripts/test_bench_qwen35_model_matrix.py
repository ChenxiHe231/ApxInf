"""CPU validation of measurement/evidence contracts, not simulated GPU scores."""
import copy
import fcntl
import os
import signal
import threading
import time
from unittest.mock import patch
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import bench_qwen35_model_matrix as bench


def args(**changes):
    base=dict(run_mode='formal',warmups=2,repeats=5,mode='mtp',mtp_depth=3,kv_cache='bf16',moe_activation='nvfp4',
              env=[],context_binary='/bin/context',decode_binary='/bin/decode',checkpoint='/model',tuning_dir='/recipes',
              workdir='/',ssh_host=None,timeout_sec=2,lock_timeout_sec=3,nsys='nsys',profile_dir='/trace',allow_repeat_divergence=False)
    base.update(changes)
    return SimpleNamespace(**base)


def fixture(case, opts=None, values=None):
    opts=opts or args()
    is_decode=case['group']=='decode';tune=opts.run_mode=='tune';profile=opts.run_mode=='profile'
    n=case['initial_kv_tokens']+(0 if is_decode else case['new_input_tokens'])
    repeats=opts.repeats if opts.run_mode=='formal' else 1;warmups=0 if tune else opts.warmups
    identity=bench.input_identity(n);prefix='apxinf.qwen35.benchmark.' if is_decode else 'apxinf.qwen35.context_benchmark.'
    output=1024 if is_decode else 1
    flags=dict(mode=opts.mode,autotune=tune,performance_eligible=not tune and not profile,profiling=profile,strict_recipe_required=not tune)
    config=dict(schema=prefix+'config.v1',**flags,output_tokens=output,warmups=warmups,repeats=repeats,
                qwen35_kv_cache=opts.kv_cache,qwen35_moe_activation=opts.moe_activation,temperature=0,sampling_mode='greedy',
                tuning_dir=opts.tuning_dir,input_digest_fnv1a64=identity['fnv1a64'],input_seed_hex=identity['seed_hex'],
                mtp_depth=3,mtp_strategy='fixed',draft_kv_dtype='fp8',draft_kv_scale=1.0)
    config['allow_repeat_divergence']=opts.allow_repeat_divergence
    if is_decode:config.update(prompt_tokens=n,context_reuse=False)
    else:config.update(prefix_tokens=case['initial_kv_tokens'],input_tokens=case['new_input_tokens'],total_context_tokens=n)
    iterations=[]
    values=values or list(range(100,100+repeats))
    for i in range(warmups+repeats):
        value=50 if i<warmups else values[i-warmups]
        row=dict(schema=prefix+'iteration.v1',**flags,iteration=i,warmup=i<warmups,output_tokens=output,
                 decode_intervals=output-1,token_digest_fnv1a64='0123456789abcdef')
        if is_decode:
            decode_ms=(output-1)*1000/value
            row.update(prompt_tokens=n,computed_prompt_tokens=n,reused_prompt_tokens=0,generation_tokens_per_second=value,
                       ttft_ms=200,decode_ms=decode_ms,total_latency_ms=200+decode_ms)
        else:row.update(prefix_tokens=case['initial_kv_tokens'],input_tokens=case['new_input_tokens'],total_context_tokens=n,
                        initial_kv_tokens=case['initial_kv_tokens'],computed_input_tokens=case['new_input_tokens'],
                        committed_context_tokens_after_prefill=n,prefill_ms=value,decode_ms=None,decode_tokens_per_second=None)
        iterations.append(row)
    summary=dict(schema=prefix+'summary.v1',**flags,divergent_repeats=0,output_tokens=output,decode_intervals=output-1,warmups=warmups,repeats=repeats)
    if is_decode:summary.update(prompt_tokens=n,generation_tokens_per_second={'median':bench.statistics.median(values)})
    else:summary.update(prefix_tokens=case['initial_kv_tokens'],input_tokens=case['new_input_tokens'],total_context_tokens=n,prefill_ms={'median':bench.statistics.median(values)})
    return [config,*iterations,summary]


def divergent_fixture(case, opts, indices=(2, 4)):
    records=fixture(case,opts)
    iterations=records[1:-1];events=[]
    for index in indices:
        row=iterations[index];row['repeat_diverged']=True;row['token_digest_fnv1a64']='f'*16
        events.append(dict(schema='apxinf.qwen35.benchmark.repeat_divergence.v1',benchmark='decode' if case['group']=='decode' else 'context',
                           mode=opts.mode,iteration=index,warmup=row['warmup'],allow_repeat_divergence=opts.allow_repeat_divergence,first_diff=0,
                           expected_output_tokens=row['output_tokens'],actual_output_tokens=row['output_tokens'],
                           expected_digest_fnv1a64=iterations[0]['token_digest_fnv1a64'],actual_digest_fnv1a64=row['token_digest_fnv1a64'],
                           expected_first_diff_token=220,actual_first_diff_token=198,
                           expected_window=dict(start=0,tokens=[220]),actual_window=dict(start=0,tokens=[198])))
    records[-1]['divergent_warmups']=sum(iterations[i]['warmup'] for i in indices)
    records[-1]['divergent_repeats']=len(indices)-records[-1]['divergent_warmups']
    return records[:-1]+events+[records[-1]]


def raw_output(records,code=0,sha='a'*64,report=''):
    process={'pid':123,'kind':'benchmark','sha256':sha,'exe':'/bin/context'}
    recipe={'count':1,'manifest_sha256':'x'}
    lock={'path':bench.GPU_LOCK_PATH,'acquired':True,'owner_pid':42,'wait_ms':0,'recipes_before':recipe,'target_binary_sha256':sha}
    receipt={'returncode':code,'timed_out':False,'child_reaped':True,'lock_held_until_reaped':True,'recipes_after':recipe,'target_binary_sha256':sha}
    return bench.LOCK_MARKER+json.dumps(lock)+'\n'+bench.SOURCE_MARKER+json.dumps(process)+'\n'+report+'\n'.join(json.dumps(row) for row in records)+'\n'+bench.EXIT_MARKER+json.dumps(receipt)+'\n'


class MatrixContracts(unittest.TestCase):
    def fake_nsys(self, root, *, stats_code=0, version_code=0, csv=True, mutate=False):
        script=root/'fake-nsys'
        script.write_text('#!'+sys.executable+'\n'+
            'import os,pathlib,sys\n'+
            'print("CPU fixture subprocess pid="+str(os.getpid()),file=sys.stderr)\n'+
            'if sys.argv[1:]==["--version"]:\n'+
            ' print("NVIDIA Nsight Systems version CPU-test-fixture");sys.exit('+str(version_code)+')\n'+
            'assert sys.argv[1]=="stats", "test must never execute profiler/GPU"\n'+
            ('pathlib.Path(sys.argv[-1]).write_bytes(b"changed during export")\n' if mutate else '')+
            ('print("Time (%),Total Time (ns),Instances,Name\\n100.0,12000,3,CPU_fixture_kernel")\n' if csv else 'print("no timing records collected")\n')+
            'sys.exit('+str(stats_code)+')\n')
        script.chmod(0o755)
        return str(script)

    def test_profile_requires_real_nonempty_report_and_independent_stats_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=str(root/'trace');report=root/'trace.nsys-rep'
            report.write_bytes(b'CPU trace artifact fixture, not a real GPU capture')
            opts=args(nsys=self.fake_nsys(root))
            receipt=bench.Transport(opts).profile_artifacts(base,root/'case')
            self.assertTrue(receipt['valid'],receipt['failures'])
            self.assertTrue(receipt['cpu_only'])
            self.assertEqual(receipt['report']['size_bytes'],report.stat().st_size)
            self.assertEqual(receipt['report']['sha256'],bench.sha256_file(report))
            self.assertEqual(receipt['report'],receipt['report_after_export'])
            self.assertEqual(receipt['nsys_version']['argv'],[opts.nsys,'--version'])
            self.assertEqual(receipt['nsys_version']['returncode'],0)
            self.assertEqual(receipt['stats']['argv'][1],'stats')
            self.assertEqual(receipt['stats']['argv'][-1],str(report))
            self.assertEqual(receipt['stats']['returncode'],0)
            self.assertIn('CPU fixture subprocess pid=',receipt['stats']['stderr'])
            self.assertNotIn('pid='+str(os.getpid())+'\n',receipt['stats']['stderr'])
            self.assertEqual(receipt['timing_summary_rows'],1)
            csv_path=Path(receipt['stats_csv']['path'])
            self.assertGreater(receipt['stats_csv']['size_bytes'],0)
            self.assertEqual(receipt['stats_csv']['sha256'],bench.sha256_file(csv_path))
            self.assertEqual(csv_path.read_text(),receipt['stats']['stdout'])
            report.unlink()
            missing=bench.Transport(opts).profile_artifacts(base,root/'missing')
            self.assertFalse(missing['valid']);self.assertNotIn('stats',missing)
            report.touch()
            empty=bench.Transport(opts).profile_artifacts(base,root/'empty')
            self.assertFalse(empty['valid']);self.assertNotIn('stats',empty)
            self.assertTrue(any('empty' in f for f in empty['failures']))

    def test_profile_rejects_stats_or_version_failure_and_no_timing_csv(self):
        for options,expected in [(dict(stats_code=7),'stats export failed'),(dict(version_code=8),'version command failed'),(dict(csv=False),'no valid timing')]:
            with self.subTest(options=options),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'trace.nsys-rep').write_bytes(b'CPU artifact fixture')
                receipt=bench.Transport(args(nsys=self.fake_nsys(root,**options))).profile_artifacts(str(root/'trace'),root/'case')
                self.assertFalse(receipt['valid'])
                self.assertTrue(any(expected in f for f in receipt['failures']),receipt['failures'])
                self.assertTrue((root/'case.stats.csv').exists())
                self.assertIn('returncode',receipt['stats'])

    def test_profile_rejects_report_mutated_during_cpu_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'trace.nsys-rep').write_bytes(b'CPU artifact fixture')
            receipt=bench.Transport(args(nsys=self.fake_nsys(root,mutate=True))).profile_artifacts(str(root/'trace'),root/'case')
            self.assertFalse(receipt['valid'])
            self.assertTrue(any('changed during' in f for f in receipt['failures']))

    def test_missing_profile_artifact_invalidates_otherwise_valid_capture_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'result';metadata=root/'expected.json'
            source=dict(source_commit='a'*40,source_dirty=False,source_patch_sha256='b'*64,source_untracked_sha256={})
            recipes=dict(count=1,manifest_sha256='x')
            actual=dict(source=source,binary_sha256=dict(context='a'*64,decode='a'*64),model_sha256={'model.safetensors':'f'*64},recipes=recipes)
            metadata.write_text(json.dumps(actual))
            case=bench.canonical_cases()[0]
            raw=raw_output(fixture(case,args(run_mode='profile')),report='[apxinf-gemm] test source=recipe\n[apxinf-attn] test source=recipe\n')
            raw=raw.replace('"kind": "benchmark"','"kind": "profiler_launcher"')
            def execute(_transport,argv,env,log,profile=False):
                self.assertTrue(profile);log.write_text(raw)
                return dict(transport_returncode=0,transport_timeout=False)
            command=['matrix','--run-mode','profile','--case',case['id'],'--workdir','/','--context-binary','/bin/context',
                     '--decode-binary','/bin/decode','--checkpoint','/model','--tuning-dir','/recipes','--metadata',str(metadata),
                     '--output',str(out),'--profile-dir','/trace']
            with patch.object(sys,'argv',command),patch.object(bench.Transport,'snapshot',lambda _s,kind='full':recipes if kind=='recipes' else actual),\
                 patch.object(bench.Transport,'execute',execute),patch.object(bench.Transport,'profile_artifacts',return_value=dict(valid=False,failures=['missing .nsys-rep'])):
                self.assertEqual(bench.main(),1)
            summary=json.loads((out/'summary.json').read_text())
            self.assertFalse(summary['all_cases_valid']);self.assertFalse(summary['performance_accepted'])
            row=summary['results'][0]
            self.assertIn('observed',row) # Native timing was valid; missing trace still fails the capture.
            self.assertFalse(row['valid']);self.assertIn('missing .nsys-rep',row['failures'])

    def test_exact_canonical_fifteen_cases_and_no_silent_shape_change(self):
        rows=bench.canonical_cases();self.assertEqual(len(rows),15)
        self.assertEqual([len([r for r in rows if r['group']==g]) for g in ('cold_prefill','append_prefill','decode')],[4,5,6])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'matrix.json';path.write_text(json.dumps({'cases':rows}))
            self.assertEqual(bench.load_cases(path),rows)
            rows[-1]['output_tokens']=1000;path.write_text(json.dumps({'cases':rows}))
            with self.assertRaisesRegex(ValueError,'canonical'):bench.load_cases(path)
        with self.assertRaisesRegex(ValueError,'duplicate'):bench.selected_cases(bench.canonical_cases(),'cold-i2k,cold-i2k')

    def test_tune_formal_profile_are_distinct_commands_and_policies(self):
        case=bench.canonical_cases()[0]
        formal=args();tune=args(run_mode='tune');profile=args(run_mode='profile')
        self.assertNotIn('--autotune',bench.command_for_case(formal,case))
        command=bench.command_for_case(tune,case);self.assertIn('--autotune',command)
        self.assertEqual(command[command.index('--warmups')+1],'0')
        self.assertEqual(command[command.index('--repeats')+1],'1')
        self.assertIn('--profile-measured',bench.command_for_case(profile,case))
        self.assertEqual(bench.environment(formal)['APXINF_REQUIRE_RECIPES'],'1')
        self.assertNotIn('APXINF_GEMM_REPORT',bench.environment(formal))
        self.assertNotIn('APXINF_REQUIRE_RECIPES',bench.environment(tune))
        self.assertEqual(bench.environment(profile)['APXINF_ATTENTION_REPORT'],'1')

    def test_input_identity_matches_u32_little_endian_fnv_contract(self):
        value=bench.input_identity(2)
        state=bench.SEED;tokens=[]
        for _ in range(2):
            state=(state*6364136223846793005+1442695040888963407)&bench.MASK64
            tokens.append((state>>17)%(248320-1000))
        raw=b''.join(x.to_bytes(4,'little') for x in tokens)
        h=bench.FNV_OFFSET
        for x in raw:h=((h^x)*bench.FNV_PRIME)&bench.MASK64
        self.assertEqual(value['fnv1a64'],f'{h:016x}')
        self.assertEqual(value['sha256'],bench.hashlib.sha256(raw).hexdigest())

    def test_prefill_resident_kv_and_actual_input_counts_are_verified(self):
        case=bench.canonical_cases()[4];opts=args();records=fixture(case,opts)
        result=bench.validate_records(records,case,opts)
        self.assertEqual(result['observed_median'],102)
        for key in ('initial_kv_tokens','computed_input_tokens','committed_context_tokens_after_prefill'):
            bad=copy.deepcopy(records);bad[1][key]+=1
            with self.assertRaisesRegex(ValueError,'actual KV'):bench.validate_records(bad,case,opts)

    def test_even_median_and_warmup_exclusion(self):
        case=bench.canonical_cases()[0];opts=args(repeats=4)
        result=bench.validate_records(fixture(case,opts,[100,200,300,400]),case,opts)
        self.assertEqual(result['observed_median'],250)
        self.assertEqual(result['measured_iterations'],4)
        decoded=bench.canonical_cases()[-1]
        metrics=bench.validate_records(fixture(decoded,opts,[100,200,300,400]),decoded,opts)
        self.assertAlmostEqual(metrics['decode_tpot_ms'],bench.statistics.median([10,5,1000/300,2.5]))
        self.assertNotEqual(metrics['decode_tpot_ms'],1000/metrics['observed_median'])

    def test_decode_counts_first_token_boundary_and_tps_consistency(self):
        case=bench.canonical_cases()[-1];opts=args();records=fixture(case,opts)
        result=bench.validate_records(records,case,opts)
        self.assertEqual(result['decode_tpot_ms'],1000/102)
        for key,value in [('decode_intervals',1024),('output_tokens',1000),('reused_prompt_tokens',1),('computed_prompt_tokens',1),('decode_ms',2)]:
            bad=copy.deepcopy(records);bad[1][key]=value
            with self.assertRaises(ValueError):bench.validate_records(bad,case,opts)

    def test_autotune_profile_cannot_be_formal_and_missing_rows_rejected(self):
        case=bench.canonical_cases()[0];opts=args()
        for changed in ['autotune','performance_eligible','strict_recipe_required']:
            records=fixture(case,opts);records[1][changed]=not records[1][changed]
            with self.assertRaises(ValueError):bench.validate_records(records,case,opts)
        with self.assertRaisesRegex(ValueError,'iteration'):
            records=fixture(case,opts);records.pop(1);bench.validate_records(records,case,opts)
        for mode in ('tune','profile'):
            result=bench.validate_records(fixture(case,args(run_mode=mode)),case,args(run_mode=mode))
            self.assertEqual(result['measured_iterations'],1)

    def test_stale_summary_or_wrong_even_median_cannot_bless_raw_rows(self):
        case=bench.canonical_cases()[0];opts=args(repeats=4)
        for key,value in [('repeats',3),('input_tokens',1000),('prefill_ms',{'median':200})]:
            records=fixture(case,opts,[100,200,300,400]);records[-1][key]=value
            with self.assertRaisesRegex(ValueError,'summary'):bench.validate_records(records,case,opts)

    def test_greedy_repeat_divergence_and_nan_are_not_hidden(self):
        case=bench.canonical_cases()[0];opts=args();records=fixture(case,opts)
        records[1]['token_digest_fnv1a64']='f'*16
        with self.assertRaisesRegex(ValueError,'diverged'):bench.validate_records(records,case,opts)
        records=fixture(case,opts);records[2]['prefill_ms']=float('nan')
        with self.assertRaisesRegex(ValueError,'invalid metric'):bench.validate_records(records,case,opts)
        records=fixture(case,opts);records[0]['allow_repeat_divergence']=True
        with self.assertRaisesRegex(ValueError,'escape'):bench.validate_records(records,case,opts)

    def test_explicit_divergence_policy_keeps_seven_raw_rounds_but_rejects_formal_score(self):
        recipes=dict(count=1,manifest_sha256='x')
        for case in (bench.canonical_cases()[0],bench.canonical_cases()[-1]):
            with self.subTest(case=case['id']):
                opts=args(allow_repeat_divergence=True)
                self.assertIn('--allow-repeat-divergence',bench.command_for_case(opts,case))
                self.assertNotIn('--allow-repeat-divergence',bench.command_for_case(args(),case))
                records=divergent_fixture(case,opts,indices=(1,2,4))
                result=bench.validate_run(raw_output(records),case,opts,recipes,recipes,'a'*64,dict(transport_returncode=0,transport_timeout=False))
                self.assertTrue(result['valid'],result['failures']);self.assertTrue(result['measurement_valid'])
                self.assertFalse(result['performance_accepted']);self.assertEqual(result['accuracy_status'],'unresolved')
                self.assertEqual(result['comparison_policy'],'diagnostic-repeat-divergence-v1')
                self.assertEqual(len(result['native_iterations']),7);self.assertEqual(len(result['repeat_divergences']),3)
                observed=result['observed']
                self.assertEqual(len(observed['output_digests']),7)
                self.assertEqual(len(observed['measured_primary_values']),5)
                self.assertEqual(observed['observed_median'],102)
                if case['group']=='decode':
                    self.assertEqual(len(observed['measured_decode_tpot_ms']),5)
                    self.assertAlmostEqual(observed['decode_tpot_distribution']['median'],1000/102)
                strict=bench.validate_run(raw_output(records),case,args(),recipes,recipes,'a'*64,dict(transport_returncode=0,transport_timeout=False))
                self.assertFalse(strict['valid']);self.assertFalse(strict['performance_accepted'])
                self.assertEqual(len(strict['repeat_divergences']),3) # Evidence survives rejection.
        consistent=bench.validate_run(raw_output(fixture(bench.canonical_cases()[0],opts)),bench.canonical_cases()[0],opts,recipes,recipes,'a'*64,dict(transport_returncode=0,transport_timeout=False))
        self.assertTrue(consistent['valid']);self.assertFalse(consistent['performance_accepted'])
        self.assertEqual(consistent['accuracy_status'],'unresolved') # Opt-in never certifies a formal score.

    def test_diagnostic_policy_still_rejects_missing_divergence_evidence_or_wrong_counts(self):
        case=bench.canonical_cases()[-1];opts=args(allow_repeat_divergence=True)
        for mutation in ('no_event','bad_first','bad_summary','bad_shape','no_flag'):
            with self.subTest(mutation=mutation):
                records=divergent_fixture(case,opts)
                if mutation=='no_event':records=[r for r in records if r.get('schema')!='apxinf.qwen35.benchmark.repeat_divergence.v1']
                elif mutation=='bad_first':records[-2]['first_diff']=1024
                elif mutation=='bad_summary':records[-1]['divergent_repeats']=0
                elif mutation=='bad_shape':records[3]['output_tokens']=1023
                elif mutation=='no_flag':records[3]['repeat_diverged']=False
                with self.assertRaises(ValueError):bench.validate_records(records,case,opts)

    def test_diagnostic_main_renders_only_diagnostic_median_and_tpot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'result';metadata=root/'expected.json'
            source=dict(source_commit='a'*40,source_dirty=False,source_patch_sha256='b'*64,source_untracked_sha256={})
            recipes=dict(count=1,manifest_sha256='x')
            actual=dict(source=source,binary_sha256=dict(context='a'*64,decode='a'*64),model_sha256={'model.safetensors':'f'*64},recipes=recipes)
            metadata.write_text(json.dumps(actual));case=bench.canonical_cases()[-1]
            raw=raw_output(divergent_fixture(case,args(allow_repeat_divergence=True)))
            def execute(_transport,argv,env,log,profile=False):
                self.assertIn('--allow-repeat-divergence',argv);log.write_text(raw)
                return dict(transport_returncode=0,transport_timeout=False)
            command=['matrix','--allow-repeat-divergence','--case',case['id'],'--workdir','/','--context-binary','/bin/context',
                     '--decode-binary','/bin/decode','--checkpoint','/model','--tuning-dir','/recipes','--metadata',str(metadata),'--output',str(out)]
            with patch.object(sys,'argv',command),patch.object(bench.Transport,'snapshot',lambda _s,kind='full':recipes if kind=='recipes' else actual),patch.object(bench.Transport,'execute',execute):
                self.assertEqual(bench.main(),0) # Valid diagnostic acquisition, not accepted performance.
            summary=json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['measurement_valid']);self.assertFalse(summary['performance_accepted'])
            self.assertEqual(summary['accuracy_status'],'unresolved')
            self.assertEqual(json.loads((out/'metadata.json').read_text())['comparison_policy'],summary['comparison_policy'])
            table=(out/'summary.md').read_text()
            self.assertIn('Diagnostic median',table);self.assertIn('Diagnostic TPOT P90',table)
            self.assertIn('102.000000 tok/s',table);self.assertIn('accuracy unresolved',table)
            self.assertNotIn('| Formal N |',table);self.assertNotIn('| passed |',table)

    def test_fixed_depth_target_and_draft_precision_and_input_are_bound(self):
        case=bench.canonical_cases()[0];opts=args()
        for key,value in [('mtp_depth',2),('draft_kv_scale',0.5),('qwen35_kv_cache','fp8'),('input_digest_fnv1a64','f'*16),('sampling_mode','sampling')]:
            records=fixture(case,opts);records[0][key]=value
            with self.assertRaises(ValueError):bench.validate_records(records,case,opts)

    def test_recipe_audit_rejects_fallback_even_without_directory_change(self):
        report='[apxinf-gemm] m=1 | implementation source=recipe\n[apxinf-attn] q=1 | FA2 source=memory-recipe\n'
        self.assertEqual(bench.audit_recipe_sources(report,True)['dispatches'],2)
        for source in ['fallback','fallback-after-tune','tuned:1.2ms']:
            with self.assertRaisesRegex(ValueError,'non-recipe'):bench.audit_recipe_sources(report.replace('source=recipe','source='+source),True)
        with self.assertRaisesRegex(ValueError,'both'):bench.audit_recipe_sources('',True)

    def test_actual_process_receipt_and_recipe_inventory_required(self):
        case=bench.canonical_cases()[0];opts=args();records=fixture(case,opts)
        before={'count':1,'manifest_sha256':'x'};outcome={'transport_timeout':False,'transport_returncode':0}
        good=bench.validate_run(raw_output(records),case,opts,before,before,'a'*64,outcome)
        self.assertTrue(good['performance_accepted'])
        for after,sha,transport in [({'count':1,'manifest_sha256':'changed'},'a'*64,outcome),(before,'b'*64,outcome),(before,'a'*64,{'transport_timeout':True,'transport_returncode':124})]:
            value=bench.validate_run(raw_output(records),case,opts,before,after,sha,transport)
            self.assertFalse(value['valid'])
            self.assertFalse(value['performance_accepted'])
        bad=bench.validate_run(raw_output(records),case,opts,{'count':0,'manifest_sha256':'x'},before,'a'*64,outcome)
        self.assertFalse(bad['valid'])
        with self.assertRaisesRegex(ValueError,'receipt'):bench.parse_output('\n'.join(json.dumps(r) for r in records))

    def test_real_launcher_timeout_reaps_only_its_child(self):
        # CPU subprocess only: prove the timeout path does not silently advance.
        with tempfile.TemporaryDirectory() as directory:
            opts=args(workdir=directory,timeout_sec=0.1,context_binary=sys.executable,decode_binary=sys.executable,tuning_dir=directory)
            log=Path(directory)/'log'
            with patch.object(bench,'GPU_LOCK_PATH',str(Path(directory)/'gpu.lock')):
                result=bench.Transport(opts).execute([sys.executable,'-c','import time; time.sleep(10)'],{},log)
            self.assertEqual(result['transport_returncode'],124)
            receipt=[json.loads(line[len(bench.EXIT_MARKER):]) for line in log.read_text().splitlines() if line.startswith(bench.EXIT_MARKER)][0]
            self.assertTrue(receipt['timed_out']);self.assertTrue(receipt['child_reaped'])

    def test_real_flock_serializes_two_workers_and_records_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lock_path=root/'gpu.lock';events=root/'events'
            opts=args(workdir=directory,context_binary=sys.executable,decode_binary=sys.executable,tuning_dir=str(root/'recipes'))
            def child(name,delay):
                code="import time; from pathlib import Path; p=Path(%r); f=p.open('a'); f.write(%r); f.flush(); time.sleep(%r); f.write(%r); f.close()"%(str(events),name+'-start\n',delay,name+'-end\n')
                return [sys.executable,'-c',code]
            results={}
            def run(name,delay):results[name]=bench.Transport(opts).execute(child(name,delay),{},root/(name+'.log'))
            with patch.object(bench,'GPU_LOCK_PATH',str(lock_path)):
                first=threading.Thread(target=run,args=('first',0.35));first.start()
                deadline=time.monotonic()+5
                while (not events.exists() or 'first-start' not in events.read_text()) and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(events.exists())
                second=threading.Thread(target=run,args=('second',0));second.start()
                first.join(6);second.join(6)
            self.assertFalse(first.is_alive());self.assertFalse(second.is_alive())
            self.assertEqual(events.read_text().splitlines(),['first-start','first-end','second-start','second-end'])
            lock=[json.loads(line[len(bench.LOCK_MARKER):]) for line in (root/'second.log').read_text().splitlines() if line.startswith(bench.LOCK_MARKER)][0]
            self.assertTrue(lock['acquired']);self.assertGreater(lock['wait_ms'],100)
            self.assertIsInstance(lock['owner_pid'],int)
            self.assertIn('hardware_before',lock);self.assertIn('recipes_before',lock)
            self.assertTrue(all(row['transport_returncode']==0 for row in results.values()))

    def test_lock_timeout_never_starts_payload_and_failure_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lock_path=root/'gpu.lock';flag=root/'must-not-start'
            opts=args(workdir=directory,context_binary=sys.executable,decode_binary=sys.executable,tuning_dir=str(root/'recipes'),lock_timeout_sec=.1)
            with patch.object(bench,'GPU_LOCK_PATH',str(lock_path)),lock_path.open('a+') as held:
                fcntl.flock(held,fcntl.LOCK_EX)
                result=bench.Transport(opts).execute([sys.executable,'-c',"from pathlib import Path; Path(%r).touch()"%str(flag)],{},root/'blocked.log')
                self.assertEqual(result['transport_returncode'],75);self.assertFalse(flag.exists())
                fcntl.flock(held,fcntl.LOCK_UN)
                failed=bench.Transport(opts).execute([sys.executable,'-c','raise SystemExit(3)'],{},root/'failed.log')
                self.assertEqual(failed['transport_returncode'],3)
                fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(held,fcntl.LOCK_UN)

    def test_timeout_releases_lock_only_after_child_is_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lock_path=root/'gpu.lock'
            opts=args(workdir=directory,context_binary=sys.executable,decode_binary=sys.executable,tuning_dir=str(root/'recipes'),timeout_sec=.1)
            with patch.object(bench,'GPU_LOCK_PATH',str(lock_path)):
                bench.Transport(opts).execute([sys.executable,'-c','import time; time.sleep(10)'],{},root/'timeout.log')
            receipt=[json.loads(line[len(bench.EXIT_MARKER):]) for line in (root/'timeout.log').read_text().splitlines() if line.startswith(bench.EXIT_MARKER)][0]
            self.assertTrue(receipt['child_reaped']);self.assertTrue(receipt['lock_held_until_reaped'])
            with lock_path.open('a+') as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)

    def test_direct_child_keeps_lock_if_worker_is_killed(self):
        # Real POSIX descriptors: no GPU work, no reliance on lockfile contents.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lock_path=root/'gpu.lock';proc=None;child_pid=None
            payload=dict(argv=[sys.executable,'-c','import time; time.sleep(10)'],environment={},cwd=directory,timeout=20,
                         profile=False,lock_path=str(lock_path),lock_timeout=2,tuning_dir=str(root/'recipes'),benchmark_binary=sys.executable)
            try:
                proc=subprocess.Popen([sys.executable,'-c',bench.LAUNCH_CODE,json.dumps(payload)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for _ in range(5):
                    line=proc.stdout.readline()
                    if line.startswith(bench.SOURCE_MARKER):
                        child_pid=json.loads(line[len(bench.SOURCE_MARKER):])['pid'];break
                self.assertIsNotNone(child_pid)
                proc.kill();proc.wait(timeout=3)
                with lock_path.open('a+') as held:
                    with self.assertRaises(BlockingIOError):fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    os.killpg(child_pid,signal.SIGTERM)
                    deadline=time.monotonic()+3
                    while True:
                        try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:
                            if time.monotonic()>deadline:raise
                            time.sleep(.01)
                    fcntl.flock(held,fcntl.LOCK_UN)
            finally:
                if proc and proc.poll() is None:proc.kill();proc.wait()
                if child_pid:
                    try:os.killpg(child_pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                if proc and proc.stdout:proc.stdout.close()

    def test_local_inspection_hashes_untracked_files_and_model_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.mkdir();recipes=root/'recipes';recipes.mkdir()
            (model/'model.safetensors').write_bytes(b'CPU identity fixture, not a model')
            (model/'config.json').write_text('{}')
            (recipes/'one.recipe').write_bytes(b'recipe fixture')
            (root/'untracked.rs').write_text('fn fixture() {}')
            subprocess.run(['git','init','-q',directory],check=True)
            subprocess.run(['git','-C',directory,'-c','user.name=ChenxiHe231','-c','user.email=chenxi_he@foxmail.com','commit','--allow-empty','-qm','CPU identity fixture'],check=True)
            opts=args(workdir=directory,checkpoint=str(model),tuning_dir=str(recipes),context_binary=sys.executable,decode_binary=sys.executable)
            snapshot=bench.Transport(opts).snapshot()
            self.assertTrue(snapshot['source']['source_dirty'])
            self.assertIn('untracked.rs',snapshot['source']['source_untracked_sha256'])
            self.assertEqual(snapshot['recipes']['count'],1)
            self.assertEqual(snapshot['model_sha256']['model.safetensors'],bench.sha256_file(model/'model.safetensors'))
            self.assertEqual(snapshot['binary_sha256']['context'],bench.sha256_file(sys.executable))

    def test_fingerprint_binds_dirty_patch_and_untracked_files(self):
        source=dict(source_commit='a'*40,source_dirty=True,source_patch_sha256='b'*64,source_untracked_sha256={'x.rs':'c'*64})
        snapshot={'source':source,'binary_sha256':{'context':'d'*64,'decode':'e'*64},'model_sha256':{'model.safetensors':'f'*64}}
        bench.verify_identity(snapshot,snapshot)
        bad=copy.deepcopy(snapshot);bad['source']['source_untracked_sha256']['x.rs']='0'*64
        with self.assertRaisesRegex(ValueError,'untracked'):bench.verify_identity(bad,snapshot)


if __name__=='__main__':unittest.main()
