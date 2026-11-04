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
              workdir='/',ssh_host=None,timeout_sec=2,lock_timeout_sec=3,nsys='nsys',profile_dir='/trace',allow_repeat_divergence=False,
              mtp_strategy='fixed',scalar_calibration=None)
    base.update(changes)
    return SimpleNamespace(**base)


def recipe_inventory_fixture(files=None):
    rows=copy.deepcopy(files if files is not None else [dict(path='scalar.recipe',bytes=5,sha256='c'*64)])
    digest=bench.hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(path='/recipes',files=rows,count=sum(row['path'].endswith('.recipe') for row in rows),manifest_sha256=digest)


def fixture(case, opts=None, values=None):
    opts=opts or args()
    is_decode=case['group']=='decode';tune=opts.run_mode=='tune';profile=opts.run_mode=='profile'
    n=case['initial_kv_tokens']+(0 if is_decode else case['new_input_tokens'])
    repeats=opts.repeats if opts.run_mode=='formal' else 1;warmups=0 if tune else opts.warmups
    identity=bench.input_identity(n);prefix='apxinf.qwen35.benchmark.' if is_decode else 'apxinf.qwen35.context_benchmark.'
    output=1024 if is_decode else 1
    version='v2' if is_decode else 'v1'
    flags=dict(mode=opts.mode,autotune=tune,performance_eligible=not tune and not profile,profiling=profile,strict_recipe_required=not tune)
    config=dict(schema=prefix+'config.'+version,**flags,output_tokens=output,warmups=warmups,repeats=repeats,
                qwen35_kv_cache=opts.kv_cache,qwen35_moe_activation=opts.moe_activation,temperature=0,sampling_mode='greedy',
                tuning_dir=opts.tuning_dir,input_digest_fnv1a64=identity['fnv1a64'],input_seed_hex=identity['seed_hex'],
                mtp_depth=3,mtp_strategy='fixed',draft_kv_dtype='fp8',draft_kv_scale=1.0)
    config['allow_repeat_divergence']=opts.allow_repeat_divergence
    strategy=opts.mtp_strategy if opts.mode=='mtp' else 'disabled'
    depth=opts.mtp_depth if strategy=='fixed' else None
    config.update(mtp_enabled=opts.mode=='mtp',mtp_strategy=strategy,mtp_depth=depth,fixed_depth=depth)
    if is_decode:config.update(prompt_tokens=n,context_reuse=False,
        timing_method=bench.DECODE_TIMING_METHOD,timing_clock='std::time::Instant',
        timing_origin='benchmark-generate-entry',timing_clock_reads_per_generate=4,
        ttft_scope='generate_entry_to_first_callback',decode_scope='first_to_last_callback_intervals',
        post_last_callback_scope='last_callback_to_generate_return')
    else:config.update(prefix_tokens=case['initial_kv_tokens'],input_tokens=case['new_input_tokens'],total_context_tokens=n)
    iterations=[]
    values=values or list(range(100,100+repeats))
    for i in range(warmups+repeats):
        value=50 if i<warmups else values[i-warmups]
        row=dict(schema=prefix+'iteration.'+version,**flags,iteration=i,warmup=i<warmups,output_tokens=output,
                 decode_intervals=output-1,token_digest_fnv1a64='0123456789abcdef')
        if is_decode:
            decode_ns=round((output-1)*1e9/value)
            decode_ms=decode_ns/1e6
            legacy_decode_ms=decode_ms+7
            row.update(prompt_tokens=n,computed_prompt_tokens=n,reused_prompt_tokens=0,decode_tokens_per_second=value,
                       ttft_ms=200,decode_interval_ms=decode_ms,post_last_callback_ms=5,total_latency_ms=205+decode_ms,
                       timing_method=bench.DECODE_TIMING_METHOD,
                       callback_timing=dict(clock='std::time::Instant',origin='benchmark-generate-entry',clock_reads=4,
                           callback_count=1024,decode_intervals=1023,first_callback_elapsed_ns=200_000_000,
                           last_callback_elapsed_ns=200_000_000+decode_ns,generate_return_elapsed_ns=205_000_000+decode_ns),
                       legacy_engine_profile=dict(schema='apxinf.qwen35.legacy-engine-profile.v1',
                           scope='first_model_token_ready_to_engine_finalize_callbacks_excluded',ttft_ms=190,
                           total_latency_ms=190+legacy_decode_ms,decode_including_finalize_ms=legacy_decode_ms,
                           generation_tokens_per_second=(output-1)*1000/legacy_decode_ms))
            used=opts.mtp_depth if opts.mtp_strategy=='fixed' else 3
            by_depth=[1 if d<used else 0 for d in range(3)]
            row['speculation']=dict(draft_steps=3,proposed_tokens=sum(by_depth),proposed_by_depth=by_depth,
                                    matched_by_depth=by_depth,accepted_prefix_by_depth=by_depth) if opts.mode=='mtp' else None
        else:row.update(prefix_tokens=case['initial_kv_tokens'],input_tokens=case['new_input_tokens'],total_context_tokens=n,
                        initial_kv_tokens=case['initial_kv_tokens'],computed_input_tokens=case['new_input_tokens'],
                        committed_context_tokens_after_prefill=n,prefill_ms=value,decode_ms=None,decode_tokens_per_second=None)
        iterations.append(row)
    summary=dict(schema=prefix+'summary.'+version,**flags,divergent_repeats=0,output_tokens=output,decode_intervals=output-1,warmups=warmups,repeats=repeats)
    if is_decode:
        summary.update(prompt_tokens=n,decode_tokens_per_second={'median':bench.statistics.median(values)},
                       timing_method=bench.DECODE_TIMING_METHOD)
        measured=[row for row in iterations if not row['warmup']]
        for key in ('ttft_ms','decode_interval_ms','post_last_callback_ms','total_latency_ms'):
            summary[key]={'median':bench.statistics.median(row[key] for row in measured)}
        summary['legacy_engine_profile']=dict(scope='first_model_token_ready_to_engine_finalize_callbacks_excluded',
            generation_tokens_per_second={'median':bench.statistics.median(row['legacy_engine_profile']['generation_tokens_per_second'] for row in measured)})
        summary['speculation']={key:([x*repeats for x in value] if isinstance(value,list) else value*repeats)
                                for key,value in iterations[-1]['speculation'].items() if key!='draft_steps'} if opts.mode=='mtp' else None
    else:summary.update(prefix_tokens=case['initial_kv_tokens'],input_tokens=case['new_input_tokens'],total_context_tokens=n,prefill_ms={'median':bench.statistics.median(values)})
    return [config,*iterations,summary]


def divergent_fixture(case, opts, indices=(2, 4), values=None):
    records=fixture(case,opts,values)
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


def raw_output(records,code=0,sha='a'*64,report='',effective_env=None,recipe_inventory=None):
    process={'pid':123,'kind':'benchmark','sha256':sha,'exe':'/bin/context'}
    config=records[0]
    controls=effective_env if effective_env is not None else ({bench.MTP_FIXED_ENV:str(config['mtp_depth'])} if config.get('mtp_strategy')=='fixed' else {})
    process['effective_mtp_environment']={key:controls.get(key) for key in (bench.MTP_FIXED_ENV,bench.MTP_SCALAR_TPOT_ENV)}
    recipe=recipe_inventory_fixture() if recipe_inventory is None else recipe_inventory
    lock={'path':bench.GPU_LOCK_PATH,'acquired':True,'owner_pid':42,'wait_ms':0,'recipes_before':recipe,'target_binary_sha256':sha}
    receipt={'returncode':code,'timed_out':False,'child_reaped':True,'lock_held_until_reaped':True,'recipes_after':recipe,'target_binary_sha256':sha}
    return bench.LOCK_MARKER+json.dumps(lock)+'\n'+bench.SOURCE_MARKER+json.dumps(process)+'\n'+report+'\n'.join(json.dumps(row) for row in records)+'\n'+bench.EXIT_MARKER+json.dumps(receipt)+'\n'


def scalar_calibration_fixture(root, cases, diagnostic=False, env=None):
    """CPU artifact fixture with actual runner receipts, never a GPU score."""
    root.mkdir(parents=True,exist_ok=True)
    opts=args(mode='scalar',repeats=6,allow_repeat_divergence=diagnostic,env=env or [])
    source=dict(source_commit='a'*40,source_dirty=False,source_patch_sha256='b'*64,source_untracked_sha256={})
    identity=dict(source=source,binary_sha256=dict(context='a'*64,decode='a'*64),model_sha256={'model.safetensors':'f'*64})
    recipes=recipe_inventory_fixture()
    env=bench.environment(opts);rows=[]
    for index,case in enumerate(cases):
        values=[(index+1)*value for value in [10,20,40,80,160,320]]
        records=divergent_fixture(case,opts,indices=(1,2,4),values=values) if diagnostic else fixture(case,opts,values)
        raw=raw_output(records);log=root/(case['id']+'.log');log.write_text(raw)
        transport=dict(transport_returncode=0,transport_timeout=False)
        validation=bench.validate_run(raw,case,opts,recipes,recipes,'a'*64,transport)
        assert validation['measurement_valid'] and validation['performance_accepted'] is (not diagnostic),validation['failures']
        rows.append(dict(case=case['id'],group='decode',run_mode='formal',argv=bench.command_for_case(opts,case),
                         log_sha256=bench.sha256_file(log),recipes_before=recipes,recipes_after=recipes,transport=transport,**validation))
    summary=dict(schema=bench.SCHEMA,run_mode='formal',all_cases_valid=True,measurement_valid=True,performance_accepted=not diagnostic,
                 source_unchanged=True,accuracy_status='unresolved' if diagnostic else 'repeat_consistent',comparison_policy=bench.comparison_policy(opts),allow_repeat_divergence=diagnostic,
                 observed_after=identity,results=rows,requested_cases=[case['id'] for case in cases],completed_cases=len(cases))
    metadata=dict(schema=bench.SCHEMA,run_mode='formal',comparison_policy=bench.comparison_policy(opts),allow_repeat_divergence=diagnostic,
                  observed_before=identity,expected=identity,environment=env)
    (root/'summary.json').write_text(json.dumps(summary));(root/'metadata.json').write_text(json.dumps(metadata))
    return root/'summary.json',identity


class CallbackTimingV2Contracts(unittest.TestCase):
    def test_callback_metric_excludes_tail_but_preserves_legacy_separately(self):
        case=bench.canonical_cases()[9];opts=args();records=fixture(case,opts)
        result=bench.validate_records(records,case,opts)
        self.assertEqual(result['observed_median'],102)
        self.assertEqual(result['decode_timing_method'],bench.DECODE_TIMING_METHOD)
        self.assertEqual(len(result['measured_callback_timing']),5)
        self.assertEqual(result['measured_callback_timing'][0]['tail_ms'],5)
        self.assertNotEqual(result['measured_callback_timing'][0]['tps'],result['measured_callback_timing'][0]['legacy_tps'])
        self.assertIn('not_per_token_latency',result['decode_tpot_distribution']['sample_scope'])

    def test_warmups_have_the_same_strict_1024_callbacks_and_clock_contract(self):
        for change in ('callbacks','intervals','clock_reads','origin','clock','monotonic','nanoseconds_type'):
            case=bench.canonical_cases()[9];opts=args();records=fixture(case,opts)
            timing=records[1]['callback_timing'] # warmup0, not a measured repeat
            if change=='callbacks':timing['callback_count']=1023
            elif change=='intervals':timing['decode_intervals']=1024
            elif change=='clock_reads':timing['clock_reads']=1026
            elif change=='origin':timing['origin']='model_profile_start'
            elif change=='clock':timing['clock']='wall_clock'
            elif change=='monotonic':timing['last_callback_elapsed_ns']=timing['first_callback_elapsed_ns']
            else:timing['first_callback_elapsed_ns']=True
            with self.subTest(change=change),self.assertRaises(ValueError):bench.validate_records(records,case,opts)

    def test_tail_not_added_to_decode_denominator_or_subtracted_by_uniform_estimate(self):
        case=bench.canonical_cases()[9];opts=args()
        for change in ('tail_in_denominator','total_without_tail','uniform_subtraction','legacy_at_top','legacy_scope','summary_tail'):
            records=fixture(case,opts);row=records[3]
            if change=='tail_in_denominator':row['decode_interval_ms']+=row['post_last_callback_ms']
            elif change=='total_without_tail':row['total_latency_ms']-=row['post_last_callback_ms']
            elif change=='uniform_subtraction':row['decode_interval_ms']-=1023*.001
            elif change=='legacy_at_top':row['generation_tokens_per_second']=row['legacy_engine_profile']['generation_tokens_per_second']
            elif change=='legacy_scope':row['legacy_engine_profile']['scope']='same_as_callback'
            else:records[-1]['post_last_callback_ms']['median']=0
            with self.subTest(change=change),self.assertRaises(ValueError):bench.validate_records(records,case,opts)

    def test_native_v1_or_mixed_decode_schemas_cannot_be_relabelled(self):
        case=bench.canonical_cases()[9];opts=args()
        for mode in ('all','mixed','future'):
            records=fixture(case,opts)
            if mode=='all':
                for row in records:row['schema']=row['schema'].replace('.v2','.v1')
            elif mode=='mixed':records.insert(1,dict(records[1],schema='apxinf.qwen35.benchmark.iteration.v1'))
            else:records[1]['schema']='apxinf.qwen35.benchmark.iteration.v3'
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'legacy Decode v1'):bench.validate_records(records,case,opts)

    def test_finalization_may_be_zero_but_primary_interval_must_be_positive(self):
        case=bench.canonical_cases()[9];opts=args();records=fixture(case,opts)
        for row in records[1:-1]:
            row['post_last_callback_ms']=0
            row['total_latency_ms']-=5
            row['callback_timing']['generate_return_elapsed_ns']=row['callback_timing']['last_callback_elapsed_ns']
        records[-1]['post_last_callback_ms']['median']=0
        records[-1]['total_latency_ms']['median']-=5
        bench.validate_records(records,case,opts)
        records[1]['decode_interval_ms']=0
        with self.assertRaises(ValueError):bench.validate_records(records,case,opts)


class MatrixContracts(unittest.TestCase):
    def test_decode_policy_selection_rejects_prefill_relabeling_and_implicit_adaptive_cost(self):
        decode=bench.canonical_cases()[9:];prefill=bench.canonical_cases()[0]
        for depth in (1,2,3):
            opts=args(mtp_depth=depth);bench.validate_selection_policy(opts,decode)
            self.assertEqual(bench.environment(opts)[bench.MTP_FIXED_ENV],str(depth))
            if depth!=3:
                with self.assertRaisesRegex(ValueError,'Decode-only'):bench.validate_selection_policy(opts,[prefill,*decode])
        adaptive=args(mtp_strategy='adaptive',scalar_calibration=Path('/fixture/scalar/summary.json'))
        bench.validate_selection_policy(adaptive,decode)
        env=bench.environment(adaptive,18.75)
        self.assertNotIn(bench.MTP_FIXED_ENV,env);self.assertEqual(env[bench.MTP_SCALAR_TPOT_ENV],'18.75')
        self.assertIsNone(bench.mtp_policy(adaptive,18.75)['fixed_depth'])
        for changes in ({'scalar_calibration':None},{'mtp_depth':1},{'mode':'scalar'}):
            bad=copy.copy(adaptive)
            for key,value in changes.items():setattr(bad,key,value)
            with self.assertRaises(ValueError):bench.validate_selection_policy(bad,decode)
        diagnostic=copy.copy(adaptive);diagnostic.allow_repeat_divergence=True
        bench.validate_selection_policy(diagnostic,decode)
        with self.assertRaisesRegex(ValueError,'Decode-only'):bench.validate_selection_policy(adaptive,[prefill])
        with self.assertRaisesRegex(ValueError,'calibrated scalar TPOT'):bench.environment(adaptive)
        for key in (bench.MTP_FIXED_ENV,bench.MTP_SCALAR_TPOT_ENV):
            with self.assertRaisesRegex(ValueError,'controlled'):bench.environment(args(env=[key+'=1']))
        bench.validate_selection_policy(args(mode='scalar'),decode)
        self.assertNotIn(bench.MTP_FIXED_ENV,bench.environment(args(mode='scalar')))

    def test_native_decode_strategy_and_actual_depth_counters_are_verified(self):
        case=bench.canonical_cases()[9]
        for opts in [args(mtp_depth=depth) for depth in (1,2,3)]+[args(mtp_strategy='adaptive'),args(mode='scalar')]:
            observed=bench.validate_records(fixture(case,opts),case,opts)
            expected=0 if opts.mode=='scalar' else (3 if opts.mtp_strategy=='adaptive' else opts.mtp_depth)
            self.assertEqual(observed['observed_max_proposed_depth'],expected)
            self.assertEqual(observed['mtp_policy']['fixed_depth'],opts.mtp_depth if opts.mode=='mtp' and opts.mtp_strategy=='fixed' else None)
        for mutation in ('draft4','depth4','beyond_fixed','scalar_stats','adaptive_fixed_label','aggregate'):
            with self.subTest(mutation=mutation):
                opts=args(mtp_depth=1)
                if mutation=='scalar_stats':opts=args(mode='scalar')
                if mutation=='adaptive_fixed_label':opts=args(mtp_strategy='adaptive')
                records=fixture(case,opts)
                if mutation=='draft4':records[1]['speculation']['draft_steps']=4
                elif mutation=='depth4':records[1]['speculation']['proposed_by_depth']=[1,0,0,1]
                elif mutation=='beyond_fixed':records[1]['speculation']['proposed_by_depth']=[1,1,0]
                elif mutation=='scalar_stats':records[1]['speculation']={}
                elif mutation=='adaptive_fixed_label':records[0]['fixed_depth']=3
                elif mutation=='aggregate':records[-1]['speculation']['proposed_by_depth']=[6,0,0];records[-1]['speculation']['proposed_tokens']=6
                with self.assertRaises(ValueError):bench.validate_records(records,case,opts)

    def test_scalar_calibration_uses_per_case_median_of_measured_tpot_not_inverse_tps(self):
        with tempfile.TemporaryDirectory() as directory:
            cases=bench.canonical_cases()[9:11];path,identity=scalar_calibration_fixture(Path(directory),cases)
            opts=args(mtp_strategy='adaptive',scalar_calibration=path)
            receipt=bench.load_scalar_calibration(opts,cases)
            self.assertEqual(receipt['summary_sha256'],bench.sha256_file(path))
            self.assertEqual(receipt['identity'],identity)
            for case,tpot in zip(cases,[18.75,9.375]):
                row=receipt['cases'][case['id']]
                self.assertEqual(row['scalar_tpot_ms'],tpot);self.assertEqual(row['decode_intervals'],1023)
                self.assertEqual(len(row['measured_decode_tpot_ms']),6)
            self.assertNotAlmostEqual(receipt['cases'][cases[0]['id']]['scalar_tpot_ms'],1000/60)

    def test_scalar_calibration_rejects_unaccepted_diagnostic_wrong_precision_or_incomplete_raw(self):
        variants=('diagnostic','not_accepted','unresolved','missing_case','changed_source','wrong_mode','raw_hash',
                  'short_output','stored_tpot','wrong_kv','wrong_activation','kernel_env','too_few_repeats')
        for variant in variants:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                case=bench.canonical_cases()[9];path,_=scalar_calibration_fixture(Path(directory),[case])
                summary=json.loads(path.read_text());opts=args(mtp_strategy='adaptive',scalar_calibration=path)
                selected=[case];log=Path(directory)/(case['id']+'.log')
                if variant=='diagnostic':summary['allow_repeat_divergence']=True
                elif variant=='not_accepted':summary['performance_accepted']=False
                elif variant=='unresolved':summary['accuracy_status']='unresolved'
                elif variant=='missing_case':selected=bench.canonical_cases()[9:11]
                elif variant=='changed_source':summary['observed_after']['source']['source_commit']='f'*40
                elif variant=='raw_hash':log.write_text(log.read_text()+'\n')
                elif variant=='stored_tpot':summary['results'][0]['observed']['measured_decode_tpot_ms'][0]=999
                elif variant=='wrong_kv':opts.kv_cache='fp8'
                elif variant=='wrong_activation':opts.moe_activation='bf16'
                elif variant=='kernel_env':opts.env=['APXINF_QWEN35_CUDA_GRAPHS=1']
                else:
                    records,_,_,_=bench.parse_output(log.read_text())
                    if variant=='wrong_mode':records[0]['mode']='mtp'
                    elif variant=='too_few_repeats':records[0]['repeats']=4
                    elif variant=='short_output':records[1]['output_tokens']=1023
                    log.write_text(raw_output(records));summary['results'][0]['log_sha256']=bench.sha256_file(log)
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):bench.load_scalar_calibration(opts,selected)

    def test_adaptive_main_checks_source_elf_weight_identity_before_any_execution(self):
        for changed in ('source','elf','weights'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);case=bench.canonical_cases()[9]
                path,identity=scalar_calibration_fixture(root/'scalar',[case]);actual=copy.deepcopy(identity)
                actual['recipes']=recipe_inventory_fixture()
                if changed=='source':actual['source']['source_patch_sha256']='0'*64
                elif changed=='elf':actual['binary_sha256']['decode']='0'*64
                else:actual['model_sha256']['model.safetensors']='0'*64
                metadata=root/'target.json';metadata.write_text(json.dumps(actual))
                command=['matrix','--mtp-strategy','adaptive','--scalar-calibration',str(path),'--case',case['id'],
                         '--workdir','/','--context-binary','/bin/context','--decode-binary','/bin/decode','--checkpoint','/model',
                         '--tuning-dir','/recipes','--metadata',str(metadata),'--output',str(root/'result')]
                with patch.object(sys,'argv',command),patch.object(bench.Transport,'snapshot',return_value=actual),patch.object(bench.Transport,'execute') as execute:
                    with self.assertRaisesRegex(ValueError,'identity mismatch'):bench.main()
                    execute.assert_not_called()

    def test_adaptive_main_records_each_case_cost_without_claiming_fixed_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cases=bench.canonical_cases()[9:11];path,identity=scalar_calibration_fixture(root/'scalar',cases)
            actual=copy.deepcopy(identity);recipes=recipe_inventory_fixture();actual['recipes']=recipes
            metadata=root/'target.json';metadata.write_text(json.dumps(actual));out=root/'adaptive';seen=[]
            def execute(_transport,argv,env,log,profile=False):
                case=next(case for case in cases if str(case['initial_kv_tokens'])==argv[argv.index('--prompt-tokens')+1])
                seen.append(env.copy());self.assertNotIn(bench.MTP_FIXED_ENV,env)
                log.write_text(raw_output(fixture(case,args(mtp_strategy='adaptive')),effective_env=env))
                return dict(transport_returncode=0,transport_timeout=False)
            command=['matrix','--mtp-strategy','adaptive','--scalar-calibration',str(path),'--cases',','.join(c['id'] for c in cases),
                     '--workdir','/','--context-binary','/bin/context','--decode-binary','/bin/decode','--checkpoint','/model',
                     '--tuning-dir','/recipes','--metadata',str(metadata),'--output',str(out)]
            with patch.object(sys,'argv',command),patch.object(bench.Transport,'snapshot',lambda _s,kind='full':recipes if kind=='recipes' else actual),patch.object(bench.Transport,'execute',execute):
                self.assertEqual(bench.main(),0)
            self.assertEqual([float(env[bench.MTP_SCALAR_TPOT_ENV]) for env in seen],[18.75,9.375])
            summary=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'metadata.json').read_text())
            self.assertTrue(summary['performance_accepted']);self.assertTrue(manifest['calibration_target_identity_verified'])
            for row in summary['results']:
                self.assertEqual(row['mtp_policy']['strategy'],'adaptive');self.assertIsNone(row['mtp_policy']['fixed_depth'])
                self.assertEqual(row['scalar_calibration']['calibration_summary_sha256'],bench.sha256_file(path))
                self.assertEqual(row['process']['effective_mtp_environment'],{key:row['environment'].get(key) for key in (bench.MTP_FIXED_ENV,bench.MTP_SCALAR_TPOT_ENV)})

    def test_diagnostic_adaptive_requires_matching_policy_and_never_becomes_official_performance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=bench.canonical_cases()[9]
            path,identity=scalar_calibration_fixture(root/'diagnostic-scalar',[case],diagnostic=True)
            opts=args(mtp_strategy='adaptive',scalar_calibration=path,allow_repeat_divergence=True)
            receipt=bench.load_scalar_calibration(opts,[case])
            self.assertFalse(receipt['performance_accepted']);self.assertEqual(receipt['accuracy_status'],'unresolved')
            calibrated=receipt['cases'][case['id']]
            self.assertEqual(calibrated['scalar_tpot_ms'],18.75);self.assertEqual(len(calibrated['repeat_divergences']),3)
            self.assertEqual(len(calibrated['output_digests']),8);self.assertFalse(calibrated['performance_accepted'])
            with self.assertRaisesRegex(ValueError,'same explicit'):bench.load_scalar_calibration(args(mtp_strategy='adaptive',scalar_calibration=path),[case])
            strict_path,_=scalar_calibration_fixture(root/'strict-scalar',[case])
            with self.assertRaisesRegex(ValueError,'same explicit'):bench.load_scalar_calibration(args(mtp_strategy='adaptive',scalar_calibration=strict_path,allow_repeat_divergence=True),[case])
            actual=copy.deepcopy(identity);recipes=recipe_inventory_fixture();actual['recipes']=recipes
            metadata=root/'target.json';metadata.write_text(json.dumps(actual));out=root/'diagnostic-adaptive'
            def execute(_transport,argv,env,log,profile=False):
                self.assertIn('--allow-repeat-divergence',argv);self.assertEqual(env[bench.MTP_SCALAR_TPOT_ENV],'18.75')
                log.write_text(raw_output(divergent_fixture(case,opts),effective_env=env))
                return dict(transport_returncode=0,transport_timeout=False)
            command=['matrix','--mtp-strategy','adaptive','--allow-repeat-divergence','--scalar-calibration',str(path),'--case',case['id'],
                     '--workdir','/','--context-binary','/bin/context','--decode-binary','/bin/decode','--checkpoint','/model',
                     '--tuning-dir','/recipes','--metadata',str(metadata),'--output',str(out)]
            with patch.object(sys,'argv',command),patch.object(bench.Transport,'snapshot',lambda _s,kind='full':recipes if kind=='recipes' else actual),patch.object(bench.Transport,'execute',execute):
                self.assertEqual(bench.main(),0)
            summary=json.loads((out/'summary.json').read_text());row=summary['results'][0]
            self.assertTrue(summary['measurement_valid']);self.assertFalse(summary['performance_accepted'])
            self.assertEqual(summary['accuracy_status'],'unresolved');self.assertFalse(row['performance_accepted'])
            self.assertEqual(row['comparison_policy'],'diagnostic-repeat-divergence-v1')
            self.assertEqual(len(row['native_iterations']),7);self.assertEqual(len(row['repeat_divergences']),2)
            self.assertFalse(row['scalar_calibration']['performance_accepted'])
            table=(out/'summary.md').read_text()
            self.assertIn('Diagnostic median',table);self.assertNotIn('| Formal N |',table);self.assertNotIn('| passed |',table)

    def test_diagnostic_calibration_rejects_policy_upgrade_or_incomplete_divergence_evidence(self):
        variants=('summary_official','row_official','mixed_policy','wrong_accuracy','stored_events','stored_digests',
                  'raw_firstdiff','raw_digest','missing_event','short_output','recipe_changed','wrong_lock')
        for variant in variants:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                case=bench.canonical_cases()[9];path,_=scalar_calibration_fixture(Path(directory),[case],diagnostic=True)
                summary=json.loads(path.read_text());row=summary['results'][0]
                opts=args(mtp_strategy='adaptive',scalar_calibration=path,allow_repeat_divergence=True)
                log=Path(directory)/(case['id']+'.log')
                if variant=='summary_official':summary['performance_accepted']=True
                elif variant=='row_official':row['performance_accepted']=True
                elif variant=='mixed_policy':row['comparison_policy']='strict-repeat-consistency-v1'
                elif variant=='wrong_accuracy':summary['accuracy_status']='repeat_consistent'
                elif variant=='stored_events':row['repeat_divergences']=[]
                elif variant=='stored_digests':row['observed']['output_digests'][1]='0'*16
                elif variant=='recipe_changed':row['recipes_after']['manifest_sha256']='changed'
                elif variant=='wrong_lock':
                    raw=log.read_text().replace('"path": "/tmp/apxinf-gpu.lock"','"path": "/tmp/wrong.lock"')
                    log.write_text(raw);row['log_sha256']=bench.sha256_file(log)
                else:
                    records,_,_,_=bench.parse_output(log.read_text())
                    events=[r for r in records if r.get('schema')=='apxinf.qwen35.benchmark.repeat_divergence.v1']
                    if variant=='raw_firstdiff':events[0]['first_diff']=1024
                    elif variant=='raw_digest':events[0]['actual_digest_fnv1a64']='0'*16
                    elif variant=='missing_event':records.remove(events[0])
                    elif variant=='short_output':records[1]['output_tokens']=1023
                    log.write_text(raw_output(records));row['log_sha256']=bench.sha256_file(log)
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):bench.load_scalar_calibration(opts,[case])

    def test_profile_and_audit_can_add_nvtx_to_uninstrumented_scalar_cost_with_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            case=bench.canonical_cases()[9];path,_=scalar_calibration_fixture(Path(directory),[case],env=['APXINF_NVTX=0'])
            recipes=recipe_inventory_fixture()
            for mode in ('profile','audit'):
                opts=args(run_mode=mode,mtp_strategy='adaptive',scalar_calibration=path,env=['APXINF_NVTX=1'])
                receipt=bench.load_scalar_calibration(opts,[case])['cases'][case['id']]
                self.assertEqual(receipt['scalar_tpot_ms'],18.75)
                observers=receipt['observer_environment_comparison']
                self.assertEqual(observers['run_mode'],mode)
                self.assertEqual(observers['original_target_env_arguments'],['APXINF_NVTX=1'])
                self.assertEqual(observers['calibration_original_environment']['APXINF_NVTX'],'0')
                self.assertEqual(observers['calibration_observer_environment']['APXINF_NVTX'],'0')
                self.assertEqual(observers['current_observer_environment']['APXINF_NVTX'],'1')
                self.assertIn('APXINF_NVTX',observers['ignored_observer_keys'])
                self.assertEqual(observers['current_observer_environment']['APXINF_GEMM_REPORT'],'1')
                env=bench.environment(opts,receipt['scalar_tpot_ms'])
                report='[apxinf-gemm] source=recipe\n[apxinf-attn] source=recipe\n'
                raw=raw_output(fixture(case,opts),effective_env=env,report=report)
                if mode=='profile':raw=raw.replace('"kind": "benchmark"','"kind": "profiler_launcher"')
                result=bench.validate_run(raw,case,opts,recipes,recipes,'a'*64,dict(transport_returncode=0,transport_timeout=False),env)
                self.assertTrue(result['measurement_valid'],result['failures']);self.assertFalse(result['performance_accepted'])

    def test_nvtx_exception_does_not_allow_formal_or_math_timing_precision_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            case=bench.canonical_cases()[9];path,_=scalar_calibration_fixture(Path(directory),[case],env=['APXINF_NVTX=0'])
            formal=args(mtp_strategy='adaptive',scalar_calibration=path,env=['APXINF_NVTX=1'])
            with self.assertRaisesRegex(ValueError,'kernel environment differs'):bench.load_scalar_calibration(formal,[case])
            for changes in ({'env':['APXINF_NVTX=1','APXINF_QWEN35_CUDA_GRAPHS=1']},
                            {'env':['APXINF_NVTX=1','APXINF_QWEN35_ROUTER_GEMV=1']},
                            {'env':['APXINF_NVTX=1','APXINF_QWEN35_VERIFY_LAYER_TIMING=1']},
                            {'kv_cache':'fp8'},{'moe_activation':'bf16'}):
                opts=args(run_mode='profile',mtp_strategy='adaptive',scalar_calibration=path,env=['APXINF_NVTX=1'])
                for key,value in changes.items():setattr(opts,key,value)
                with self.subTest(changes=changes),self.assertRaises(ValueError):bench.load_scalar_calibration(opts,[case])

    def test_scalar_recipe_subset_allows_new_verifier_keys_but_rejects_late_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            case=bench.canonical_cases()[9];path,_=scalar_calibration_fixture(Path(directory),[case])
            opts=args(mtp_strategy='adaptive',scalar_calibration=path)
            calibrated=bench.load_scalar_calibration(opts,[case])['cases'][case['id']]
            required=calibrated['recipe_subset'];original=recipe_inventory_fixture()
            self.assertEqual(required['files'],original['files'])
            self.assertEqual(required['manifest_sha256'],original['manifest_sha256'])
            verifier=dict(path='verify/rows4.recipe',bytes=8,sha256='d'*64)
            expanded=recipe_inventory_fixture([*original['files'],verifier])
            self.assertNotEqual(expanded['manifest_sha256'],required['manifest_sha256'])
            check=bench.verify_recipe_subset(required,expanded)
            self.assertTrue(check['verified']);self.assertEqual(check['added_recipe_keys'],['verify/rows4.recipe'])
            env=bench.environment(opts,calibrated['scalar_tpot_ms']);transport=dict(transport_returncode=0,transport_timeout=False)
            raw=raw_output(fixture(case,opts),effective_env=env,recipe_inventory=expanded)
            result=bench.validate_run(raw,case,opts,expanded,expanded,'a'*64,transport,env,required)
            self.assertTrue(result['performance_accepted'],result['failures'])
            self.assertEqual(result['scalar_recipe_subset_locked_check']['matched_count'],1)
            # This is a late rewrite after the initial snapshot, but stable
            # throughout the candidate process: only the new subset gate
            # distinguishes it from the previously accepted recipe inventory.
            changed=recipe_inventory_fixture([dict(original['files'][0],sha256='e'*64),verifier])
            late=raw_output(fixture(case,opts),effective_env=env,recipe_inventory=changed)
            self.assertTrue(bench.validate_run(late,case,opts,changed,changed,'a'*64,transport,env)['valid'])
            rejected=bench.validate_run(late,case,opts,changed,changed,'a'*64,transport,env,required)
            self.assertFalse(rejected['valid']);self.assertTrue(any('recipe changed or missing' in f for f in rejected['failures']))
            with self.assertRaisesRegex(ValueError,'changed or missing'):bench.verify_recipe_subset(required,recipe_inventory_fixture([verifier]))
            # Adding a key during the formal process still fails the existing
            # whole-directory before/after gate; additions are allowed only
            # between scalar calibration and the candidate acquisition.
            during=bench.validate_run(raw,case,opts,original,expanded,'a'*64,transport,env,required)
            self.assertFalse(during['valid']);self.assertTrue(any('recipes changed' in f for f in during['failures']))

    def test_adaptive_preflight_rejects_changed_or_deleted_scalar_recipe_before_execution(self):
        for change in ('rewrite','delete'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);case=bench.canonical_cases()[9];path,identity=scalar_calibration_fixture(root/'scalar',[case])
                original=recipe_inventory_fixture()['files'][0]
                files=[dict(original,bytes=6)] if change=='rewrite' else [dict(path='new-verifier.recipe',bytes=8,sha256='d'*64)]
                actual=copy.deepcopy(identity);actual['recipes']=recipe_inventory_fixture(files)
                metadata=root/'target.json';metadata.write_text(json.dumps(actual))
                command=['matrix','--mtp-strategy','adaptive','--scalar-calibration',str(path),'--case',case['id'],
                         '--workdir','/','--context-binary','/bin/context','--decode-binary','/bin/decode','--checkpoint','/model',
                         '--tuning-dir','/recipes','--metadata',str(metadata),'--output',str(root/'result')]
                with patch.object(sys,'argv',command),patch.object(bench.Transport,'snapshot',return_value=actual),patch.object(bench.Transport,'execute') as execute:
                    with self.assertRaisesRegex(ValueError,'recipe changed or missing'):bench.main()
                    execute.assert_not_called()

    def test_adaptive_plan_marks_recipe_subset_pending_without_contacting_target(self):
        with tempfile.TemporaryDirectory() as directory:
            case=bench.canonical_cases()[9];path,_=scalar_calibration_fixture(Path(directory),[case])
            command=['matrix','--plan','--mtp-strategy','adaptive','--scalar-calibration',str(path),'--case',case['id'],
                     '--workdir','/','--context-binary','/bin/context','--decode-binary','/bin/decode','--checkpoint','/model','--tuning-dir','/recipes']
            with patch.object(sys,'argv',command),patch.object(sys,'stdout',bench.io.StringIO()) as stdout,patch.object(bench,'Transport') as transport:
                self.assertEqual(bench.main(),0);transport.assert_not_called();plan=json.loads(stdout.getvalue())
            self.assertFalse(plan['calibration_target_recipes_verified'])
            cost=plan['plan'][0]['scalar_calibration']
            self.assertEqual(cost['recipe_subset_target_status'],'pending')
            self.assertEqual(cost['recipe_subset']['count'],1)
            self.assertRegex(cost['recipe_subset']['manifest_sha256'],'^[0-9a-f]{64}$')

    def test_scalar_calibration_requires_hashed_locked_recipe_file_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            case=bench.canonical_cases()[9];path,_=scalar_calibration_fixture(Path(directory),[case])
            summary=json.loads(path.read_text());log=Path(directory)/(case['id']+'.log')
            records,_,_,_=bench.parse_output(log.read_text())
            inventory=recipe_inventory_fixture();inventory.pop('files')
            log.write_text(raw_output(records,recipe_inventory=inventory))
            summary['results'][0]['log_sha256']=bench.sha256_file(log);path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'per-file byte/hash evidence'):
                bench.load_scalar_calibration(args(mtp_strategy='adaptive',scalar_calibration=path),[case])

    def test_worker_clears_inherited_mtp_controller_variables_before_explicit_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lock=root/'gpu.lock'
            opts=args(workdir=directory,context_binary=sys.executable,decode_binary=sys.executable,tuning_dir=str(root/'recipes'))
            child='import os,json;print(json.dumps({k:os.environ.get(k) for k in ["APXINF_QWEN35_MTP_FIXED_DEPTH","APXINF_QWEN35_SCALAR_TPOT_MS"]}))'
            for index,env in enumerate(({}, {bench.MTP_FIXED_ENV:'2'}, {bench.MTP_SCALAR_TPOT_ENV:'18.75'})):
                log=root/(str(index)+'.log')
                with patch.dict(os.environ,{bench.MTP_FIXED_ENV:'1',bench.MTP_SCALAR_TPOT_ENV:'19.42'}),patch.object(bench,'GPU_LOCK_PATH',str(lock)):
                    outcome=bench.Transport(opts).execute([sys.executable,'-c',child],env,log)
                self.assertEqual(outcome['transport_returncode'],0)
                raw=log.read_text();_,process,_,_=bench.parse_output(raw)
                expected={key:env.get(key) for key in (bench.MTP_FIXED_ENV,bench.MTP_SCALAR_TPOT_ENV)}
                self.assertEqual(process['effective_mtp_environment'],expected)
                self.assertIn(json.dumps(expected),raw)

    def test_adaptive_rejects_missing_or_wrong_effective_controller_receipt(self):
        case=bench.canonical_cases()[9];opts=args(mtp_strategy='adaptive')
        recipes=recipe_inventory_fixture();expected=bench.environment(opts,18.75)
        for env in ({},{bench.MTP_FIXED_ENV:'3',bench.MTP_SCALAR_TPOT_ENV:'18.75'},{bench.MTP_SCALAR_TPOT_ENV:'19.42'}):
            with self.subTest(env=env):
                result=bench.validate_run(raw_output(fixture(case,opts),effective_env=env),case,opts,recipes,recipes,'a'*64,
                                          dict(transport_returncode=0,transport_timeout=False),expected)
                self.assertFalse(result['valid']);self.assertFalse(result['performance_accepted'])
                self.assertTrue(any('environment mismatch' in failure for failure in result['failures']))

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
            recipes=recipe_inventory_fixture()
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
        self.assertAlmostEqual(result['decode_tpot_ms'],1000/102,delta=1e-6/1023) # <= one nanosecond of total interval quantization
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
        recipes=recipe_inventory_fixture()
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
            recipes=recipe_inventory_fixture()
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

    def test_profile_worker_creates_target_parent_before_child_and_exports_fresh_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/"nested profile 'quoted'; $literal [x]";base=parent/'cold-i2k'
            opts=args(workdir=directory,context_binary=sys.executable,decode_binary=sys.executable,
                      tuning_dir=str(root/'recipes'),profile_dir=str(parent),nsys=self.fake_nsys(root))
            child='import pathlib,sys; p=pathlib.Path(sys.argv[1].split("=",1)[1]); assert p.parent.is_dir(); pathlib.Path(str(p)+".nsys-rep").write_bytes(b"CPU fixture, not GPU capture")'
            with patch.object(bench,'GPU_LOCK_PATH',str(root/'gpu.lock')):
                outcome=bench.Transport(opts).execute([sys.executable,'-c',child,'--output='+str(base)],{},root/'launch.log',profile=True)
            self.assertEqual(outcome['transport_returncode'],0)
            records,process,exit_receipt,lock=bench.parse_output((root/'launch.log').read_text())
            self.assertTrue(exit_receipt['child_reaped']);self.assertTrue(lock['acquired'])
            self.assertEqual(process['profile_output']['base'],str(base));self.assertFalse(process['profile_output']['parent_preexisted'])
            artifact=bench.Transport(opts).profile_artifacts(str(base),root/'case')
            self.assertTrue(artifact['valid'],artifact['failures'])
            self.assertEqual(artifact['report']['sha256'],bench.sha256_file(Path(str(base)+'.nsys-rep')))
            self.assertEqual(artifact['stats']['argv'][-1],str(base)+'.nsys-rep')

    def test_profile_worker_rejects_any_stale_prefix_before_child_without_overwrite(self):
        for suffix,kind in [('.nsys-rep','file'),('.sqlite','file'),('.qdstrm','file'),('','directory'),('.nsys-rep','dangling_symlink')]:
            with self.subTest(suffix=suffix,kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);parent=root/'traces';parent.mkdir();base=parent/'cold-i2k';stale=Path(str(base)+suffix)
                if kind=='file':stale.write_bytes(b'old evidence')
                elif kind=='directory':stale.mkdir()
                else:stale.symlink_to(root/'nonexistent')
                flag=root/'child-started';opts=args(workdir=directory,context_binary=sys.executable,decode_binary=sys.executable,tuning_dir=str(root/'recipes'),profile_dir=str(parent))
                child='from pathlib import Path; Path('+repr(str(flag))+').touch()'
                with patch.object(bench,'GPU_LOCK_PATH',str(root/'gpu.lock')):
                    result=bench.Transport(opts).execute([sys.executable,'-c',child,'--output='+str(base)],{},root/'blocked.log',profile=True)
                self.assertEqual(result['transport_returncode'],73);self.assertFalse(flag.exists())
                _,process,exit_receipt,_=bench.parse_output((root/'blocked.log').read_text())
                self.assertEqual(process['kind'],'not_started');self.assertIn('prefix already exists',exit_receipt['profile_prelaunch_error'])
                if kind=='file':self.assertEqual(stale.read_bytes(),b'old evidence')
                if kind=='dangling_symlink':self.assertTrue(stale.is_symlink())

    def test_profile_mkdir_failure_has_not_started_receipt_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'file-not-directory';parent.write_text('preserve');flag=root/'child-started'
            opts=args(workdir=directory,context_binary=sys.executable,decode_binary=sys.executable,tuning_dir=str(root/'recipes'),profile_dir=str(parent))
            with patch.object(bench,'GPU_LOCK_PATH',str(root/'gpu.lock')):
                result=bench.Transport(opts).execute([sys.executable,'-c','from pathlib import Path; Path('+repr(str(flag))+').touch()','--output='+str(parent/'cold-i2k')],{},root/'failure.log',profile=True)
            self.assertEqual(result['transport_returncode'],73);self.assertFalse(flag.exists());self.assertEqual(parent.read_text(),'preserve')
            _,process,receipt,_=bench.parse_output((root/'failure.log').read_text());self.assertEqual(process['kind'],'not_started');self.assertTrue(receipt['child_reaped'])

    def test_formal_and_tune_never_create_configured_profile_parent(self):
        for mode in ('formal','tune'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);parent=root/'profile-must-not-exist'
                opts=args(run_mode=mode,workdir=directory,context_binary=sys.executable,decode_binary=sys.executable,tuning_dir=str(root/'recipes'),profile_dir=str(parent))
                with patch.object(bench,'GPU_LOCK_PATH',str(root/'gpu.lock')):
                    result=bench.Transport(opts).execute([sys.executable,'-c','pass'],{},root/'run.log')
                self.assertEqual(result['transport_returncode'],0);self.assertFalse(parent.exists())

    def test_profile_transport_rejects_missing_duplicate_relative_or_wrong_parent_output(self):
        for outputs in [[],['--output=relative/cold-i2k'],['--output=/elsewhere/cold-i2k'],['--output=/trace/one','--output=/trace/two']]:
            with self.subTest(outputs=outputs),tempfile.TemporaryDirectory() as directory:
                with patch.object(bench.subprocess,'run') as run:
                    with self.assertRaisesRegex(ValueError,'profile'):
                        bench.Transport(args(profile_dir='/trace')).execute([sys.executable,*outputs],{},Path(directory)/'log',profile=True)
                run.assert_not_called();self.assertFalse((Path(directory)/'log').exists())

    def test_profile_ssh_payload_quotes_remote_path_and_does_not_mkdir_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/"remote 'quoted'; $(not-a-command) [x]";base=parent/'cold-i2k'
            opts=args(ssh_host='thor-u',profile_dir=str(parent))
            with patch.object(bench.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
                bench.Transport(opts).execute(['nsys','profile','--output='+str(base),'/bin/context'],{},root/'transport.log',profile=True)
            argv=run.call_args.args[0];self.assertEqual(argv[:2],['ssh','thor-u'])
            remote=bench.shlex.split(argv[2]);self.assertEqual(remote[:3],['python3','-c',bench.LAUNCH_CODE])
            payload=json.loads(remote[3]);self.assertEqual(payload['profile_output'],str(base));self.assertEqual(payload['argv'][2],'--output='+str(base));self.assertFalse(parent.exists())

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
