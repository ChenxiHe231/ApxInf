"""CPU-only launcher contracts. No real sudo/pool/cargo/GPU operations."""
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import build_qwen35_thor_benchmarks as build


def memory(pages=32768, used=0, reserved=0, available=42 * 1024 ** 3):
    return dict(MemAvailable=available, HugePages_Total=pages, HugePages_Free=pages-used,
                HugePages_Rsvd=reserved, HugePages_Surp=0, Hugepagesize=2*1024**2,
                nr_hugepages=pages, used_pages=used, committed_or_reserved_pages=used+reserved)


class Contracts(unittest.TestCase):
    def test_plan_is_pure_and_fixed(self):
        args=build.parser().parse_args(["--plan"])
        with mock.patch.object(build,"write_pool",side_effect=AssertionError("sudo")), \
             mock.patch.object(build,"OwnedBuild",side_effect=AssertionError("cargo")), \
             mock.patch.object(build,"acquire_lock",side_effect=AssertionError("lock")), \
             mock.patch.object(build,"save_receipt",side_effect=AssertionError("write")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(build.execute(args),0)
        plan=json.loads(output.getvalue())
        self.assertEqual(plan['lock_path'],'/tmp/apxinf-gpu.lock')
        self.assertEqual(plan['environment'],dict(CARGO_BUILD_JOBS='2',APXINF_NVCC_PARALLELISM='2'))
        self.assertEqual(plan['argv'].count('--example'),2)
        self.assertIn('qwen35-cuda',plan['argv']);self.assertFalse(plan['execute'])

    def test_cli_has_no_lock_or_proc_root_bypass_and_never_zero(self):
        for argv in [['--lock-path','/tmp/other'],['--proc-root','/fake'],['--inference-pages','0'],['--timeout-sec','nan']]:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): build.main(argv)

    def test_platform_root_and_non_thor_rejected(self):
        for system,machine,uid,model in [('Darwin','arm64',501,'Thor'),('Linux','x86_64',1000,'Thor'),('Linux','aarch64',0,'Thor'),('Linux','aarch64',1000,'Jetson Orin')]:
            with mock.patch.object(build.platform,'system',return_value=system), \
                 mock.patch.object(build.platform,'machine',return_value=machine), \
                 mock.patch.object(build.os,'geteuid',return_value=uid), \
                 mock.patch.object(Path,'read_text',return_value=model),self.assertRaises(build.GuardError):
                build.validate_platform()

    def test_reserved_is_added_once_and_small_readback_is_strict(self):
        build.check_memory(memory(32768,1000,1000))
        with self.assertRaises(build.GuardError):build.check_memory(memory(32768,1000,1049))
        with self.assertRaises(build.GuardError):build.check_memory(memory(32768,2049))
        with self.assertRaises(build.GuardError):build.check_memory(memory(4096),small=True)
        bad=memory(2048);bad['HugePages_Free']=3000
        with self.assertRaises(build.GuardError):build.check_memory(bad,small=True)

    def test_unknown_start_live_group_fails_closed_without_signal_but_absent_is_empty(self):
        owned=build.OwnedBuild.__new__(build.OwnedBuild)
        owned.proc=SimpleNamespace(pid=99999999,returncode=0,poll=lambda:0,wait=lambda timeout:0)
        owned.pgid=99999999;owned.start_ticks=None;owned.scan=build.scan_group
        with mock.patch.object(owned,'reap_descendants',return_value=[]),mock.patch.object(build.os,'killpg',return_value=None) as kill:
            receipt=owned.cleanup(grace=.01)
        self.assertFalse(receipt['proven']);self.assertEqual(kill.call_args_list,[mock.call(99999999,0)])
        with mock.patch.object(build.os,'killpg',side_effect=ProcessLookupError):
            self.assertEqual(build.scan_group(99999999,None),[])

    def workflow(self, *, available=42*1024**3, used=0, reserved=0, build_code=0, pool_auth=True, inference=False, inference_auth=True, inference_pages=4096):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name)
        (root/'crates/apxinf-model').mkdir(parents=True);(root/'crates/apxinf-model/Cargo.toml').write_text('[package]')
        argv=['--execute','--workdir',str(root),'--target-dir',str(root/'target'),'--receipt',str(root/'receipt.json')]
        if inference:argv+=['--inference-pages',str(inference_pages)]
        args=build.parser().parse_args(argv);state=dict(pages=32768,ended=False);events=[]
        def snapshot():return memory(state['pages'],used,reserved,available)
        def pool(pages):
            events.append(('pool',pages))
            if not pool_auth or (pages>2048 and not inference_auth):return dict(ok=False,returncode=1,stderr='sudo: a password is required')
            if pages>2048:self.assertTrue(state['ended'],'expanded pool before owned cleanup')
            state['pages']=pages;return dict(ok=True,actual=pages,returncode=0)
        class FakeOwned:
            def __init__(self,*a,**kw):events.append(('spawn',));self.proc=SimpleNamespace(pid=12345,returncode=build_code,wait=lambda timeout:build_code);self.pgid=12345;self.start_ticks=1
            def members(self):return []
            def cleanup(self):state['ended']=True;events.append(('cleanup',));return dict(proven=True,direct_child_reaped=True,members_after=[])
        patches=[mock.patch.object(build,'GPU_LOCK_PATH',root/'private-test.lock'),
                 mock.patch.object(build,'validate_platform',return_value=dict(model='CPU mock Thor')),
                 mock.patch.object(build,'enable_subreaper'),mock.patch.object(build.signal,'signal'),
                 mock.patch.object(build,'source_snapshot',return_value=dict(commit='a'*40,patch_sha256='b'*64)),
                 mock.patch.object(build,'memory_snapshot',side_effect=snapshot),
                 mock.patch.object(build,'write_pool',side_effect=pool),
                 mock.patch.object(build.shutil,'which',return_value='/mock/cargo'),
                 mock.patch.object(build,'digest',return_value='d'*64),
                 mock.patch.object(build,'OwnedBuild',FakeOwned),
                 mock.patch.object(build,'artifact_receipts',return_value={x:dict(sha256='c'*64) for x in build.EXAMPLES})]
        with contextlib.ExitStack() as stack:
            for patch in patches:stack.enter_context(patch)
            code=build.execute(args)
        return code,json.loads((root/'receipt.json').read_text()),events

    def test_success_default_small_pool_and_opt_in_only_after_cleanup(self):
        code,receipt,events=self.workflow()
        self.assertEqual(code,0);self.assertEqual(receipt['memory_final']['nr_hugepages'],2048)
        self.assertEqual([e for e in events if e[0]=='pool'],[('pool',2048)])
        code,receipt,events=self.workflow(inference=True)
        self.assertEqual(code,0);self.assertLess(events.index(('cleanup',)),events.index(('pool',4096)))
        self.assertTrue(receipt['explicit_unlock']);self.assertEqual(len(receipt['artifacts']),2)

    def test_no_spawn_on_used_reserved_auth_or_memory_failure(self):
        for options in [dict(used=2049),dict(used=1000,reserved=1049),dict(pool_auth=False),dict(available=24*1024**3-1)]:
            code,receipt,events=self.workflow(**options)
            self.assertEqual(code,1);self.assertNotIn(('spawn',),events)
            self.assertEqual(receipt['status'],'failed')

    def test_failed_build_never_expands_pool(self):
        code,receipt,events=self.workflow(build_code=101,inference=True)
        self.assertEqual(code,1);self.assertEqual(receipt['build_exit_code'],101)
        self.assertNotIn(('pool',4096),events);self.assertTrue(receipt['cleanup']['proven'])

    def test_expired_optional_authorization_fails_with_small_pool(self):
        code,receipt,events=self.workflow(inference=True,inference_auth=False)
        self.assertEqual(code,1);self.assertEqual(receipt['memory_final']['nr_hugepages'],2048)
        self.assertTrue(receipt['cleanup']['proven']);self.assertFalse(receipt['inference_pool']['ok'])

    def test_oversized_inference_request_is_rejected_before_pool_write(self):
        code,receipt,events=self.workflow(inference=True,inference_pages=100000)
        self.assertEqual(code,1);self.assertEqual([e for e in events if e[0]=='pool'],[('pool',2048)])
        self.assertEqual(receipt['memory_final']['nr_hugepages'],2048)

    def test_noninteractive_sudo_failure_and_fixed_privileged_argv(self):
        result=subprocess.CompletedProcess([],1,'','sudo: a password is required')
        with mock.patch.object(build.subprocess,'run',return_value=result) as run:
            receipt=build.write_pool(2048)
        self.assertFalse(receipt['ok']);self.assertEqual(run.call_args.args[0][:5],['sudo','-n','--','/usr/bin/python3','-I'])

    def test_pool_writer_partial_growth_rolls_back_in_same_authorization(self):
        state=dict(actual=2048,writes=[])
        def write(value):state['writes'].append(value);state['actual']=3072 if value=='4096\n' else int(value)
        fake=SimpleNamespace(write_text=write,read_text=lambda:str(state['actual']))
        def paths(path):return SimpleNamespace(read_text=lambda:'MemAvailable: 44040192 kB\n') if path=='/proc/meminfo' else fake
        with mock.patch('pathlib.Path',side_effect=paths),mock.patch.object(sys,'argv',['writer','4096']), \
             contextlib.redirect_stdout(io.StringIO()) as output,self.assertRaises(SystemExit) as exit_:
            exec(build.POOL_CODE,{})
        self.assertEqual(exit_.exception.code,1)
        self.assertEqual(state['writes'],['4096\n','2048\n']);self.assertTrue(json.loads(output.getvalue())['small_pool_proven'])

    def test_pool_writer_low_post_growth_memory_rolls_back(self):
        state=dict(actual=2048,writes=[])
        def write(value):state['writes'].append(value);state['actual']=int(value)
        fake=SimpleNamespace(write_text=write,read_text=lambda:str(state['actual']))
        def paths(path):
            if path=='/proc/meminfo':return SimpleNamespace(read_text=lambda:'MemAvailable: '+str((40 if state['actual']==2048 else 6)*1024**2)+' kB\n')
            return fake
        with mock.patch('pathlib.Path',side_effect=paths),mock.patch.object(sys,'argv',['writer','4096']), \
             contextlib.redirect_stdout(io.StringIO()) as output,self.assertRaises(SystemExit):exec(build.POOL_CODE,{})
        self.assertEqual(state['writes'],['4096\n','2048\n']);self.assertTrue(json.loads(output.getvalue())['small_pool_proven'])

    def test_pool_writer_oversized_request_does_not_write(self):
        writes=[];fake=SimpleNamespace(write_text=writes.append,read_text=lambda:'2048')
        def paths(path):return SimpleNamespace(read_text=lambda:'MemAvailable: 44040192 kB\n') if path=='/proc/meminfo' else fake
        with mock.patch('pathlib.Path',side_effect=paths),mock.patch.object(sys,'argv',['writer','100000']), \
             contextlib.redirect_stdout(io.StringIO()) as output,self.assertRaises(SystemExit):exec(build.POOL_CODE,{})
        self.assertEqual(writes,[]);self.assertTrue(json.loads(output.getvalue())['small_pool_proven'])

    def test_elf_receipt_uses_actual_cargo_json_path_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d).resolve();rows=[]
            for name in build.EXAMPLES:
                path=target/name;path.write_bytes(b'\x7fELF\x02\x01'+b'\0'*12+(183).to_bytes(2,'little')+b'CPU fixture')
                rows.append(dict(reason='compiler-artifact',target=dict(name=name),executable=str(path)))
            log=target/'build.log';log.write_text('\n'.join(json.dumps(x) for x in rows))
            receipt=build.artifact_receipts(log,target)
            self.assertEqual(receipt[build.EXAMPLES[0]]['sha256'],build.digest(target/build.EXAMPLES[0]))
            (target/build.EXAMPLES[0]).write_bytes(b'not ELF')
            with self.assertRaises(build.GuardError):build.artifact_receipts(log,target)

    def test_quarantine_records_failure_and_retains_lock_without_restore(self):
        with tempfile.TemporaryDirectory() as d,open(Path(d)/'lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);receipt={}
            with mock.patch.object(build.signal,'signal'),mock.patch.object(build.time,'sleep',side_effect=KeyboardInterrupt), \
                 contextlib.redirect_stderr(io.StringIO()),self.assertRaises(KeyboardInterrupt):
                build.quarantine(lock,receipt,Path(d)/'receipt.json')
            self.assertFalse(receipt['explicit_unlock'])
            with open(Path(d)/'lock','a+') as other,self.assertRaises(BlockingIOError):
                fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)


@unittest.skipUnless(sys.platform.startswith('linux'),'real /proc child/grandchild tests run on main Linux CPU host')
class RealCPUProcessGroups(unittest.TestCase):
    def setUp(self):
        build.enable_subreaper();self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.lock=open(self.root/'lock','a+');fcntl.flock(self.lock,fcntl.LOCK_EX)
        self.addCleanup(lambda:self.lock.close());self.log=open(self.root/'log','w');self.addCleanup(self.log.close)

    def start(self,code,*args):
        return build.OwnedBuild([sys.executable,'-c',code,*map(str,args)],self.root,os.environ.copy(),self.log,self.lock.fileno())

    def ready(self,path):
        until=time.monotonic()+5
        while not path.exists() and time.monotonic()<until:time.sleep(.01)
        self.assertTrue(path.exists(),'CPU child failed to start')

    def test_direct_child_inherits_lock_and_is_reaped(self):
        ready=self.root/'ready'
        with build.defer_stop_signals():
            child=self.start('import os,pathlib,sys,time;os.fstat(int(sys.argv[1]));pathlib.Path(sys.argv[2]).write_text("ready");time.sleep(30)',self.lock.fileno(),ready)
        try:
            self.ready(ready);self.lock.close()
            with open(self.root/'lock','a+') as other,self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            receipt=child.cleanup(grace=.5);self.assertTrue(receipt['proven'],receipt)
            self.assertEqual(child.proc.returncode,-signal.SIGTERM,'child inherited a blocked stop signal')
            with open(self.root/'lock','a+') as other:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:
            child.cleanup(grace=.5)
            if child.proc.poll() is None:child.proc.kill();child.proc.wait()

    def test_orphan_grandchild_is_reaped_before_unlock_and_other_job_survives(self):
        ready=self.root/'grandchild'
        code='import os,pathlib,sys,time;os.fstat(int(sys.argv[1]));pathlib.Path(sys.argv[2]).write_text(str(os.getpid()));time.sleep(30)'
        leader='import pathlib,subprocess,sys,time;p=subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2],sys.argv[3]],pass_fds=(int(sys.argv[2]),));\nwhile not pathlib.Path(sys.argv[3]).exists():time.sleep(.01)'
        child=self.start(leader,code,self.lock.fileno(),ready)
        other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        try:
            self.ready(ready);grandchild=int(ready.read_text());self.assertEqual(child.proc.wait(timeout=5),0)
            self.assertTrue(child.members(),'leader exit alone must not mean cleanup')
            receipt=child.cleanup(grace=1);self.assertTrue(receipt['proven'],receipt)
            self.assertIn(grandchild,[r['pid'] for r in receipt['descendants_reaped']])
            self.assertFalse(Path(f'/proc/{grandchild}').exists());self.assertIsNone(other.poll())
            with open(self.root/'lock','a+') as probe,self.assertRaises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:
            child.cleanup(grace=1)
            if child.proc.poll() is None:child.proc.kill();child.proc.wait()
            other.terminate();other.wait(timeout=5)

    def test_term_ignored_escalates_only_owned_group_and_reaps(self):
        ready=self.root/'ready'
        child=self.start('import pathlib,signal,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(sys.argv[1]).write_text("ready");time.sleep(30)',ready)
        try:
            self.ready(ready);receipt=child.cleanup(grace=.1)
            self.assertTrue(receipt['proven'],receipt);self.assertEqual(child.proc.returncode,-signal.SIGKILL)
        finally:
            child.cleanup(grace=.5)
            if child.proc.poll() is None:child.proc.kill();child.proc.wait()


if __name__=='__main__':unittest.main()
