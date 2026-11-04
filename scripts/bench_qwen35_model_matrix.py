#!/usr/bin/env python3
"""Run the 15 resident-weight Qwen35 microbenchmarks with auditable recipes.

This tool never builds CUDA, changes clocks/hugepages, or stops existing tasks.
Tune, formal, recipe-audit and Nsight processes are separate invocations. Formal
results require strict recipe-only native execution and unchanged recipe files.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

SCHEMA = "apxinf.model-matrix.run.v1"
SEED = 0x35A3BF0440960001
VOCAB = 248320
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
SOURCE_MARKER = "APXINF_MATRIX_PROCESS "
EXIT_MARKER = "APXINF_MATRIX_EXIT "
LOCK_MARKER = "APXINF_MATRIX_LOCK "
GPU_LOCK_PATH = "/tmp/apxinf-gpu.lock"
MTP_FIXED_ENV = "APXINF_QWEN35_MTP_FIXED_DEPTH"
MTP_SCALAR_TPOT_ENV = "APXINF_QWEN35_SCALAR_TPOT_MS"
COLLECTOR_SOURCE = Path(__file__).read_bytes()
COLLECTOR_SHA256 = hashlib.sha256(COLLECTOR_SOURCE).hexdigest()


def canonical_cases():
    rows = []
    for n in (2048, 4096, 8192, 16384):
        rows.append(dict(id=f"cold-i{n//1024}k", group="cold_prefill", initial_kv_tokens=0, new_input_tokens=n))
    for n in (4096, 8192, 16384, 32768, 49152):
        rows.append(dict(id=f"append-kv{n//1024}k-i4k", group="append_prefill", initial_kv_tokens=n, new_input_tokens=4096))
    for n in (4096, 8192, 16384, 24576, 32768, 49152):
        rows.append(dict(id=f"decode-kv{n//1024}k-o1k", group="decode", initial_kv_tokens=n, output_tokens=1024, decode_intervals=1023))
    return rows


def load_cases(path=None):
    canonical = canonical_cases()
    if path is None:
        return canonical
    data = json.loads(Path(path).read_text())
    rows = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != 15:
        raise ValueError("matrix must contain all 15 cases")
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
    if len(by_id) != 15:
        raise ValueError("matrix contains missing or duplicate IDs")
    for expected in canonical:
        actual = by_id.get(expected["id"], {})
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("matrix changed a canonical case: " + expected["id"])
    return [by_id[row["id"]] for row in canonical]


def input_identity(tokens):
    state = SEED
    digest = FNV_OFFSET
    sha = hashlib.sha256()
    for _ in range(tokens):
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
        value = (state >> 17) % (VOCAB - 1000)
        raw = value.to_bytes(4, "little")
        sha.update(raw)
        for byte in raw:
            digest = ((digest ^ byte) * FNV_PRIME) & MASK64
    return dict(seed_hex=f"{SEED:016x}", tokens=tokens, fnv1a64=f"{digest:016x}", sha256=sha.hexdigest())


def selected_cases(rows, selection):
    if not selection:
        return rows
    wanted = selection.split(",")
    if len(set(wanted)) != len(wanted) or set(wanted) - {row["id"] for row in rows}:
        raise ValueError("unknown or duplicate case selection")
    return [row for row in rows if row["id"] in wanted]


def validate_selection_policy(args, cases):
    strategy = getattr(args, "mtp_strategy", "fixed")
    if any(case["group"] != "decode" for case in cases) and (strategy != "fixed" or args.mtp_depth != 3):
        raise ValueError("Prefill hook is fixed depth3; non-fixed3 policies require an explicit Decode-only subset")
    if strategy == "adaptive":
        if args.mode != "mtp" or args.mtp_depth != 3:
            raise ValueError("adaptive requires --mode mtp and maximum depth3; --mtp-depth only selects fixed depth")
        if not getattr(args, "scalar_calibration", None):
            raise ValueError("adaptive requires explicit --scalar-calibration; the runtime's default scalar TPOT is forbidden")
    elif getattr(args, "scalar_calibration", None):
        raise ValueError("--scalar-calibration is used only with --mtp-strategy adaptive")
    if args.mode == "scalar" and args.mtp_depth != 3:
        raise ValueError("scalar disables MTP; do not label it with a fixed draft depth")


def environment(args, scalar_tpot_ms=None):
    env = {}
    for item in args.env:
        if "=" not in item:
            raise ValueError("--env must be KEY=VALUE")
        key, value = item.split("=", 1)
        if not re.fullmatch(r"(?:APXINF|CUDA)_[A-Z0-9_]+", key):
            raise ValueError("only explicit APXINF_/CUDA_ environment controls are accepted")
        if key in env:
            raise ValueError("duplicate environment key")
        if key in (MTP_FIXED_ENV, MTP_SCALAR_TPOT_ENV):
            raise ValueError("MTP controller environment is controlled by --mtp-strategy/--mtp-depth/--scalar-calibration")
        env[key] = value
    # Presence alone enables native reporting; do not set these to '0'.
    env.pop("APXINF_GEMM_REPORT", None)
    env.pop("APXINF_ATTENTION_REPORT", None)
    env.pop("APXINF_REQUIRE_RECIPES", None)
    if args.run_mode != "tune":
        env["APXINF_REQUIRE_RECIPES"] = "1"
    if args.run_mode in ("audit", "profile"):
        env["APXINF_GEMM_REPORT"] = "1"
        env["APXINF_ATTENTION_REPORT"] = "1"
    if args.mode == "mtp":
        if getattr(args, "mtp_strategy", "fixed") == "fixed":
            env[MTP_FIXED_ENV] = str(args.mtp_depth)
        else:
            env[MTP_SCALAR_TPOT_ENV] = format(finite_number(scalar_tpot_ms, "calibrated scalar TPOT"), ".17g")
    return env


def mtp_policy(args, scalar_tpot_ms=None):
    enabled = args.mode == "mtp"
    strategy = getattr(args, "mtp_strategy", "fixed") if enabled else "disabled"
    return dict(strategy=strategy, fixed_depth=args.mtp_depth if strategy == "fixed" else None,
                maximum_draft_depth=3 if enabled else None, scalar_tpot_ms=scalar_tpot_ms if strategy == "adaptive" else None)


def command_for_case(args, case):
    warmups = args.warmups if args.run_mode != "tune" else 0
    repeats = args.repeats if args.run_mode == "formal" else 1
    is_decode = case["group"] == "decode"
    binary = args.decode_binary if is_decode else args.context_binary
    argv = [binary, args.checkpoint, "--mode", args.mode, "--qwen35-kv-cache", args.kv_cache,
            "--qwen35-moe-activation", args.moe_activation, "--tuning-dir", args.tuning_dir,
            "--warmups", str(warmups), "--repeats", str(repeats)]
    if is_decode:
        argv += ["--prompt-tokens", str(case["initial_kv_tokens"]), "--max-new", "1024"]
    else:
        argv += ["--prefix-tokens", str(case["initial_kv_tokens"]), "--input-tokens", str(case["new_input_tokens"]), "--max-new", "1"]
    if args.run_mode == "tune":
        argv.append("--autotune")
    if args.run_mode == "profile":
        argv.append("--profile-measured")
    if getattr(args, "allow_repeat_divergence", False):
        argv.append("--allow-repeat-divergence")
    return argv


def comparison_policy(args):
    return "diagnostic-repeat-divergence-v1" if getattr(args, "allow_repeat_divergence", False) else "strict-repeat-consistency-v1"


SNAPSHOT_CODE = r'''
import hashlib,json,os,pathlib,platform,subprocess,sys
kind,workdir,checkpoint,context,decode,tuning=sys.argv[1:]
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def recipes():
    p=pathlib.Path(tuning)
    rows=[]
    if p.exists():
        for x in sorted(p.rglob('*')):
            if x.is_file():rows.append({'path':str(x.relative_to(p)),'bytes':x.stat().st_size,'sha256':digest(x)})
    return {'path':str(p),'files':rows,'count':sum(x['path'].endswith('.recipe') for x in rows),'manifest_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
if kind=='recipes':print(json.dumps(recipes()));sys.exit(0)
os.chdir(workdir)
def git(*args):return subprocess.check_output(['git',*args])
untracked=[x.decode() for x in git('ls-files','--others','--exclude-standard','-z').split(b'\0') if x]
patch=git('diff','HEAD','--binary')
status=git('status','--porcelain','--untracked-files=all').decode()
source={'source_commit':git('rev-parse','HEAD').decode().strip(),'source_dirty':bool(status),'source_status':status,'source_patch_sha256':hashlib.sha256(patch).hexdigest(),'source_untracked_sha256':{x:digest(x) for x in untracked if pathlib.Path(x).is_file()}}
result={'source':source,'binary_sha256':{'context':digest(context),'decode':digest(decode)},'recipes':recipes()}
if kind=='full':
    p=pathlib.Path(checkpoint)
    model=[x for x in sorted(p.iterdir()) if x.is_file() and (x.suffix in ('.safetensors','.json','.jinja'))]
    result['model_sha256']={x.name:digest(x) for x in model}
    clocks=[]
    for gpu in pathlib.Path('/sys/class/devfreq').glob('*gpu*'):
        item={'path':str(gpu)}
        for name in ('min_freq','max_freq','cur_freq','governor'):
            try:item[name]=(gpu/name).read_text().strip()
            except OSError:pass
        clocks.append(item)
    mem={}
    try:
        for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(('HugePages_','Hugepagesize:','Cma')):
                k,v=line.split(':',1);mem[k]=v.strip()
    except OSError:pass
    result['hardware_observed']={'hostname':platform.node(),'platform':platform.platform(),'gpu_clocks':clocks,'memory':mem}
print(json.dumps(result))
'''


LAUNCH_CODE = r'''
import fcntl,hashlib,json,os,pathlib,signal,subprocess,sys,time
p=json.loads(sys.argv[1]);env=dict(os.environ)
for key in ('APXINF_GEMM_REPORT','APXINF_ATTENTION_REPORT','APXINF_REQUIRE_RECIPES','APXINF_QWEN35_MTP_FIXED_DEPTH','APXINF_QWEN35_SCALAR_TPOT_MS'):env.pop(key,None)
env.update(p['environment'])
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def inventory():
    root=pathlib.Path(p['tuning_dir']);rows=[]
    if root.exists():
        for x in sorted(root.rglob('*')):
            if x.is_file():rows.append({'path':str(x.relative_to(root)),'bytes':x.stat().st_size,'sha256':digest(x)})
    return {'path':str(root),'files':rows,'count':sum(x['path'].endswith('.recipe') for x in rows),'manifest_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
def hardware():
    mem={};clocks=[]
    try:
        for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(('HugePages_','Hugepagesize:','Cma')):
                k,v=line.split(':',1);mem[k]=v.strip()
    except OSError:pass
    for gpu in pathlib.Path('/sys/class/devfreq').glob('*gpu*'):
        item={'path':str(gpu)}
        for key in ('governor','min_freq','max_freq','cur_freq'):
            try:item[key]=(gpu/key).read_text().strip()
            except OSError:pass
        clocks.append(item)
    return {'memory':mem,'gpu_clocks':clocks}
def holder(fd):
    try:
        st=os.fstat(fd)
        for line in pathlib.Path('/proc/locks').read_text().splitlines():
            columns=line.split()
            if 'FLOCK' not in columns:continue
            index=columns.index('FLOCK');devino=columns[index+4].split(':')
            if len(devino)==3 and (int(devino[0],16),int(devino[1],16),int(devino[2]))==(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino):
                return {'pid':int(columns[index+3]),'record':line}
    except (OSError,ValueError,IndexError):pass
    return None
lock=open(p['lock_path'],'a+');acquired=False;proc=None
started=time.monotonic();prior_holder=None;last_notice=-30.0
def terminate_owned_child():
    if proc is None or proc.poll() is not None:return True
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:return False
    return proc.poll() is not None
try:
    while True:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True;break
        except BlockingIOError:
            elapsed=time.monotonic()-started
            prior_holder=holder(lock.fileno()) or prior_holder
            if elapsed>=p['lock_timeout']:
                print('APXINF_MATRIX_LOCK '+json.dumps({'path':p['lock_path'],'acquired':False,'wait_ms':elapsed*1000,'prior_holder':prior_holder,'owner_pid':os.getpid()}),flush=True)
                print('APXINF_MATRIX_PROCESS '+json.dumps({'kind':'not_started','pid':None}),flush=True)
                print('APXINF_MATRIX_EXIT '+json.dumps({'returncode':75,'timed_out':False,'lock_timed_out':True,'child_reaped':True}),flush=True)
                sys.exit(75)
            if elapsed-last_notice>=30:
                print('APXINF_MATRIX_LOCK_WAIT '+json.dumps({'path':p['lock_path'],'wait_ms':elapsed*1000,'holder':prior_holder}),flush=True);last_notice=elapsed
            time.sleep(min(0.1,max(0.001,p['lock_timeout']-elapsed)))
    print('APXINF_MATRIX_LOCK '+json.dumps({'path':p['lock_path'],'acquired':True,'wait_ms':(time.monotonic()-started)*1000,'prior_holder':prior_holder,'owner_pid':os.getpid(),'recipes_before':inventory(),'hardware_before':hardware(),'target_binary_sha256':digest(p['benchmark_binary'])}),flush=True)
    # The direct benchmark inherits this same open-file-description. It keeps
    # the lock after abrupt worker loss. A profiler may close inherited FDs in
    # descendants: only the live worker/launcher lifetime is guaranteed there.
    proc=subprocess.Popen(p['argv'],cwd=p['cwd'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
    def interrupted(signum,frame):
        terminate_owned_child()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    identity={'pid':proc.pid,'kind':'profiler_launcher' if p['profile'] else 'benchmark','direct_child_inherits_gpu_lock':not p['profile'],
              'effective_mtp_environment':{key:env.get(key) for key in ('APXINF_QWEN35_MTP_FIXED_DEPTH','APXINF_QWEN35_SCALAR_TPOT_MS')}}
    try:
        path='/proc/%d/exe'%proc.pid
        identity['exe']=os.readlink(path);identity['sha256']=digest(path)
    except OSError as error:identity['error']=str(error)
    print('APXINF_MATRIX_PROCESS '+json.dumps(identity),flush=True)
    timed_out=False
    try:raw,_=proc.communicate(timeout=p['timeout'])
    except subprocess.TimeoutExpired:
        timed_out=True
        terminate_owned_child()
        raw,_=proc.communicate()
    sys.stdout.buffer.write(raw);sys.stdout.flush()
    print('\nAPXINF_MATRIX_EXIT '+json.dumps({'returncode':proc.returncode,'timed_out':timed_out,'child_reaped':proc.poll() is not None,'lock_held_until_reaped':True,'backend_task_may_be_live':bool(p['profile'] and timed_out),'recipes_after':inventory(),'hardware_after':hardware(),'target_binary_sha256':digest(p['benchmark_binary'])}),flush=True)
    sys.exit(124 if timed_out else proc.returncode)
finally:
    reaped=terminate_owned_child()
    # LOCK_UN affects the shared open-file-description, including a live child.
    # If reaping is unproven, close only this descriptor; do not release its lock.
    if acquired and reaped:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    lock.close()

'''


PROFILE_ARTIFACT_CODE = r'''
import hashlib,json,pathlib,stat,subprocess,sys
p=json.loads(sys.argv[1])
report=pathlib.Path(p['report'])
result={'report':{'path':str(report)},'failures':[]}
def identity():
    info=report.lstat()
    if not stat.S_ISREG(info.st_mode):raise ValueError('nsys report is not a regular file')
    if not info.st_size:raise ValueError('nsys report is empty')
    h=hashlib.sha256()
    with report.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return {'path':str(report),'size_bytes':info.st_size,'sha256':h.hexdigest()}
def run(argv):
    receipt={'argv':argv,'returncode':None,'timed_out':False}
    try:
        proc=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=p['timeout'])
        receipt.update(returncode=proc.returncode,stdout=proc.stdout.decode('utf-8','replace'),stderr=proc.stderr.decode('utf-8','replace'))
    except subprocess.TimeoutExpired as error:
        receipt.update(timed_out=True,stdout=(error.stdout or b'').decode('utf-8','replace'),stderr=(error.stderr or b'').decode('utf-8','replace'))
    except OSError as error:receipt['error']=str(error)
    return receipt
result['nsys_version']=run([p['nsys'],'--version'])
try:result['report']=identity()
except (OSError,ValueError) as error:result['failures'].append(str(error))
if not result['failures']:
    result['stats']=run([p['nsys'],'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum','--format','csv','--output','-','--force-export=true',str(report)])
    try:
        after=identity()
        result['report_after_export']=after
        if after!=result['report']:result['failures'].append('nsys report changed during CPU stats export')
    except (OSError,ValueError) as error:result['failures'].append(str(error))
print(json.dumps(result))
'''


def validate_profile_artifacts(artifacts):
    failures = list(artifacts.get('failures', []))
    report = artifacts.get('report', {})
    if type(report.get('size_bytes')) is not int or report['size_bytes'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}', report.get('sha256', '')):
        failures.append('missing nonempty hashed .nsys-rep artifact')
    version = artifacts.get('nsys_version', {})
    if version.get('returncode') != 0 or version.get('timed_out') or not version.get('stdout', '').strip():
        failures.append('nsys version command failed or returned no version')
    stats = artifacts.get('stats', {})
    if stats.get('returncode') != 0 or stats.get('timed_out'):
        failures.append('independent CPU nsys stats export failed')
    # nsys emits several CSV tables plus processing/status lines. Require at
    # least one real timing header followed by a numeric nonempty summary row.
    header = None
    timing_rows = 0
    for row in csv.reader(io.StringIO(stats.get('stdout', ''))):
        if 'Total Time (ns)' in row and 'Name' in row:
            header = row
        elif header and len(row) == len(header):
            try:
                elapsed = float(row[header.index('Total Time (ns)')].replace(',', ''))
                if math.isfinite(elapsed) and elapsed > 0 and row[header.index('Name')].strip():
                    timing_rows += 1
            except ValueError:
                pass
    if not timing_rows:
        failures.append('nsys stats CSV contains no valid timing summary rows')
    artifacts['timing_summary_rows'] = timing_rows
    return failures


class Transport:
    def __init__(self, args):
        self.args = args

    def python_argv(self, code, values):
        argv = ["python3", "-c", code, *map(str, values)]
        return ["ssh", self.args.ssh_host, shlex.join(argv)] if self.args.ssh_host else argv

    def snapshot(self, kind="full"):
        args = self.args
        values = [kind, args.workdir, args.checkpoint, args.context_binary, args.decode_binary, args.tuning_dir]
        result = subprocess.run(self.python_argv(SNAPSHOT_CODE, values), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return json.loads(result.stdout)

    def execute(self, argv, env, log, profile=False):
        target_binary = self.args.decode_binary if "--prompt-tokens" in argv else self.args.context_binary
        payload = json.dumps(dict(argv=argv, environment=env, cwd=self.args.workdir, timeout=self.args.timeout_sec, profile=profile,
                                  lock_path=GPU_LOCK_PATH, lock_timeout=self.args.lock_timeout_sec, tuning_dir=self.args.tuning_dir, benchmark_binary=target_binary))
        started = time.monotonic()
        with log.open("wb") as stream:
            try:
                process = subprocess.run(self.python_argv(LAUNCH_CODE, [payload]), stdout=stream, stderr=subprocess.STDOUT, timeout=self.args.lock_timeout_sec+self.args.timeout_sec+90)
                return dict(transport_returncode=process.returncode, wall_ms=(time.monotonic()-started)*1000, transport_timeout=False)
            except subprocess.TimeoutExpired:
                return dict(transport_returncode=124, wall_ms=(time.monotonic()-started)*1000, transport_timeout=True,
                            backend_task_may_be_live=True, error="transport timed out; remote task cancellation is unproven")

    def profile_artifacts(self, report_base, output_base):
        """Inspect/export an existing trace in a separate CPU-only process."""
        payload = json.dumps(dict(report=report_base+'.nsys-rep', nsys=self.args.nsys, timeout=self.args.timeout_sec))
        argv = self.python_argv(PROFILE_ARTIFACT_CODE, [payload])
        artifacts = dict(failures=[], collector_argv=argv, cpu_only=True)
        try:
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2*self.args.timeout_sec+30)
            artifacts.update(collector_returncode=proc.returncode, collector_stderr=proc.stderr.decode('utf-8','replace'))
            if proc.returncode:
                artifacts['failures'].append('profile artifact collector exited unsuccessfully')
            else:
                artifacts.update(json.loads(proc.stdout))
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            artifacts['failures'].append('profile artifact collection failed: '+str(error))
        stats = artifacts.get('stats', {})
        for key, suffix in [('stdout', '.stats.csv'), ('stderr', '.stats.stderr.log')]:
            path = Path(str(output_base)+suffix)
            path.write_bytes(stats.get(key, '').encode('utf-8'))
            artifacts['stats_csv' if key == 'stdout' else 'stats_stderr'] = dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha256_file(path))
        artifacts['failures'] = validate_profile_artifacts(artifacts)
        artifacts['valid'] = not artifacts['failures']
        return artifacts


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe_subset(inventory):
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("recipe inventory lacks per-file byte/hash evidence")
    entries = {}
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("invalid recipe inventory entry")
        path = row.get("path", "")
        if not isinstance(path, str):
            raise ValueError("invalid recipe inventory path")
        if not path.endswith(".recipe"):
            continue
        if Path(path).is_absolute() or any(part in ("", ".", "..") for part in path.split('/')) or path in entries or type(row.get("bytes")) is not int or row["bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            raise ValueError("invalid recipe inventory entry: " + path)
        entries[path] = dict(path=path, bytes=row["bytes"], sha256=row["sha256"])
    rows = [entries[key] for key in sorted(entries)]
    if len(rows) != inventory.get("count"):
        raise ValueError("recipe inventory count differs from per-file evidence")
    return dict(files=rows, count=len(rows), manifest_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def verify_recipe_subset(required, current_inventory):
    current = recipe_subset(current_inventory)
    by_path = {row["path"]:row for row in current["files"]}
    for row in required["files"]:
        if by_path.get(row["path"]) != row:
            raise ValueError("scalar calibration recipe changed or missing: " + row["path"])
    return dict(verified=True, required_subset_manifest_sha256=required["manifest_sha256"],
                matched_count=required["count"], current_recipe_count=current["count"],
                added_recipe_keys=sorted(set(by_path)-{row["path"] for row in required["files"]}))


def verify_identity(expected, actual):
    if not isinstance(expected, dict):
        raise ValueError("metadata must be an object")
    source = expected.get("source", expected)
    if type(source.get("source_dirty")) is not bool:
        raise ValueError("metadata source_dirty must be a boolean")
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("source_commit", "")):
        raise ValueError("metadata source_commit is invalid")
    required = ("source_commit", "source_dirty", "source_patch_sha256", "source_untracked_sha256")
    for key in required:
        if key not in source or source[key] != actual["source"].get(key):
            raise ValueError("source identity mismatch: " + key)
    for key in ("binary_sha256", "model_sha256"):
        if not expected.get(key) or expected[key] != actual.get(key):
            raise ValueError("identity mismatch: " + key)
    if not actual["model_sha256"] or not any(x.endswith(".safetensors") for x in actual["model_sha256"]):
        raise ValueError("checkpoint has no verified safetensors")


def parse_output(raw):
    records = []
    processes = []
    exits = []
    locks = []
    for line in raw.splitlines():
        if line.startswith(SOURCE_MARKER):
            processes.append(json.loads(line[len(SOURCE_MARKER):]))
        elif line.startswith(LOCK_MARKER):
            locks.append(json.loads(line[len(LOCK_MARKER):]))
        elif line.startswith(EXIT_MARKER):
            exits.append(json.loads(line[len(EXIT_MARKER):]))
        elif line.startswith("{"):
            data = json.loads(line)
            if isinstance(data, dict) and str(data.get("schema", "")).startswith("apxinf.qwen35."):
                records.append(data)
    if len(processes) != 1 or len(exits) != 1 or len(locks) != 1:
        raise ValueError("missing unique process identity/exit receipt")
    return records, processes[0], exits[0], locks[0]


def audit_recipe_sources(raw, required):
    dispatches = []
    for line in raw.splitlines():
        if line.startswith(("[apxinf-gemm]", "[apxinf-attn]")):
            match = re.search(r"\bsource=(\S+)", line)
            if not match:
                raise ValueError("dispatch report lacks recipe source")
            if match[1] not in ("recipe", "memory-recipe"):
                raise ValueError("non-recipe execution: " + match[1])
            dispatches.append(dict(kind="gemm" if line.startswith("[apxinf-gemm]") else "attention", source=match[1]))
    if required and {row["kind"] for row in dispatches} != {"gemm", "attention"}:
        raise ValueError("recipe audit did not observe both GEMM and attention")
    if re.search(r"recipe miss|recipe is missing|fallback-after-tune", raw, re.I):
        raise ValueError("benchmark reported missing recipe/fallback")
    return dict(dispatches=len(dispatches), recipe=sum(row["source"] == "recipe" for row in dispatches),
                memory_recipe=sum(row["source"] == "memory-recipe" for row in dispatches))


def finite_number(value, name, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError("invalid metric " + name)
    return float(value)


def validate_divergence_evidence(records, iterations, summary, output, args, decode):
    events = [row for row in records if row.get("schema") == "apxinf.qwen35.benchmark.repeat_divergence.v1"]
    by_iteration = {}
    baseline = iterations[0]["token_digest_fnv1a64"]
    for event in events:
        index = event.get("iteration")
        if type(index) is not int or not 0 < index < len(iterations) or index in by_iteration:
            raise ValueError("missing/duplicate/out-of-range divergence iteration")
        row = iterations[index]
        if event.get("mode") != args.mode or event.get("benchmark") != ("decode" if decode else "context") or event.get("warmup") is not row["warmup"] or event.get("allow_repeat_divergence") is not True:
            raise ValueError("structured divergence policy/mode mismatch")
        if event.get("expected_output_tokens") != output or event.get("actual_output_tokens") != output or event.get("expected_digest_fnv1a64") != baseline or event.get("actual_digest_fnv1a64") != row["token_digest_fnv1a64"]:
            raise ValueError("structured divergence output counts/digests mismatch")
        first = event.get("first_diff")
        expected = event.get("expected_first_diff_token")
        actual = event.get("actual_first_diff_token")
        if type(first) is not int or not 0 <= first < output or any(type(x) is not int or not 0 <= x < VOCAB for x in (expected, actual)) or expected == actual:
            raise ValueError("structured divergence lacks a valid first differing token")
        for key, token in (("expected_window", expected), ("actual_window", actual)):
            window = event.get(key, {})
            if not isinstance(window, dict):
                raise ValueError("structured divergence token window is not an object")
            start, tokens = window.get("start"), window.get("tokens")
            if type(start) is not int or not isinstance(tokens, list) or not 0 <= start <= first or not first-start < len(tokens) or start+len(tokens) > output or tokens[first-start] != token or any(type(x) is not int or not 0 <= x < VOCAB for x in tokens):
                raise ValueError("structured divergence token window mismatch")
        by_iteration[index] = event
    for index, row in enumerate(iterations):
        changed = row["token_digest_fnv1a64"] != baseline
        reported = row.get("repeat_diverged", False)
        if type(reported) is not bool or (changed and not reported) or reported != (index in by_iteration):
            raise ValueError("diverged iteration lacks matching flag/structured evidence")
    warm = sum(iterations[i]["warmup"] for i in by_iteration)
    if summary.get("divergent_warmups", 0) != warm or summary.get("divergent_repeats", 0) != len(events)-warm:
        raise ValueError("summary divergence counts differ from raw evidence")
    return events


def validate_decode_speculation(stats, args, aggregate=False):
    if args.mode == "scalar":
        if stats is not None:
            raise ValueError("scalar Decode unexpectedly reported MTP counters")
        return 0
    if not isinstance(stats, dict):
        raise ValueError("MTP Decode lacks actual speculative counters")
    if not aggregate and (type(stats.get("draft_steps")) is not int or not 1 <= stats["draft_steps"] <= 3):
        raise ValueError("native MTP maximum draft steps exceeds supported depth3")
    arrays = []
    for name in ("proposed_by_depth", "matched_by_depth", "accepted_prefix_by_depth"):
        values = stats.get(name)
        if not isinstance(values, list) or len(values) != 3 or any(type(x) is not int or x < 0 for x in values):
            raise ValueError("invalid native MTP depth counters: " + name)
        if getattr(args, "mtp_strategy", "fixed") == "fixed" and any(values[args.mtp_depth:]):
            raise ValueError("native MTP counters proposed/matched beyond requested fixed depth")
        arrays.append(values)
    proposed, matched, accepted = arrays
    if sum(proposed) != stats.get("proposed_tokens") or any(not 0 <= a <= m <= p for p, m, a in zip(proposed, matched, accepted)):
        raise ValueError("native MTP depth counters disagree with proposal totals")
    return max((index+1 for index, count in enumerate(proposed) if count), default=0)


def validate_records(records, case, args):
    decode = case["group"] == "decode"
    prefix = "apxinf.qwen35.benchmark." if decode else "apxinf.qwen35.context_benchmark."
    configs = [r for r in records if r.get("schema") == prefix+"config.v1"]
    iterations = [r for r in records if r.get("schema") == prefix+"iteration.v1"]
    summaries = [r for r in records if r.get("schema") == prefix+"summary.v1"]
    if len(configs) != 1 or len(summaries) != 1:
        raise ValueError("missing/duplicate config or summary")
    config, summary = configs[0], summaries[0]
    tune = args.run_mode == "tune"
    profile = args.run_mode == "profile"
    allow_divergence = getattr(args, "allow_repeat_divergence", False)
    warmups = 0 if tune else args.warmups
    repeats = args.repeats if args.run_mode == "formal" else 1
    output = 1024 if decode else 1
    total_input = case["initial_kv_tokens"] + (0 if decode else case["new_input_tokens"])
    identity = input_identity(total_input)
    for field, value in {"autotune":tune, "strict_recipe_required":not tune, "mode":args.mode,
                         "performance_eligible":not tune and not profile, "output_tokens":output,
                         "warmups":warmups, "repeats":repeats}.items():
        matches = (config.get(field) is value) if isinstance(value, bool) else (config.get(field) == value)
        if not matches:
            raise ValueError(f"config {field} mismatch: {config.get(field)!r} != {value!r}")
    if config.get("profiling", False) is not profile:
        raise ValueError("profile mode identity mismatch")
    if config.get("temperature") != 0 or config.get("sampling_mode") != "greedy":
        raise ValueError("sampling semantics changed")
    if config.get("tuning_dir") != args.tuning_dir:
        raise ValueError("tuning directory mismatch")
    if config.get("allow_repeat_divergence", False) and not allow_divergence:
        raise ValueError("repeat-divergence escape hatch is not a correctness-certified run")
    if config.get("allow_repeat_divergence", False) is not allow_divergence:
        raise ValueError("explicit repeat-divergence policy does not match benchmark config")
    if "allow_repeat_divergence" in summary and summary["allow_repeat_divergence"] is not allow_divergence:
        raise ValueError("summary repeat-divergence policy mismatch")
    if not allow_divergence and any(row.get("schema") == "apxinf.qwen35.benchmark.repeat_divergence.v1" for row in records):
        raise ValueError("greedy output divergence reported by benchmark")
    dtype = config.get("qwen35_kv_cache", config.get("target_kv_dtype"))
    activation = config.get("qwen35_moe_activation", config.get("moe_activation"))
    if dtype != args.kv_cache or activation != args.moe_activation:
        raise ValueError("precision/activation configuration mismatch")
    if config.get("input_digest_fnv1a64") != identity["fnv1a64"] or str(config.get("input_seed_hex", "")).lower().removeprefix("0x") != identity["seed_hex"]:
        raise ValueError("input token identity mismatch")
    if decode:
        if config.get("prompt_tokens") != total_input or config.get("context_reuse") is not False:
            raise ValueError("decode prompt/reuse configuration mismatch")
    elif (config.get("prefix_tokens"), config.get("input_tokens"), config.get("total_context_tokens")) != (case["initial_kv_tokens"], case["new_input_tokens"], total_input):
        raise ValueError("prefill geometry mismatch")
    policy = mtp_policy(args)
    if config.get("mtp_strategy") != policy["strategy"] or config.get("mtp_enabled") is not (args.mode == "mtp"):
        raise ValueError("MTP strategy/enabled configuration mismatch")
    if config.get("mtp_depth") != policy["fixed_depth"] or config.get("fixed_depth") != policy["fixed_depth"]:
        raise ValueError("MTP strategy/depth mismatch")
    if args.mode == "mtp":
        if not decode and policy["fixed_depth"] != 3:
            raise ValueError("MTP strategy/depth mismatch")
        if config.get("draft_kv_dtype") != "fp8" or config.get("draft_kv_scale") != 1.0:
            raise ValueError("draft KV precision missing or changed")
    if len(iterations) != warmups+repeats:
        raise ValueError("missing/extra measured iteration")
    values = []
    tpots = []
    digests = []
    observed_depths = []
    for index, row in enumerate(iterations):
        if row.get("iteration") != index or row.get("warmup") is not (index < warmups):
            raise ValueError("iteration ordering/warmup classification mismatch")
        if row.get("autotune") is not tune or row.get("performance_eligible") is not (not tune and not profile):
            raise ValueError("iteration tuning/eligibility mismatch")
        if row.get("strict_recipe_required") is not (not tune) or row.get("profiling", False) is not profile:
            raise ValueError("iteration strict recipe/profile mode mismatch")
        if row.get("mode") != args.mode or row.get("output_tokens") != output or row.get("decode_intervals") != output-1:
            raise ValueError("iteration output/mode/intervals mismatch")
        if decode:
            if row.get("prompt_tokens") != total_input or row.get("reused_prompt_tokens") != 0 or row.get("computed_prompt_tokens") != total_input:
                raise ValueError("decode prompt was truncated or unexpectedly reused")
            tps = finite_number(row.get("generation_tokens_per_second"), "decode TPS")
            ttft = finite_number(row.get("ttft_ms"), "TTFT")
            decode_ms = finite_number(row.get("decode_ms"), "Decode")
            total_ms = finite_number(row.get("total_latency_ms"), "total latency")
            if not math.isclose(tps, (output-1)*1000/decode_ms, rel_tol=1e-9, abs_tol=1e-6) or not math.isclose(total_ms, ttft+decode_ms, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("Decode intervals, TPS and measured times disagree")
            value = tps
            observed_depths.append(validate_decode_speculation(row.get("speculation"), args))
        else:
            if (row.get("prefix_tokens"), row.get("input_tokens"), row.get("total_context_tokens")) != (case["initial_kv_tokens"], case["new_input_tokens"], total_input):
                raise ValueError("prefill iteration geometry changed")
            if (row.get("initial_kv_tokens"), row.get("computed_input_tokens"), row.get("committed_context_tokens_after_prefill")) != (case["initial_kv_tokens"], case["new_input_tokens"], total_input):
                raise ValueError("actual KV position/input counts differ from case")
            value = finite_number(row.get("prefill_ms"), "prefill")
            if row.get("decode_ms") is not None or row.get("decode_tokens_per_second") is not None:
                raise ValueError("prefill-only run unexpectedly measured decode")
        if row.get("repeat_diverged", False) and not allow_divergence:
            raise ValueError("greedy output divergence reported by benchmark")
        digest = row.get("token_digest_fnv1a64")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{16}", digest):
            raise ValueError("missing actual output token digest")
        digests.append(digest)
        if index >= warmups:
            values.append(value)
            if decode:
                tpots.append(decode_ms / (output-1))
    if not allow_divergence and (len(set(digests)) != 1 or summary.get("divergent_repeats",0) != 0 or summary.get("divergent_warmups",0) != 0):
        raise ValueError("greedy outputs diverged; retain observations, not a correctness-certified score")
    divergence_events = validate_divergence_evidence(records, iterations, summary, output, args, decode) if allow_divergence else []
    if summary.get("performance_eligible") is not (not tune and not profile) or summary.get("autotune") is not tune:
        raise ValueError("summary eligibility/tuning mismatch")
    for field, value in {"mode":args.mode,"warmups":warmups,"repeats":repeats,"output_tokens":output,"decode_intervals":output-1}.items():
        if summary.get(field) != value:
            raise ValueError("summary counts/mode mismatch: " + field)
    if summary.get("strict_recipe_required") is not (not tune) or summary.get("profiling", False) is not profile:
        raise ValueError("summary strict recipe/profile mode mismatch")
    shape_fields = {"prompt_tokens":total_input} if decode else {"prefix_tokens":case["initial_kv_tokens"],"input_tokens":case["new_input_tokens"],"total_context_tokens":total_input}
    if any(summary.get(key) != value for key,value in shape_fields.items()):
        raise ValueError("summary geometry differs from iteration records")
    if decode:
        validate_decode_speculation(summary.get("speculation"), args, aggregate=True)
        if args.mode == "mtp":
            measured_stats = [row["speculation"] for row in iterations if not row["warmup"]]
            for key in ("proposed_by_depth", "matched_by_depth", "accepted_prefix_by_depth"):
                if summary["speculation"][key] != [sum(row[key][depth] for row in measured_stats) for depth in range(3)]:
                    raise ValueError("summary MTP counters disagree with raw measured rounds")
    metric_key = "generation_tokens_per_second" if decode else "prefill_ms"
    reported_median = finite_number((summary.get(metric_key) or {}).get("median"), "summary median")
    if not math.isclose(reported_median, statistics.median(values), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("summary median differs from formal raw observations")
    return dict(input_identity=identity, measured_iterations=repeats, primary_metric="decode_tokens_per_second" if decode else "prefill_ttft_ms",
                observed_median=statistics.median(values), observed_p90=sorted(values)[math.ceil(.9*len(values))-1], min=min(values),max=max(values),
                decode_tpot_ms=statistics.median(tpots) if decode else None, output_digest=digests[0], output_digests=digests,
                measured_primary_values=values, measured_decode_tpot_ms=tpots, total_iterations=len(iterations), warmup_iterations=warmups,
                decode_tpot_distribution=dict(median=statistics.median(tpots),p90=sorted(tpots)[math.ceil(.9*len(tpots))-1],min=min(tpots),max=max(tpots)) if decode else None,
                repeat_divergences=divergence_events, comparison_policy=comparison_policy(args), diagnostic_only=allow_divergence,
                mtp_policy=policy, observed_max_proposed_depth=max(observed_depths, default=0) if decode else None)


def validate_run(raw, case, args, recipe_before, recipe_after, expected_binary_sha, transport, expected_environment=None, expected_recipe_subset=None):
    failures = []
    details = {}
    exit_receipt = {}
    try:
        records, process, exit_receipt, lock = parse_output(raw)
        details.update(native_iterations=[row for row in records if row.get("schema", "").endswith("iteration.v1")],
                       repeat_divergences=[row for row in records if row.get("schema") == "apxinf.qwen35.benchmark.repeat_divergence.v1"])
        details.update(process=process, exit_receipt=exit_receipt, gpu_lock=lock, backend_task_may_be_live=bool(exit_receipt.get("backend_task_may_be_live")))
        if lock.get("path") != GPU_LOCK_PATH or lock.get("acquired") is not True:
            raise ValueError("shared GPU lock was not acquired")
        if expected_recipe_subset is not None:
            details["scalar_recipe_subset_locked_check"] = verify_recipe_subset(expected_recipe_subset, lock.get("recipes_before", {}))
        if exit_receipt.get("lock_held_until_reaped") is not True or exit_receipt.get("child_reaped") is not True:
            raise ValueError("GPU lock lifetime/child cleanup is unproven")
        if lock.get("target_binary_sha256") != expected_binary_sha or exit_receipt.get("target_binary_sha256") != expected_binary_sha:
            raise ValueError("locked binary identity changed")
        if transport.get("transport_timeout") or transport["transport_returncode"] != 0 or exit_receipt.get("returncode") != 0 or exit_receipt.get("timed_out"):
            raise ValueError("benchmark/transport exited unsuccessfully")
        if process.get("kind") != ("profiler_launcher" if args.run_mode=="profile" else "benchmark"):
            raise ValueError("unexpected process role")
        if expected_environment is not None and process.get("effective_mtp_environment") != {key:expected_environment.get(key) for key in (MTP_FIXED_ENV, MTP_SCALAR_TPOT_ENV)}:
            raise ValueError("worker effective MTP controller environment mismatch")
        if args.run_mode != "profile" and process.get("sha256") != expected_binary_sha:
            raise ValueError("actual /proc executable differs from verified binary")
        if args.run_mode != "tune":
            if recipe_before.get("count",0) < 1:
                raise ValueError("formal/audit/profile requires pre-existing recipes")
            if recipe_before["manifest_sha256"] != recipe_after["manifest_sha256"]:
                raise ValueError("recipes changed during non-tuning process")
            locked_before=lock.get("recipes_before",{});locked_after=exit_receipt.get("recipes_after",{})
            if locked_before.get("count",0)<1 or locked_before.get("manifest_sha256") != locked_after.get("manifest_sha256"):
                raise ValueError("recipes missing or changed inside GPU lock")
            details["recipe_audit"] = audit_recipe_sources(raw, args.run_mode in ("audit","profile"))
        details["observed"] = validate_records(records,case,args)
    except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
        failures.append(str(error))
        if exit_receipt.get("child_reaped") is not True:
            details["backend_task_may_be_live"] = True
    accepted = not failures
    diagnostic = getattr(args, "allow_repeat_divergence", False)
    return dict(valid=accepted, measurement_valid=accepted, performance_accepted=accepted and args.run_mode=="formal" and not diagnostic,
                accuracy_status="unresolved" if diagnostic else ("repeat_consistent" if accepted else "failed_or_incomplete"),
                comparison_policy=comparison_policy(args), failures=failures, **details)


def load_scalar_calibration(args, cases):
    """Validate scalar formal-stage evidence under the same explicit accuracy policy."""
    summary_path = Path(args.scalar_calibration).resolve()
    metadata_path = summary_path.parent/'metadata.json'
    summary_bytes, metadata_bytes = summary_path.read_bytes(), metadata_path.read_bytes()
    summary = json.loads(summary_bytes)
    metadata = json.loads(metadata_bytes)
    policy = comparison_policy(args)
    diagnostic = getattr(args, "allow_repeat_divergence", False)
    expected_accuracy = "unresolved" if diagnostic else "repeat_consistent"
    for document in (summary, metadata):
        if document.get("schema") != SCHEMA or document.get("run_mode") != "formal" or document.get("comparison_policy") != policy or document.get("allow_repeat_divergence", False) is not diagnostic:
            raise ValueError("scalar calibration requires the same explicit strict/diagnostic formal-stage measurement policy")
    if any(summary.get(key) is not True for key in ("all_cases_valid", "measurement_valid", "source_unchanged")) or summary.get("performance_accepted") is not (not diagnostic):
        raise ValueError("scalar calibration acceptance status does not match the strict/diagnostic policy")
    if summary.get("accuracy_status") != expected_accuracy:
        raise ValueError("scalar calibration accuracy status does not match the strict/diagnostic policy")
    identity = metadata.get("observed_before", {})
    verify_identity(identity, identity)
    verify_identity(metadata.get("expected", {}), identity)
    final = summary.get("observed_after", {})
    if final.get("source") != identity["source"] or final.get("binary_sha256") != identity["binary_sha256"]:
        raise ValueError("scalar calibration source/ELF changed during acquisition")
    if not isinstance(summary.get("results"), list):
        raise ValueError("scalar calibration has no case results")
    by_case = {row.get("case"):row for row in summary["results"]}
    if len(by_case) != len(summary["results"]):
        raise ValueError("duplicate scalar calibration case")
    current_env = environment(args, 1.0)
    controlled = {MTP_FIXED_ENV, MTP_SCALAR_TPOT_ENV, "APXINF_GEMM_REPORT", "APXINF_ATTENTION_REPORT", "APXINF_REQUIRE_RECIPES"}
    observer_keys = {"APXINF_GEMM_REPORT", "APXINF_ATTENTION_REPORT"}
    if args.run_mode in ("profile", "audit"):
        # NVTX markers may be added for an observer invocation. Other timing
        # switches can add synchronizations and remain part of kernel_env.
        controlled.add("APXINF_NVTX")
        observer_keys.add("APXINF_NVTX")
    def kernel_env(values):
        if not isinstance(values, dict):
            raise ValueError("scalar calibration lacks a recorded kernel environment")
        return {key:value for key,value in values.items() if key not in controlled}
    env_by_case = {entry["case"]:entry["environment"] for entry in metadata.get("case_execution", [])}
    receipt = dict(summary_path=str(summary_path), summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
                   metadata_path=str(metadata_path), metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(), identity=identity,
                   comparison_policy=policy,accuracy_status=expected_accuracy,performance_accepted=not diagnostic,cases={})
    for case in cases:
        row = by_case.get(case["id"])
        if not row or row.get("group") != "decode" or row.get("run_mode") != "formal" or any(row.get(key) is not True for key in ("valid", "measurement_valid")) or row.get("performance_accepted") is not (not diagnostic) or row.get("comparison_policy") != policy or row.get("accuracy_status") != expected_accuracy:
            raise ValueError("missing scalar calibration with the matching acceptance policy for " + case["id"])
        calibration_env = env_by_case.get(case["id"], metadata.get("environment", {}))
        if kernel_env(calibration_env) != kernel_env(current_env):
            raise ValueError("scalar calibration kernel environment differs for " + case["id"])
        log = summary_path.parent/(case["id"]+'.log')
        raw_bytes = log.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != row.get("log_sha256"):
            raise ValueError("scalar calibration raw log hash mismatch")
        raw = raw_bytes.decode()
        records, _, _, _ = parse_output(raw)
        configs = [item for item in records if item.get("schema") == "apxinf.qwen35.benchmark.config.v1"]
        if len(configs) != 1 or configs[0].get("mode") != "scalar":
            raise ValueError("calibration case must be actual scalar Decode")
        config = configs[0]
        if type(config.get("warmups")) is not int or config["warmups"] < 2 or type(config.get("repeats")) is not int or config["repeats"] < 5:
            raise ValueError("scalar calibration requires at least warmups2/measured5")
        scalar_args = argparse.Namespace(**vars(args))
        scalar_args.mode, scalar_args.mtp_strategy, scalar_args.mtp_depth = "scalar", "fixed", 3
        scalar_args.run_mode, scalar_args.allow_repeat_divergence = "formal", diagnostic
        scalar_args.warmups, scalar_args.repeats = config["warmups"], config["repeats"]
        validation = validate_run(raw, case, scalar_args, row["recipes_before"], row["recipes_after"],
                                  identity["binary_sha256"]["decode"], row["transport"])
        if not validation["measurement_valid"] or validation["performance_accepted"] is not (not diagnostic):
            raise ValueError("scalar calibration raw validation failed: " + '; '.join(validation["failures"]))
        observed = validation["observed"]
        locked_recipes = validation["exit_receipt"].get("recipes_after", {})
        required_recipes = recipe_subset(locked_recipes)
        if not required_recipes["count"] or row.get("exit_receipt", {}).get("recipes_after") != locked_recipes:
            raise ValueError("scalar calibration lacks matching locked post-run recipe evidence")
        verify_recipe_subset(required_recipes, validation["gpu_lock"].get("recipes_before", {}))
        values = row.get("observed", {}).get("measured_decode_tpot_ms")
        if values != observed["measured_decode_tpot_ms"] or row.get("native_iterations") != validation["native_iterations"] or row.get("repeat_divergences") != validation["repeat_divergences"] or row.get("observed", {}).get("repeat_divergences") != observed["repeat_divergences"] or row.get("observed", {}).get("output_digests") != observed["output_digests"]:
            raise ValueError("scalar calibration stored measurements disagree with raw 1023-interval Decode")
        scalar_tpot = finite_number(statistics.median(values), "scalar calibration median TPOT")
        receipt["cases"][case["id"]] = dict(scalar_tpot_ms=scalar_tpot, measured_decode_tpot_ms=values,
                                              warmups=config["warmups"], measured_iterations=config["repeats"],
                                              decode_intervals=1023, log_sha256=row["log_sha256"], comparison_policy=policy,
                                              accuracy_status=expected_accuracy,performance_accepted=not diagnostic,
                                              output_digests=observed["output_digests"],repeat_divergences=validation["repeat_divergences"],
                                              recipe_subset=required_recipes,recipe_subset_target_status="pending",
                                              observer_environment_comparison=dict(
                                                  run_mode=args.run_mode,original_target_env_arguments=list(args.env),
                                                  calibration_original_environment=calibration_env,
                                                  calibration_observer_environment={key:calibration_env.get(key) for key in sorted(observer_keys)},
                                                  current_observer_environment={key:current_env.get(key) for key in sorted(observer_keys)},
                                                  ignored_observer_keys=sorted(observer_keys),
                                                  note="Profile/audit instrumentation can change controller observations; its throughput is never accepted formal performance."),
                                              calibration_summary_sha256=receipt["summary_sha256"],calibration_metadata_sha256=receipt["metadata_sha256"])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-mode",choices=("tune","formal","audit","profile"),default="formal")
    parser.add_argument("--matrix",type=Path)
    parser.add_argument("--case","--cases",dest="cases")
    parser.add_argument("--ssh-host")
    for flag in ("workdir","context-binary","decode-binary","checkpoint","tuning-dir"):
        parser.add_argument("--"+flag,required=True)
    parser.add_argument("--metadata",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--inspect",action="store_true",help="Read source/binary/model hashes only; does not execute a benchmark")
    parser.add_argument("--plan",action="store_true",help="Print exact invocation plan without contacting the target")
    parser.add_argument("--mode",choices=("mtp","scalar"),default="mtp")
    parser.add_argument("--allow-repeat-divergence",action="store_true",help="Explicit diagnostic collection only; preserve divergence evidence and never accept formal performance")
    parser.add_argument("--mtp-depth",type=int,choices=(1,2,3),default=3)
    parser.add_argument("--mtp-strategy",choices=("fixed","adaptive"),default="fixed")
    parser.add_argument("--scalar-calibration",type=Path,help="Validated scalar formal-stage summary.json under the same strict/diagnostic policy, with sibling metadata.json and hashed raw case logs; required for adaptive")
    parser.add_argument("--kv-cache",choices=("bf16","fp8"),default="bf16")
    parser.add_argument("--moe-activation",choices=("nvfp4","bf16"),default="nvfp4")
    parser.add_argument("--warmups",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--timeout-sec",type=int,default=1800)
    parser.add_argument("--lock-timeout-sec",type=int,default=3600,help="Separate wait for shared /tmp/apxinf-gpu.lock; benchmark timeout begins after acquisition")
    parser.add_argument("--env",action="append",default=[])
    parser.add_argument("--nsys",default="nsys")
    parser.add_argument("--profile-dir",help="Absolute directory on benchmark host for nsys reports")
    args = parser.parse_args()
    minimum_warmups = 0 if args.run_mode == "tune" else 2
    if args.warmups < minimum_warmups or args.repeats < 1 or args.timeout_sec < 1 or args.lock_timeout_sec < 1:
        parser.error("warmups>=2 for non-tune, repeats>=1, timeout-sec>=1 required")
    try:
        cases = selected_cases(load_cases(args.matrix),args.cases)
        validate_selection_policy(args, cases)
        calibration = load_scalar_calibration(args, cases) if args.mtp_strategy == "adaptive" else None
        plan = []
        for case in cases:
            calibrated_case = calibration["cases"][case["id"]] if calibration else None
            scalar_tpot = calibrated_case["scalar_tpot_ms"] if calibrated_case else None
            plan.append(dict(case=case,argv=command_for_case(args,case),environment=environment(args,scalar_tpot),
                             mtp_policy=mtp_policy(args,scalar_tpot),scalar_calibration=calibrated_case))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.plan:
        print(json.dumps(dict(schema=SCHEMA,run_mode=args.run_mode,comparison_policy=comparison_policy(args),plan=plan,
                              scalar_calibration=calibration,calibration_target_identity_verified=False if calibration else None,
                              calibration_target_recipes_verified=False if calibration else None),ensure_ascii=False,indent=2));return 0
    transport = Transport(args)
    actual = transport.snapshot()
    if args.inspect:
        print(json.dumps(actual,ensure_ascii=False,indent=2));return 0
    if calibration:
        verify_identity(calibration["identity"], actual)
        for calibrated_case in calibration["cases"].values():
            calibrated_case["target_snapshot_recipe_subset_check"] = verify_recipe_subset(calibrated_case["recipe_subset"], actual.get("recipes", {}))
            calibrated_case["recipe_subset_target_status"] = "verified_snapshot; lock_check_required"
    if not args.metadata or not args.output:
        parser.error("execution requires --metadata and --output")
    expected=json.loads(args.metadata.read_text())
    verify_identity(expected,actual)
    if args.run_mode != "tune" and actual["recipes"]["count"] < 1:
        parser.error("recipe directory is empty: run separate tune phase first")
    if args.run_mode=="profile" and (not args.profile_dir or not Path(args.profile_dir).is_absolute()):
        parser.error("profile mode requires an absolute --profile-dir on target host")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'collector.py').write_bytes(COLLECTOR_SOURCE)
    (args.output/'matrix.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    metadata=dict(schema=SCHEMA,run_mode=args.run_mode,expected=expected,observed_before=actual,
                  environment=plan[0]["environment"] if not calibration else None,
                  case_execution=[dict(case=row["case"]["id"],environment=row["environment"],mtp_policy=row["mtp_policy"],scalar_calibration=row["scalar_calibration"]) for row in plan],
                  scalar_calibration=calibration,calibration_target_identity_verified=True if calibration else None,
                  calibration_target_recipes_verified=True if calibration else None,
                  comparison_policy=comparison_policy(args),allow_repeat_divergence=args.allow_repeat_divergence,
                  accuracy_status="unresolved" if args.allow_repeat_divergence else "pending_repeat_consistency",
                  collector_sha256=COLLECTOR_SHA256,collector_python=sys.version,utc_started=dt.datetime.now(dt.timezone.utc).isoformat(),
                  timing_scope="resident weights; cold KV or append timer; output1 prefill /1024 decode; first token excluded from1023 decode intervals")
    (args.output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    results=[]
    for entry in plan:
        case=entry['case'];argv=entry['argv'];before=transport.snapshot('recipes')
        if args.run_mode!='tune' and before['count']<1:raise RuntimeError('recipes vanished before execution')
        profile_report=None
        if args.run_mode=='profile':
            profile_report=str(Path(args.profile_dir)/case['id'])
            argv=[args.nsys,'profile','--trace=cuda,nvtx,osrt','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none',
                  '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output='+profile_report]+argv
        log=args.output/(case['id']+'.log')
        outcome=transport.execute(argv,entry['environment'],log,args.run_mode=='profile')
        after=transport.snapshot('recipes')
        binary_role='decode' if case['group']=='decode' else 'context'
        validation=validate_run(log.read_text(errors='replace'),case,args,before,after,actual['binary_sha256'][binary_role],outcome,entry['environment'],
                                entry['scalar_calibration']['recipe_subset'] if entry['scalar_calibration'] else None)
        if entry['scalar_calibration']:
            locked_subset_check=validation.get('scalar_recipe_subset_locked_check')
            entry['scalar_calibration']['recipe_subset_target_status']='verified_snapshot_and_lock' if locked_subset_check else 'snapshot_verified; lock_failed_or_unproven'
            entry['scalar_calibration']['locked_recipe_subset_check']=locked_subset_check
        profile_artifacts=None
        if profile_report:
            if validation.get('backend_task_may_be_live') or outcome.get('backend_task_may_be_live'):
                profile_artifacts=dict(valid=False,failures=['profile export skipped: profiler task cleanup is unproven'])
            else:
                # The inference worker has reaped its child and released the GPU
                # lock. Stats only reads/exports the trace; it never runs CUDA.
                profile_artifacts=transport.profile_artifacts(profile_report,args.output/case['id'])
            if not profile_artifacts['valid']:
                validation['valid']=False
                validation['measurement_valid']=False
                validation['performance_accepted']=False
                validation['failures'].extend(profile_artifacts['failures'])
        result=dict(case=case['id'],group=case['group'],argv=argv,run_mode=args.run_mode,recipes_before=before,recipes_after=after,
                    environment=entry['environment'],mtp_policy=entry['mtp_policy'],scalar_calibration=entry['scalar_calibration'],
                    log_sha256=sha256_file(log),transport=outcome,profile_report=profile_report,profile_artifacts=profile_artifacts,**validation)
        results.append(result)
        (args.output/(case['id']+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('case','valid','performance_accepted','accuracy_status','comparison_policy','failures')},ensure_ascii=False),flush=True)
        if not validation['valid'] or outcome.get('backend_task_may_be_live'):
            break
    final=transport.snapshot('source')
    source_unchanged=final['source']==actual['source'] and final['binary_sha256']==actual['binary_sha256']
    if not source_unchanged:
        for row in results:
            row["valid"]=False;row["measurement_valid"]=False;row["performance_accepted"]=False
            row["failures"].append("source/binary changed during acquisition")
            (args.output/(row["case"]+".json")).write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n")
    complete=len(results)==len(cases)
    accepted=complete and source_unchanged and all(row['valid'] for row in results)
    performance_accepted=accepted and args.run_mode=='formal' and all(row['performance_accepted'] for row in results)
    summary=dict(schema=SCHEMA,run_mode=args.run_mode,requested_cases=[x['id'] for x in cases],completed_cases=len(results),
                 case_execution=metadata['case_execution'],scalar_calibration=calibration,
                 calibration_target_identity_verified=True if calibration else None,
                 calibration_target_recipes_verified=True if calibration else None,
                 all_cases_valid=accepted,measurement_valid=accepted,full_matrix_requested=len(cases)==15,full_matrix_valid=accepted and len(cases)==15,performance_accepted=performance_accepted,source_unchanged=source_unchanged,
                 comparison_policy=comparison_policy(args),allow_repeat_divergence=args.allow_repeat_divergence,
                 accuracy_status="unresolved" if args.allow_repeat_divergence else ("repeat_consistent" if accepted else "failed_or_incomplete"),
                 comparison_notice="Baseline and candidate require the same declared comparison_policy; diagnostic timings do not establish accuracy or accepted performance.",
                 observed_after=final,results=results,skipped_cases=[x['id'] for x in cases[len(results):]],
                 backend_task_may_be_live=any(row.get('backend_task_may_be_live') or row['transport'].get('backend_task_may_be_live') for row in results))
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# Qwen35 model matrix','',f'Run mode: {args.run_mode}; accepted formal performance: {summary["performance_accepted"]}.',
           f'Requested {len(cases)} cases; completed {len(results)}; full15-case measurements valid: {summary["full_matrix_valid"]}.',
           f'Comparison policy: {summary["comparison_policy"]}; accuracy status: {summary["accuracy_status"]}.',summary['comparison_notice'],
           'Each row is a separate case. No mixed latency distribution or autotune time is reported as formal performance.','']
    if args.allow_repeat_divergence:
        lines+=['| Case | Status | Diagnostic measured N | Diagnostic median | Diagnostic P90 | Diagnostic TPOT median | Diagnostic TPOT P90 |',
                '|---|---|---:|---:|---:|---:|---:|']
    else:
        lines+=['| Case | Status | Formal N | Median | P90 |','|---|---|---:|---:|---:|']
    for row in results:
        obs=row.get('observed');eligible=row['performance_accepted']
        if obs and row['valid'] and args.allow_repeat_divergence:
            unit='tok/s' if row['group']=='decode' else 'ms'
            tpot=obs.get('decode_tpot_distribution')
            tail=f'{tpot["median"]:.6f} ms | {tpot["p90"]:.6f} ms' if tpot else '— | —'
            lines.append(f'| {row["case"]} | diagnostic; accuracy unresolved | {obs["measured_iterations"]} | {obs["observed_median"]:.6f} {unit} | {obs["observed_p90"]:.6f} {unit} | {tail} |')
        elif obs and eligible:
            unit='tok/s' if row['group']=='decode' else 'ms'
            lines.append(f'| {row["case"]} | passed | {obs["measured_iterations"]} | {obs["observed_median"]:.6f} {unit} | {obs["observed_p90"]:.6f} {unit} |')
        else:lines.append(f'| {row["case"]} | {"diagnostic only" if row["valid"] else "failed"} | — | — | — |'+(' — | — |' if args.allow_repeat_divergence else ''))
    for case_id in summary['skipped_cases']:
        lines.append(f'| {case_id} | not run after prior failure | — | — | — |'+(' — | — |' if args.allow_repeat_divergence else ''))
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    return 0 if accepted else 1


if __name__=='__main__':
    raise SystemExit(main())
