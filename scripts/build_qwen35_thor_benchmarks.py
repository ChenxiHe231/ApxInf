#!/usr/bin/env python3
"""Compile both Thor model benchmarks in a locked, small-HugeTLB build phase.

Default is plan-only. This does not run inference, stop services or kill other
jobs. Complete all compilation before starting the matrix; do not interleave
native compilation with a large inference HugeTLB pool.
"""
import argparse
import contextlib
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

GPU_LOCK_PATH = Path("/tmp/apxinf-gpu.lock")
POOL_PATH = Path("/proc/sys/vm/nr_hugepages")
BUILD_PAGES = 2048
MIN_AVAILABLE = 24 * 1024 ** 3
INFERENCE_RESERVE = 8 * 1024 ** 3
EXAMPLES = ("qwen35_context_bench", "qwen35_mtp_bench")
SCHEMA = "apxinf.qwen35.thor-build.v1"

# Fixed privileged operation: pool write/readback only, never zero. If an
# optional increase is partial/fails, rollback within this same authorized
# process, so sudo credential expiry cannot prevent a second rollback sudo.
POOL_CODE = r'''
import json,pathlib,sys
p=pathlib.Path('/proc/sys/vm/nr_hugepages');target=int(sys.argv[1])
if target<2048:raise SystemExit('pool target must be >=2048')
result={'requested':target,'rollback_requested':False};wrote=False
def available():
 for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
 raise RuntimeError('MemAvailable unavailable')
try:
 result['before']=int(p.read_text());growing=target>result['before']
 if growing:
  result['available_before']=available()
  if (target-result['before'])*2097152>result['available_before']-8589934592:raise RuntimeError('inference growth exceeds 8GiB reserve budget')
 p.write_text(str(target)+'\n');wrote=True;result['actual']=int(p.read_text())
 if result['actual']!=target:raise RuntimeError('pool readback differs')
 if growing:
  result['available_after']=available()
  if result['available_after']<8589934592:raise RuntimeError('inference growth left less than 8GiB available')
 result['ok']=True
except Exception as error:
 result.update(ok=False,error=str(error))
 try:
  if wrote or int(p.read_text())!=2048:result['rollback_requested']=True;p.write_text('2048\n')
  result['actual']=int(p.read_text());result['small_pool_proven']=result['actual']==2048
 except Exception as rollback:result.update(rollback_error=str(rollback),small_pool_proven=False)
print(json.dumps(result),flush=True)
sys.exit(0 if result['ok'] else 1)
'''


class GuardError(RuntimeError):
    pass


@contextlib.contextmanager
def defer_stop_signals():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try: yield
    finally: signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def source_snapshot(root):
    def git(*args): return subprocess.check_output(["git", "-C", str(root), *args])
    untracked = [x.decode() for x in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if x]
    return dict(commit=git("rev-parse", "HEAD").decode().strip(),
                status=git("status", "--porcelain", "--untracked-files=all").decode(),
                patch_sha256=hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
                untracked_sha256={x: digest(root / x) for x in untracked if (root / x).is_file()})


def validate_platform():
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise GuardError("execute requires local Linux aarch64 Jetson Thor")
    if os.geteuid() == 0: raise GuardError("run launcher/cargo as the ordinary user, not root")
    model = Path("/proc/device-tree/model").read_bytes().decode().rstrip("\0").strip()
    compatible = [x.decode() for x in Path("/proc/device-tree/compatible").read_bytes().split(b"\0") if x]
    named_thor = "thor" in model.lower()
    observed_pair = model == "p3960-0010" and {"nvidia,p3960", "nvidia,tegra264"}.issubset(compatible)
    if not (named_thor or observed_pair): raise GuardError("local device is not a recognized Thor model/board+SoC pair")
    return dict(system=platform.system(), machine=platform.machine(), model=model, compatible=compatible,
                identification="thor_model_string" if named_thor else "observed_p3960-0010_tegra264_pair", uid=os.geteuid())


def memory_snapshot():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "Hugepagesize", "HugePages_Total", "HugePages_Free", "HugePages_Rsvd", "HugePages_Surp"):
            fields[key] = int(value.split()[0]) * (1024 if "kB" in value else 1)
    for key in ("MemAvailable", "HugePages_Total", "HugePages_Free", "HugePages_Rsvd", "Hugepagesize"):
        if key not in fields: raise GuardError("missing /proc/meminfo field " + key)
    fields["used_pages"] = fields["HugePages_Total"] - fields["HugePages_Free"]
    # Rsvd is usually still counted in Free: add it once to mapped usage.
    fields["committed_or_reserved_pages"] = fields["used_pages"] + fields["HugePages_Rsvd"]
    fields["nr_hugepages"] = int(POOL_PATH.read_text())
    return fields


def check_memory(snapshot, *, small=False):
    total, free, reserved = (snapshot[k] for k in ("HugePages_Total", "HugePages_Free", "HugePages_Rsvd"))
    if not 0 <= reserved <= free <= total or snapshot["Hugepagesize"] != 2 * 1024 ** 2:
        raise GuardError("invalid HugePages counts or non-2MiB pool")
    if snapshot["committed_or_reserved_pages"] > BUILD_PAGES:
        raise GuardError("mapped plus reserved HugePages exceeds2048; refusing shrink")
    if small and (snapshot["nr_hugepages"] != BUILD_PAGES or total != BUILD_PAGES):
        raise GuardError("small build pool/readback was not established")


def write_pool(pages):
    if pages < BUILD_PAGES: raise GuardError("pool must never be set below2048")
    argv = ["sudo", "-n", "--", "/usr/bin/python3", "-I", "-c", POOL_CODE, str(pages)]
    # No timeout: killing a root pool writer mid-write cannot establish the
    # pool's state. Only the fixed pool operation runs privileged.
    result = subprocess.run(argv, text=True, capture_output=True)
    try: receipt = json.loads(result.stdout)
    except ValueError: receipt = dict(ok=False, error="sudo/pool writer produced no JSON")
    receipt.update(returncode=result.returncode, stderr=result.stderr, argv=argv)
    return receipt


def proc_stat(path):
    raw = path.read_text()
    pid = int(raw.split("(", 1)[0]); rest = raw.rsplit(")", 1)[1].split()
    return dict(pid=pid, state=rest[0], pgid=int(rest[2]), sid=int(rest[3]), start_ticks=int(rest[19]))


def enable_subreaper():
    # Adopt orphaned compiler grandchildren so cleanup can reap the complete
    # owned group, rather than assume PID1 will promptly reap them for us.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0: raise GuardError("cannot enable Linux child subreaper")


def scan_group(pgid, start_ticks):
    """Fixed real /proc only. Unknown membership/provenance fails closed."""
    try: os.killpg(pgid, 0)
    except ProcessLookupError: return []
    except PermissionError as error: raise GuardError("cannot verify owned process group") from error
    if start_ticks is None:
        raise GuardError("live group has no original leader starttime; refusing to signal it")
    rows = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit(): continue
        try: row = proc_stat(path / "stat")
        except FileNotFoundError: continue
        except OSError as error: raise GuardError("cannot inspect process membership") from error
        if row["pgid"] != pgid: continue
        if row["sid"] != pgid or (row["pid"] == pgid and start_ticks is not None and row["start_ticks"] != start_ticks):
            raise GuardError("process group identity changed; refusing to signal it")
        rows.append(row)
    if not rows:
        try: os.killpg(pgid, 0)
        except ProcessLookupError: return []
        raise GuardError("group still exists but membership was not established")
    return rows


def unblock_child_signals():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})


class OwnedBuild:
    def __init__(self, argv, cwd, env, log, lock_fd, scanner=scan_group):
        self.proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=(lock_fd,), preexec_fn=unblock_child_signals)
        self.pgid = self.proc.pid
        self.scan = scanner
        try: self.start_ticks = proc_stat(Path(f"/proc/{self.pgid}/stat"))["start_ticks"]
        except (OSError, ValueError): self.start_ticks = None

    def members(self): return self.scan(self.pgid, self.start_ticks)

    def reap_descendants(self):
        reaped = []
        # Popen must collect its own direct child status first.
        if self.proc.poll() is None: return reaped
        while True:
            try: pid, status = os.waitpid(-self.pgid, os.WNOHANG)
            except ChildProcessError: break
            if pid == 0: break
            reaped.append(dict(pid=pid, wait_status=status))
        return reaped

    def cleanup(self, grace=10.0):
        result = dict(pid=self.proc.pid, pgid=self.pgid, direct_child_inherits_lock=True, proven=False, descendants_reaped=[])
        try:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                self.proc.poll()
                result["descendants_reaped"] += self.reap_descendants()
                rows = self.members()
                if not rows: break
                live = [row for row in rows if row["state"] not in ("Z", "X")]
                if live:
                    try: os.killpg(self.pgid, sig)  # only the group created above
                    except ProcessLookupError: pass
                until = time.monotonic() + grace
                while time.monotonic() < until:
                    self.proc.poll()
                    result["descendants_reaped"] += self.reap_descendants()
                    if not self.members(): break
                    time.sleep(0.05)
            self.proc.wait(timeout=grace)
            result["descendants_reaped"] += self.reap_descendants()
            rows = self.members()
            result.update(returncode=self.proc.returncode, members_after=rows,
                          direct_child_reaped=True,
                          proven=not rows)
        except (OSError, GuardError, subprocess.TimeoutExpired) as error:
            result["error"] = str(error)
        return result


def acquire_lock(stream, timeout):
    until = time.monotonic() + timeout
    while True:
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); return
        except BlockingIOError:
            if time.monotonic() >= until: raise GuardError("shared GPU lock is busy; no build/pool change")
            time.sleep(0.1)


def artifact_receipts(log, target):
    artifacts = {}
    for line in log.read_text(errors="replace").splitlines():
        try: row = json.loads(line)
        except ValueError: continue
        name = row.get("target", {}).get("name")
        if row.get("reason") != "compiler-artifact" or name not in EXAMPLES or not row.get("executable"): continue
        path = Path(row["executable"]).resolve()
        if target not in path.parents: raise GuardError("cargo artifact is outside explicit target directory")
        with path.open("rb") as stream: header = stream.read(20)
        if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183:
            raise GuardError("artifact is not a native aarch64 ELF: " + str(path))
        artifacts[name] = dict(path=str(path), sha256=digest(path), size_bytes=path.stat().st_size)
    if set(artifacts) != set(EXAMPLES): raise GuardError("cargo JSON did not identify both actual benchmark ELFs")
    return artifacts


def build_plan(args):
    root = Path(args.workdir).resolve(); target = Path(args.target_dir).resolve()
    argv = ["cargo", "build", "--release", "-p", "apxinf-model", "--no-default-features",
            "--features", "qwen35-cuda", "--example", EXAMPLES[0], "--example", EXAMPLES[1],
            "--jobs", "2", "--target-dir", str(target), "--message-format=json-render-diagnostics"]
    return dict(schema=SCHEMA, execute=args.execute, workdir=str(root), target_dir=str(target), argv=argv,
                environment=dict(CARGO_BUILD_JOBS="2", APXINF_NVCC_PARALLELISM="2"),
                lock_path=str(GPU_LOCK_PATH), build_pages=BUILD_PAGES,
                required_available_bytes=MIN_AVAILABLE, inference_pages=args.inference_pages,
                inference_reserve_bytes=INFERENCE_RESERVE,
                receipt=str(Path(args.receipt).resolve()))


def save_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n"); tmp.replace(path)


def quarantine(lock, receipt, path):
    """Fail closed: a still-live guardian retains the lock, never grows pool."""
    receipt.update(status="unsafe_state_lock_retained", explicit_unlock=False, exit_code=70)
    save_receipt(path, receipt)
    print("Safety state is unproven; guard retains shared lock. See " + str(path), file=sys.stderr, flush=True)
    # A signal must not accidentally release the lock in this unsafe state.
    signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True: time.sleep(60)


def execute(args):
    receipt = build_plan(args); path = Path(receipt["receipt"])
    if not args.execute: print(json.dumps(receipt, indent=2)); return 0
    lock = None; acquired = False; owned = None; proven = True
    receipt.update(status="preflight", owner_pid=os.getpid(), launcher_sha256=digest(__file__),
                   started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    def interrupted(signum, frame): raise GuardError("interrupted by signal " + str(signum))
    old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        receipt["platform"] = validate_platform()
        enable_subreaper()
        root = Path(receipt["workdir"]); target = Path(receipt["target_dir"])
        if not (root / "crates/apxinf-model/Cargo.toml").is_file(): raise GuardError("not an ApxInf worktree")
        env = os.environ.copy(); env.update(receipt["environment"])
        env["PATH"] = str(Path.home() / ".cargo/bin") + ":/usr/local/cuda/bin:" + env.get("PATH", "")
        cargo = shutil.which("cargo", path=env["PATH"])
        if cargo is None: raise GuardError("cargo not found in ordinary user's PATH")
        receipt["argv"][0] = cargo
        receipt["cargo_sha256"] = digest(cargo)
        lock = GPU_LOCK_PATH.open("a+"); acquire_lock(lock, args.lock_timeout_sec); acquired = True
        receipt["lock_acquired"] = True
        receipt["source_before"] = source_snapshot(root)
        before = receipt["memory_before"] = memory_snapshot()
        check_memory(before)
        # Finish the small fixed privileged operation and record its outcome
        # before an interrupt may unwind the guard. Cargo never runs as root.
        receipt["status"] = "shrinking_pool"; save_receipt(path, receipt)
        with defer_stop_signals():
            receipt["shrink_pool"] = write_pool(BUILD_PAGES)
            small = receipt["memory_build_ready"] = memory_snapshot()
        if not receipt["shrink_pool"].get("ok"): raise GuardError("sudo/small-pool request failed")
        check_memory(small, small=True)
        if small["MemAvailable"] < MIN_AVAILABLE: raise GuardError("MemAvailable below24GiB; build refused")
        save_receipt(path, receipt)
        log_path = path.parent / "cargo-build.jsonl"; receipt["build_log"] = str(log_path)
        with log_path.open("w") as log:
            # Bind the owned object before pending signals can unwind us;
            # otherwise a signal between spawn and assignment loses cleanup.
            with defer_stop_signals():
                owned = OwnedBuild(receipt["argv"], root, env, log, lock.fileno())
            receipt.update(build_pid=owned.proc.pid, build_pgid=owned.pgid,
                           build_start_ticks=owned.start_ticks, status="building")
            save_receipt(path, receipt)
            try:
                receipt["build_exit_code"] = owned.proc.wait(timeout=args.timeout_sec)
                receipt["unexpected_live_descendants"] = any(r["state"] not in ("Z", "X") for r in owned.members())
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                raise
            finally:
                receipt["cleanup"] = owned.cleanup(); proven = receipt["cleanup"]["proven"]
                receipt.setdefault("build_exit_code", owned.proc.returncode)
        if not proven: quarantine(lock, receipt, path)
        if receipt["build_exit_code"] != 0 or receipt["unexpected_live_descendants"]:
            raise GuardError("cargo failed or left live owned descendants")
        receipt["artifacts"] = artifact_receipts(log_path, target)
        receipt["source_after"] = source_snapshot(root)
        if receipt["source_after"] != receipt["source_before"]: raise GuardError("source changed during build")
        receipt["memory_after_build"] = memory_snapshot()
        if args.inference_pages is not None:
            current = receipt["memory_after_build"]
            additional = max(0, args.inference_pages-current["HugePages_Total"]) * current["Hugepagesize"]
            if additional > current["MemAvailable"]-INFERENCE_RESERVE:
                raise GuardError("inference pool exceeds available memory minus8GiB reserve; no pool write")
            receipt["status"] = "setting_inference_pool"; save_receipt(path, receipt)
            with defer_stop_signals():
                receipt["inference_pool"] = write_pool(args.inference_pages)
                after = receipt["memory_after_inference_pool"] = memory_snapshot()
            if not receipt["inference_pool"].get("ok") or after["nr_hugepages"] != args.inference_pages or after["HugePages_Total"] != args.inference_pages:
                if after["nr_hugepages"] != BUILD_PAGES or after["HugePages_Total"] != BUILD_PAGES:
                    with defer_stop_signals():
                        receipt["inference_rollback"] = write_pool(BUILD_PAGES)
                        after = receipt["memory_after_rollback"] = memory_snapshot()
                    if after["nr_hugepages"] != BUILD_PAGES or after["HugePages_Total"] != BUILD_PAGES:
                        receipt["error"] = "inference pool failed and small rollback unproven"
                        quarantine(lock, receipt, path)
                raise GuardError("inference pool request failed; no inference was started")
            check_memory(after)
        receipt.update(status="success", exit_code=0)
        return 0
    except Exception as error:
        receipt.update(status="failed", error=str(error), exit_code=1)
        return 1
    finally:
        for s in (signal.SIGINT, signal.SIGTERM): signal.signal(s, signal.SIG_IGN)
        # Covers constructor/wait/interrupt exceptions. No signal is sent to
        # jobs outside the uniquely created owned group.
        if owned is not None and not receipt.get("cleanup", {}).get("proven"):
            receipt["cleanup"] = owned.cleanup(); proven = receipt["cleanup"]["proven"]
        if acquired:
            if not proven: quarantine(lock, receipt, path)
            try: receipt["memory_final"] = memory_snapshot()
            except Exception as error: receipt["memory_final_error"] = str(error)
            receipt["explicit_unlock"] = True
            receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            save_receipt(path, receipt)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        else: save_receipt(path, receipt)
        if lock is not None: lock.close()
        for s, handler in old_handlers.items(): signal.signal(s, handler)


def parser():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(); mode.add_argument("--execute", action="store_true")
    mode.add_argument("--plan", action="store_true", help="default; no lock, sudo, pool writes or build")
    p.add_argument("--workdir", default=str(root)); p.add_argument("--target-dir", default=os.environ.get("CARGO_TARGET_DIR", str(root / "target")))
    p.add_argument("--receipt", default=str(root / "devlocal/model-microbench/safe-build" / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}") / "receipt.json"))
    p.add_argument("--inference-pages", type=int); p.add_argument("--timeout-sec", type=float, default=7200)
    p.add_argument("--lock-timeout-sec", type=float, default=60)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.inference_pages is not None and args.inference_pages < BUILD_PAGES: p.error("--inference-pages must be >=2048, never zero")
    if not math.isfinite(args.timeout_sec) or not math.isfinite(args.lock_timeout_sec) or args.timeout_sec <= 0 or args.lock_timeout_sec < 0: p.error("invalid timeout")
    return execute(args)


if __name__ == "__main__": sys.exit(main())
