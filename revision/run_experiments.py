"""Resumable multi-GPU experiment runner.

Examples
--------
    # all 4 GPUs, 2 workers per GPU, 24 extra CPU workers
    python run_experiments.py run --profile full --gpus 0,1,2,3 --workers-per-gpu 2 --cpu-workers 24

    # only GPUs 0 and 2
    python run_experiments.py run --profile full --gpus 0,2

    # CPU only
    python run_experiments.py run --profile full --gpus none --cpu-workers 30

    # progress
    python run_experiments.py status --profile full

Crash safety
------------
* every finished job is stored atomically in ``jobs/<id>.json``;
* every objective evaluation is appended to ``cache/<id>.jsonl``; an
  interrupted tuning run is replayed from this cache and continues where it
  stopped;
* simply re-run the same command after a power cut / reboot: finished jobs
  are skipped, stale locks are cleared and failed jobs are retried.
* a heartbeat file prevents two runners from working on the same results
  folder at the same time.

No NVLink is required: each job uses at most one GPU.  The GPUs are shared
by running several independent jobs in parallel.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rev.config import PROFILES  # noqa: E402
from rev.io_utils import atomic_write_json, read_json  # noqa: E402
from rev.jobs import enumerate_jobs, paths  # noqa: E402
from rev.models import GPU_CAPABLE  # noqa: E402

HEARTBEAT_STALE_S = 180


def detect_gpus() -> list:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
        return [l.strip() for l in out.splitlines() if l.strip().isdigit()]
    except Exception:
        return []


def parse_gpus(arg: str) -> list:
    if arg in ("none", "", "cpu"):
        return []
    if arg == "auto":
        return detect_gpus()
    gpus = [g.strip() for g in arg.split(",") if g.strip()]
    avail = detect_gpus()
    missing = [g for g in gpus if avail and g not in avail]
    if missing:
        sys.exit(f"Requested GPU(s) {missing} not found. Available: {avail}")
    if not avail:
        print("WARNING: nvidia-smi not found; trusting --gpus as given.")
    return gpus


def pid_alive(pid: int) -> bool:
    """True if a process with this PID is running (never sends a signal)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)  # POSIX: signal 0 only checks existence
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class RunnerLock:
    def __init__(self, path: Path):
        self.path = path

    def acquire(self) -> None:
        info = read_json(self.path)
        if info and time.time() - info.get("heartbeat", 0) < HEARTBEAT_STALE_S:
            same_host = info.get("host") == platform.node()
            # after a reboot the old PID is gone -> the lock is stale even if the heartbeat is recent
            if not same_host or pid_alive(int(info.get("pid", -1))):
                sys.exit(f"Another runner seems active (host={info.get('host')}, pid={info.get('pid')}). "
                         f"If you are sure it is not, delete {self.path}")
        self.beat()

    def beat(self) -> None:
        atomic_write_json(self.path, {"host": platform.node(), "pid": os.getpid(), "heartbeat": time.time()})

    def release(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def summarise(jobs, P) -> dict:
    done = {p.stem for p in P["jobs"].glob("*.json")}
    failed = {p.stem for p in P["errors"].glob("*.txt")}
    running = {p.stem for p in P["locks"].glob("*.lock")}
    by_block = Counter()
    done_block = Counter()
    for j in jobs:
        by_block[j["block"]] += 1
        if j["id"] in done:
            done_block[j["block"]] += 1
    return {
        "total": len(jobs), "done": len(done & {j["id"] for j in jobs}),
        "failed": len(failed), "running": len(running),
        "blocks": {b: f"{done_block[b]}/{n}" for b, n in by_block.items()},
    }


def cmd_status(args) -> None:
    profile = PROFILES[args.profile]
    P = paths(profile)
    manifest = P["root"] / "manifest.json"
    jobs = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else \
        enumerate_jobs(profile, [], False)
    s = summarise(jobs, P)
    print(json.dumps(s, indent=1))
    if s["failed"]:
        print(f"Failed job tracebacks: {P['errors']}")


def cmd_run(args) -> None:
    profile = PROFILES[args.profile]
    P = paths(profile)
    gpus = parse_gpus(args.gpus)
    gpu_models = [m.strip() for m in args.gpu_models.split(",") if m.strip()]
    bad = [m for m in gpu_models if m not in GPU_CAPABLE and not (m == "LightGBM" and args.lgbm_gpu)]
    if bad:
        sys.exit(f"Models without GPU support in this setup: {bad}")
    if args.lgbm_gpu and "LightGBM" not in gpu_models:
        gpu_models.append("LightGBM")

    lock = RunnerLock(P["root"] / "runner.heartbeat.json")
    lock.acquire()

    # stale claims from an interrupted run
    for f in P["locks"].glob("*.lock"):
        f.unlink()
    if not args.no_retry_errors:
        for f in P["errors"].glob("*.txt"):
            f.unlink()

    jobs = enumerate_jobs(profile, gpu_models, use_gpu=bool(gpus))
    if args.blocks:
        keep = set(args.blocks.split(","))
        jobs = [j for j in jobs if j["block"] in keep]
    manifest = P["root"] / "manifest.json"
    atomic_write_json(manifest, jobs)
    atomic_write_json(P["root"] / "run_settings.json", {
        "profile": profile.name, "budget": profile.budget_hash(), "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu, "cpu_workers": args.cpu_workers,
        "threads": args.threads, "gpu_models": gpu_models, "host": platform.node(),
        "python": sys.version, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    s = summarise(jobs, P)
    print(f"Results folder: {P['root']}")
    print(f"Jobs: {s['done']}/{s['total']} already done. GPUs: {gpus or 'none'}; GPU models: {gpu_models}")

    cpu_workers = args.cpu_workers
    if cpu_workers < 0:
        cpu_workers = max(1, (os.cpu_count() or 2) // max(1, args.threads) - len(gpus) * args.workers_per_gpu)

    slots = []
    for g in gpus:
        for w in range(args.workers_per_gpu):
            slots.append((f"gpu{g}_w{w}", g))
    for w in range(cpu_workers):
        slots.append((f"cpu_w{w}", None))
    if not slots:
        sys.exit("No worker slots configured.")

    procs = {}

    def start(slot, gpu):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else ""
        env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [sys.executable, "-m", "rev.worker", "--profile", profile.name, "--manifest", str(manifest),
               "--slot", slot, "--threads", str(args.threads),
               "--gpu-ram-part", str(args.gpu_ram_part)]
        if gpu is not None:
            cmd.append("--gpu")
        if args.lgbm_gpu:
            cmd.append("--lgbm-gpu")
        log = open(P["logs"] / f"{slot}.log", "a", encoding="utf-8")
        log.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.flush()
        procs[slot] = (subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=log, stderr=subprocess.STDOUT),
                       gpu, log)

    for slot, gpu in slots:
        start(slot, gpu)
    print(f"Started {len(slots)} workers ({sum(1 for _, g in slots if g is not None)} on GPU). "
          f"Logs: {P['logs']}")

    restarts = Counter()
    last_report = 0.0
    retries_left = args.error_retries
    try:
        while True:
            if not procs:
                failed = list(P["errors"].glob("*.txt"))
                if failed and retries_left > 0:
                    retries_left -= 1
                    print(f"{len(failed)} job(s) failed; retrying them ({args.error_retries - retries_left}/"
                          f"{args.error_retries}).")
                    for f in failed:
                        f.unlink()
                    for slot, gpu in slots:
                        start(slot, gpu)
                else:
                    break
            time.sleep(5)
            lock.beat()
            for slot in list(procs):
                p, gpu, log = procs[slot]
                rc = p.poll()
                if rc is None:
                    continue
                log.close()
                del procs[slot]
                # release claims left behind by a crashed worker
                for lf in P["locks"].glob("*.lock"):
                    info = read_json(lf)
                    if info and info.get("slot") == slot:
                        try:
                            lf.unlink()
                        except OSError:
                            pass
                if rc != 0 and restarts[slot] < args.max_restarts:
                    restarts[slot] += 1
                    print(f"Worker {slot} exited with code {rc}; restarting ({restarts[slot]})")
                    start(slot, gpu)
            if time.time() - last_report > args.report_every:
                last_report = time.time()
                s = summarise(jobs, P)
                atomic_write_json(P["root"] / "status.json", s)
                print(f"[{time.strftime('%H:%M:%S')}] done {s['done']}/{s['total']}  running {s['running']}  "
                      f"failed {s['failed']}  {s['blocks']}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted - stopping workers (finished jobs are kept; re-run to resume).")
        for p, _, log in procs.values():
            p.terminate()
        for p, _, log in procs.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            log.close()
    finally:
        s = summarise(jobs, P)
        atomic_write_json(P["root"] / "status.json", s)
        print(json.dumps(s, indent=1))
        lock.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", default="full", choices=list(PROFILES))
    r.add_argument("--gpus", default="auto", help="'0,1,2,3', 'auto' (all visible) or 'none'")
    r.add_argument("--workers-per-gpu", type=int, default=2)
    r.add_argument("--cpu-workers", type=int, default=-1, help="-1 = all remaining cores")
    r.add_argument("--threads", type=int, default=1, help="threads per job")
    r.add_argument("--gpu-models", default="XGBoost,CatBoost",
                   help="models that run on GPU slots (benchmark with bench_devices.py)")
    r.add_argument("--gpu-ram-part", type=float, default=0.2, help="CatBoost GPU RAM share per worker")
    r.add_argument("--lgbm-gpu", action="store_true", help="LightGBM on GPU (needs a GPU-enabled build)")
    r.add_argument("--blocks", default="", help="comma-separated subset of blocks")
    r.add_argument("--no-retry-errors", action="store_true")
    r.add_argument("--max-restarts", type=int, default=20)
    r.add_argument("--error-retries", type=int, default=2, help="retry rounds for failed jobs within one run")
    r.add_argument("--report-every", type=float, default=60)
    st = sub.add_parser("status")
    st.add_argument("--profile", default="full", choices=list(PROFILES))
    args = ap.parse_args()
    {"run": cmd_run, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
