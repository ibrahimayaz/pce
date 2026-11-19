"""Worker process: repeatedly claims and runs pending jobs.

Launched by ``run_experiments.py`` as a separate process.  The parent sets
``CUDA_VISIBLE_DEVICES`` to exactly one physical GPU for GPU slots (no NVLink:
a job never spans more than one card) and to an empty string for CPU slots.

Claiming uses lock files created with O_CREAT|O_EXCL, which is atomic on both
Windows and Linux.  Stale locks of a previous (crashed / powered-off) run are
removed by the parent before workers are started.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _claim(lock: Path, slot: str) -> bool:
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"slot": slot, "pid": os.getpid(), "t": time.time()}))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--gpu", action="store_true", help="this slot owns one GPU")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--gpu-ram-part", type=float, default=0.2)
    ap.add_argument("--lgbm-gpu", action="store_true")
    args = ap.parse_args()

    from rev.models import set_thread_env
    set_thread_env(args.threads)

    from rev.jobs import get_profile, paths, record_error, run_job

    profile = get_profile(args.profile)
    P = paths(profile)
    jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    # GPU slots take GPU jobs first and help with CPU jobs when idle.
    # CPU slots only take CPU jobs.
    if args.gpu:
        order = [j for j in jobs if j["device"] == "cuda"] + [j for j in jobs if j["device"] != "cuda"]
    else:
        order = [j for j in jobs if j["device"] != "cuda"]

    print(f"[{args.slot}] started, gpu={args.gpu}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    while True:
        progressed = False
        for job in order:
            res = P["jobs"] / f"{job['id']}.json"
            lock = P["locks"] / f"{job['id']}.lock"
            err = P["errors"] / f"{job['id']}.txt"
            # failed jobs are not retried within the same run (they are retried on the next run)
            if res.exists() or lock.exists() or err.exists():
                continue
            if not _claim(lock, args.slot):
                continue
            try:
                if res.exists():
                    continue
                job = dict(job)
                if job["device"] == "cuda" and not args.gpu:
                    job["device"] = "cpu"
                t0 = time.time()
                print(f"[{args.slot}] START {job['id']} ({job['device']})", flush=True)
                run_job(job, profile, args.threads, args.gpu_ram_part, args.lgbm_gpu, args.slot)
                print(f"[{args.slot}] DONE  {job['id']} in {time.time() - t0:.1f}s", flush=True)
                progressed = True
            except KeyboardInterrupt:
                raise
            except Exception:
                traceback.print_exc()
                record_error(job, profile)
                print(f"[{args.slot}] FAIL  {job['id']} (see errors/)", flush=True)
                progressed = True
            finally:
                try:
                    lock.unlink()
                except OSError:
                    pass
            # re-scan from the top so that high-priority jobs are taken first
            break
        if not progressed:
            remaining = [j for j in order if not (P["jobs"] / f"{j['id']}.json").exists()
                         and not (P["errors"] / f"{j['id']}.txt").exists()
                         and not (P["locks"] / f"{j['id']}.lock").exists()]
            if not remaining:
                print(f"[{args.slot}] no more jobs, exiting", flush=True)
                return
            time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
