"""Power-cut simulation: a tuning job is killed half-way and restarted.

The resumed result must be identical to an uninterrupted run, and the restart
must reuse the evaluations stored before the kill.

    python tests/test_resume.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

CHILD = r'''
import sys, json
sys.path.insert(0, r"{here}")
from rev.config import PROFILES, Profile
from rev.jobs import enumerate_jobs, run_job
p = Profile(name="{name}", eo_pop=10, eo_epoch=12, rkf_folds=5, rkf_repeats=1, inner_folds=3,
            blocks=[b for b in PROFILES["smoke"].blocks if b.name in ("rkf_main", "rkf_tuners")])
PROFILES[p.name] = p
job = [j for j in enumerate_jobs(p, [], False) if j["model"] == "CatBoost" and j["tuner"] == "{tuner}"][0]
r = run_job(job, p, 2, 0.2, False, "test")
print(json.dumps({{"params": r["best_params"], "metrics": r["metrics"], "hits": r["n_cache_hits"],
                  "evals": r["n_evals"]}}))
'''


def run(name, tuner, kill_after=None):
    code = CHILD.format(here=str(HERE), name=name, tuner=tuner)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if kill_after:
        time.sleep(kill_after)
        proc.kill()  # simulated power cut (no clean-up code runs)
        proc.wait()
        return None
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError(err)
    return json.loads(out.strip().splitlines()[-1])


def main():
    for d in ("resA", "resB"):
        shutil.rmtree(HERE / f"results_{d}", ignore_errors=True)
    ok = True
    for tuner in ("eo", "tpe", "random"):
        for d in ("resA", "resB"):
            shutil.rmtree(HERE / f"results_{d}", ignore_errors=True)
        ref = run("resA", tuner)
        run("resB", tuner, kill_after=12)
        cache = list((HERE / "results_resB").glob("budget_*/cache/*.jsonl"))
        n_before = sum(1 for _ in open(cache[0], encoding="utf-8")) if cache else 0
        res = run("resB", tuner)
        same = ref["params"] == res["params"] and abs(ref["metrics"]["RMSE"] - res["metrics"]["RMSE"]) < 1e-9
        ok &= same and res["hits"] > 0
        print(f"{tuner:6s}: evaluations={res['evals']}, cached before kill={n_before}, reused after restart="
              f"{res['hits']}, identical to uninterrupted run: {same}")
    for d in ("resA", "resB"):
        shutil.rmtree(HERE / f"results_{d}", ignore_errors=True)
    print("RESUME TEST PASSED" if ok else "RESUME TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
