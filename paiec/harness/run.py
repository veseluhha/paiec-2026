import argparse
import importlib.util
import json
import multiprocessing as mp
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FOLDS, REPO, RUNS, load
from .sim import BUDGETS, WEIGHTS, Bench, Submission, episode, run

G = {}


def fit(model_dir, out, train, variant=None):
    shutil.copytree(model_dir, out, ignore=shutil.ignore_patterns("__pycache__"))
    if variant is not None:
        (out / "variant.json").write_text(json.dumps(variant))
    f = out / "fit.py"
    if f.exists():
        spec = importlib.util.spec_from_file_location(f"fit_{out.parent.name}", f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.fit(train, out)
    return out


def _task(args):
    fold, e, seed = args
    rng = np.random.default_rng([seed, FOLDS.index(fold), e])
    if fold not in G["bench"]:
        G["bench"] = {fold: Bench(G["data"], fold)}
    ps = episode(G["bench"][fold], rng)
    B, got, st = run(Submission(G["subs"][fold]), ps, rng)
    return [{"fold": fold, "episode": e, "pair": p.sid, "n_eval": len(p.ev), "n_acq": len(p.acq),
             "labels": g, **{f"b{b}": B[k, j] for j, b in enumerate(BUDGETS)}, "alc": float(WEIGHTS @ B[k]),
             "seconds": st["seconds"], "max_call": st["max_call"]} for k, (p, g) in enumerate(zip(ps, got))]


def summarize(df):
    cols = [f"b{b}" for b in BUDGETS] + ["alc"]
    by = df.groupby("fold")[cols].mean()
    rng = np.random.default_rng(0)
    eps = [g for _, g in df.groupby(["fold", "episode"])]
    boot = [pd.concat([eps[i] for i in rng.integers(len(eps), size=len(eps))]).alc.mean() for _ in range(300)]
    return {"alc": df.alc.mean(), "alc_macro": by.alc.mean(), "ci95": list(np.percentile(boot, [2.5, 97.5])),
            "budgets": df[cols[:-1]].mean().to_dict(), "folds": by.round(4).to_dict("index"), "pairs": len(df),
            "max_call": df.max_call.max(), "episode_seconds": df.groupby(["fold", "episode"]).seconds.first().mean()}


def main():
    ap = argparse.ArgumentParser(description="leave-one-benchmark-out evaluation of a submission directory")
    ap.add_argument("model")
    ap.add_argument("--variant", type=json.loads, default=None, help="json written to variant.json before fit")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default=",".join(FOLDS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    model = Path(a.model).resolve()
    rid = "_".join(x for x in (time.strftime("%m%d-%H%M%S"), model.name, a.tag) if x)
    out = RUNS / rid
    out.mkdir(parents=True)
    git = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    (out / "config.json").write_text(json.dumps({**vars(a), "git": git, "run": rid}, indent=1))
    t0 = time.time()
    data, folds = load(), a.folds.split(",")
    G["data"], G["bench"] = data, {}
    G["subs"] = {f: fit(model, out / f"fold_{f}", data.without(f), a.variant) for f in folds}
    t1 = time.time()
    tasks = [(f, e, a.seed) for f in folds for e in range(a.episodes)]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
            rows = [r for rs in ex.map(_task, tasks) for r in rs]
    else:
        rows = [r for t in tasks for r in _task(t)]
    df = pd.DataFrame(rows)
    df.to_csv(out / "pairs.csv", index=False)
    s = {**summarize(df), "fit_seconds": t1 - t0, "total_seconds": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(rid)
    print(f"alc {s['alc']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] macro {s['alc_macro']:.4f} pairs {s['pairs']}")
    print(" ".join(f"{k} {v:.4f}" for k, v in s["budgets"].items()))
    print(pd.DataFrame(s["folds"]).T.to_string())


if __name__ == "__main__":
    main()
