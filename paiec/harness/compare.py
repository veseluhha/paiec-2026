import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import RUNS

COLS = ["b0", "b1", "b3", "b7", "b15", "b31", "alc"]


def _dir(r):
    return Path(r) if Path(r).is_dir() else RUNS / r


def compare(a, b):
    ca, cb = (json.loads((_dir(r) / "config.json").read_text()) for r in (a, b))
    diff = [k for k in ("seed", "episodes", "folds") if ca.get(k) != cb.get(k)]
    if diff:
        raise SystemExit(f"configs differ: {diff}")
    key = ["fold", "episode", "pair", "n_eval", "n_acq"]
    x = pd.read_csv(_dir(a) / "pairs.csv").merge(pd.read_csv(_dir(b) / "pairs.csv"), on=key)
    rows = []
    for name, g in [("all", x), *x.groupby("fold")]:
        r = {"fold": name, "n": len(g)}
        for c in COLS:
            d = (g[f"{c}_y"] - g[f"{c}_x"]).groupby([g.fold, g.episode]).mean()
            r[c] = d.mean()
            if c == "alc":
                r["se"] = d.std(ddof=1) / np.sqrt(len(d))
        rows.append(r)
    df = pd.DataFrame(rows).set_index("fold")
    f = df.alc.iloc[1:]
    df.loc["all", "se_folds"] = f.std(ddof=1) / np.sqrt(len(f))
    df.loc["all", "better_folds"] = int((f < 0).sum())
    return df


if __name__ == "__main__":
    print(f"{sys.argv[2]} minus {sys.argv[1]} (negative = better)")
    print(compare(sys.argv[1], sys.argv[2]).round(4).to_string())
