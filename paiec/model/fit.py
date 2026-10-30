import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

LM = "google/gemma-4-31B-it"
MIN_ITEMS, MIN_SUBJECTS = 80, 5
KEYS = ["normalized_name", "provider", "release_date", "access_date", "harness", "harness_version",
        "reasoning_effort", "subject_features_extra"]
spec = importlib.util.spec_from_file_location("subj_fit", Path(__file__).parent / "subj.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def rasch(s, i, y, ns, ni, it=25):
    sw2, sb2, x = 1.0, 1.0, np.zeros(1 + ns + ni)

    def f(x):
        d, b = x[1:ns + 1], x[ns + 1:]
        z = x[0] + d[s] - b[i]
        r = expit(z) - y
        nll = np.sum(np.logaddexp(0, z) - y * z) + d @ d / 2 / sw2 + b @ b / 2 / sb2
        return nll, np.concatenate([[r.sum()], np.bincount(s, r, ns) + d / sw2, -np.bincount(i, r, ni) + b / sb2])

    for _ in range(it):
        x = minimize(f, x, jac=True, method="L-BFGS-B").x
        d, b = x[1:ns + 1], x[ns + 1:]
        p = expit(x[0] + d[s] - b[i])
        w = p * (1 - p)
        hd, hb = np.bincount(s, w, ns) + 1 / sw2, np.bincount(i, w, ni) + 1 / sb2
        sw2, sb2 = np.mean(d ** 2 + 1 / hd), np.mean(b ** 2 + 1 / hb)
    return x[0] + d, 1 / hd, b, sb2


def bench_fit(g):
    c = g.groupby(["subject_id", "item_id"]).y.mean().reset_index()
    n = c.groupby("subject_id").size()
    el = n.index[n >= MIN_ITEMS]
    if len(el) < MIN_SUBJECTS:
        return None
    sc, ic = c.subject_id.astype("category"), c.item_id.astype("category")
    th, vt, b, sb2 = rasch(sc.cat.codes.to_numpy(), ic.cat.codes.to_numpy(), c.y.to_numpy(),
                           len(sc.cat.categories), len(ic.cat.categories))
    k = sc.cat.categories.get_indexer(el)
    mu = th[k].mean()
    stats = {"mu": mu, "sw2": np.mean((th[k] - mu) ** 2 + vt[k]), "sb2": sb2}
    return stats, dict(zip(el, th[k])), dict(zip(ic.cat.categories, b))


def regress(rows, t0):
    X = np.array([S.design(r["s"], t0) for r in rows]).reshape(len(rows), -1)
    z = np.array([r["z"] for r in rows])
    bs = np.array([r["bench"] for r in rows])
    c = X.mean(0)
    for b in np.unique(bs):
        X[bs == b] -= X[bs == b].mean(0)
        z = np.where(bs == b, z - z[bs == b].mean(), z)
    return {"t0": t0, "c": c, "beta": np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ z)}


def prior_params(rows):
    ts = np.array([t for r in rows if (t := S.year(r["s"]["release_date"])) is not None])
    t0 = float(np.median(ts))
    e, nd = [], []
    for b in sorted({r["bench"] for r in rows}):
        Q = regress([r for r in rows if r["bench"] != b], t0)
        hold = [r for r in rows if r["bench"] == b]
        z = np.array([r["z"] for r in hold])
        f = np.array([S.score(r["s"], Q) for r in hold])
        e += list((z - z.mean() - f + f.mean()) ** 2)
        nd += [S.year(r["s"]["release_date"]) is None for r in hold]
    e, nd = np.array(e), np.array(nd)
    P = regress(rows, t0)
    return {"t0": t0, "c": P["c"].tolist(), "beta": P["beta"].tolist(),
            "res_d": float(np.mean(e[~nd] if (~nd).any() else e)), "res_n": float(np.mean(e[nd])) if nd.any() else 1.0}


def src_params(data, bs):
    f = data.items.set_index(["bench", "item_id"]).item_features.fillna("").str.split(";").str[0]
    out = []
    for bench, bi in bs.items():
        s, b = f.loc[bench].reindex(list(bi)).fillna("").to_numpy(), np.array(list(bi.values()))
        s, b = s[s != ""], b[s != ""]
        ks, n = np.unique(s, return_counts=True)
        if len(ks) < 2:
            continue
        m = np.array([b[s == k].mean() for k in ks])
        se2 = sum(((b[s == k] - mk) ** 2).sum() for k, mk in zip(ks, m)) / (len(b) - len(ks))
        su2 = max(0.0, (n @ (m - b.mean()) ** 2 - (len(ks) - 1) * se2) / (len(b) - n @ n / len(b)))
        if np.isfinite(su2 / (su2 + se2)):
            out.append(su2 / (su2 + se2))
    return {"r": float(np.mean(out)) if out else 0.0}


def judge_params(bs, sb2):
    sc = pd.read_parquet(Path(os.environ["PAIEC_JUDGE_CACHE"]) / "scores.parquet").groupby(["bench", "item_id"])[["gemma4_31b"]].mean()
    a, se2 = [], []
    for bench, bi in bs.items():
        bi = pd.Series(bi)
        if bench not in sc.index.get_level_values(0):
            continue
        r = sc.loc[bench].reindex(bi.index).dropna()
        if len(r) < 20:
            continue
        z = ((r - r.mean()) / r.std()).mean(1)
        z = (z - z.mean()) / z.std()
        b = bi.loc[r.index].to_numpy()
        k = np.polyfit(z, b, 1)
        a.append(k[0])
        se2.append(np.var(b - np.polyval(k, z)) / len(b))
    s2 = (np.var(a, ddof=1) if len(a) > 1 else 0.0) + np.mean(se2)
    g = sc[sc.index.get_level_values(0).isin(list(bs))].to_numpy(float)
    g = g[~np.isnan(g).any(1)]
    return {"gmu": g.mean(0).tolist(), "gsd": g.std(0).tolist(), "b0": float(np.mean(a)), "eb": [0.0, float(s2 / sb2)]}


def params(data, judge):
    sub = data.subjects.set_index(["bench", "subject_id"])
    stats, bs, rows, extra = [], {}, [], []
    for bench, g in data.responses.groupby("bench"):
        r = bench_fit(g)
        if r is None:
            continue
        st, th, b = r
        if not g.binary.iloc[0]:
            extra.append(st["mu"])
            continue
        stats.append(st)
        bs[bench] = b
        for s, t in th.items():
            m = sub.loc[(bench, s)]
            rows.append({"bench": bench, "s": {k: m[k] for k in KEYS}, "z": (t - st["mu"]) / np.sqrt(st["sw2"])})
    mus = np.array([v["mu"] for v in stats] + extra)
    P = {"mu0": 0.0, "tau2": np.mean(mus ** 2), "sw2": np.mean([v["sw2"] for v in stats]),
         "sb2": np.mean([v["sb2"] for v in stats]), "prior": prior_params(rows), "src": src_params(data, bs)}
    if judge:
        P["judge"] = judge_params(bs, P["sb2"])
    return P


def fit(data, out):
    v = out / "variant.json"
    judge = json.loads(v.read_text()).get("judge", False) if v.exists() else False
    (out / "params.json").write_text(json.dumps(params(data, judge), indent=1, default=float))
    if judge:
        (out / "models.txt").write_text(LM + "\n")
