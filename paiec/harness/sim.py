import copy
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SUBJECT_KEYS

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
MAX_LABELS = 31
MIN_ITEMS = 80
EVAL_SIZES = {44: 144, 50: 13, 53: 23, 54: 38, 56: 47, 57: 3, 58: 58, 60: 66, 68: 4, 71: 11, 72: 1, 73: 1,
              74: 9, 75: 13, 76: 11, 113: 12}
PAIRS_PER_BENCH = {1: 274, 2: 90}


def _draw(rng, dist):
    k = np.array(list(dist))
    w = np.array(list(dist.values()), float)
    return int(rng.choice(k, p=w / w.sum()))


def _anon(prefix, key):
    return f"{prefix}_{int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 900000 + 100000}"


class Bench:
    def __init__(self, data, bench):
        r = data.responses[data.responses.bench == bench]
        it = data.items[data.items.bench == bench].set_index("item_id")
        sb = data.subjects[data.subjects.bench == bench].set_index("subject_id")
        self.name = bench
        self.items = {i: {"item_content": row.item_content, "item_features": row.item_features,
                          "benchmark_id": row.benchmark_id} for i, row in it.iterrows()}
        self.subjects = {s: {k: row[k] for k in SUBJECT_KEYS} for s, row in sb.iterrows()}
        self.ys, self.inter = {}, {}
        codes, keys = pd.MultiIndex.from_arrays([r.subject_id.to_numpy(), r.item_id.to_numpy()]).factorize()
        o = np.argsort(codes, kind="stable")
        y, inter = r.y.to_numpy(), r.interactors.to_numpy()
        for (s, i), k in zip(keys, np.split(o, np.flatnonzero(np.diff(codes[o])) + 1)):
            self.ys.setdefault(s, {})[i] = y[k]
            self.inter[s, i] = inter[k[0]]
        self.order = sorted(self.items)
        self.eligible = sorted(s for s, d in self.ys.items() if len(d) >= MIN_ITEMS)

    def item(self, s, i):
        return {**self.items[i], "interactors": self.inter[s, i]}


@dataclass
class Pair:
    sid: str
    subject: dict
    acq: list
    ev: list
    acq_ids: list
    ev_ids: list


def episode(bench, rng, n_pairs=None, size=None):
    n_pairs = min(n_pairs or _draw(rng, PAIRS_PER_BENCH), len(bench.eligible))
    m = size or 2 * _draw(rng, EVAL_SIZES)
    rank = {i: k for k, i in enumerate(rng.permutation(bench.order))}
    pairs = []
    for s in rng.choice(bench.eligible, n_pairs, replace=False):
        own = sorted(bench.ys[s], key=rank.__getitem__)[:m]
        own = [own[k] for k in rng.permutation(len(own))]
        h = len(own) // 2
        acq = [(bench.item(s, i), float(rng.choice(bench.ys[s][i]))) for i in own[:h]]
        ev = [(bench.item(s, i), bench.ys[s][i]) for i in own[h:]]
        pairs.append(Pair(_anon("subject", f"{bench.name}/{s}"), bench.subjects[s], acq, ev, own[:h], own[h:]))
    return pairs


class Submission:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def _load(self, name):
        f = self.path / f"{name}.py"
        if not f.exists():
            return None
        spec = importlib.util.spec_from_file_location(f"sub_{name}_{time.monotonic_ns()}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def fresh(self):
        root = str(self.path)
        for k, m in list(sys.modules.items()):
            if (getattr(m, "__file__", None) or "").startswith(root):
                del sys.modules[k]
        if root not in sys.path:
            sys.path.insert(0, root)
        model, labeling = self._load("model"), self._load("labeling")
        return model.predict, getattr(labeling, "acquisition_function", None) if labeling else None


def _hook_mode(fn):
    if fn is None:
        return None, False
    sig = inspect.signature(fn)
    ps = sig.parameters
    names = ("prediction", "labeled", "context")
    kw = any(p.kind == p.VAR_KEYWORD for p in ps.values())
    keys = tuple(n for n in names if kw or (n in ps and ps[n].kind != inspect.Parameter.POSITIONAL_ONLY))
    shapes = [(False, names), (True, ())] + ([(False, keys)] if keys and keys != names else [])
    for positional, supplied in shapes:
        try:
            sig.bind(None, None, None, None) if positional else sig.bind(None, **dict.fromkeys(supplied))
        except TypeError:
            continue
        return (positional, supplied), positional or "prediction" in supplied
    raise ValueError("legacy_rank hooks are not simulated")


def _call_hook(hook, mode, inp, pred, labeled, ctx):
    positional, supplied = mode
    if positional:
        return hook(inp, pred, labeled, ctx)
    vals = {"prediction": pred, "labeled": labeled, "context": ctx}
    return hook(inp, **{k: vals[k] for k in supplied})


def _prob(v):
    if isinstance(v, bool):
        raise ValueError("predict returned bool")
    p = float(v)
    if not math.isfinite(p) or p < 0 or p > 1:
        raise ValueError(f"invalid probability {v!r}")
    return p


def _random_query(context, inp):
    key = json.dumps([context, inp], sort_keys=True, separators=(",", ":"))
    u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
    return u < min(1, context["labels_remaining"] / context["items_remaining"])


def run(sub, pairs, rng):
    predict, hook = sub.fresh()
    mode, needs_pred = _hook_mode(hook)
    labeled, pos, got = [], [0] * len(pairs), [0] * len(pairs)
    out = np.zeros((len(pairs), len(BUDGETS)))
    order = rng.permutation(len(pairs))
    stats = {"acq_calls": 0, "eval_calls": 0, "max_call": 0.0}
    t0 = time.perf_counter()
    for b, budget in enumerate(BUDGETS):
        for k in order:
            p = pairs[k]
            while got[k] < budget and pos[k] < len(p.acq):
                item, y = p.acq[pos[k]]
                inp = [dict(p.subject), dict(item)]
                ctx = {"subject_id": p.sid, "benchmark_id": item["benchmark_id"], "labels_acquired": got[k],
                       "labels_remaining": MAX_LABELS - got[k], "max_labels": MAX_LABELS,
                       "items_remaining": len(p.acq) - pos[k]}
                pos[k] += 1
                if mode is None:
                    q = _random_query(ctx, inp)
                else:
                    pred = _prob(predict(copy.deepcopy(inp), copy.deepcopy(labeled))) if needs_pred else None
                    q = _call_hook(hook, mode, copy.deepcopy(inp), pred, copy.deepcopy(labeled), dict(ctx))
                    if type(q) is not bool:
                        raise ValueError("acquisition_function must return bool")
                    stats["acq_calls"] += 1
                if q:
                    labeled.append([[dict(p.subject), dict(item)], int(y)])
                    got[k] += 1
        ev_predict, cache = sub.fresh()[0], {}
        for k, p in enumerate(pairs):
            se, n = 0.0, 0
            for item, ys in p.ev:
                inp = [dict(p.subject), dict(item)]
                key = json.dumps(inp, sort_keys=True)
                if key not in cache:
                    c = time.perf_counter()
                    cache[key] = _prob(ev_predict(inp, copy.deepcopy(labeled)))
                    stats["max_call"] = max(stats["max_call"], time.perf_counter() - c)
                    stats["eval_calls"] += 1
                q = cache[key]
                se += float(((q - ys) ** 2).sum())
                n += len(ys)
            out[k, b] = se / n
    stats["seconds"] = time.perf_counter() - t0
    return out, got, stats
