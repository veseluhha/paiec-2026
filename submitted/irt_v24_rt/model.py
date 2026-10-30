import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import pickle
import re
import sys
import time
import types
import zlib
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

D = Path(__file__).parent
P = json.loads((D / "params.json").read_text())
V = {"cross": True, "pool": True, "year": True, "others": False, **json.loads((D / "variant.json").read_text())}
XB, WB = hermegauss(24)
WB = WB / WB.sum()
B = math.sqrt(P["sb2"]) * XB
LAM = P.get("lam", [[1.0, 1.0]])
CAL = P.get("cal")
H = np.linspace(-20, 20, 801)
Q = {1: 24, 2: 16, 3: 10}
C = {}
OT = {"rho": 0.5, "ell": 0.5, "med": 0.87, "nmin": 4, "dim": 16384, "chars": 4000, **V.get("ot", {})}
E = {}
W4 = sys.modules.setdefault("paiec_w4", types.ModuleType("paiec_w4")).__dict__
SRC = {"r": P.get("src", {}).get("r", 0.0), "eb": [1.0], **V.get("src", {})}
JD = {"eb": [0.0, 0.03, 0.1], "b0": 0.0, "sec": 30, **P.get("judge", {}), **V.get("judge", {})}
RT = "judge" in V and not os.environ.get("W4_ITEMS")
LM = tuple(JD.get("lm", ("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct")))
HEAD, TAIL, BUDGET = 1000, 2000, 24000
TASK = "Below is a task from an AI evaluation benchmark.\n\n{meta}<task>\n{t}\n</task>\n\n"
ASK = ("How hard is this task for a strong AI system (a frontier LLM or LLM agent) to solve correctly? "
       "Answer with a single digit from 1 (trivial, almost every system solves it) to 9 "
       "(extremely hard, almost no system solves it).")
J = sys.modules.setdefault("paiec_judge", types.ModuleType("paiec_judge")).__dict__
J.setdefault("s", {})
J.setdefault("t0", 0.0)
if "prior" in V:
    spec = importlib.util.spec_from_file_location("subj", D / "subj.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    VP = {"t": "lin", "dm": True, "rel": False, "prov": False, "meta": False, **V["prior"]}
    PP = {**P["prior"], "beta": np.array(P["prior"]["beta"]), "c": np.array(P["prior"]["c"])}


def sig(z):
    return 1 / (1 + np.exp(-z))


def lse(x):
    m = x.max()
    return m + math.log(np.exp(x - m).sum())


def skey(s):
    return tuple(sorted(s.items()))


def ikey(i):
    return i.get("item_content", ""), i.get("item_features", ""), i.get("benchmark_id", "")


def delta(s, others=()):
    if "prior" in V:
        m, v = S.prior(s, others, VP, PP)
        return m * math.sqrt(P["sw2"]), v * P["sw2"]
    if V["year"]:
        try:
            t = int(s["release_date"][:4]) + (int(s["release_date"][5:7]) - 1) / 12 - P["year"]["t0"]
            return P["year"]["slope"] * t * math.sqrt(P["sw2"]), P["year"]["res"] * P["sw2"]
        except (KeyError, TypeError, ValueError):
            pass
    return 0.0, P["sw2"]


def wmap(name):
    if name not in W4:
        W4[name] = pickle.loads((Path(os.environ["W4_ITEMS"]) / name).read_bytes())
    return W4[name]


def src(k):
    if "perm" in SRC:
        return wmap(SRC["perm"]).get((k[0], k[1]), "")
    return k[1].split(";")[0]


def txt(c, f):
    return wmap(OT["perm"]).get((c, f), c) if "perm" in OT else c


def hashed(t):
    w = re.findall(r"\w+" if OT.get("w") else r"[a-z0-9_]+", (t[:OT["chars"]] + " " + t[-OT["chars"]:]).lower())
    x = np.zeros(OT["dim"], np.float32)
    for g in w + [a + " " + b for a, b in zip(w, w[1:])]:
        x[zlib.crc32(g.encode()) % OT["dim"]] += 1
    return np.log1p(x)


def embs(texts):
    ks = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    for k, t in zip(ks, texts):
        if k not in E:
            E[k] = hashed(t)
    return np.stack([E[k] for k in ks])


def hid(keys):
    z = np.zeros(OT["hdim"], np.float32)
    if RT:
        return np.stack([J.get("h", {}).get(jh(c, f), z) for c, f in keys])
    mp = wmap(OT["hid"])
    return np.stack([np.asarray(mp.get((c, f), z), np.float32) for c, f in keys])


def unit(X, idf):
    X = X * idf
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def jh(c, f):
    return hashlib.sha256(f"{c}\0{f}".encode()).hexdigest()


def jopen():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        return None
    paths, lm = json.loads(os.environ.get("PAIEC_JUDGE_PATHS", "{}")), []
    for name in LM[:JD.get("m", 1)]:
        tok = AutoTokenizer.from_pretrained(paths.get(name, name))
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(paths.get(name, name), dtype=torch.bfloat16, device_map="cuda")
        lm.append((tok, m.eval(), [tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)]))
        if len(lm) == 1 and "hid" in OT:
            L = max((x for n, x in m.named_modules() if isinstance(x, torch.nn.ModuleList) and n.endswith("layers")), key=len)
            L[round(0.75 * len(L)) - 1].register_forward_hook(
                lambda mod, a, o: J.__setitem__("hk", (o[0] if isinstance(o, tuple) else o)[:, -1].float()))
            J["R"] = np.random.default_rng(0).standard_normal((m.config.get_text_config().hidden_size, OT["hdim"])).astype(np.float32) / np.sqrt(OT["hdim"])
    return lm


def jload():
    if "lm" not in J:
        try:
            J["lm"] = jopen()
        except Exception:
            J["lm"] = None
        if J["lm"] is None and "torch" in sys.modules:
            gc.collect()
            sys.modules["torch"].cuda.empty_cache()
    return J["lm"]


def jprompt(tok, c, f):
    ids = tok.encode(c, add_special_tokens=False)
    t = c if len(ids) <= HEAD + TAIL else tok.decode(ids[:HEAD]) + "\n[...]\n" + tok.decode(ids[-TAIL:])
    meta = f"Metadata: {f}\n\n" if f else ""
    msg = [{"role": "user", "content": TASK.format(meta=meta, t=t) + ASK}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def batches(enc):
    order = np.argsort([len(e) for e in enc])
    b, mx = [], 0
    for j in order:
        mx = max(mx, len(enc[j]))
        if b and mx * (len(b) + 1) > JD.get("budget", BUDGET):
            yield b
            b, mx = [], len(enc[j])
        b.append(j)
    if b:
        yield b


def jrun(items, cap):
    lm = jload()
    if not lm or not items:
        return
    import torch
    enc = [[tok(jprompt(tok, c, f), add_special_tokens=False).input_ids for c, f in items] for tok, _, _ in lm]
    for b in batches(enc[0]):
        if cap and time.time() - J["t0"] > JD["sec"]:
            J["cut"] = True
            return
        r = np.full((len(b), len(lm)), np.nan)
        hv = np.zeros((len(b), OT.get("hdim", 1)), np.float32)
        try:
            for j, (tok, m, dg) in enumerate(lm):
                x = tok.pad({"input_ids": [enc[j][i] for i in b]}, return_tensors="pt").to(m.device)
                pos = (x.attention_mask.cumsum(-1) - 1).clamp(min=0)
                with torch.no_grad():
                    lo = m(**x, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1]
                    pr = torch.softmax(lo.float(), -1)[:, dg].cpu().numpy()
                    if j == 0 and "hid" in OT:
                        hv[:] = J["hk"].cpu().numpy() @ J["R"]
                pr[:, 0] = 0
                r[:, j] = (pr / pr.sum(1, keepdims=True)) @ np.arange(10)
        except Exception:
            r[:] = np.nan
        for i, v, w in zip(b, r, hv):
            J["s"][jh(*items[i])] = v
            if "hid" in OT and np.isfinite(v).all():
                J.setdefault("h", {})[jh(*items[i])] = w


def jneed(keys, cap):
    jrun(list(dict.fromkeys((k[0], k[1]) for k in keys if jh(k[0], k[1]) not in J["s"])), cap)


def jraw(k):
    if RT:
        return J["s"].get(jh(k[0], k[1]), np.full(JD.get("m", 1), np.nan))
    v = wmap(JD["map"]).get((k[0], k[1]))
    return np.full(JD.get("m", 1), np.nan) if v is None else np.asarray(v, float)


def jz(R, mu, sd, s2):
    z = np.nanmean((R - mu) / sd, -1) / s2
    return np.clip(np.nan_to_num(z), -3, 3)


def jlo(item):
    k = (item.get("item_content", ""), item.get("item_features", ""))
    if RT:
        jneed([k], False)
    return float(np.clip(np.nan_to_num(np.nanmean((jraw(k) - np.array(JD["gmu"])) / np.array(JD["gsd"]))), -3, 3))


def kern(G, X, sx, su2, rv, s2=0.0, z=0.0):
    k = su2 * ((G["src"] == sx) & (sx != ""))
    if X is not None:
        k = k + rv * G["t"] * np.exp(-np.clip(1 - G["X"] @ X, 0, None) / G["sc"])
    if s2:
        k = k + s2 * G["z"] * z
    return k


def gp(W, M, tm):
    keys = list(M)
    t = np.array([k[0] not in tm for k in keys])
    tx = "ot" in V and t.sum() >= OT["nmin"]
    sv = np.array([src(k) for k in keys])
    sx = "src" in V and SRC["r"] > 0 and len(set(sv[sv != ""])) >= max(SRC.get("kmin", 1), 1)
    jx = None
    if "judge" in V and keys:
        if RT and len(keys) >= OT["nmin"]:
            jneed(keys, True)
        R = np.stack([jraw(k) for k in keys])
        ok = ~np.isnan(R).any(-1)
        if ok.sum() >= OT["nmin"]:
            mu, sd = R[ok].mean(0), R[ok].std(0) + 1e-9
            s2 = np.nanmean((R[ok] - mu) / sd, -1).std() + 1e-9
            jx = (mu, sd, s2)
    if not keys or not (tx or sx or jx):
        return None
    mb, lam = [], []
    for k in keys:
        m, L = M[k]
        q = (W / L) @ m * WB
        mu = q @ B
        v = max(q @ (B - mu) ** 2, 1e-9)
        lam.append(max(1 / v - 1 / P["sb2"], 1e-9))
        mb.append(mu / v / lam[-1])
    mb, lam = np.array(mb), np.array(lam)
    G = {"src": sv, "t": t, "X": None, "sc": 1.0, "idf": None, "j": jx, "z": jz(R, *jx) if jx else None}
    E_ = np.zeros((len(keys), len(keys)))
    if tx:
        if "hid" in OT:
            X = hid([k[:2] for k in keys])
            t = t & (np.abs(X).sum(1) > 0)
            G["t"] = t
            mu = X[t].mean(0) if t.any() else X.mean(0)
            idf, X = mu, unit(X - mu, 1.0)
        else:
            X = embs([txt(*k[:2]) for k in keys])
            idf = np.log((1 + t.sum()) / (1 + (X[t] > 0).sum(0))) + 1
            X = unit(X, idf)
        Dm = np.clip(1 - X @ X.T, 0, None)
        u = Dm[np.ix_(t, t)][np.triu_indices(t.sum(), 1)]
        u = u[u > 1e-6]
        G.update(X=X, idf=idf, sc=OT["ell"] * (np.median(u) if len(u) >= 3 else OT["med"]))
        E_ = np.exp(-Dm / G["sc"]) * np.outer(t, t)
    S_ = (sv[:, None] == sv[None, :]) & (sv[:, None] != "")
    fits = []
    for c, e, j in itertools.product(SRC["eb"] if sx else [0.0], OT.get("eb", [1.0]) if tx else [0.0],
                                     JD["eb"] if jx else [0.0]):
        su2 = c * SRC["r"] * P["sb2"]
        rv = e * OT["rho"] * (P["sb2"] - su2)
        s2 = j * P["sb2"]
        Cv = su2 * S_ + rv * E_
        nug = P["sb2"] - np.diag(Cv)
        y = mb
        if jx:
            Cv = Cv + s2 * np.outer(G["z"], G["z"])
            nug = np.maximum(P["sb2"] - np.diag(Cv), 0)
            y = mb - JD["b0"] * (j > 0) * G["z"]
        A = Cv + np.diag(1 / lam + nug)
        L = np.linalg.cholesky(A)
        z = np.linalg.solve(L, y)
        fits.append((su2, rv, s2, L, np.linalg.solve(L.T, z), -0.5 * z @ z - np.log(np.diag(L)).sum()))
    ev = np.array([f[5] for f in fits])
    G["fits"] = [(f[:5], w) for f, w in zip(fits, np.exp(ev - ev.max()) / np.exp(ev - ev.max()).sum())]
    return G


def bprior(G, item, tm):
    if G is None:
        return B, WB
    c = item.get("item_content", "")
    tx = c not in tm
    f = item.get("item_features", "")
    if RT and "hid" in OT and G["X"] is not None and tx:
        jneed([(c, f)], False)
    hq = hid([(c, f)]) if "hid" in OT and G["X"] is not None and tx else None
    X = None if G["X"] is None or not tx or (hq is not None and not np.abs(hq).sum()) else (unit(hq - G["idf"], 1.0)[0] if "hid" in OT else unit(embs([txt(c, f)]), G["idf"])[0])
    sx = src((c, item.get("item_features", "")))
    if RT and G["j"]:
        jneed([(c, item.get("item_features", ""))], False)
    zx = jz(jraw((c, item.get("item_features", ""))), *G["j"]) if G["j"] else 0.0
    bs, ws = [], []
    for (su2, rv, s2, L, al), w in G["fits"]:
        k = kern(G, X, sx, su2, rv, s2, zx)
        v = np.linalg.solve(L, k)
        kxx = su2 * (sx != "") + (rv if X is not None else 0.0)
        if G["j"]:
            kxx += s2 * zx * zx
            bs.append(k @ al + JD["b0"] * (s2 > 0) * zx + math.sqrt(max(kxx - v @ v, 0) + max(P["sb2"] - kxx, 0)) * XB)
        else:
            bs.append(k @ al + math.sqrt(max(kxx - v @ v, 0) + P["sb2"] - kxx) * XB)
        ws.append(w * WB)
    return np.concatenate(bs)[None, :], np.concatenate(ws)


def loglik(g, ys):
    p = np.clip(sig(g[:, None] - B) @ WB, 1e-12, 1 - 1e-12)
    k = sum(ys)
    return k * np.log(p) + (len(ys) - k) * np.log(1 - p)


def level(others, t2):
    g = P["mu0"] + math.sqrt(t2) * np.linspace(-7, 7, 281)
    lp = -0.5 * (g - P["mu0"]) ** 2 / t2 - 0.5 * math.log(2 * math.pi * t2) + math.log(g[1] - g[0])
    dh = H[1] - H[0]
    for d, v, ys in others:
        ll = loglik(H, ys)
        k = -0.5 * (H[None, :] - g[:, None] - d) ** 2 / v - 0.5 * math.log(2 * math.pi * v) + math.log(dh)
        z = k + ll
        mz = z.max(1, keepdims=True)
        lp = lp + (mz[:, 0] + np.log(np.exp(z - mz).sum(1)))
    lz = lse(lp)
    w = np.exp(lp - lz)
    m = w @ g
    return m, w @ (g - m) ** 2, lz


def post(obs, mu, cov):
    K = len(mu)
    m1, s1 = np.zeros(K), np.zeros(K)
    for j in range(K):
        g = mu[j] + math.sqrt(cov[j, j]) * np.linspace(-6, 6, 400)
        lp = -0.5 * (g - mu[j]) ** 2 / cov[j, j] + loglik(g, [y for o in obs.values() for k, y in o if k == j])
        w = np.exp(lp - lp.max())
        w /= w.sum()
        m1[j] = w @ g
        s1[j] = math.sqrt(w @ (g - m1[j]) ** 2) * 1.2
    x, wq = hermegauss(Q[K])
    X = np.array(list(itertools.product(x, repeat=K)))
    T = m1 + s1 * X
    lw = np.array([sum(np.log(wq[a]) for a in c) for c in itertools.product(range(Q[K]), repeat=K)])
    lw += 0.5 * (X ** 2).sum(1)
    R = T - mu
    lw -= 0.5 * np.einsum("nk,kl,nl->n", R, np.linalg.inv(cov), R)
    M = {}
    for i, o in obs.items():
        m = np.ones((len(T), len(B)))
        for k, y in o:
            p = sig(T[:, k, None] - B)
            m *= p if y else 1 - p
        L = np.clip(m @ WB, 1e-300, None)
        M[i] = (m, L)
        lw += np.log(L)
    lz = lse(lw) + np.log(s1).sum() - 0.5 * np.linalg.slogdet(2 * math.pi * cov)[1]
    W = np.exp(lw - lw.max())
    return T[:, 0], W / W.sum(), M, lz


def fit_all(rows, target, tm):
    subj = {target[0]: target[1]}
    obs = {}
    for (k, s), i, y in rows:
        subj.setdefault(k, s)
        obs.setdefault(i, []).append((k, y))
    shared = {k: sum(any(a == target[0] for a, _ in o) and any(a == k for a, _ in o) for o in obs.values())
              for k in subj}
    rest = sorted((k for k in subj if k != target[0]), key=lambda k: -shared[k])
    keep, out = [target[0]] + rest[:2], rest[2:] if V["others"] and V["pool"] else []
    ix = {k: j for j, k in enumerate(keep)}
    ko = {i: [(ix[k], y) for k, y in o if k in ix] for i, o in obs.items()}
    ko = {i: o for i, o in ko.items() if o}
    ss = list(subj.values())
    oth = [(*delta(subj[k], ss), [y for o in obs.values() for a, y in o if a == k]) for k in out]
    dm = np.array([delta(subj[k], ss) for k in keep])
    K = len(keep)
    res = []
    for s, w in LAM:
        m, t2, lz = level(oth, P["tau2"] * s) if oth else (P["mu0"], P["tau2"] * s, 0.0)
        cov = t2 * (np.ones((K, K)) if V["pool"] else np.eye(K)) + np.diag(dm[:, 1])
        r = post(ko, m + dm[:, 0], cov)
        res.append(((*r[:3], gp(*r[1:3], tm)), math.log(w) + lz + r[3]))
    lw = np.array([x[1] for x in res])
    om = np.exp(lw - lw.max())
    n = sum(1 for (k, _), _, _ in rows if k == target[0])
    return [x[0] for x in res], om / om.sum(), n


def calibrate(p, n):
    if not CAL:
        return p
    j = max(k for k, b in enumerate(CAL["n"]) if b <= n)
    a, c = CAL["a"][j], CAL["c"][j]
    if CAL["kind"] == "shrink":
        return 0.5 + a * (p - 0.5)
    z = math.log(max(p, 1e-9) / max(1 - p, 1e-9))
    return 1 / (1 + math.exp(-(a * z + c)))


def clean(d):
    return {k: "" if v is None else str(v) for k, v in (d or {}).items()}


def remap(i):
    c, f = wmap(V["stress"]).get((i.get("item_content", ""), i.get("item_features", "")), (None, None))
    return i if c is None else {**i, "item_content": c, "item_features": f}


def templates(lab):
    if not V.get("tmpl"):
        return set(), set()
    keys, per = {}, {}
    for k, i, _ in lab:
        keys.setdefault(i[0], set()).add(i)
        per[k, i] = per.get((k, i), 0) + 1
    cnt = {}
    for (k, i), n in per.items():
        cnt[k, i[0]] = cnt.get((k, i[0]), 0) + n
    tm = {c for c, ks in keys.items() if len(ks) >= V["tmpl"]} | {c for (_, c), n in cnt.items() if n >= V["tmpl"]}
    return tm, {i for (_, i), n in per.items() if n >= 2 and i[0] in tm} if V.get("split") else set()


def predict(input, labeled=None):
    subject, item = map(clean, input)
    labeled = [((clean(s), clean(i)), y) for (s, i), y in labeled or []]
    if V.get("stress"):
        item = remap(item)
        labeled = [((s, remap(i)), y) for (s, i), y in labeled]
    bid = item.get("benchmark_id", "")
    tk = skey(subject)
    lab = [(skey(s), s, ikey(i), y) for (s, i), y in labeled if i.get("benchmark_id", "") == bid]
    if not V["cross"]:
        lab = [r for r in lab if r[0] == tk]
    tm, sp = templates([(k, i, y) for k, _, i, y in lab])
    rows = [((k, s), i + (j,) if i in sp else i, y) for j, (k, s, i, y) in enumerate(lab)]
    key = (bid, tk, len(rows), hash(tuple((r[0][0], r[1], r[2]) for r in rows)))
    if key not in C or C[key][3]:
        J["t0"], J["cut"] = time.time(), False
        C[key] = (*fit_all(rows, (tk, subject), tm), J["cut"])
    res, om, n, _ = C[key]
    p = 0.0
    for (t, W, M, G), o in zip(res, om):
        m, L = M.get(ikey(item), (None, None))
        if m is None:
            b, wb = bprior(G, item, tm)
            z = jlo(item) if "lo" in JD and not (G and G["j"]) else 0.0
            if z:
                b, wb = np.concatenate([b, b + JD["b0"] * z], -1), np.concatenate([wb, wb]) / 2
            p += o * (W @ (sig(t[:, None] - b) @ wb))
        else:
            p += o * (W @ (((m * sig(t[:, None] - B)) @ WB) / L))
    return float(np.clip(calibrate(float(p), n), 0, 1))
