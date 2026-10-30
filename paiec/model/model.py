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
JUDGE = "judge" in P
CACHE = os.environ.get("PAIEC_JUDGE_CACHE")
RT = JUDGE and not CACHE
LM = (D / "models.txt").read_text().split()[0] if JUDGE else None
JD = P.get("judge")
XB, WB = hermegauss(24)
WB = WB / WB.sum()
B = math.sqrt(P["sb2"]) * XB
H = np.linspace(-20, 20, 801)
Q = {1: 24, 2: 16, 3: 10}
RHO, ELL, MED, NMIN, DIM, CHARS = 0.5, 0.5, 0.87, 4, 16384, 4000
TXT_EB, SRC_EB, SRC_K, SRC_R = [0, 0.5, 1], [0, 0.5, 1, 2], 4, P["src"]["r"]
TMPL, SEC, HEAD, TAIL, BUDGET = 3, 30, 1000, 2000, 6000
TASK = "Below is a task from an AI evaluation benchmark.\n\n{meta}<task>\n{t}\n</task>\n\n"
ASK = ("How hard is this task for a strong AI system (a frontier LLM or LLM agent) to solve correctly? "
       "Answer with a single digit from 1 (trivial, almost every system solves it) to 9 "
       "(extremely hard, almost no system solves it).")
C, E = {}, {}
J = sys.modules.setdefault("paiec_judge", types.ModuleType("paiec_judge")).__dict__
J.setdefault("s", {})
J.setdefault("t0", 0.0)
spec = importlib.util.spec_from_file_location("subj", D / "subj.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
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


def delta(s):
    m, v = S.prior(s, PP)
    return m * math.sqrt(P["sw2"]), v * P["sw2"]


def src(k):
    return k[1].split(";")[0]


def hashed(t):
    w = re.findall(r"\w+", (t[:CHARS] + " " + t[-CHARS:]).lower())
    x = np.zeros(DIM, np.float32)
    for g in w + [a + " " + b for a, b in zip(w, w[1:])]:
        x[zlib.crc32(g.encode()) % DIM] += 1
    return np.log1p(x)


def embs(texts):
    ks = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    for k, t in zip(ks, texts):
        if k not in E:
            E[k] = hashed(t)
    return np.stack([E[k] for k in ks])


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
    path = os.environ.get("PAIEC_JUDGE_PATH", LM)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    return tok, m.eval(), [tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)]


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
    b, mx = [], 0
    for j in np.argsort([len(e) for e in enc]):
        mx = max(mx, len(enc[j]))
        if b and mx * (len(b) + 1) > BUDGET:
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
    tok, m, dg = lm
    enc = [tok(jprompt(tok, c, f), add_special_tokens=False).input_ids for c, f in items]
    for b in batches(enc):
        if cap and time.time() - J["t0"] > SEC:
            J["cut"] = True
            return
        r = np.full((len(b), 1), np.nan)
        try:
            x = tok.pad({"input_ids": [enc[i] for i in b]}, return_tensors="pt").to(m.device)
            pos = (x.attention_mask.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                lo = m(**x, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1]
                pr = torch.softmax(lo.float(), -1)[:, dg].cpu().numpy()
            pr[:, 0] = 0
            r[:, 0] = (pr / pr.sum(1, keepdims=True)) @ np.arange(10)
        except Exception:
            r[:] = np.nan
        for i, v in zip(b, r):
            J["s"][jh(*items[i])] = v


def jneed(keys, cap):
    jrun(list(dict.fromkeys((k[0], k[1]) for k in keys if jh(k[0], k[1]) not in J["s"])), cap)


def jraw(k):
    if RT:
        v = J["s"].get(jh(k[0], k[1]))
    else:
        if "map" not in J:
            J["map"] = pickle.loads((Path(CACHE) / "g31.pkl").read_bytes())
        v = J["map"].get((k[0], k[1]))
    return np.full(1, np.nan) if v is None else np.asarray(v, float)


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
    tx = t.sum() >= NMIN
    sv = np.array([src(k) for k in keys])
    sx = SRC_R > 0 and len(set(sv[sv != ""])) >= SRC_K
    jx = None
    if JUDGE and keys:
        if RT and len(keys) >= NMIN:
            jneed(keys, True)
        R = np.stack([jraw(k) for k in keys])
        ok = ~np.isnan(R).any(-1)
        if ok.sum() >= NMIN:
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
        X = embs([k[0] for k in keys])
        idf = np.log((1 + t.sum()) / (1 + (X[t] > 0).sum(0))) + 1
        X = unit(X, idf)
        Dm = np.clip(1 - X @ X.T, 0, None)
        u = Dm[np.ix_(t, t)][np.triu_indices(t.sum(), 1)]
        u = u[u > 1e-6]
        G.update(X=X, idf=idf, sc=ELL * (np.median(u) if len(u) >= 3 else MED))
        E_ = np.exp(-Dm / G["sc"]) * np.outer(t, t)
    S_ = (sv[:, None] == sv[None, :]) & (sv[:, None] != "")
    fits = []
    for c, e, j in itertools.product(SRC_EB if sx else [0.0], TXT_EB if tx else [0.0], JD["eb"] if jx else [0.0]):
        su2 = c * SRC_R * P["sb2"]
        rv = e * RHO * (P["sb2"] - su2)
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
    c, f = item.get("item_content", ""), item.get("item_features", "")
    X = unit(embs([c]), G["idf"])[0] if G["X"] is not None and c not in tm else None
    sx = src((c, f))
    if RT and G["j"]:
        jneed([(c, f)], False)
    zx = jz(jraw((c, f)), *G["j"]) if G["j"] else 0.0
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
    w = np.exp(lp - lse(lp))
    m = w @ g
    return m, w @ (g - m) ** 2


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
    W = np.exp(lw - lw.max())
    return T[:, 0], W / W.sum(), M


def fit_all(rows, target, tm):
    subj = {target[0]: target[1]}
    obs = {}
    for (k, s), i, y in rows:
        subj.setdefault(k, s)
        obs.setdefault(i, []).append((k, y))
    shared = {k: sum(any(a == target[0] for a, _ in o) and any(a == k for a, _ in o) for o in obs.values())
              for k in subj}
    rest = sorted((k for k in subj if k != target[0]), key=lambda k: -shared[k])
    keep, out = [target[0]] + rest[:2], rest[2:]
    ix = {k: j for j, k in enumerate(keep)}
    ko = {i: [(ix[k], y) for k, y in o if k in ix] for i, o in obs.items()}
    ko = {i: o for i, o in ko.items() if o}
    oth = [(*delta(subj[k]), [y for o in obs.values() for a, y in o if a == k]) for k in out]
    dm = np.array([delta(subj[k]) for k in keep])
    m, t2 = level(oth, P["tau2"]) if oth else (P["mu0"], P["tau2"])
    T, W, M = post(ko, m + dm[:, 0], t2 * np.ones((len(keep), len(keep))) + np.diag(dm[:, 1]))
    return T, W, M, gp(W, M, tm)


def clean(d):
    return {k: "" if v is None else str(v) for k, v in (d or {}).items()}


def templates(lab):
    keys, per = {}, {}
    for k, i, _ in lab:
        keys.setdefault(i[0], set()).add(i)
        per[k, i] = per.get((k, i), 0) + 1
    cnt = {}
    for (k, i), n in per.items():
        cnt[k, i[0]] = cnt.get((k, i[0]), 0) + n
    tm = {c for c, ks in keys.items() if len(ks) >= TMPL} | {c for (_, c), n in cnt.items() if n >= TMPL}
    return tm, {i for (_, i), n in per.items() if n >= 2 and i[0] in tm}


def predict(input, labeled=None):
    subject, item = map(clean, input)
    labeled = [((clean(s), clean(i)), y) for (s, i), y in labeled or []]
    bid = item.get("benchmark_id", "")
    tk = skey(subject)
    lab = [(skey(s), s, ikey(i), y) for (s, i), y in labeled if i.get("benchmark_id", "") == bid]
    tm, sp = templates([(k, i, y) for k, _, i, y in lab])
    rows = [((k, s), i + (j,) if i in sp else i, y) for j, (k, s, i, y) in enumerate(lab)]
    key = (bid, tk, len(rows), hash(tuple((r[0][0], r[1], r[2]) for r in rows)))
    if key not in C or C[key][-1]:
        J["t0"], J["cut"] = time.time(), False
        C[key] = (*fit_all(rows, (tk, subject), tm), J["cut"])
    t, W, M, G, _ = C[key]
    m, L = M.get(ikey(item), (None, None))
    if m is not None:
        return float(np.clip(W @ (((m * sig(t[:, None] - B)) @ WB) / L), 0, 1))
    b, wb = bprior(G, item, tm)
    z = jlo(item) if JUDGE and not (G and G["j"]) else 0.0
    if z:
        b, wb = np.concatenate([b, b + JD["b0"] * z], -1), np.concatenate([wb, wb]) / 2
    return float(np.clip(W @ (sig(t[:, None] - b) @ wb), 0, 1))
