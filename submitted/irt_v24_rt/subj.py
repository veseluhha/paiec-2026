import math
import re

import numpy as np

EFF = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}
SMALL = re.compile(r"\b(mini|flash|lite|nano|small|haiku|tiny)\b")
LARGE = re.compile(r"\b(pro|max|opus|large|ultra|plus|heavy)\b")
SIZE = re.compile(r"\b(\d+(?:\.\d+)?)b\b")
NOTHINK = re.compile(r"no[-_ ]?think|non[-_ ]?think")
THINK = re.compile(r"think|reason")


def year(t):
    try:
        return int(t[:4]) + (int(t[5:7]) - 1) / 12
    except (TypeError, ValueError):
        return None


def meta(s):
    x = f"{s.get('normalized_name', '')} {s.get('subject_features_extra', '')}".lower()
    e = s.get("reasoning_effort", "").lower()
    m = SIZE.findall(x)
    think = -1.0 if NOTHINK.search(x) else float(bool(THINK.search(x) or e))
    return [think, (EFF[e] - 3) / 2 if e in EFF else 0.0, float(bool(SMALL.search(x))),
            float(bool(LARGE.search(x))), math.log10(float(m[-1])) - 2 if m else 0.0, float(bool(m))]


def design(s, V, P):
    t, cols = year(s.get("release_date", "")), []
    if V["t"] != "none":
        if t is None:
            cols += [0.0] * (2 if V["t"] == "quad" else 1) + [1.0]
        else:
            u = (math.floor(t) if V["t"] == "int" else t) - P["t0"]
            if V["t"] == "clip":
                u = min(max(u, P["lo"]), P["hi"])
            if V["t"] == "cap":
                u = min(u, 0.0)
            cols += [u, u * u, 0.0] if V["t"] == "quad" else [u, 0.0]
    if V["meta"]:
        cols += meta(s)
    return np.array(cols)


def score(s, V, P):
    z = design(s, V, P) - P["c"]
    if V["t"] != "none" and year(s.get("release_date", "")) is None:
        z[:3 if V["t"] == "quad" else 2] = 0.0
    if V["t"] != "none":
        z[:2 if V["t"] == "quad" else 1] *= V.get("k", 1.0)
    return float(z @ P["beta"]) + (P["prov"].get(s.get("provider", "").lower(), 0.0) if V["prov"] else 0.0)


def prior(s, others, V, P):
    nd = year(s.get("release_date", "")) is None
    res = P["res_n"] if nd and V["t"] != "none" else P["res_d"]
    if not V["rel"]:
        m = score(s, V, P)
        c = V.get("sat")
        if V.get("down"):
            f = V.get("floor")
            lo = f * math.tanh(min(m, 0.0) / f) if f else min(m, 0.0)
            up = P["res_n"] if V.get("pvar") == "n" and m > 0 else res + max(m, 0.0) ** 2
            return lo, up + (min(m, 0.0) - lo) ** 2
        return (c * math.tanh(m / c), res + (m - c * math.tanh(m / c)) ** 2) if c else (m, res)
    g = [score(o, V, P) for o in others]
    return score(s, V, P) - np.mean(g), res + (1 - res) / len(g)
