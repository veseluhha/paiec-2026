import numpy as np


def year(t):
    try:
        return int(t[:4]) + (int(t[5:7]) - 1) / 12
    except (TypeError, ValueError):
        return None


def design(s, t0):
    t = year(s.get("release_date", ""))
    return np.array([0.0, 1.0] if t is None else [t - t0, 0.0])


def score(s, P):
    z = design(s, P["t0"]) - P["c"]
    if year(s.get("release_date", "")) is None:
        z[:2] = 0.0
    return float(z @ P["beta"])


def prior(s, P):
    m = score(s, P)
    res = P["res_n"] if year(s.get("release_date", "")) is None else P["res_d"]
    return min(m, 0.0), res + max(m, 0.0) ** 2
