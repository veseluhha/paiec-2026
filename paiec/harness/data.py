import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("PAIEC_DATA_DIR", REPO / "data"))
RUNS = Path(os.environ.get("PAIEC_RUNS_DIR", REPO / "runs"))
REV = "54074453da7b9911be0f48e4c560be547bc84d21"
RAW = DATA / "measurement-db" / REV
CACHE = DATA / "cache" / REV[:8]
SUBJECT_KEYS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                "harness_version", "reasoning_effort", "subject_features_extra"]
FOLDS = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "agentdojo", "agi_elo", "algotune",
         "appworld", "benger", "dtap_bench", "editbench", "engdesign", "helm_bbq", "helm_thaiexam", "hle",
         "icpc2_code_selector", "ikp", "kernelbench", "llms_lean_formalization", "mathconstraint", "medhelm_med_qa",
         "medhelm_medcalc_bench", "mmlupro", "naturebench", "openbiorq", "perfcodebench", "robust_reasoning_benchmark",
         "software_world_models", "swebench_live", "swepolybench", "tau2_bench", "workbench_revisited"]
AUX = ["swe_rebench", "mmdocrag"]
TABLES = ("benchmarks", "subjects", "items", "responses")


def anon(name):
    return "benchmark_" + str(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 900000 + 100000)


def _str(s):
    return s.astype(object).where(s.notna(), "").astype(str)


def _binary_items(items):
    def ok(g):
        try:
            sc = json.loads(g)["response_scale"]
        except (TypeError, ValueError, KeyError):
            return False
        return sc.get("kind") == "discrete" and sorted(sc.get("values", [])) == [0.0, 1.0]
    return items[items.grading_criterion.map(ok)]


def _sha(t):
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def dup_keys(i):
    c = i.content.map(lambda t: re.sub(r"\s+", " ", str(t)).strip().lower())
    inst = i.raw_item_id.astype(str).str.split("/").str[-1].str.lower()
    return [" ".join(k for k in (_sha(t[:200]) if len(t) >= 20 else "", "i:" + r if "__" in r else "") if k)
            for t, r in zip(c, inst)]


def build():
    subs, its, rs = [], [], []
    for b in FOLDS + AUX:
        bm, s, i, r = (pd.read_parquet(RAW / b / "formatted_tables" / f"{t}.parquet") for t in TABLES)
        bm = bm.iloc[0]
        r = r[r.response.notna()]
        if bm.response_type == "mixed":
            i = i if set(r.response) <= {0.0, 1.0} else _binary_items(i)
            r = r[r.item_id.isin(i.item_id) & r.response.isin([0.0, 1.0])]
        s = s.assign(bench=b, **{k: _str(s[k]) for k in SUBJECT_KEYS})[["subject_id", "bench", *SUBJECT_KEYS]]
        i = pd.DataFrame({"item_id": i.item_id, "bench": b, "item_content": _str(i.content),
                          "item_features": _str(i.item_features), "benchmark_id": anon(b), "dup": dup_keys(i)})
        r = pd.DataFrame({"bench": b, "subject_id": r.subject_id, "item_id": r.item_id, "trial": r.trial,
                          "test_condition": _str(r.test_condition), "interactors": _str(r.interactors),
                          "y": r.response.astype(float), "binary": bm.response_type in ("binary", "mixed")})
        subs.append(s), its.append(i), rs.append(r)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.concat(subs).drop_duplicates(["subject_id", "bench"]).to_parquet(CACHE / "subjects.parquet")
    pd.concat(its).drop_duplicates(["item_id", "bench"]).to_parquet(CACHE / "items.parquet")
    pd.concat(rs).reset_index(drop=True).to_parquet(CACHE / "responses.parquet")


@dataclass
class Data:
    subjects: pd.DataFrame
    items: pd.DataFrame
    responses: pd.DataFrame

    def only(self, benches):
        k = set(benches)
        return Data(self.subjects[self.subjects.bench.isin(k)], self.items[self.items.bench.isin(k)],
                    self.responses[self.responses.bench.isin(k)])

    def without(self, bench):
        d = self.only(set(self.responses.bench.unique()) - {bench})
        held = {k for ks in self.items.dup[self.items.bench == bench] for k in ks.split()}
        bad = d.items[d.items.dup.map(lambda ks: any(k in held for k in ks.split()))]
        if len(bad):
            key = pd.MultiIndex.from_frame(bad[["bench", "item_id"]])
            d.items = d.items[~pd.MultiIndex.from_frame(d.items[["bench", "item_id"]]).isin(key)]
            d.responses = d.responses[~pd.MultiIndex.from_frame(d.responses[["bench", "item_id"]]).isin(key)]
        return d


def load():
    if not (CACHE / "responses.parquet").exists():
        build()
    ts = [pd.read_parquet(CACHE / f"{t}.parquet") for t in TABLES[1:]]
    return Data(*(x.astype({c: object for c in x.columns if x[c].dtype == "str"}) for x in ts))


if __name__ == "__main__":
    build()
    d = load()
    print(d.responses.groupby("bench").agg(n=("y", "size"), p=("y", "mean"), s=("subject_id", "nunique"),
                                           i=("item_id", "nunique")).to_string())
