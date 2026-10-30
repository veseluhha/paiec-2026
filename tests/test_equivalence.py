import importlib.util
import json
import os
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[1] / "paiec" / "model"
FX = json.loads((Path(__file__).parent / "fixtures" / "equivalence.json").read_text())


def expand(case):
    s, b, k, j = case
    item = lambda b, k: dict(FX["items"][b][k])
    return ([dict(FX["subjects"][s]), item(b, k)],
            [[[dict(FX["subjects"][s]), item(b, k)], y] for s, b, k, y in FX["labeled"][j]])


class Equivalence(unittest.TestCase):
    """predictions of the release model match the research models (irt_v7_nj_v2, irt_v15) bit for bit"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "g31.pkl").write_bytes(pickle.dumps({(c, f): np.array([v]) for c, f, v in FX["scores"]}))
        os.environ["PAIEC_JUDGE_CACHE"] = str(self.tmp)

    def load(self, variant):
        d = self.tmp / variant
        d.mkdir()
        for f in ("model.py", "subj.py"):
            shutil.copy(MODEL / f, d)
        (d / "params.json").write_text(json.dumps(FX["params"][variant]))
        (d / "models.txt").write_text("google/gemma-4-31B-it\n")
        spec = importlib.util.spec_from_file_location(f"model_{variant}", d / "model.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def check(self, variant):
        m = self.load(variant)
        got = [m.predict(*expand(c)) for c in FX["cases"]]
        self.assertEqual(got, FX["expected"][variant])

    def test_numpy(self):
        self.check("numpy")

    def test_judge(self):
        self.check("judge")


if __name__ == "__main__":
    unittest.main()
