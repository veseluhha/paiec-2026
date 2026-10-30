import argparse
import json
import shutil
from pathlib import Path

from .data import load
from .run import fit

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="fit a submission directory on all training benchmarks")
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--variant", type=json.loads, default=None)
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    fit(a.model.resolve(), a.out.resolve(), load(), a.variant)
    print(a.out, sorted(p.name for p in a.out.iterdir()))
