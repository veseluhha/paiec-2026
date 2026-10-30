import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.harness.data import load

scores, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
s = pd.read_parquet(scores, columns=["bench", "item_id", "gemma4_31b"])
s.to_parquet(out / "scores.parquet", index=False)
x = s.merge(load().items[["bench", "item_id", "item_content", "item_features"]], on=["bench", "item_id"], validate="1:1")
g = x.groupby(["bench", "item_content", "item_features"], sort=False).gemma4_31b.mean().reset_index()
(out / "g31.pkl").write_bytes(pickle.dumps({(c, f): np.array([v]) for c, f, v in zip(g.item_content, g.item_features, g.gemma4_31b)}))
print(out, len(s), "items", len(g), "keys")
