import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from paiec.harness.data import load

HEAD, TAIL, CHUNK = 1000, 2000, 4000
TASK = "Below is a task from an AI evaluation benchmark.\n\n{meta}<task>\n{t}\n</task>\n\n"
ASK = ("How hard is this task for a strong AI system (a frontier LLM or LLM agent) to solve correctly? "
       "Answer with a single digit from 1 (trivial, almost every system solves it) to 9 "
       "(extremely hard, almost no system solves it).")

model, items, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
x = pd.read_parquet(items, columns=["bench", "item_id"]).merge(
    load().items[["bench", "item_id", "item_content", "item_features"]], on=["bench", "item_id"], validate="1:1")
k = x.drop_duplicates(["item_content", "item_features"])
texts = [t if len(t) <= 80000 else t[:40000] + t[-40000:] for t in k.item_content]
tok = AutoTokenizer.from_pretrained(model)
digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
llm = LLM(model, dtype="auto", max_model_len=4608, gpu_memory_utilization=0.92, enable_prefix_caching=True,
          max_logprobs=10, logprobs_mode="processed_logprobs", seed=0, tensor_parallel_size=1,
          limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
sp = SamplingParams(max_tokens=1, temperature=0, logprobs=10, allowed_token_ids=digits[1:])


def prompt(t, f):
    ids = tok.encode(t, add_special_tokens=False)
    t = t if len(ids) <= HEAD + TAIL else tok.decode(ids[:HEAD]) + "\n[...]\n" + tok.decode(ids[-TAIL:])
    meta = f"Metadata: {f}\n\n" if f else ""
    s = tok.apply_chat_template([{"role": "user", "content": TASK.format(meta=meta, t=t) + ASK}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return TokensPrompt(prompt_token_ids=tok(s, add_special_tokens=False).input_ids)


scores = []
for c in range(0, len(k), CHUNK):
    t0 = time.time()
    res = llm.generate([prompt(t, f) for t, f in zip(texts[c:c + CHUNK], k.item_features[c:c + CHUNK])], sp, use_tqdm=False)
    pr = np.array([[np.exp(o.outputs[0].logprobs[0][g].logprob) if g in o.outputs[0].logprobs[0] else 0.0 for g in digits]
                   for o in res])
    scores += list((pr / pr.sum(1, keepdims=True)) @ np.arange(10))
    print(f"{len(scores)}/{len(k)} {time.time() - t0:.0f}s", flush=True)
mp = dict(zip(zip(k.item_content, k.item_features), scores))
x["gemma4_31b"] = [mp[c, f] for c, f in zip(x.item_content, x.item_features)]
x[["bench", "item_id", "gemma4_31b"]].to_parquet(out, index=False)
