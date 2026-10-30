# PAIEC 2026: hierarchical Rasch model with an LM difficulty judge

Code for a submission to the NeurIPS 2026 Predictive AI Evaluation Competition. The task is to predict `P(correct)` for an (AI system, benchmark item) pair on benchmarks unseen in training, while labels for the pair arrive at budgets 0, 1, 3, 7, 15, 31.

## Final submission

Codabench submission 991193 is `submitted/irt_v24_rt/`, the unpacked `irt_v24_rt_20261015_f4489686.zip` (sha256 `f44896868bb9d5132d140296f1fd0fe0ca042e2cdbd5316703933e05c7d0d3f1`).

It is the judge variant described below with one change: the text distance of the online GP (step 5) is the cosine distance between hidden states of the judge, not hashed n-grams. The hidden state is taken in the same forward pass that produces the rating: last prompt token, layer `round(0.75 L) - 1` (44 of 60), projected to 256 dimensions by a fixed Gaussian matrix (`numpy.random.default_rng(0)`) and centred on the labelled items of the call. The 4-label threshold and the evidence guard are unchanged, so at budgets 0 and 1 it predicts the same as the judge variant. Without CUDA it falls back to the numpy path; this fallback was tested for the judge variant only.

| | ALC |
|---|---|
| LOBO, 32 folds, seed 0 | 0.2057 |
| minus judge variant, seeds 0 / 1 | −0.0012 / −0.0012 |
| minus numpy variant, seeds 0 / 1 | −0.0074 / −0.0073 |
| platform, mean of 20 formative loads (sd) | 0.1765 (0.016) |

The package in `paiec/` reproduces the numpy and judge variants bit for bit. The hidden-state kernel exists only in the submitted code; LOBO runs of it need a cache of hidden states (about 0.4 GB, not included), rebuilt by scoring the items with the judge.

## Approach

1. Rasch (1PL) model. Every prediction is a posterior predictive mean on quadrature grids, so it is deterministic.
2. Subject ability on the test benchmark = benchmark level + subject deviation. The level of a new benchmark has the prior `N(0, tau^2)`, with `tau^2` estimated from the training benchmarks.
3. Subject prior: ridge regression of within-benchmark ability on release date, fitted on the training benchmarks. It can only lower the prior mean; its residual variance is estimated leave-one-benchmark-out.
4. Online update from `labeled`: the target subject and the two subjects sharing most items with it are integrated jointly. The remaining labelled subjects of the benchmark inform the benchmark level through their marginal likelihood.
5. Item difficulty is pooled within the test benchmark by a Gaussian process over hashed word n-grams of the item text, plus a random effect of the item source (first field of `item_features`). Kernel scales are averaged over a small grid by evidence. Texts shared by three or more items (prompt templates) are left out of the text kernel.
6. Judge variant: `google/gemma-4-31B-it` rates every item zero-shot on a 1-9 difficulty scale (expected digit under the next-token distribution). The rating enters the GP as a linear term whose slope prior is learned on the training benchmarks. Before the GP is active, it enters as a 50/50 mixture of no shift and a shift by the prior slope times the standardised rating.
7. Each `predict` call depends only on its `input` and `labeled`. Module-level caches (posterior per labelled set, ratings per item) memoise deterministic computations.

Two variants share `paiec/model`: **numpy** (no LM, research id `irt_v7_nj_v2`) and **judge** (`irt_v15_rt`). The judge variant needs a CUDA GPU at inference; without one it reduces to the numpy variant.

## Layout

```
submitted/irt_v24_rt/  final submission, as uploaded
paiec/model/           model.py (predict), subj.py (subject prior), fit.py (training, not shipped in the zip)
paiec/harness/         leave-one-benchmark-out (LOBO) evaluation replaying the platform protocol
paiec/submission/      deterministic zip builder and validator (static checks and a smoke test in a subprocess)
scripts/               reproduction steps 1-4
judge/scores.parquet   gemma-4-31B-it ratings used for training and LOBO (66,595 items)
tests/                 harness invariants, validator, bit-exact equivalence with the research code
```

## Reproduce

Python 3.11. CPU steps: `pip install -r requirements.txt`. GPU steps (re-scoring items, running the judge variant): `requirements-judge.txt`.

Commands run from the repository root. Data go to `data/`, LOBO runs to `runs/` (override with `PAIEC_DATA_DIR`, `PAIEC_RUNS_DIR`).

1. Data: `scripts/1_data.sh` downloads 34 benchmarks of [aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db) at revision `54074453da7b9911be0f48e4c560be547bc84d21` (gated: accept the terms and run `hf auth login`; about 0.7 GB) and builds the response cache (about 0.36 GB). The 32 binary benchmarks are LOBO folds; `swe_rebench` and `mmdocrag` are used for training only.
2. Judge ratings: `scripts/2_judge.sh` builds the lookup from `judge/scores.parquet`. `RESCORE=1 scripts/2_judge.sh` recomputes the ratings with vLLM on one GPU (`JUDGE_MODEL` may point to a local snapshot; we used revision `842da3794eaa0b77d5f08bae87a17459d91ff475`). Rated items: all items of `matharena`, `multi_swebench`, `real_webagents`, `researchcodebench`; for the other 28 folds, every item in LOBO episodes of seeds 0-1 plus up to 3,000 random items per benchmark for the slope prior.
3. LOBO: `scripts/3_lobo.sh` (`EPISODES=100` by default) fits both variants per held-out benchmark, replays the episodes and prints the paired difference (`python -m paiec.harness.compare RUN_A RUN_B`).
4. Submission: `scripts/4_submission.sh` fits both variants on all 34 benchmarks (`build/`) and writes validated zips to `dist/`. Validator: `python -m paiec.submission.check_zip ZIP`.

Tests: `python -m unittest discover tests` (needs the data cache from step 1).

### LOBO harness

Each episode imitates a formative platform sample for one held-out benchmark: 1 or 2 subjects with at least 80 items and an evaluation pool size, both drawn from frequencies in public platform logs. Items of a pair are split 50/50 into acquisition and evaluation pools. Labels are bought by the platform's hash-based random rule (no `labeling.py`) at checkpoints 0/1/3/7/15/31, with a fresh import of the model at every checkpoint and `labeled` shared across pairs. Score: Brier per pair, ALC = 0.1 b0 + 0.2 (b1 + b3 + b7 + b15) + 0.1 b31. Training items that duplicate an item of the held-out benchmark (normalised text prefix or SWE instance id) are removed before fitting.

## Expected numbers

LOBO, 32 folds, 100 episodes, seed 0, 3,965 pairs; lower is better, the constant 0.5 gives 0.25. The release code reproduces the research runs bit for bit (`pairs.csv` identical in every budget column):

| variant | ALC | b0 | b31 | research run | release run |
|---|---|---|---|---|---|
| numpy | 0.2131 | 0.2478 | 0.1877 | 1008-155008 | 1011-015310 |
| judge | 0.2069 | 0.2437 | 0.1788 | 1010-220754 | 1011-022116 |

Judge minus numpy: −0.0062 ALC, better on 25 of 32 folds (paired over identical episodes). A beta-binomial baseline gives 0.2166 on the same episodes (run 1008-150145, not in this release).

## Hardware and time

- LOBO and fitting: Apple M4 Pro, 24 GB RAM, 2 worker processes. One LOBO run of one variant takes 28-31 min (fitting 32 folds 19 min, 3,200 episodes 9-12 min); the final fit on all 34 benchmarks takes 40 s.
- Item ratings: one H100 80 GB, vLLM 0.19.1, bf16, one forward pass per item (total time not recorded).
- Judge variant at inference: transformers, bf16, peak 58.3 GiB GPU memory. The first call loads the weights (about 27 s), later calls take under 0.1 s.

## Data and model disclosure

- Training data: the public measurement-db release at the revision above. No test-time information is used beyond `input` and `labeled`; hidden benchmarks and subjects are not identified from anonymous ids; platform logs are not used for training (formative submissions only checked runtime reliability).
- External model: `google/gemma-4-31B-it` (Apache-2.0), zero-shot, no fine-tuning. `judge/scores.parquet` holds its ratings of public measurement-db items.
- Not included: research ablations and oracles, other judges, calibration and acquisition-policy experiments, and an experiment that looked up item difficulty in the public measurement-db pool by near-duplicate text (not part of any submitted variant).

## License

- Code: MIT (`LICENSE`).
- `judge/scores.parquet`: CC BY-SA 4.0, derived from `aims-foundations/measurement-db` (CC BY-SA); ratings produced with `google/gemma-4-31B-it` (Apache-2.0). See `judge/LICENSE`.
