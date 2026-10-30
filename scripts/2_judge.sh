#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
if [ "${RESCORE:-0}" = 1 ]; then
  python scripts/judge_scores.py "${JUDGE_MODEL:-google/gemma-4-31B-it}" judge/scores.parquet judge/scores_rescored.parquet
  python scripts/judge_cache.py judge/scores_rescored.parquet data/judge
else
  python scripts/judge_cache.py judge/scores.parquet data/judge
fi
