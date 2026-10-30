#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
python -m paiec.harness.fit_full paiec/model build/numpy --variant '{"judge": false}'
PAIEC_JUDGE_CACHE=$PWD/data/judge python -m paiec.harness.fit_full paiec/model build/judge --variant '{"judge": true}'
python -m paiec.submission.build_zip build/numpy --out dist
python -m paiec.submission.build_zip build/judge --out dist
