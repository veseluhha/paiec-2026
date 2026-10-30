#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
EP=${EPISODES:-100}
a=$(python -m paiec.harness.run paiec/model --episodes "$EP" --tag numpy | tee /dev/stderr | sed -n 1p)
b=$(PAIEC_JUDGE_CACHE=$PWD/data/judge python -m paiec.harness.run paiec/model --variant '{"judge": true}' --episodes "$EP" --tag judge | tee /dev/stderr | sed -n 1p)
python -m paiec.harness.compare "$a" "$b"
