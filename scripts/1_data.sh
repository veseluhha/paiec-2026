#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
python scripts/download_data.py
python -m paiec.harness.data
