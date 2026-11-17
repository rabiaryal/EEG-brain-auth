#!/usr/bin/env bash
# Phase 7: full pipeline on the screened-subject config. Usage: bash scripts/run_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
C=configs/full109.yaml
$PY scripts/build_dataset.py --config $C
$PY scripts/run_baseline.py --config $C
$PY scripts/run_cnn.py --config $C
$PY scripts/run_attacks.py --config $C
$PY scripts/run_defenses.py --config $C
$PY scripts/eval_embeddings.py --config $C --checkpoint cnn
$PY scripts/eval_embeddings.py --config $C --checkpoint cnn_robust
