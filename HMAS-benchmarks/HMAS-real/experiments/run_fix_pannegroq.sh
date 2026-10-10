#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."   # mas_project_v2/
PY=python3
LOGDIR=experiments/logs
mkdir -p "$LOGDIR"

echo "=== $(date) === model=qwen/qwen3-32b arch=hybrid (refait, panne Groq)"
MAS_LLM_MODEL="qwen/qwen3-32b" MAS_RESULTS_SUFFIX="_qwen3_32b" \
  "$PY" experiments/experiment_runner.py --arch hybrid > "$LOGDIR/fix_qwen3_32b_hybrid.log" 2>&1

echo "=== $(date) === model=openai/gpt-oss-20b arch=react (refait, panne Groq)"
MAS_LLM_MODEL="openai/gpt-oss-20b" MAS_RESULTS_SUFFIX="_gptoss_20b" \
  "$PY" experiments/experiment_runner.py --arch react > "$LOGDIR/fix_gptoss_20b_react.log" 2>&1

echo "=== $(date) === TERMINÉ ==="
