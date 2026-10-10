#!/bin/bash
# Compare Hybrid BDI+Verifier et LLM seul (DemoAgent/ReAct) sur 4 modèles Groq
# supplémentaires (llama-3.3-70b déjà couvert par les résultats existants
# sans suffixe). 12 tâches, 10 shots, max_steps=10.
set -uo pipefail
cd "$(dirname "$0")/.."   # mas_project_v2/

PY=python3
LOGDIR=experiments/logs
mkdir -p "$LOGDIR"

declare -A MODELS=(
  [llama31_8b]="llama-3.1-8b-instant"
  [gptoss_20b]="openai/gpt-oss-20b"
  [llama4_scout_17b]="meta-llama/llama-4-scout-17b-16e-instruct"
  [qwen3_32b]="qwen/qwen3-32b"
)

for tag in "${!MODELS[@]}"; do
  model="${MODELS[$tag]}"
  for arch in react hybrid; do
    log="$LOGDIR/model_${tag}_${arch}.log"
    echo "=== $(date) === model=$model arch=$arch -> $log"
    MAS_LLM_MODEL="$model" MAS_RESULTS_SUFFIX="_$tag" \
      "$PY" experiments/experiment_runner.py --arch "$arch" > "$log" 2>&1
  done
done

echo "=== $(date) === TOUS LES RUNS TERMINÉS ==="
