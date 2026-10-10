import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Overridable via MAS_LLM_MODEL for model-comparison runs (see
# experiments/experiment_runner.py and experiments/react_baseline.py) —
# a run with no env var set stays on the default llama-3.3-70b-versatile.
LLM_MODEL          = os.environ.get("MAS_LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MODEL_FALLBACK = "llama-3.1-8b-instant"

# Suffix appended to results filenames so that parallel model-comparison
# runs (same architecture, different LLM_MODEL) don't clobber each other's
# results file. Empty by default -> today's filenames unchanged.
RESULTS_SUFFIX = os.environ.get("MAS_RESULTS_SUFFIX", "")

MAX_STEPS = 15

# Internal deliberation budget of the BDI agent (Rao & Georgeff 1991).
# The agent loops perceive->brf->reconsider->options->filter->execute inside
# a single run_task() call, without being recreated between attempts, until
# success or this budget is exhausted. Overridable per call via
# MASCoordinator.run_task(..., max_attempts=...), wired to --shots.
MAX_REPLANS = 3

MAX_HISTORY_LENGTH = 10

RESULTS_DIR  = "mas_project_v2/results"
HISTORY_FILE = "mas_project_v2/results/agent_histories.json"
RESULTS_FILE = "mas_project_v2/results/mas_results.csv"

# ─────────────────────────────────────────────────────────
# Feature flags — all default to False/None so that a run with no
# flags/env vars set is byte-identical to today's behavior. Each
# corresponds to a phase in the MAS enrichment plan (ported from the
# DeepJSONEval architecture's corrections, adapted to the REAL domain).
#
# Overridable via environment variable (MAS_<FLAG_NAME>=true), NOT by
# editing the hardcoded default below — this lets a single config.py
# serve BOTH the hybrid architecture's own subprocess (which should run
# enriched) AND anything else importing core/ in the SAME process, e.g.
# experiments/frering_bdi_baseline.py, which runs in-process inside
# experiment_runner.py and must stay a clean, unenriched ablation
# baseline. See experiments/experiment_runner.py's _run_hybrid_shot_subprocess
# for where these env vars are actually set (only in the hybrid
# subprocess's environment, never the parent process's os.environ).
# ─────────────────────────────────────────────────────────
def _flag(env_name: str, default: bool = False) -> bool:
    return os.environ.get(env_name, str(default)).strip().lower() in ("1", "true", "yes")

TRACE_ENABLED                  = _flag("MAS_TRACE_ENABLED")                  # per-attempt structured trace (JSONL per task)
CONFLICT_DETECTION_ENABLED     = _flag("MAS_CONFLICT_DETECTION_ENABLED")     # executor/verifier conflict desire + resolution plan
DYNAMIC_DESIRES_ENABLED        = _flag("MAS_DYNAMIC_DESIRES_ENABLED")        # escape_plateau / final_attempt / diagnose_unknown_failure / category_high_risk
UCB1_ENABLED                   = _flag("MAS_UCB1_ENABLED")                  # UCB1 arm selection for warm-start (vs. greedy argmax)
WORLD_MODEL_BAYESIAN_SMOOTHING = _flag("MAS_WORLD_MODEL_BAYESIAN_SMOOTHING") # smooth (successes+k*prior)/(n+k) vs. hard cutoff
EXECUTOR_SELF_VERIFY           = _flag("MAS_EXECUTOR_SELF_VERIFY")           # executor<->verifier loop: reward-free self-check before send_msg_to_user
LEARNED_VERIFIER_WEIGHTS       = _flag("MAS_LEARNED_VERIFIER_WEIGHTS")       # logistic-regression-fitted U_V weights from historical traces (see core/verifier_learned_weights.py)
LEARNED_PE_WEIGHTS             = _flag("MAS_LEARNED_PE_WEIGHTS")             # logistic-regression-fitted U_PE weights + calibrated KEEP_THRESHOLD from historical traces (see core/verifier_learned_weights.py)
INTERACTION_METRICS_ENABLED    = _flag("MAS_INTERACTION_METRICS_ENABLED")    # live PE/V agreement + diagnosticity tracking (see core/interaction_metrics.py)
VERIFIER_AGREEMENT_ENABLED     = _flag("MAS_VERIFIER_AGREEMENT_ENABLED")     # verifier tracks cross-attempt answer agreement (self-consistency) as a confidence/abstention signal — see core/self_consistency.py, agents/rule_based_verifier.py
EXECUTOR_LOOP_GUARD_ENABLED    = _flag("MAS_EXECUTOR_LOOP_GUARD_ENABLED")    # within-shot loop breaker: injects a corrective instruction when the executor repeats an action or swaps two values (see agents/llm_executor_agent.py)
EXECUTOR_GROUNDING_RECOVERY_ENABLED = _flag("MAS_EXECUTOR_GROUNDING_RECOVERY_ENABLED")  # on "could not find element with bid X", injects a forceful instruction to use only a bid present in the current AXTree (see agents/llm_executor_agent.py)
EXECUTOR_MULTIACTION_FIX_ENABLED    = _flag("MAS_EXECUTOR_MULTIACTION_FIX_ENABLED")     # balanced-paren action parsing: fixes the [^)]* regex that truncated actions whose argument contains a parenthesis, e.g. fill('221', 'Meeting (weekly)') (see agents/llm_executor_agent.py)
MEMORY_DIR = os.environ.get("MAS_MEMORY_DIR") or None  # override for all persistent memory files; None = today's hardcoded RESULTS_DIR paths
