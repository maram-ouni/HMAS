import os
from dotenv import load_dotenv
load_dotenv()

# ── Backend selection ──────────────────────────────────────
# "groq"  → Groq cloud API   (JSON mode only, no schema-level constraint)
# "vllm"  → vLLM local server (constrained decoding via outlines / lm-format-enforcer)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")

# ── Groq ──────────────────────────────────────────────────
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"

GROQ_MODEL          = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.environ.get("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")
# ── vLLM ──────────────────────────────────────────────────
VLLM_BASE_URL       = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY        = os.environ.get("VLLM_API_KEY", "EMPTY")
VLLM_MODEL          = os.environ.get("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
VLLM_MODEL_FALLBACK = os.environ.get("VLLM_MODEL_FALLBACK", "mistralai/Mistral-7B-Instruct-v0.3")
VLLM_GUIDED_DECODING_BACKEND = "outlines"  # "outlines" or "lm-format-enforcer"

# ── Resolved active settings (backend-agnostic, used by all agents) ─
if LLM_BACKEND == "vllm":
    LLM_API_KEY        = VLLM_API_KEY
    LLM_BASE_URL       = VLLM_BASE_URL
    LLM_MODEL          = VLLM_MODEL
    LLM_MODEL_FALLBACK = VLLM_MODEL_FALLBACK
else:  # "groq" (default)
    LLM_API_KEY        = GROQ_API_KEY
    LLM_BASE_URL       = GROQ_BASE_URL
    LLM_MODEL          = GROQ_MODEL
    LLM_MODEL_FALLBACK = GROQ_MODEL_FALLBACK

# Dataset
DATASET_NAME  = "GTSAIInfraLabSOTAS/DeepJSONEval"
DATASET_SPLIT = "train"

# Budget de délibération INTERNE à l'agent BDI (Rao & Georgeff 1991).
# L'agent boucle perceive→brf→reconsider→options→filter→execute jusqu'au
# succès ou épuisement de ce budget, SANS être recréé entre les tentatives.
# La valeur peut être surchargée par tâche via TASK_ASSIGNED["max_attempts"]
# (câblé sur --shots). Auparavant 0 : la boucle vivait à tort dans le harnais.
MAX_REPLANS = 5

# NB : les poids (WEIGHTS_PE/WEIGHTS_V) et les seuils (V_THRESHOLD/
# V_SUCCESS_THRESHOLD) réellement utilisés par le vérificateur vivent dans
# core/payoff_calculator.py — seule source de vérité. D'anciennes constantes
# dupliquées (AGENT_PAYOFF_WEIGHTS, VERIFIER_THRESHOLD/SUCCESS_THRESHOLD)
# existaient ici avec les mêmes valeurs mais n'étaient lues par aucun agent
# (confirmé par grep) — supprimées le 2026-07-11 (V2.1, changement #8) pour
# éviter une double source de vérité qui aurait pu tromper une future
# modification (éditer ces constantes-ci n'aurait eu AUCUN effet).

MAX_HISTORY_LENGTH = 10

RESULTS_DIR  = "mas_project_v2/results"
HISTORY_FILE = "mas_project_v2/results/agent_histories.json"
RESULTS_FILE = "mas_project_v2/results/mas_results.csv"

# ── Mécanismes Verifier/décision portés depuis agisdk/mas_project_v2 ──────
# Flags d'environnement, off par défaut — comportement identique à avant
# tant qu'aucun n'est activé. Convention identique à agisdk (MAS_<FLAG>).
def _flag(env_name: str, default: bool = False) -> bool:
    return os.environ.get(env_name, str(default)).strip().lower() in ("1", "true", "yes")

LEARNED_VERIFIER_WEIGHTS    = _flag("MAS_LEARNED_VERIFIER_WEIGHTS")
LEARNED_PE_WEIGHTS          = _flag("MAS_LEARNED_PE_WEIGHTS")
INTERACTION_METRICS_ENABLED = _flag("MAS_INTERACTION_METRICS_ENABLED")
VERIFIER_AGREEMENT_ENABLED  = _flag("MAS_VERIFIER_AGREEMENT_ENABLED")
