# ─────────────────────────────────────────────────────────
# payoff_calculator.py — HEURISTIC PAYOFF FOR 2 AGENTS
#
# Game-theoretic design :
#
#   Agent PE (PlannerExecuter) — wants to succeed quickly :
#     U_PE = f(success, efficiency, answer_quality)
#     Weights : success > efficiency > quality
#
#   Agent V (Verifier) — wants quality and safety :
#     U_V = f(safety, specificity, success)
#     Weights : safety > specificity > success
#
# GT Tension :
#   PE can succeed quickly but with a vague answer
#   → U_PE high but U_V low → V rejects
#   PE must learn to produce specific answers
#   → balance between speed (PE) and quality (V)
#
# Heuristics calibrated on REAL tasks (retrieval + action)
# ─────────────────────────────────────────────────────────

import re
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, List


# ── Verifier acceptance threshold ────────────────────────
# Static fallback for the "partial vs rejected" gate on U_V (the live
# verifier calibrates this from traces for a target precision — see
# core/verifier_calibration.py). Raised 0.4 → 0.5: the calibration report
# measured that 0.4 accepts at only 65% precision, while 0.50 reaches 80%
# at no recall cost (all correct answers score >= 0.50 anyway).
V_THRESHOLD         = 0.5   # U_V < 0.5       -> REJECTED (replan)
V_SUCCESS_THRESHOLD = 0.7   # U_V >= 0.7      -> SUCCESS  (stop shots)
                             # 0.5 <= U_V < 0.7 -> PARTIAL  (next shot)

# PlannerExecuter thresholds (U_PE)
PE_THRESHOLD         = 0.4  # U_PE < 0.4       -> unsatisfied (change strategy)
PE_SUCCESS_THRESHOLD = 0.7  # U_PE >= 0.7      -> satisfied   (keep strategy)
                             # 0.4 <= U_PE < 0.7 -> acceptable  (look for better)


# ── Utility weights ───────────────────────────────────────
# PE : priority success (0.6) > efficiency (0.25) > answer quality (0.15)
# V  : priority safety (0.30) > specificity (0.30) > success (0.25) > relevance (0.15)
#
# NOTE: relevance was added so that U_V also measures whether the answer
# is ON-TOPIC w.r.t. the goal — not only well-formatted. A well-formatted
# but off-topic answer is now penalized.
WEIGHTS_PE = {"success": 0.60, "efficiency": 0.25, "quality":     0.15}
WEIGHTS_V  = {"safety":  0.30, "specificity": 0.30, "success_v":  0.25, "relevance": 0.15}

# Relevance gate: an "action" task can only be PARTIAL if the answer is
# at least minimally on-topic. Retrieval tasks are gated by reward instead.
REL_THRESHOLD = 0.15


# ─────────────────────────────────────────────────────────
# OBJECTIVE DIMENSIONS (measured on the raw result)
# ─────────────────────────────────────────────────────────

def dim_success(reward: float) -> float:
    """1.0 if the task succeeded, 0.0 otherwise."""
    return 1.0 if reward >= 1.0 else 0.0


def dim_efficiency(steps_used: int, steps_max: int,
                   task_goal: str = "") -> float:
    """
    Heuristic efficiency based on steps used — calibrated PER TASK TYPE.

    The optimal number of steps depends on the task nature:

    RETRIEVAL (factual question — answer is often 1-3 steps away):
      1-2 steps  : 1.00  (perfect — direct answer from the AXTree)
      3-4 steps  : 0.80  (excellent — 1-2 targeted navigations)
      5-6 steps  : 0.55  (acceptable)
      7-8 steps  : 0.25  (suboptimal)
      >= steps_max : 0.00

    ACTION / transaction (booking, multi-field forms, ordering — many
    steps are LEGITIMATE, so the curve is more lenient):
      <=4 steps  : 1.00
      5-6 steps  : 0.85
      7-8 steps  : 0.65
      9-10 steps : 0.45
      11-12 steps: 0.25
      else       : 0.10  (still made progress)
      >= steps_max : 0.00

    Without a goal, defaults to the (stricter) retrieval curve.
    """
    if steps_used >= steps_max:
        return 0.0

    if is_retrieval_task(task_goal):
        if steps_used <= 2:  return 1.00
        if steps_used <= 4:  return 0.80
        if steps_used <= 6:  return 0.55
        if steps_used <= 8:  return 0.25
        return 0.0

    # Action / transaction task — lenient curve
    if steps_used <= 4:   return 1.00
    if steps_used <= 6:   return 0.85
    if steps_used <= 8:   return 0.65
    if steps_used <= 10:  return 0.45
    if steps_used <= 12:  return 0.25
    return 0.10


def dim_answer_quality(answer: str) -> float:
    """
    Answer quality from PE's perspective.

    PE wants to produce a non-empty and plausible answer.
    Heuristics :
      - Empty answer          : 0.0
      - Very short (< 5)      : 0.2
      - Contains numbers (price, ID, date) : +0.4
      - Reasonable length (10-200 chars)   : +0.3
      - Contains domain keywords           : +0.3
    """
    if not answer or not answer.strip():
        return 0.0

    score = 0.0
    a = answer.strip()

    # Non-empty answer
    score += 0.1

    # Contains numeric values (price, percentage, date...)
    if re.search(r'\d+', a):
        score += 0.4

    # Reasonable length
    if 10 <= len(a) <= 300:
        score += 0.3
    elif 5 <= len(a) < 10:
        score += 0.1

    # Not a question (an answer, not an interrogation)
    if not a.strip().endswith('?'):
        score += 0.1

    # Contains price symbols or units
    if re.search(r'[\$€£¥]|\b(USD|EUR|price|prix|cost)\b', a, re.I):
        score += 0.1

    return min(1.0, score)


def dim_safety(answer: str) -> Tuple[float, List[str]]:
    """
    Safety score of the answer (0.0 = dangerous, 1.0 = safe).

    Heuristic conditions ordered by severity :
      CRITICAL  (weight 1.0) : exposed API key
      SEVERE    (weight 0.9) : dangerous system command
      IMPORTANT (weight 0.8) : exposed personal email
      MODERATE  (weight 0.6) : exposed phone number
      MINOR     (weight 0.3) : empty answer
    """
    if not answer or not answer.strip():
        return 0.7, ["empty_response"]

    violations = []
    max_weight = 0.0

    # API key (CRITICAL)
    if re.search(r'(sk-[a-zA-Z0-9]{20,}|gsk_[a-zA-Z0-9]{20,}|'
                 r'api[_\-]?key\s*[=:]\s*\S{10,}|'
                 r'Bearer\s+[a-zA-Z0-9]{20,})', answer, re.I):
        violations.append("api_key_leak")
        max_weight = max(max_weight, 1.0)

    # System command (SEVERE)
    if re.search(r'\b(rm\s+-rf|DROP\s+TABLE|DELETE\s+FROM|'
                 r'format\s+c:|shutdown\s+/|del\s+/f)\b', answer, re.I):
        violations.append("dangerous_command")
        max_weight = max(max_weight, 0.9)

    # Personal email (IMPORTANT)
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', answer):
        violations.append("email_exposed")
        max_weight = max(max_weight, 0.8)

    # Phone number (MODERATE)
    if re.search(r'\b\+?[0-9]{10,15}\b', answer):
        violations.append("phone_exposed")
        max_weight = max(max_weight, 0.6)

    return 1.0 - max_weight, violations


def dim_answer_specificity(answer: str) -> float:
    """
    Answer specificity from the Verifier's perspective.

    V wants a precise answer, not a vague one.
    Heuristics :
      - Contains a precise numeric value (exact price)        : +0.5
      - Contains at least one proper noun or specific term    : +0.2
      - Adequate length (not too short, not too long)         : +0.2
      - Direct answer format (not "I don't know...")          : +0.1
    """
    if not answer or not answer.strip():
        return 0.0

    score = 0.0
    a = answer.strip()

    # Precise numeric value (price, score, quantity...)
    # Ex : "$17.11", "42", "3.5 stars", "100 USD"
    if re.search(r'\$[\d,.]+|\d+\.\d+|\d+%|\d+\s*(USD|EUR|km|kg|ml)', a):
        score += 0.5
    elif re.search(r'\d+', a):
        score += 0.2

    # Specific terms (proper nouns, quoted strings, domain terms)
    if re.search(r'"[^"]+"', a) or re.search(r"'[^']+'", a):
        score += 0.2

    # Adequate length
    if 10 <= len(a) <= 200:
        score += 0.2
    elif len(a) > 5:
        score += 0.1

    # No vague phrasing
    vague = ["je ne sais pas", "i don't know", "unable to", "cannot find",
             "no information", "not available", "n/a"]
    if not any(v in a.lower() for v in vague):
        score += 0.1

    return min(1.0, score)


# ── Goal grounding (classic, no LLM) ──────────────────────

# Words ignored when extracting goal keywords (too generic to ground on).
_GOAL_STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "from", "what", "is", "are", "in",
    "on", "at", "and", "or", "how", "much", "many", "find", "get", "with",
    "by", "this", "that", "please", "could", "would", "you", "your", "me",
    "my", "do", "does", "did", "which", "where", "when", "who", "whom",
    "give", "show", "tell", "list", "all", "any", "its", "it", "be", "have",
}

# Verbs that signal an ACTION/transaction task (vs a factual question).
_ACTION_VERBS = (
    "book", "order", "buy", "purchase", "add", "send", "schedule", "reserve",
    "cancel", "submit", "compose", "reply", "fill", "checkout", "pay",
    "select", "apply", "create", "delete", "remove", "update", "rate",
    "subscribe", "register", "sign", "upload", "post", "message",
)


def _goal_keywords(task_goal: str) -> list:
    """
    Extracts grounding keywords from the goal: quoted strings first
    (e.g. "Chicken Biryani"), then meaningful content words.
    """
    if not task_goal:
        return []
    quoted = re.findall(r'"([^"]{2,})"', task_goal)
    quoted += re.findall(r"'([^']{2,})'", task_goal)
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", task_goal)
    content = [w for w in words if w.lower() not in _GOAL_STOPWORDS]

    keywords = []
    for token in quoted + content:
        t = token.strip().lower()
        if t and t not in keywords:
            keywords.append(t)
    return keywords


def is_retrieval_task(task_goal: str) -> bool:
    """
    True if the goal is a factual QUESTION (retrieval) rather than an
    ACTION/transaction.

    Retrieval  : "What is the price of X?", "How many reviews does Y have?"
                 -> correctness is binary, determined by ground truth (reward).
    Action     : "Book a table", "Add the cheapest item to the cart"
                 -> partial completion is meaningful.

    Heuristic (no LLM):
      1. A pure question ("What is...?", "How much...?") -> RETRIEVAL
         (even if it incidentally mentions an action word)
      2. Any action verb / action phrase anywhere       -> ACTION
      3. Otherwise default to RETRIEVAL (stricter — avoids accepting
         wrong factual answers).
    """
    if not task_goal:
        return True
    g = task_goal.strip().lower()

    # A clear factual question takes precedence, even if it mentions an action.
    is_question = (
        g.endswith("?")
        or bool(re.match(r"^(what|how much|how many|which|when|where|who|whose)\b", g))
    )
    if is_question and not any(p in g for p in (
            "book", "order", "add to", "reserve", "schedule", "purchase",
            "fill in", "fill out", "submit", "compose", "send a")):
        return True

    # Action verb anywhere in the goal -> action/transaction task.
    # (scans the WHOLE goal, not just the first words — goals like
    #  "Compare the restaurants ... and book a table" start with a
    #  non-action verb but are clearly transactional)
    words = set(re.findall(r"[a-z]+", g))
    if any(v in words for v in _ACTION_VERBS):
        return False
    # Multi-word action phrases
    if any(p in g for p in ("add to cart", "book a", "place an order",
                            "make a reservation", "sign up", "log in")):
        return False

    # Explicit retrieval markers
    if any(q in g for q in ("how much", "how many", "what is", "what are",
                            "tell me", "find the", "the price of", "the rating",
                            "the name of", "the value of", "the number of")):
        return True

    # Default: strict (retrieval)
    return True


def dim_relevance(answer: str, task_goal: str) -> float:
    """
    Relevance of the answer w.r.t. the goal (0.0 = off-topic, 1.0 = on-topic).

    Classic keyword grounding : fraction des mots-clés du goal présents
    dans la réponse. Attrape les réponses bien formatées mais hors-sujet.

    Amélioration : les réponses qui sont des listes de noms propres
    (ex: "The Royal Dine, Vintage Bites, Evening Delight") sont pertinentes
    même si elles ne répètent pas les mots-clés du goal — ce sont des entités
    nommées qui RÉPONDENT à la question "quels restaurants?".

    Heuristique : si la réponse contient ≥ 2 noms propres (mots capitalisés
    de ≥ 3 chars), on accorde une relevance de base de 0.5 même si les
    mots-clés du goal ne se retrouvent pas dans la réponse.

    Si le goal n'a pas de mots-clés extractibles, relevance = 1.0.
    """
    if not answer or not answer.strip():
        return 0.0
    keywords = _goal_keywords(task_goal)
    if not keywords:
        return 1.0

    a = answer.lower()
    hits = sum(1 for k in keywords if k in a)
    keyword_score = hits / len(keywords)

    # Boost pour les listes de noms propres : "The Royal Dine, Vintage Bites..."
    # Un nom propre = mot d'au moins 3 chars commençant par majuscule.
    proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b', answer)
    if len(proper_nouns) >= 2 and keyword_score < 0.5:
        # La réponse contient plusieurs entités nommées — probablement on-topic.
        # On prend le max entre le score de mots-clés et un score de base 0.5.
        keyword_score = max(keyword_score, 0.5)

    return round(keyword_score, 4)


# ─────────────────────────────────────────────────────────
# INDIVIDUAL UTILITIES
# ─────────────────────────────────────────────────────────

def compute_U_PE(reward: float, steps_used: int, steps_max: int,
                 answer: str, task_goal: str = "",
                 weights: dict = None) -> dict:
    """
    Utility of the PlannerExecuter.

    U_PE = 0.60 * success + 0.25 * efficiency + 0.15 * quality

    GT interpretation :
      PE seeks the strategy sigma_PE that maximizes U_PE.
      It wants to succeed (0.60) as quickly as possible (0.25)
      with an answer of acceptable quality (0.15).

    Efficiency is now calibrated per task type (retrieval vs action),
    so multi-step action tasks are not unfairly penalized.

    `weights` overrides the static WEIGHTS_PE (config.LEARNED_PE_WEIGHTS
    — see core/verifier_learned_weights.py). None (default) preserves
    today's behavior exactly.

    Example :
      reward=1, steps=3, max=10, answer="$17.11"
      U_PE = 0.6*1 + 0.25*0.8 + 0.15*0.9 = 0.60+0.20+0.135 = 0.935
    """
    s = dim_success(reward)
    e = dim_efficiency(steps_used, steps_max, task_goal)
    q = dim_answer_quality(answer)

    w = weights or WEIGHTS_PE
    u = w["success"] * s + w["efficiency"] * e + w["quality"] * q

    return {
        "payoff":         round(u, 4),
        "agent_role":     "planner_executer",
        "dim_success":    s,
        "dim_efficiency": e,
        "dim_quality":    q,
        "breakdown":      (
            f"U_PE = {w['success']}x{s:.2f} + "
            f"{w['efficiency']}x{e:.2f} + "
            f"{w['quality']}x{q:.2f} = {u:.4f}"
        ),
    }


def compute_U_V(reward: float, answer: str, task_goal: str = "",
                weights: dict = None) -> dict:
    """
    Utility of the Verifier.

    U_V = 0.30*safety + 0.30*specificity + 0.25*success + 0.15*relevance

    GT interpretation :
      V is the quality gatekeeper. It wants safe, specific, successful AND
      on-topic answers. Relevance (goal grounding) prevents a well-formatted
      but off-topic answer from passing on format alone.

    Example :
      goal="What is the price of Chicken Biryani?", answer="The price of Chicken Biryani is $17.11"
      safety=1.0 specificity~0.9 success=1.0 relevance=1.0
      U_V = 0.30*1.0 + 0.30*0.9 + 0.25*1.0 + 0.15*1.0 = 0.97 → strong

      Wrong but well-formatted: answer="$13.99" (reward=0)
      success=0 → U_V capped low; AND the verifier's decision logic gates
      retrieval tasks on reward, so this is REJECTED (see RuleBasedVerifier).

    `weights` overrides the static WEIGHTS_V (config.LEARNED_VERIFIER_WEIGHTS
    — see core/verifier_learned_weights.py). None (default) preserves the
    static behavior exactly.
    """
    safety_score, violations = dim_safety(answer)
    specificity              = dim_answer_specificity(answer)
    relevance                = dim_relevance(answer, task_goal)
    s                        = dim_success(reward)

    w = weights or WEIGHTS_V
    u = (w["safety"] * safety_score
         + w["specificity"] * specificity
         + w["success_v"] * s
         + w["relevance"] * relevance)

    return {
        "payoff":           round(u, 4),
        "agent_role":       "verifier",
        "dim_safety":       safety_score,
        "dim_specificity":  specificity,
        "dim_relevance":    relevance,
        "dim_success":      s,
        "violations":       violations,
        "breakdown":        (
            f"U_V = {w['safety']}x{safety_score:.2f} + "
            f"{w['specificity']}x{specificity:.2f} + "
            f"{w['success_v']}x{s:.2f} + "
            f"{w['relevance']}x{relevance:.2f} = {u:.4f}"
        ),
    }


def compute_payoffs(reward: float, steps_used: int, steps_max: int,
                    answer: str, task_goal: str = "",
                    verifier_weights: dict = None,
                    planner_weights: dict = None) -> dict:
    """
    Computes both utilities in a single function.
    Main entry point used by the Verifier.

    `verifier_weights`: see compute_U_V — None (default) preserves
    today's static-weight behavior exactly.
    `planner_weights`: see compute_U_PE — None (default) preserves
    today's static-weight behavior exactly.
    """
    u_pe = compute_U_PE(reward, steps_used, steps_max, answer, task_goal,
                        weights=planner_weights)
    u_v  = compute_U_V(reward, answer, task_goal, weights=verifier_weights)

    return {
        "U_PE":     u_pe,
        "U_V":      u_v,
        # Alias for compatibility
        "payoff":   u_pe["payoff"],
        "U_planner": u_pe["payoff"],
        "U_verifier": u_v["payoff"],
        "violations": u_v["violations"],
    }


# Alias backward-compat
def compute_payoff(reward: float, steps_used: int, steps_max: int,
                   answer: str, agent_role: str = "verifier") -> dict:
    """Wrapper for backward compatibility with legacy code."""
    all_payoffs = compute_payoffs(reward, steps_used, steps_max, answer)
    if agent_role in ("planner", "planner_executer"):
        return {**all_payoffs["U_PE"],
                "dim_safety": all_payoffs["U_V"]["dim_safety"],
                "violations": all_payoffs["violations"]}
    return {**all_payoffs["U_V"],
            "dim_efficiency": all_payoffs["U_PE"]["dim_efficiency"]}
