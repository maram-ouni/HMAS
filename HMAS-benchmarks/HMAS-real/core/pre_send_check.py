# ─────────────────────────────────────────────────────────
# pre_send_check.py — EXECUTOR SELF-VERIFICATION (before sending)
#
# Breaks the strict pipeline (executor -> verifier -> planner) by letting
# the LLM executor consult the verifier's DETERMINISTIC criteria on its
# OWN candidate answer BEFORE committing it via send_msg_to_user.
#
# Key constraint (honest): the RuleBasedVerifier's final decision for
# retrieval tasks is gated on `reward` (ground truth), which the executor
# does NOT have before sending. So a pre-send self-check can only assess
# the REWARD-FREE quality dimensions — the FORM and GROUNDING of the
# answer:
#   - non-empty            (catches the empty-answer failure mode seen on
#                           fly-unified-1 / zilloft-1)
#   - specific             (has a concrete value, not vague)
#   - on-topic (relevance) (addresses the goal, not off-topic)
#   - safe                 (no leaked secret / dangerous content)
#
# Retrieval-aware, matching rule_based_verifier.py's own philosophy: for
# RETRIEVAL tasks (factual questions), correctness is BINARY and decided by
# ground truth only — the real verifier never gates them on relevance or
# specificity. A short, correct proper-noun answer (e.g. "Phat Thai" for
# "which Thai restaurant?") can score low on keyword-overlap relevance
# without being wrong. So relevance/specificity are only enforced as a
# pre-send gate for ACTION tasks; retrieval tasks are only checked for
# emptiness and safety here (caught a real false-positive rejecting
# "Phat Thai" as "off-topic" during live testing — see docs/AVANCEMENT.md).
#
# This is a genuine executor<->verifier interaction (the verifier's rules
# are consulted mid-episode, not only at the end), implemented as a PURE,
# unit-testable function with NO LLM call and NO ground truth.
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.payoff_calculator import (
    dim_answer_specificity, dim_relevance, dim_safety, is_retrieval_task,
)


def pre_send_verdict(answer: str, task_goal: str,
                     min_specificity: float = 0.30,
                     min_relevance: float = 0.15) -> dict:
    """
    Reward-free self-check of a candidate answer before send_msg_to_user.

    Returns a dict:
      {
        "accept":  bool,            # True => safe to send
        "reason":  str,             # "" if accept, else why not + how to fix
        "dims":    {specificity, relevance, safety},
      }

    Deterministic, no LLM, no ground truth. Mirrors the reward-free part of
    the verifier's U_V (safety + specificity + relevance); it deliberately
    does NOT judge correctness (that needs the benchmark's ground truth).

    Retrieval tasks skip the relevance/specificity gate (see module
    docstring) — only emptiness and safety are checked for them.
    """
    a = (answer or "").strip()
    specificity = dim_answer_specificity(a)
    relevance   = dim_relevance(a, task_goal)
    safety, violations = dim_safety(a)
    dims = {"specificity": specificity, "relevance": relevance, "safety": safety}

    if not a:
        return {"accept": False, "dims": dims,
                "reason": "Your answer is EMPTY. Read the exact value from the "
                          "AXTree and send a concrete answer — do not send nothing."}

    if violations:
        return {"accept": False, "dims": dims,
                "reason": f"Your answer contains a safety issue ({', '.join(violations)}). "
                          "Remove leaked secrets / personal data and send only the requested value."}

    if is_retrieval_task(task_goal):
        return {"accept": True, "reason": "", "dims": dims}

    if relevance < min_relevance:
        return {"accept": False, "dims": dims,
                "reason": "Your answer looks off-topic for this goal. Re-read the goal and "
                          "answer the exact entity/value it asks for."}

    if specificity < min_specificity:
        return {"accept": False, "dims": dims,
                "reason": "Your answer is too vague. Provide a specific value "
                          "(a number, an exact price, a precise name) read from the page."}

    return {"accept": True, "reason": "", "dims": dims}
