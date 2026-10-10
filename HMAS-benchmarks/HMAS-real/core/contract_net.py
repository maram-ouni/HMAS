# ─────────────────────────────────────────────────────────
# contract_net.py — CONTRACT-NET TASK ALLOCATION (multi-model executors)
#
# Turns the fixed single-executor pipeline into a genuine NEGOTIATION:
# the planner issues a call-for-proposals for a task; several LLM
# executors (one per Groq model — llama-3.3-70b, llama-4-scout-17b,
# llama-3.1-8b, …) each BID their expected reliability on that task's
# category; the planner AWARDS the task to the highest bidder. This is
# the classic Contract-Net Protocol (Smith, 1980) — a real MAS
# coordination mechanism, not a hard-coded route.
#
# The bids are DATA-DRIVEN: they come from each model's measured success
# rate per category in the model-comparison runs (react_results_<tag>.json
# / hybrid_results_<tag>.json). So the planner routes each category to the
# model empirically most likely to succeed on it.
#
# This module is the PURE, unit-testable core (bidding + award). Wiring
# several live executors behind it needs Groq; the protocol itself does not.
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Pure bidding / award core ─────────────────────────────

def compute_bid(successes: int, n: int, prior: float = 0.5, k: float = 4.0) -> float:
    """
    An executor's bid = its confidence-smoothed success rate on the category.

    Bayesian smoothing (successes + k*prior)/(n + k): with no evidence
    (n=0) the bid equals `prior`; as evidence grows it converges to the
    raw success rate. This prevents a model with 1 lucky success from
    outbidding a model proven reliable over many tasks.
    """
    return (successes + k * prior) / (n + k)


def award(bids: dict) -> str:
    """
    Planner awards the task to the highest bidder.

    Deterministic tie-break: the lexicographically smallest model id among
    the top bids wins (reproducibility). Raises on an empty bid set.
    """
    if not bids:
        raise ValueError("No bids received for the call-for-proposals.")
    return max(sorted(bids), key=lambda m: bids[m])


def run_auction(category: str, reliability: "ModelReliability",
                models: list = None) -> dict:
    """
    One Contract-Net round for a category. Returns:
      { "winner": model, "bids": {model: bid}, "evidence": {model: (succ, n)} }
    """
    models = models or reliability.models()
    bids, evidence = {}, {}
    for m in models:
        succ, n = reliability.record(m, category)
        bids[m] = compute_bid(succ, n)
        evidence[m] = (succ, n)
    return {"winner": award(bids), "bids": bids, "evidence": evidence}


# ── Data-driven reliability (from model-comparison result files) ──

def _category_of(task_id: str) -> str:
    # "v1.dashdish-6" -> "dashdish"
    name = task_id.split(".", 1)[-1]
    idx = name.rfind("-")
    return name[:idx] if idx > 0 else name


class ModelReliability:
    """
    Per-(model, category) track record loaded from result JSON files.

    `sources` maps a model id to a list of result files (react/hybrid runs
    for that model). Success = reward >= 1.0.
    """

    def __init__(self, sources: dict):
        # model -> category -> [successes, n]
        self._stats: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for model, files in sources.items():
            for f in files:
                path = f if os.path.isabs(f) else os.path.join(_HERE, f)
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as fh:
                    for r in json.load(fh):
                        cat = _category_of(r.get("task_id", ""))
                        cell = self._stats[model][cat]
                        cell[1] += 1
                        if r.get("reward", 0.0) >= 1.0:
                            cell[0] += 1

    def models(self) -> list:
        return sorted(self._stats)

    def categories(self) -> list:
        cats = set()
        for cat_map in self._stats.values():
            cats.update(cat_map)
        return sorted(cats)

    def record(self, model: str, category: str) -> tuple:
        """(successes, n) for this model on this category; (0, 0) if unseen."""
        cell = self._stats.get(model, {}).get(category)
        return (cell[0], cell[1]) if cell else (0, 0)
