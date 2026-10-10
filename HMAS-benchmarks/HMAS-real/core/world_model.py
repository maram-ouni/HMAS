# ─────────────────────────────────────────────────────────
# world_model.py — PERSISTENT BELIEFS ABOUT THE ENVIRONMENT
# AND ABOUT THE EXECUTOR AGENT (v2 — BDI architecture improvement)
#
# Unlike BeliefBase (beliefs about the current task, reset on every
# set_task()), WorldModel persists to disk and survives across tasks
# AND across runs. It carries two kinds of beliefs that v1 lacked:
#
#   1. Beliefs about the environment: real difficulty of each task
#      category (site), measured over the full history.
#   2. Beliefs about ANOTHER agent (the LLM executor / DemoAgent):
#      empirical reliability of each plan for each category — a
#      theory-of-mind proxy ("I believe the executor will fail if I
#      send it this kind of hint for this site").
#
# `category` is keyed by extract_task_type(task_id) (e.g. "dashdish"
# from "v1.dashdish-10") — REAL has no dataset category field, so the
# site name plays that role here.
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, WORLD_MODEL_BAYESIAN_SMOOTHING


class WorldModel:
    """Beliefs shared across tasks and runs — the real BDI 'environment'."""

    def __init__(self, results_dir: str = None):
        self._lock = threading.Lock()
        # Isolated-memory override (None = today's hardcoded RESULTS_DIR,
        # shared across runs — see config.MEMORY_DIR).
        self._results_dir = results_dir or RESULTS_DIR
        self._file = os.path.join(self._results_dir, "world_model.json")
        self.category_stats: dict = {}   # category -> {"n": int, "sum_u_v": float}
        self.plan_stats: dict = {}       # "category:plan_name" -> {"n": int, "successes": int}
        self._load()

    # ── Beliefs about the environment (per category) ───────

    def record_task_outcome(self, category: str, u_v: float) -> None:
        if not category:
            return
        with self._lock:
            s = self.category_stats.setdefault(category, {"n": 0, "sum_u_v": 0.0})
            s["n"] += 1
            s["sum_u_v"] += u_v
        self._save()

    def category_avg_u_v(self, category: str) -> float:
        s = self.category_stats.get(category)
        return (s["sum_u_v"] / s["n"]) if s and s["n"] else 0.0

    def category_samples(self, category: str) -> int:
        return self.category_stats.get(category, {}).get("n", 0)

    # ── Beliefs about the executor (per category + plan) ───

    def record_plan_outcome(self, category: str, plan_name: str, success: bool) -> None:
        if not category or not plan_name:
            return
        key = f"{category}:{plan_name}"
        with self._lock:
            s = self.plan_stats.setdefault(key, {"n": 0, "successes": 0})
            s["n"] += 1
            if success:
                s["successes"] += 1
        self._save()

    def executor_reliability(self, category: str, plan_name: str,
                             min_samples: int = 4, default: float = 1.0,
                             k: float = 4.0) -> float:
        """
        Belief about the executor's reliability for this plan on this
        category.

        Default (config.WORLD_MODEL_BAYESIAN_SMOOTHING=False): hard
        cutoff — while the sample is insufficient, keep an optimistic
        default belief rather than penalizing a plan without evidence.

        Opt-in: smooth Bayesian estimate (successes + k*default)/(n + k)
        instead — degrades continuously to `default` at n=0 and converges
        to the raw success rate as n grows, avoiding the hard cutoff's
        discontinuity at min_samples. Enabling this can change plan-skip
        decisions for categories already past min_samples in an existing
        world_model.json — kept behind its own flag for that reason.
        """
        key = f"{category}:{plan_name}"
        s = self.plan_stats.get(key)
        if not s:
            return default
        if WORLD_MODEL_BAYESIAN_SMOOTHING:
            return (s["successes"] + k * default) / (s["n"] + k)
        if s["n"] < min_samples:
            return default
        return s["successes"] / s["n"]

    # ── Persistence ─────────────────────────────────────────

    def _save(self) -> None:
        os.makedirs(self._results_dir, exist_ok=True)
        with self._lock:
            data = {"category_stats": self.category_stats, "plan_stats": self.plan_stats}
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file, encoding="utf-8") as f:
                data = json.load(f)
            self.category_stats = data.get("category_stats", {})
            self.plan_stats     = data.get("plan_stats", {})
            print(f"  [WorldModel] Loaded beliefs on "
                  f"{len(self.category_stats)} categories, "
                  f"{len(self.plan_stats)} (category, plan) pairs")
        except Exception as e:
            print(f"  [WorldModel] Load failed: {e}")
