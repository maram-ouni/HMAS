# ─────────────────────────────────────────────────────────
# interaction_metrics.py — LIVE INTER-AGENT INTERACTION METRICS (opt-in)
#
# Until now, the only place that measured how PE (Planner-Executor) and
# V (Verifier) relate to each other was a live but ephemeral counter
# (BeliefBase.consecutive_conflict, core/bdi_model.py) used purely to
# trigger a replan, plus a fully offline, post-hoc script
# (experiments/mas_metrics.py) run manually after a batch of tasks.
# Neither gives a running, inspectable picture of the interaction DURING
# a session.
#
# This module is that missing piece: a small, file-backed accumulator
# (one writer, this thread) that records every (U_PE, U_V, reward,
# quality) tuple and reports, at any point:
#   - conflict_rate         : same definition as BeliefBase's live
#                             counter (quality != "success" and
#                             U_PE - U_V > 0.3), but as a RATE over the
#                             whole recorded history, not a streak.
#   - mean_disagreement     : mean(|U_PE - U_V|) — a continuous version
#                             of the same idea, not gated by the 0.3
#                             threshold or the quality!="success" guard.
#   - pe_reward_gap / v_reward_gap : mean(utility | reward>=1) minus
#                             mean(utility | reward<1) for each agent's
#                             own utility. Whichever gap is larger is,
#                             empirically, the more diagnostic judge of
#                             actual ground-truth success — a concrete,
#                             measurable answer to "which agent's
#                             self-assessment can we trust more".
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

# Same threshold as BeliefBase.brf()'s live conflict counter
# (core/bdi_model.py) — kept identical so "conflict" means one thing
# across the codebase, not two slightly-different heuristics.
CONFLICT_GAP = 0.3


class InteractionMetrics:
    """Accumulates (U_PE, U_V, reward, quality) tuples for one job and
    reports summary statistics about how PE and V relate to each other."""

    def __init__(self, results_dir: str = None):
        self._results_dir = results_dir or RESULTS_DIR
        self._file = os.path.join(self._results_dir, "interaction_metrics.json")
        self._lock = threading.Lock()
        self._records: list = []
        self._load()

    def record(self, u_pe: float, u_v: float, reward: float, quality: str) -> None:
        with self._lock:
            self._records.append({
                "u_pe": u_pe, "u_v": u_v, "reward": reward, "quality": quality,
            })
        self._save()

    def summary(self) -> dict:
        with self._lock:
            records = list(self._records)
        n = len(records)
        if n == 0:
            return {
                "n": 0, "conflict_rate": 0.0, "mean_disagreement": 0.0,
                "pe_reward_gap": 0.0, "v_reward_gap": 0.0,
            }

        n_conflict = sum(
            1 for r in records
            if r["quality"] != "success" and (r["u_pe"] - r["u_v"]) > CONFLICT_GAP
        )
        mean_disagreement = sum(abs(r["u_pe"] - r["u_v"]) for r in records) / n

        succeeded = [r for r in records if r["reward"] >= 1.0]
        failed    = [r for r in records if r["reward"] < 1.0]

        def _gap(key: str) -> float:
            if not succeeded or not failed:
                return 0.0
            mean_succ = sum(r[key] for r in succeeded) / len(succeeded)
            mean_fail = sum(r[key] for r in failed) / len(failed)
            return mean_succ - mean_fail

        return {
            "n":                  n,
            "conflict_rate":      round(n_conflict / n, 4),
            "mean_disagreement":  round(mean_disagreement, 4),
            "pe_reward_gap":      round(_gap("u_pe"), 4),
            "v_reward_gap":       round(_gap("u_v"), 4),
        }

    def summary_line(self) -> str:
        s = self.summary()
        if s["n"] == 0:
            return "  [Interaction] No history yet."
        more_diagnostic = "U_V" if s["v_reward_gap"] > s["pe_reward_gap"] else "U_PE"
        return (
            f"  [Interaction] n={s['n']} | conflict_rate={s['conflict_rate']:.1%} | "
            f"mean|U_PE-U_V|={s['mean_disagreement']:.3f} | "
            f"PE_reward_gap={s['pe_reward_gap']:+.3f} | "
            f"V_reward_gap={s['v_reward_gap']:+.3f} | "
            f"more diagnostic of success: {more_diagnostic}"
        )

    # ── Persistence ───────────────────────────────────────────

    def _save(self) -> None:
        os.makedirs(self._results_dir, exist_ok=True)
        with self._lock:
            records = list(self._records)
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file, encoding="utf-8") as f:
                self._records = json.load(f)
        except Exception as e:
            print(f"  [InteractionMetrics] Load failed: {e}")
