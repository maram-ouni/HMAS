# ─────────────────────────────────────────────────────────
# verifier_calibration.py — IS THE VERIFIER TRUSTWORTHY?
#
# The whole point of the Verifier is that "accepted ≈ correct". This
# module MEASURES that — something never quantified before: given the
# continuous U_V score and the ground-truth reward on historical attempts,
#
#   - reliability_bins()        : P(correct | U_V in bucket) — the
#                                 reliability diagram. A well-calibrated
#                                 verifier's buckets rise monotonically.
#   - precision_recall_at()     : if we accept when U_V >= t, what fraction
#                                 of accepted answers are actually correct
#                                 (precision) and what fraction of correct
#                                 answers we catch (recall).
#   - threshold_for_precision() : the LOWEST threshold whose accepted set is
#                                 correct at least `target` of the time —
#                                 i.e. "only accept when P(correct) >= 0.8".
#                                 This turns "accepted" into a calibrated,
#                                 trustworthy signal instead of a hand-picked
#                                 cutoff (config.V_THRESHOLD = 0.4, never
#                                 validated against reward).
#
# Pure Python, no dependency, no LLM call — operates on (u_v, reward) pairs
# already collected. Deterministic.
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


def _binary(reward: float) -> int:
    return 1 if reward >= 1.0 else 0


@dataclass
class Bin:
    lo: float
    hi: float
    n: int
    n_correct: int

    @property
    def p_correct(self) -> float:
        return self.n_correct / self.n if self.n else 0.0


def reliability_bins(pairs: List[Tuple[float, float]], n_bins: int = 5) -> List[Bin]:
    """
    Buckets (u_v, reward) pairs into `n_bins` equal-width U_V intervals over
    [0, 1] and reports P(correct) per bucket. Empty buckets are kept (n=0)
    so the diagram has a fixed x-axis. A trustworthy verifier's p_correct
    should increase with the U_V bucket.
    """
    width = 1.0 / n_bins
    bins = [Bin(lo=i * width, hi=(i + 1) * width, n=0, n_correct=0) for i in range(n_bins)]
    for u_v, reward in pairs:
        idx = min(int(u_v / width), n_bins - 1) if u_v == u_v else 0  # NaN-safe
        idx = max(0, min(idx, n_bins - 1))
        bins[idx].n += 1
        bins[idx].n_correct += _binary(reward)
    return bins


@dataclass
class PRPoint:
    threshold: float
    n_accepted: int
    n_correct_accepted: int
    n_correct_total: int

    @property
    def precision(self) -> float:
        # P(correct | accepted) — the trustworthiness of an "accept".
        return self.n_correct_accepted / self.n_accepted if self.n_accepted else 0.0

    @property
    def recall(self) -> float:
        # fraction of the correct answers that get accepted.
        return self.n_correct_accepted / self.n_correct_total if self.n_correct_total else 0.0


def precision_recall_at(pairs: List[Tuple[float, float]], threshold: float) -> PRPoint:
    """Precision/recall of the rule 'accept iff U_V >= threshold'."""
    n_correct_total = sum(_binary(r) for _, r in pairs)
    accepted = [(u, r) for u, r in pairs if u >= threshold]
    n_correct_accepted = sum(_binary(r) for _, r in accepted)
    return PRPoint(threshold=threshold, n_accepted=len(accepted),
                   n_correct_accepted=n_correct_accepted,
                   n_correct_total=n_correct_total)


def threshold_for_precision(pairs: List[Tuple[float, float]],
                            target_precision: float,
                            min_accepts: int = 5) -> Optional[float]:
    """
    Returns the LOWEST U_V threshold whose accepted set is correct at least
    `target_precision` of the time (and accepts at least `min_accepts`
    answers, so the estimate isn't from a handful of points). None if no
    threshold reaches the target — an honest "the verifier cannot guarantee
    this precision at any cutoff on this data".

    Candidate thresholds are the observed U_V values (accepting at each
    distinct score), scanned low→high so the first qualifying one is the
    most permissive (best recall) meeting the precision bar.
    """
    if not pairs:
        return None
    candidates = sorted({round(u, 4) for u, _ in pairs})
    for t in candidates:
        pr = precision_recall_at(pairs, t)
        if pr.n_accepted >= min_accepts and pr.precision >= target_precision:
            return t
    return None


def load_uv_reward_from_traces(trace_dir: str,
                               retrieval: Optional[bool] = None) -> List[Tuple[float, float]]:
    """
    Reads (U_V, reward) pairs from traces/*.jsonl. `retrieval` (True/False)
    restricts to rows tagged with that task nature (`is_retrieval`, written
    since 2026-07-21); rows without the tag are skipped when scoping is
    requested. Uses the STORED U_V (what the verifier actually produced),
    so there is no chicken-and-egg with the current weights.
    """
    pairs = []
    if not os.path.isdir(trace_dir):
        return pairs
    for path in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "U_V" not in row or "reward" not in row:
                    continue
                if retrieval is not None and row.get("is_retrieval") != retrieval:
                    continue
                pairs.append((float(row["U_V"]), float(row["reward"])))
    return pairs


def precision_calibrated_threshold(trace_dir: str,
                                   static_fallback: float,
                                   target_precision: float = 0.8,
                                   retrieval: Optional[bool] = None,
                                   min_accepts: int = 15) -> float:
    """
    The verifier's acceptance threshold, calibrated for a TARGET PRECISION
    from historical (U_V, reward) traces (optionally scoped by retrieval vs
    action). Replaces a Youden's-J objective — which maximizes TPR−FPR and
    on this data LOWERED the threshold (0.31), hurting precision — with the
    trustworthiness-aligned objective "only accept when P(correct) is high".
    Degenerates to `static_fallback` when there isn't enough data.
    """
    pairs = load_uv_reward_from_traces(trace_dir, retrieval=retrieval)
    n_correct = sum(1 for _, r in pairs if r >= 1.0)
    n_wrong = len(pairs) - n_correct
    if n_correct < min_accepts or n_wrong < min_accepts:
        return static_fallback
    t = threshold_for_precision(pairs, target_precision, min_accepts=min_accepts)
    return t if t is not None else static_fallback
