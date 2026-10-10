# ─────────────────────────────────────────────────────────
# self_consistency.py — MAJORITY-VOTE SELF-CONSISTENCY + CONFIDENCE
#
# Reliability mechanism for the stochastic LLM Executor: instead of
# committing a single attempt's answer, aggregate k independent attempts
# and keep the MOST FREQUENT answer (self-consistency). Crucially, it also
# returns an AGREEMENT ratio used as a confidence signal — a trustworthy
# system commits on high agreement and abstains / flags low agreement.
#
# Empirical note (see docs/AVANCEMENT.md, 21 July): on this benchmark,
# majority voting beats a single attempt (first-shot) but NOT the oracle
# "any-shot" upper bound, and it CANNOT rescue tasks where the model is
# systematically wrong (the frequent answer is then a wrong answer). The
# confidence ratio is what lets the caller tell those cases apart: a
# confident-but-wrong task shows HIGH agreement on a wrong answer, an
# unstable task shows LOW agreement — both are actionable signals the raw
# majority answer alone hides.
#
# Pure Python, no dependency, no LLM call: it only aggregates answers that
# were already produced. Deterministic given its inputs.
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional


def normalize_answer(answer: Optional[str]) -> str:
    """
    Canonical form for comparing free-text answers: lowercased, whitespace
    collapsed, surrounding quotes/punctuation stripped. Two answers that
    differ only in formatting vote together. Empty/None → "".
    """
    a = (answer or "").strip().lower()
    a = re.sub(r"\s+", " ", a)
    a = re.sub(r"[\"'.,:;!?()]", "", a)
    return a.strip()


@dataclass
class VoteResult:
    answer: str            # the winning ORIGINAL answer text ("" if no valid votes)
    agreement: float       # winning votes / total non-empty votes, in [0, 1]
    n_votes: int           # number of non-empty answers that participated
    n_total: int           # total attempts considered (including empty)
    is_confident: bool     # agreement >= threshold AND enough votes
    abstain: bool          # True → caller should flag low confidence / not commit


def majority_vote(answers: List[Optional[str]],
                  min_agreement: float = 0.5,
                  min_votes: int = 2) -> VoteResult:
    """
    Aggregates `answers` (one per independent attempt) into a single voted
    answer plus a confidence verdict.

    - Empty/whitespace-only answers are treated as ABSTENTIONS: they count
      toward n_total (so an executor that rarely answers scores as low
      confidence) but never win the vote.
    - The winner is the most frequent NORMALIZED answer; the returned text
      is the first original spelling of that normalized form (readability).
    - `is_confident` requires BOTH agreement >= min_agreement AND at least
      `min_votes` non-empty answers — a lone answer among abstentions is not
      confidence, it is thin evidence.
    - `abstain` is the inverse: the caller should surface uncertainty rather
      than commit a low-confidence answer (the trustworthiness lever).

    Ties are broken by the answer that appeared earliest (stable, so the
    result is deterministic given the input order).
    """
    n_total = len(answers)
    pairs = [(a, normalize_answer(a)) for a in answers]
    valid = [(orig, norm) for orig, norm in pairs if norm]
    n_votes = len(valid)

    if n_votes == 0:
        return VoteResult(answer="", agreement=0.0, n_votes=0, n_total=n_total,
                          is_confident=False, abstain=True)

    counts = Counter(norm for _, norm in valid)
    top_norm, top_n = counts.most_common(1)[0]
    # Deterministic tie-break: earliest-appearing normalized form among the maxima.
    best = max(counts.values())
    if list(counts.values()).count(best) > 1:
        for _, norm in valid:
            if counts[norm] == best:
                top_norm = norm
                top_n = best
                break
    # First original spelling of the winning normalized answer.
    winner_text = next(orig for orig, norm in valid if norm == top_norm)

    agreement = top_n / n_votes
    is_confident = agreement >= min_agreement and n_votes >= min_votes
    return VoteResult(
        answer=winner_text,
        agreement=round(agreement, 4),
        n_votes=n_votes,
        n_total=n_total,
        is_confident=is_confident,
        abstain=not is_confident,
    )
