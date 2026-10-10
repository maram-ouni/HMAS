# ─────────────────────────────────────────────────────────
# verifier_learned_weights.py — LEARNED VERIFIER/PLANNER-EXECUTER
# DIMENSION WEIGHTS (opt-in)
#
# Also calibrates the acceptance threshold itself (calibrate_threshold):
# a retrospective replay showed that reweighting alone almost never
# crosses a FIXED threshold gate, so it changes the continuous payoff
# score but essentially never changes an actual accept/reject decision.
# Calibrating the threshold from the same historical data is the
# complementary, lower-dimensional fix.
#
# (Historical note: this superseded and REPLACED an earlier
# self-referential heuristic — a dimension whose OWN historical scores
# were low gained weight, without ever checking whether it actually
# predicted the ground-truth outcome (reward). That heuristic was removed.)
#
# This module closes that gap with a small, auditable model instead of
# a heuristic: a logistic regression fit on real historical traces
# (config.TRACE_ENABLED, <memory_dir>/traces/<task_id>.jsonl), predicting
# "did this task eventually succeed" from the non-success dimension
# scores observed at a NOT-YET-successful attempt. Used for BOTH:
#   - U_V (config.LEARNED_VERIFIER_WEIGHTS)  : safety/specificity/relevance
#   - U_PE (config.LEARNED_PE_WEIGHTS)       : efficiency/quality
#
# Why not regress directly on reward with the "success" dimension as a
# feature? dim_success IS reward (1.0 if reward>=1.0 else 0.0), so that
# would trivially learn weight=1 on success and 0 elsewhere — circular,
# not a finding. Training is therefore restricted to rows where
# reward<1.0 (dim_success constant at 0 there, dropped from the feature
# vector), with the label taken from whether the SAME task_id eventually
# reached reward>=1.0 later in its trace file. The success weight is
# never learned — it stays pinned at its static prior, since it is
# structurally uninformative in the only regime trained on here.
#
# The model is intentionally tiny (a handful of coefficients, no bias
# term used downstream, pure-Python gradient descent — no numpy
# dependency) so the result stays interpretable: each dimension's share
# of the learned weight literally IS "how much this signal correlated
# with eventual success, historically" — not a black box.
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import glob
import json
import math
import os

# U_V's non-success dimensions (default — preserves the original API).
FEATURES = ("safety", "specificity", "relevance")

# U_PE's non-success dimensions.
PE_FEATURES = ("efficiency", "quality")

# Below this many examples per class (task eventually succeeded / never
# did), the fit would mostly capture noise — fall back to the static
# prior rather than produce weights that look precise but aren't.
MIN_EXAMPLES_PER_CLASS = 15


def _load_training_examples(trace_dir: str, feature_keys: tuple = FEATURES,
                            retrieval: bool = None) -> list:
    """
    Reads every traces/<task_id>.jsonl file and builds (features, label)
    pairs. label=1 if that task_id ever reached reward>=1.0 across any
    recorded attempt in the file (append-only across shots/runs), else 0.
    Only rows with reward<1.0 become training examples. `feature_keys`
    selects which `dim_<key>` fields to read (defaults to U_V's).

    `retrieval`, when given (True/False), restricts training to rows from
    the SAME task nature — factual-question retrieval tasks vs
    action/transaction tasks (core.payoff_calculator.is_retrieval_task) —
    rather than by site family: a single global fit mixing every site
    together let one site's behavior push weights around for a completely
    unrelated one (see AVANCEMENT.md), and site family isn't even the
    axis that matters — agents/rule_based_verifier.py's own quality
    decision already branches on retrieval vs action, not on site.
    Rows written before this field existed (no "is_retrieval" key) are
    skipped when scoping is requested, same as any other missing feature.
    Filtering means fewer available examples, so this typically falls back
    to the static prior more often (MIN_EXAMPLES_PER_CLASS unchanged) —
    intentional: a noisy scoped fit is worse than no fit at all.
    """
    examples = []
    if not os.path.isdir(trace_dir):
        return examples

    for path in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            continue

        task_succeeded = any(r.get("reward", 0.0) >= 1.0 for r in rows)
        label = 1 if task_succeeded else 0
        for r in rows:
            if r.get("reward", 0.0) >= 1.0:
                continue
            if retrieval is not None and r.get("is_retrieval") != retrieval:
                continue
            features = [float(r.get(f"dim_{key}", 0.0)) for key in feature_keys]
            examples.append((features, label))
    return examples


def _fit_logistic_regression(examples: list, n_features: int,
                             lr: float = 0.3, epochs: int = 800) -> list:
    """
    Plain batch gradient descent on an n-feature logistic regression.
    Returns the fitted coefficients in feature order — the bias term is
    fit (to center the model) but discarded downstream: only the
    RELATIVE magnitude of each coefficient is used to redistribute
    weight, not the model's standalone predictions.
    """
    weights = [0.0] * n_features
    bias = 0.0
    n = len(examples)

    for _ in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for features, label in examples:
            z = bias + sum(w * x for w, x in zip(weights, features))
            z = max(-30.0, min(30.0, z))
            pred = 1.0 / (1.0 + math.exp(-z))
            error = pred - label
            for i, x in enumerate(features):
                grad_w[i] += error * x
            grad_b += error
        weights = [w - lr * (g / n) for w, g in zip(weights, grad_w)]
        bias = bias - lr * (grad_b / n)

    return weights


def _compute_learned_weights(trace_dir: str, static_weights: dict,
                             feature_keys: tuple, pinned_key: str,
                             retrieval: bool = None) -> dict:
    """Generic fit shared by U_V and U_PE — see public wrappers below."""
    examples = _load_training_examples(trace_dir, feature_keys, retrieval)
    n_pos = sum(1 for _, y in examples if y == 1)
    n_neg = sum(1 for _, y in examples if y == 0)
    if n_pos < MIN_EXAMPLES_PER_CLASS or n_neg < MIN_EXAMPLES_PER_CLASS:
        return dict(static_weights)

    coefs = _fit_logistic_regression(examples, len(feature_keys))
    magnitudes = [abs(c) for c in coefs]
    total_mag = sum(magnitudes)
    if total_mag <= 0:
        return dict(static_weights)

    pinned_prior = static_weights.get(pinned_key, 0.0)
    remaining = 1.0 - pinned_prior
    learned = {
        dim: remaining * (mag / total_mag)
        for dim, mag in zip(feature_keys, magnitudes)
    }
    learned[pinned_key] = pinned_prior
    return {dim: round(w, 4) for dim, w in learned.items()}


def compute_learned_weights(trace_dir: str, static_weights: dict,
                            retrieval: bool = None) -> dict:
    """
    Fits weights for U_V's safety/specificity/relevance from real
    historical traces; success_v is always pinned to
    `static_weights["success_v"]`.

    `retrieval` (True/False, from core.payoff_calculator.is_retrieval_task)
    restricts the fit to that task nature alone, so a factual-question
    task's failure profile can't push weights around for an unrelated
    action/transaction task.

    Degenerates to `static_weights` unchanged when there isn't enough
    data of both classes (task eventually succeeded / never did) to fit
    a meaningful signal.
    """
    return _compute_learned_weights(trace_dir, static_weights, FEATURES,
                                    "success_v", retrieval)


def compute_learned_pe_weights(trace_dir: str, static_weights: dict,
                               retrieval: bool = None) -> dict:
    """
    Fits weights for U_PE's efficiency/quality from real historical
    traces; success is always pinned to `static_weights["success"]`.
    Same method as compute_learned_weights, applied to the
    PlannerExecuter's utility instead of the Verifier's.

    `retrieval` restricts the fit to that task nature alone — see
    compute_learned_weights.

    Degenerates to `static_weights` unchanged when there isn't enough
    data of both classes.
    """
    return _compute_learned_weights(trace_dir, static_weights, PE_FEATURES,
                                    "success", retrieval)


def _weighted_score(weights: dict, feature_keys: tuple, features: list) -> float:
    # The pinned (success) dimension's contribution is always 0 here —
    # see module docstring: dim_success is constant (0) in the
    # reward<1.0 training subset.
    return sum(weights[key] * value for key, value in zip(feature_keys, features))


def calibrate_threshold(trace_dir: str, weights: dict, static_threshold: float,
                        feature_keys: tuple = FEATURES,
                        retrieval: bool = None) -> float:
    """
    Picks the acceptance threshold (U_V's "partial vs rejected" gate in
    agents/rule_based_verifier.py — config's V_THRESHOLD — or U_PE's
    "keep vs explore" gate in core/strategy_learner.py — config's
    PE_SUCCESS_THRESHOLD / KEEP_THRESHOLD) that best separates "task
    eventually succeeded" from "never did" on real historical traces,
    using `weights` (static or learned) to score each example.
    Complements compute_learned_weights/compute_learned_pe_weights:
    reweighting alone can shift every score by roughly the same amount
    without ever crossing a FIXED threshold — calibrating the threshold
    itself is what actually changes accept/reject decisions.
    `feature_keys` must match the dimensions used to build `weights`
    (FEATURES for U_V, PE_FEATURES for U_PE). `retrieval` restricts to
    that task nature alone — see compute_learned_weights.

    Threshold search uses Youden's J statistic (TPR - FPR), which is
    robust to the severe class imbalance here (few tasks eventually
    succeed): plain accuracy would be maximized by a threshold that
    rejects everything, which is useless. Ties broken by picking the
    candidate closest to `static_threshold`, so an equally-good fit
    doesn't drift arbitrarily from today's behavior.

    Degenerates to `static_threshold` unchanged when there isn't enough
    data of both classes.
    """
    examples = _load_training_examples(trace_dir, feature_keys, retrieval)
    n_pos = sum(1 for _, y in examples if y == 1)
    n_neg = sum(1 for _, y in examples if y == 0)
    if n_pos < MIN_EXAMPLES_PER_CLASS or n_neg < MIN_EXAMPLES_PER_CLASS:
        return static_threshold

    scored = [
        (_weighted_score(weights, feature_keys, features), label)
        for features, label in examples
    ]
    candidates = sorted({score for score, _ in scored})
    if not candidates:
        return static_threshold

    best_j, best_thresholds = None, []
    for t in candidates:
        tp = sum(1 for score, label in scored if label == 1 and score >= t)
        fn = sum(1 for score, label in scored if label == 1 and score < t)
        fp = sum(1 for score, label in scored if label == 0 and score >= t)
        tn = sum(1 for score, label in scored if label == 0 and score < t)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if best_j is None or j > best_j:
            best_j, best_thresholds = j, [t]
        elif j == best_j:
            best_thresholds.append(t)

    return round(min(best_thresholds, key=lambda t: abs(t - static_threshold)), 4)
