"""
Tests for core/verifier_learned_weights.py (config.LEARNED_VERIFIER_WEIGHTS
and config.LEARNED_PE_WEIGHTS).
"""
import json

import pytest

from core.verifier_learned_weights import (
    compute_learned_weights, compute_learned_pe_weights, calibrate_threshold,
    _load_training_examples, MIN_EXAMPLES_PER_CLASS, PE_FEATURES,
)

STATIC = {"safety": 0.30, "specificity": 0.30, "success_v": 0.25, "relevance": 0.15}
STATIC_PE = {"success": 0.60, "efficiency": 0.25, "quality": 0.15}


def _write_trace(trace_dir, task_id, rows):
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{task_id}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_degenerates_to_static_with_no_traces(tmp_path):
    w = compute_learned_weights(str(tmp_path / "traces"), STATIC)
    assert w == STATIC


def test_degenerates_to_static_below_min_examples_per_class(tmp_path):
    trace_dir = tmp_path / "traces"
    # Only a handful of never-succeeded tasks — well under
    # MIN_EXAMPLES_PER_CLASS, and zero successful ones.
    for i in range(3):
        _write_trace(trace_dir, f"task{i}", [
            {"reward": 0.0, "dim_safety": 1.0, "dim_specificity": 0.5,
             "dim_relevance": 0.5},
        ])
    w = compute_learned_weights(str(trace_dir), STATIC)
    assert w == STATIC


def test_success_v_never_learned_stays_pinned_to_prior(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.0, "dim_safety": 0.5, "dim_specificity": 0.5,
             "dim_relevance": 0.9},
            {"reward": 1.0, "dim_safety": 1.0, "dim_specificity": 1.0,
             "dim_relevance": 1.0},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.0, "dim_safety": 0.5, "dim_specificity": 0.5,
             "dim_relevance": 0.1},
        ])
    w = compute_learned_weights(str(trace_dir), STATIC)
    assert w["success_v"] == STATIC["success_v"]


def test_weights_sum_to_one_when_fitted(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.0, "dim_safety": 0.5, "dim_specificity": 0.5,
             "dim_relevance": 0.9},
            {"reward": 1.0, "dim_safety": 1.0, "dim_specificity": 1.0,
             "dim_relevance": 1.0},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.0, "dim_safety": 0.5, "dim_specificity": 0.5,
             "dim_relevance": 0.1},
        ])
    w = compute_learned_weights(str(trace_dir), STATIC)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)


def test_predictive_dimension_gains_weight(tmp_path):
    # relevance cleanly separates eventual-success from never-succeeded;
    # safety/specificity are pure noise (identical across both classes).
    # The fitted weight on relevance should end up higher than on the
    # noise dimensions, and higher than its own static prior.
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 10
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.0, "dim_safety": 0.5, "dim_specificity": 0.5,
             "dim_relevance": 0.9},
            {"reward": 1.0, "dim_safety": 0.5, "dim_specificity": 0.5,
             "dim_relevance": 0.9},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.0, "dim_safety": 0.5, "dim_specificity": 0.5,
             "dim_relevance": 0.05},
        ])
    w = compute_learned_weights(str(trace_dir), STATIC)
    assert w["relevance"] > STATIC["relevance"]
    assert w["relevance"] > w["safety"]
    assert w["relevance"] > w["specificity"]


def test_rows_with_reward_at_least_one_excluded_from_training(tmp_path):
    # Only successful rows (reward>=1.0) exist — no reward<1.0 examples
    # at all, so training data is empty regardless of task count.
    trace_dir = tmp_path / "traces"
    for i in range(MIN_EXAMPLES_PER_CLASS + 5):
        _write_trace(trace_dir, f"task{i}", [
            {"reward": 1.0, "dim_safety": 1.0, "dim_specificity": 1.0,
             "dim_relevance": 1.0},
        ])
    examples = _load_training_examples(str(trace_dir))
    assert examples == []


def test_malformed_trace_lines_are_skipped(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / "task_bad.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write("not valid json\n")
        f.write(json.dumps({"reward": 0.0, "dim_safety": 0.5,
                            "dim_specificity": 0.5, "dim_relevance": 0.5}) + "\n")
    examples = _load_training_examples(str(trace_dir))
    assert len(examples) == 1


# ── calibrate_threshold ──────────────────────────────────────────────

# Isolates dim_safety as the only contributor to U_V so test scores can
# be set directly via dim_safety, independent of the weight-learning tests.
SAFETY_ONLY = {"safety": 1.0, "specificity": 0.0, "success_v": 0.0, "relevance": 0.0}


def test_threshold_degenerates_to_static_with_no_traces(tmp_path):
    t = calibrate_threshold(str(tmp_path / "traces"), SAFETY_ONLY, static_threshold=0.4)
    assert t == 0.4


def test_threshold_degenerates_to_static_below_min_examples_per_class(tmp_path):
    trace_dir = tmp_path / "traces"
    for i in range(3):
        _write_trace(trace_dir, f"task{i}", [
            {"reward": 0.0, "dim_safety": 0.9, "dim_specificity": 0.0, "dim_relevance": 0.0},
        ])
    t = calibrate_threshold(str(trace_dir), SAFETY_ONLY, static_threshold=0.4)
    assert t == 0.4


def test_threshold_separates_cleanly_separable_classes(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.0, "dim_safety": 1.0, "dim_specificity": 0.0, "dim_relevance": 0.0},
            {"reward": 1.0, "dim_safety": 1.0, "dim_specificity": 0.0, "dim_relevance": 0.0},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.0, "dim_safety": 0.0, "dim_specificity": 0.0, "dim_relevance": 0.0},
        ])
    # Perfect separation at score=1.0 (only reachable by the eventually-
    # succeeded class) vs score=0.0 (everyone) — the calibrated threshold
    # must land on the unique perfectly-separating candidate, far from an
    # arbitrary static default.
    t = calibrate_threshold(str(trace_dir), SAFETY_ONLY, static_threshold=0.4)
    assert t == pytest.approx(1.0)


def _write_tied_threshold_dataset(trace_dir):
    # Two candidate thresholds (0.5 and 0.8) both maximize Youden's J:
    #   t=0.5 -> TPR=1.0 (20/20), FPR=0.5 (10/20)  -> J=0.5
    #   t=0.8 -> TPR=0.5 (10/20), FPR=0.0 (0/20)   -> J=0.5
    for i in range(10):
        _write_trace(trace_dir, f"succ_mid{i}", [
            {"reward": 0.0, "dim_safety": 0.5, "dim_specificity": 0.0, "dim_relevance": 0.0},
            {"reward": 1.0, "dim_safety": 1.0, "dim_specificity": 0.0, "dim_relevance": 0.0},
        ])
    for i in range(10):
        _write_trace(trace_dir, f"succ_high{i}", [
            {"reward": 0.0, "dim_safety": 0.8, "dim_specificity": 0.0, "dim_relevance": 0.0},
            {"reward": 1.0, "dim_safety": 1.0, "dim_specificity": 0.0, "dim_relevance": 0.0},
        ])
    for i in range(10):
        _write_trace(trace_dir, f"fail_low{i}", [
            {"reward": 0.0, "dim_safety": 0.2, "dim_specificity": 0.0, "dim_relevance": 0.0},
        ])
    for i in range(10):
        _write_trace(trace_dir, f"fail_mid{i}", [
            {"reward": 0.0, "dim_safety": 0.5, "dim_specificity": 0.0, "dim_relevance": 0.0},
        ])


def test_threshold_tie_broken_toward_nearby_static_default_low(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_tied_threshold_dataset(trace_dir)
    t = calibrate_threshold(str(trace_dir), SAFETY_ONLY, static_threshold=0.6)
    assert t == pytest.approx(0.5)


def test_threshold_tie_broken_toward_nearby_static_default_high(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_tied_threshold_dataset(trace_dir)
    t = calibrate_threshold(str(trace_dir), SAFETY_ONLY, static_threshold=0.9)
    assert t == pytest.approx(0.8)


# ── compute_learned_pe_weights (U_PE — config.LEARNED_PE_WEIGHTS) ────

def test_pe_degenerates_to_static_with_no_traces(tmp_path):
    w = compute_learned_pe_weights(str(tmp_path / "traces"), STATIC_PE)
    assert w == STATIC_PE


def test_pe_success_never_learned_stays_pinned_to_prior(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.0, "dim_efficiency": 0.5, "dim_quality": 0.9},
            {"reward": 1.0, "dim_efficiency": 1.0, "dim_quality": 1.0},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.0, "dim_efficiency": 0.5, "dim_quality": 0.1},
        ])
    w = compute_learned_pe_weights(str(trace_dir), STATIC_PE)
    assert w["success"] == STATIC_PE["success"]


def test_pe_weights_sum_to_one_when_fitted(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.0, "dim_efficiency": 0.5, "dim_quality": 0.9},
            {"reward": 1.0, "dim_efficiency": 1.0, "dim_quality": 1.0},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.0, "dim_efficiency": 0.5, "dim_quality": 0.1},
        ])
    w = compute_learned_pe_weights(str(trace_dir), STATIC_PE)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)


def test_pe_predictive_dimension_gains_weight(tmp_path):
    # quality cleanly separates eventual-success from never-succeeded;
    # efficiency is pure noise (identical across both classes).
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 10
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.0, "dim_efficiency": 0.5, "dim_quality": 0.9},
            {"reward": 1.0, "dim_efficiency": 0.5, "dim_quality": 0.9},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.0, "dim_efficiency": 0.5, "dim_quality": 0.05},
        ])
    w = compute_learned_pe_weights(str(trace_dir), STATIC_PE)
    assert w["quality"] > STATIC_PE["quality"]
    assert w["quality"] > w["efficiency"]


def test_pe_threshold_separates_cleanly_separable_classes(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.0, "dim_efficiency": 1.0, "dim_quality": 0.0},
            {"reward": 1.0, "dim_efficiency": 1.0, "dim_quality": 0.0},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.0, "dim_efficiency": 0.0, "dim_quality": 0.0},
        ])
    weights = {"efficiency": 1.0, "quality": 0.0, "success": 0.0}
    t = calibrate_threshold(str(trace_dir), weights, static_threshold=0.4,
                            feature_keys=PE_FEATURES)
    assert t == pytest.approx(1.0)


# ── retrieval scoping — fits/thresholds must be per task nature, not global ──

def _write_nature_traces(trace_dir, prefix, n, is_retrieval, succ_dims, fail_dims):
    """Writes n succeeded + n failed traces, all tagged with is_retrieval."""
    for i in range(n):
        _write_trace(trace_dir, f"v1.{prefix}-succ{i}", [
            {"reward": 0.0, "is_retrieval": is_retrieval, **succ_dims},
            {"reward": 1.0, "is_retrieval": is_retrieval, **succ_dims},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"v1.{prefix}-fail{i}", [
            {"reward": 0.0, "is_retrieval": is_retrieval, **fail_dims},
        ])


def test_retrieval_scoping_ignores_unrelated_nature(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    # Plenty of retrieval (question) data (would normally trigger a fit)...
    _write_nature_traces(trace_dir, "topwork", n, True,
                         {"dim_safety": 0.9, "dim_specificity": 0.9, "dim_relevance": 0.9},
                         {"dim_safety": 0.1, "dim_specificity": 0.1, "dim_relevance": 0.1})
    # ...but scoping to action tasks (zero examples of that nature) must
    # degenerate to the static prior, not borrow the retrieval fit.
    w = compute_learned_weights(str(trace_dir), STATIC, retrieval=False)
    assert w == STATIC


def test_retrieval_scoping_uses_only_matching_nature_data(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    # Retrieval tasks: only `safety` varies with success (specificity/
    # relevance constant -> zero predictive power). Action tasks: only
    # `relevance` varies. If scoping leaked across natures, each fit
    # would pick up the OTHER nature's signal too instead of ignoring it.
    _write_nature_traces(trace_dir, "topwork", n, True,
                         {"dim_safety": 0.9, "dim_specificity": 0.5, "dim_relevance": 0.5},
                         {"dim_safety": 0.1, "dim_specificity": 0.5, "dim_relevance": 0.5})
    _write_nature_traces(trace_dir, "dashdish", n, False,
                         {"dim_safety": 0.5, "dim_specificity": 0.5, "dim_relevance": 0.9},
                         {"dim_safety": 0.5, "dim_specificity": 0.5, "dim_relevance": 0.1})

    # Each nature's own fit should reflect ONLY its own varying dimension.
    w_retrieval = compute_learned_weights(str(trace_dir), STATIC, retrieval=True)
    w_action    = compute_learned_weights(str(trace_dir), STATIC, retrieval=False)
    assert w_retrieval["safety"] > w_retrieval["relevance"]
    assert w_action["relevance"] > w_action["safety"]

    w_global = compute_learned_weights(str(trace_dir), STATIC)  # no scoping
    assert w_global != w_retrieval  # cross-nature mixing changes the fit


def test_retrieval_none_matches_unscoped_behavior(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    _write_nature_traces(trace_dir, "topwork", n, True,
                         {"dim_safety": 0.9, "dim_specificity": 0.5, "dim_relevance": 0.5},
                         {"dim_safety": 0.1, "dim_specificity": 0.5, "dim_relevance": 0.5})
    assert (compute_learned_weights(str(trace_dir), STATIC)
            == compute_learned_weights(str(trace_dir), STATIC, retrieval=None))


def test_rows_without_is_retrieval_field_excluded_when_scoping(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    # Old-format traces (written before this field existed) must not be
    # silently included in a scoped fit — degenerates to static.
    for i in range(n):
        _write_trace(trace_dir, f"v1.old-succ{i}", [
            {"reward": 0.0, "dim_safety": 0.9, "dim_specificity": 0.9, "dim_relevance": 0.9},
            {"reward": 1.0, "dim_safety": 0.9, "dim_specificity": 0.9, "dim_relevance": 0.9},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"v1.old-fail{i}", [
            {"reward": 0.0, "dim_safety": 0.1, "dim_specificity": 0.1, "dim_relevance": 0.1},
        ])
    assert compute_learned_weights(str(trace_dir), STATIC, retrieval=True) == STATIC
    assert compute_learned_weights(str(trace_dir), STATIC, retrieval=False) == STATIC
