"""
Tests for core/world_model.py, focused on the
WORLD_MODEL_BAYESIAN_SMOOTHING opt-in (hard cutoff vs. smooth Bayesian
estimate for executor_reliability()).
"""
import pytest

import core.world_model as world_model_module
from core.world_model import WorldModel


def make_wm(tmp_path):
    return WorldModel(results_dir=str(tmp_path))


# ── category beliefs ─────────────────────────────────────────────────

def test_category_avg_u_v_empty_is_zero(tmp_path):
    wm = make_wm(tmp_path)
    assert wm.category_avg_u_v("unknown_cat") == 0.0
    assert wm.category_samples("unknown_cat") == 0


def test_category_avg_u_v_averages_recorded_outcomes(tmp_path):
    wm = make_wm(tmp_path)
    wm.record_task_outcome("cat", 0.8)
    wm.record_task_outcome("cat", 0.4)
    assert wm.category_avg_u_v("cat") == pytest.approx(0.6)
    assert wm.category_samples("cat") == 2


# ── executor_reliability(): hard cutoff (flag off, default) ──────────

def test_reliability_returns_default_below_min_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(world_model_module, "WORLD_MODEL_BAYESIAN_SMOOTHING", False)
    wm = make_wm(tmp_path)
    wm.record_plan_outcome("cat", "planA", success=False)
    wm.record_plan_outcome("cat", "planA", success=False)
    # Only 2 samples, min_samples default is 4 -> optimistic default kept
    assert wm.executor_reliability("cat", "planA", min_samples=4, default=1.0) == 1.0


def test_reliability_returns_raw_rate_at_or_above_min_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(world_model_module, "WORLD_MODEL_BAYESIAN_SMOOTHING", False)
    wm = make_wm(tmp_path)
    for success in (True, True, False, False):
        wm.record_plan_outcome("cat", "planA", success=success)
    assert wm.executor_reliability("cat", "planA", min_samples=4) == 0.5


def test_reliability_unknown_plan_returns_default(tmp_path, monkeypatch):
    monkeypatch.setattr(world_model_module, "WORLD_MODEL_BAYESIAN_SMOOTHING", False)
    wm = make_wm(tmp_path)
    assert wm.executor_reliability("cat", "never_seen", default=0.42) == 0.42


# ── executor_reliability(): Bayesian smoothing (opt-in) ───────────────

def test_bayesian_smoothing_degenerates_to_default_at_zero_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(world_model_module, "WORLD_MODEL_BAYESIAN_SMOOTHING", True)
    wm = make_wm(tmp_path)
    assert wm.executor_reliability("cat", "never_seen", default=0.7) == 0.7


def test_bayesian_smoothing_has_no_discontinuity_at_min_samples(tmp_path, monkeypatch):
    # The whole point of the opt-in: avoid the hard jump the default
    # cutoff makes right at min_samples. Compare n=min_samples-1 vs
    # n=min_samples with IDENTICAL underlying success rate — the
    # smoothed estimates should be close together (continuous), whereas
    # the hard-cutoff version jumps from `default` to the raw rate.
    monkeypatch.setattr(world_model_module, "WORLD_MODEL_BAYESIAN_SMOOTHING", True)
    wm = make_wm(tmp_path)
    for success in (True, True, True):   # n=3, all successes
        wm.record_plan_outcome("cat", "planA", success=success)
    r3 = wm.executor_reliability("cat", "planA", min_samples=4, default=1.0, k=4.0)
    wm.record_plan_outcome("cat", "planA", success=True)  # n=4, still all successes
    r4 = wm.executor_reliability("cat", "planA", min_samples=4, default=1.0, k=4.0)
    assert abs(r4 - r3) < 0.15   # smooth, not a hard jump


def test_bayesian_smoothing_converges_to_raw_rate_as_samples_grow(tmp_path, monkeypatch):
    monkeypatch.setattr(world_model_module, "WORLD_MODEL_BAYESIAN_SMOOTHING", True)
    wm = make_wm(tmp_path)
    for _ in range(200):
        wm.record_plan_outcome("cat", "planA", success=False)
    # Raw rate is 0.0 — with 200 samples the Bayesian estimate should be
    # very close to 0, regardless of an optimistic default=1.0 prior.
    r = wm.executor_reliability("cat", "planA", min_samples=4, default=1.0, k=4.0)
    assert r < 0.05


def test_persistence_round_trip(tmp_path):
    wm1 = make_wm(tmp_path)
    wm1.record_task_outcome("cat", 0.5)
    wm1.record_plan_outcome("cat", "planA", success=True)
    wm2 = make_wm(tmp_path)
    assert wm2.category_avg_u_v("cat") == 0.5
    assert wm2.executor_reliability("cat", "planA", min_samples=1) == 1.0
