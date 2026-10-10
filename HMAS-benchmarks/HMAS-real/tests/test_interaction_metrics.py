"""
Tests for core/interaction_metrics.py (config.INTERACTION_METRICS_ENABLED).
"""
import pytest

from core.interaction_metrics import InteractionMetrics, CONFLICT_GAP


@pytest.fixture
def im(tmp_path):
    return InteractionMetrics(results_dir=str(tmp_path))


def test_empty_summary(im):
    s = im.summary()
    assert s == {"n": 0, "conflict_rate": 0.0, "mean_disagreement": 0.0,
                 "pe_reward_gap": 0.0, "v_reward_gap": 0.0}


def test_empty_summary_line_does_not_crash(im):
    assert "No history" in im.summary_line()


def test_conflict_rate_matches_bdi_model_definition(im):
    # Same rule as BeliefBase.brf() (core/bdi_model.py): conflict iff
    # quality != "success" and (u_pe - u_v) > CONFLICT_GAP.
    im.record(u_pe=0.9, u_v=0.9 - CONFLICT_GAP - 0.01, reward=0.0, quality="rejected")  # conflict
    im.record(u_pe=0.9, u_v=0.9 - CONFLICT_GAP + 0.01, reward=0.0, quality="rejected")  # not (gap too small)
    im.record(u_pe=0.9, u_v=0.1, reward=1.0, quality="success")                          # not (success)
    s = im.summary()
    assert s["n"] == 3
    assert s["conflict_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_mean_disagreement_is_symmetric_absolute_gap(im):
    im.record(u_pe=0.8, u_v=0.5, reward=0.0, quality="rejected")   # gap 0.3
    im.record(u_pe=0.2, u_v=0.6, reward=0.0, quality="rejected")   # gap 0.4 (V higher this time)
    s = im.summary()
    assert s["mean_disagreement"] == pytest.approx(0.35, abs=1e-4)


def test_reward_gap_zero_without_both_classes(im):
    im.record(u_pe=0.9, u_v=0.9, reward=1.0, quality="success")
    im.record(u_pe=0.8, u_v=0.8, reward=1.0, quality="success")
    s = im.summary()
    assert s["pe_reward_gap"] == 0.0
    assert s["v_reward_gap"] == 0.0


def test_more_diagnostic_utility_identified_correctly(im):
    # U_V tracks ground truth well (high when reward=1, low when reward=0);
    # U_PE is flat/uninformative regardless of outcome.
    for _ in range(5):
        im.record(u_pe=0.7, u_v=0.9, reward=1.0, quality="success")
    for _ in range(5):
        im.record(u_pe=0.7, u_v=0.1, reward=0.0, quality="rejected")
    s = im.summary()
    assert s["v_reward_gap"] > s["pe_reward_gap"]
    assert "U_V" in im.summary_line()


def test_persistence_round_trip(tmp_path):
    im1 = InteractionMetrics(results_dir=str(tmp_path))
    im1.record(u_pe=0.5, u_v=0.6, reward=1.0, quality="success")
    im2 = InteractionMetrics(results_dir=str(tmp_path))
    assert im2.summary() == im1.summary()
