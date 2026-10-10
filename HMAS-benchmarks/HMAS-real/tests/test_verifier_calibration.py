"""
Tests for core/verifier_calibration.py — reliability diagram, precision/recall
of the U_V accept rule, and the calibrated-threshold search.
"""
from core.verifier_calibration import (
    reliability_bins, precision_recall_at, threshold_for_precision,
)


# ── reliability_bins ──────────────────────────────────────────────────

def test_bins_cover_unit_interval():
    bins = reliability_bins([], n_bins=5)
    assert len(bins) == 5
    assert bins[0].lo == 0.0 and bins[-1].hi == 1.0
    assert all(b.n == 0 for b in bins)


def test_bins_count_and_p_correct():
    # width=0.2: 0.05,0.1 → bucket 0 [0,0.2) (both wrong);
    #            0.9,0.95 → bucket 4 [0.8,1.0] (both correct)
    pairs = [(0.05, 0.0), (0.1, 0.0), (0.9, 1.0), (0.95, 1.0)]
    bins = reliability_bins(pairs, n_bins=5)
    assert bins[0].n == 2 and bins[0].p_correct == 0.0
    assert bins[4].n == 2 and bins[4].p_correct == 1.0


def test_bins_monotonic_for_well_calibrated_data():
    pairs = [(0.05, 0), (0.15, 0), (0.45, 0), (0.55, 1), (0.85, 1), (0.95, 1)]
    p = [b.p_correct for b in reliability_bins(pairs, n_bins=5)]
    # non-decreasing across populated buckets
    populated = [b.p_correct for b in reliability_bins(pairs, 5) if b.n]
    assert populated == sorted(populated)


def test_uv_of_one_lands_in_last_bin():
    bins = reliability_bins([(1.0, 1.0)], n_bins=5)
    assert bins[-1].n == 1


# ── precision_recall_at ───────────────────────────────────────────────

def test_precision_recall_basic():
    # accept iff U_V>=0.5: accepts (0.6,1),(0.8,0),(0.9,1) → 2 correct/3
    pairs = [(0.1, 0.0), (0.6, 1.0), (0.8, 0.0), (0.9, 1.0), (0.4, 1.0)]
    pr = precision_recall_at(pairs, 0.5)
    assert pr.n_accepted == 3
    assert pr.n_correct_accepted == 2
    assert round(pr.precision, 3) == round(2 / 3, 3)
    # 3 correct total (0.6,0.9,0.4); 2 of them accepted
    assert pr.n_correct_total == 3
    assert round(pr.recall, 3) == round(2 / 3, 3)


def test_precision_one_when_only_correct_accepted():
    pairs = [(0.9, 1.0), (0.95, 1.0), (0.1, 0.0)]
    pr = precision_recall_at(pairs, 0.5)
    assert pr.precision == 1.0


def test_empty_accept_set_precision_zero():
    pr = precision_recall_at([(0.1, 0.0), (0.2, 1.0)], 0.9)
    assert pr.n_accepted == 0
    assert pr.precision == 0.0


# ── threshold_for_precision ───────────────────────────────────────────

def test_finds_threshold_meeting_target():
    # below 0.5 answers are wrong, at/above 0.5 correct
    pairs = [(0.1, 0), (0.2, 0), (0.3, 0), (0.5, 1), (0.6, 1),
             (0.7, 1), (0.8, 1), (0.9, 1)]
    t = threshold_for_precision(pairs, target_precision=1.0, min_accepts=3)
    assert t is not None
    pr = precision_recall_at(pairs, t)
    assert pr.precision >= 1.0
    assert pr.n_accepted >= 3


def test_returns_none_when_target_unreachable():
    # mixed everywhere → no threshold hits precision 0.95 with >=5 accepts
    pairs = [(u / 10, (u % 2)) for u in range(10)]  # alternating correct/wrong
    assert threshold_for_precision(pairs, target_precision=0.95, min_accepts=5) is None


def test_prefers_lowest_qualifying_threshold_best_recall():
    # both 0.5 and 0.7 give precision 1.0; the lower (better recall) wins
    pairs = [(0.1, 0), (0.5, 1), (0.6, 1), (0.7, 1), (0.8, 1)]
    t = threshold_for_precision(pairs, target_precision=1.0, min_accepts=3)
    assert t == 0.5


def test_min_accepts_gate_rejects_thin_evidence():
    # only 2 accepts at high threshold, both correct, but min_accepts=5
    pairs = [(0.1, 0)] * 8 + [(0.9, 1), (0.95, 1)]
    assert threshold_for_precision(pairs, target_precision=1.0, min_accepts=5) is None
