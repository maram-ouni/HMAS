"""
Tests for core/contract_net.py — the Contract-Net bidding/award core.
"""
import pytest

from core.contract_net import compute_bid, award, run_auction, ModelReliability


# ── compute_bid ─────────────────────────────────────────────────────────

def test_bid_equals_prior_with_no_evidence():
    assert compute_bid(0, 0, prior=0.5) == 0.5
    assert compute_bid(0, 0, prior=0.3) == 0.3


def test_bid_converges_to_raw_rate_with_much_evidence():
    b = compute_bid(90, 100, prior=0.5, k=4.0)
    assert abs(b - 0.9) < 0.05


def test_bid_smoothing_dampens_a_single_lucky_success():
    # 1/1 raw = 1.0, but smoothed stays well below a model with 45/50.
    lucky = compute_bid(1, 1, prior=0.5, k=4.0)
    solid = compute_bid(45, 50, prior=0.5, k=4.0)
    assert lucky < solid


# ── award ───────────────────────────────────────────────────────────────

def test_award_picks_highest_bidder():
    assert award({"a": 0.2, "b": 0.9, "c": 0.5}) == "b"


def test_award_tie_break_is_deterministic_smallest_name():
    assert award({"zeta": 0.7, "alpha": 0.7}) == "alpha"


def test_award_empty_raises():
    with pytest.raises(ValueError):
        award({})


# ── run_auction (integration with a fake reliability) ───────────────────

class _FakeReliability:
    def __init__(self, table):
        self._t = table  # {model: {category: (succ, n)}}

    def models(self):
        return sorted(self._t)

    def record(self, model, category):
        return self._t.get(model, {}).get(category, (0, 0))


def test_auction_routes_category_to_best_model():
    rel = _FakeReliability({
        "big":   {"dashdish": (9, 10), "staynb": (0, 10)},
        "small": {"dashdish": (1, 10), "staynb": (0, 10)},
    })
    r_dash = run_auction("dashdish", rel)
    assert r_dash["winner"] == "big"          # big is far more reliable here
    assert r_dash["bids"]["big"] > r_dash["bids"]["small"]


def test_auction_unknown_category_falls_back_to_prior_tie():
    rel = _FakeReliability({"big": {}, "small": {}})
    r = run_auction("never_seen", rel)
    # no evidence for anyone -> all bid the prior -> deterministic tie-break
    assert r["winner"] == "big"
    assert r["evidence"]["big"] == (0, 0)


# ── ModelReliability loading (round-trip via a temp file) ───────────────

def test_model_reliability_loads_and_aggregates_by_category(tmp_path):
    data = [
        {"task_id": "v1.dashdish-1", "reward": 1.0},
        {"task_id": "v1.dashdish-6", "reward": 0.0},
        {"task_id": "v1.staynb-3",   "reward": 1.0},
    ]
    f = tmp_path / "modelX_results.json"
    f.write_text(__import__("json").dumps(data))
    rel = ModelReliability({"modelX": [str(f)]})
    assert rel.record("modelX", "dashdish") == (1, 2)   # 1 success / 2 tasks
    assert rel.record("modelX", "staynb") == (1, 1)
    assert rel.record("modelX", "unseen") == (0, 0)
