"""
Tests for the verifier's cross-attempt agreement signal
(config.VERIFIER_AGREEMENT_ENABLED): RuleBasedVerifierAgent._record_agreement.

Exercised on a lightweight stub carrying only the attributes the method
touches, so the full threaded agent / message bus isn't needed.
"""
import agents.rule_based_verifier as rbv
from agents.rule_based_verifier import RuleBasedVerifierAgent


class _Stub:
    _record_agreement = RuleBasedVerifierAgent._record_agreement

    def __init__(self):
        self._task_answers = {}


def test_off_returns_none_none(monkeypatch):
    monkeypatch.setattr(rbv, "VERIFIER_AGREEMENT_ENABLED", False)
    s = _Stub()
    assert s._record_agreement("t1", "Paris") == (None, None)
    # off → does not even accumulate
    assert s._task_answers == {}


def test_convergent_answers_high_agreement_confident(monkeypatch):
    monkeypatch.setattr(rbv, "VERIFIER_AGREEMENT_ENABLED", True)
    s = _Stub()
    s._record_agreement("t1", "Paris")
    agreement, abstain = s._record_agreement("t1", "paris.")  # normalizes equal
    assert agreement == 1.0
    assert abstain is False


def test_divergent_answers_abstain(monkeypatch):
    monkeypatch.setattr(rbv, "VERIFIER_AGREEMENT_ENABLED", True)
    s = _Stub()
    s._record_agreement("t1", "Paris")
    s._record_agreement("t1", "London")
    agreement, abstain = s._record_agreement("t1", "Berlin")  # 1/3 agreement
    assert round(agreement, 2) == 0.33
    assert abstain is True   # below default min_agreement 0.5


def test_tie_at_half_is_confident_by_default(monkeypatch):
    # Documents module default: 50% plurality meets min_agreement=0.5.
    monkeypatch.setattr(rbv, "VERIFIER_AGREEMENT_ENABLED", True)
    s = _Stub()
    s._record_agreement("t1", "Paris")
    agreement, abstain = s._record_agreement("t1", "London")
    assert agreement == 0.5
    assert abstain is False


def test_single_answer_abstains_thin_evidence(monkeypatch):
    monkeypatch.setattr(rbv, "VERIFIER_AGREEMENT_ENABLED", True)
    s = _Stub()
    agreement, abstain = s._record_agreement("t1", "Paris")
    # one vote → min_votes not met → abstain
    assert abstain is True


def test_accumulates_per_task(monkeypatch):
    monkeypatch.setattr(rbv, "VERIFIER_AGREEMENT_ENABLED", True)
    s = _Stub()
    s._record_agreement("t1", "A")
    s._record_agreement("t1", "A")
    s._record_agreement("t2", "B")
    assert s._task_answers["t1"] == ["A", "A"]
    assert s._task_answers["t2"] == ["B"]


def test_majority_wins_over_minority(monkeypatch):
    monkeypatch.setattr(rbv, "VERIFIER_AGREEMENT_ENABLED", True)
    s = _Stub()
    for a in ["8 items", "8 items", "8 items", "9 items"]:
        agreement, abstain = s._record_agreement("t1", a)
    assert agreement == 0.75   # 3/4
    assert abstain is False
