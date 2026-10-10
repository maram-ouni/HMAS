"""
Tests for core/self_consistency.py — majority-vote self-consistency with a
confidence/abstention verdict.
"""
from core.self_consistency import normalize_answer, majority_vote, VoteResult


# ── normalize_answer ──────────────────────────────────────────────────

def test_normalize_collapses_formatting():
    assert normalize_answer("  The  Answer.  ") == "the answer"
    assert normalize_answer('"8 restaurants"') == "8 restaurants"
    assert normalize_answer("A, B, C!") == "a b c"


def test_normalize_empty_and_none():
    assert normalize_answer("") == ""
    assert normalize_answer(None) == ""
    assert normalize_answer("   ") == ""


def test_formatting_variants_vote_together():
    r = majority_vote(['8 restaurants', '"8 Restaurants."', '  8  restaurants  '])
    assert r.n_votes == 3
    assert r.agreement == 1.0
    assert "8" in r.answer


# ── majority_vote — basic winner selection ────────────────────────────

def test_clear_majority_wins():
    r = majority_vote(["Paris", "Paris", "London"])
    assert normalize_answer(r.answer) == "paris"
    assert r.agreement == round(2 / 3, 4)
    assert r.n_votes == 3
    assert r.is_confident is True
    assert r.abstain is False


def test_returns_original_spelling_of_winner():
    # winner text is a real original answer, not the normalized form
    r = majority_vote(["The Blue House", "the blue house.", "Red"])
    assert r.answer in ("The Blue House", "the blue house.")


# ── abstentions (empty answers) ───────────────────────────────────────

def test_empty_answers_are_abstentions_not_winners():
    r = majority_vote(["", "", "Paris"])
    assert normalize_answer(r.answer) == "paris"
    assert r.n_votes == 1          # only the non-empty one voted
    assert r.n_total == 3
    # a lone answer among abstentions is NOT confidence (min_votes=2)
    assert r.is_confident is False
    assert r.abstain is True


def test_all_empty_returns_abstain():
    r = majority_vote(["", None, "   "])
    assert r.answer == ""
    assert r.n_votes == 0
    assert r.n_total == 3
    assert r.abstain is True
    assert r.is_confident is False


# ── confidence thresholds ─────────────────────────────────────────────

def test_low_agreement_abstains():
    # three different answers → agreement 1/3, below default 0.5
    r = majority_vote(["A", "B", "C"])
    assert r.agreement == round(1 / 3, 4)
    assert r.is_confident is False
    assert r.abstain is True


def test_confident_when_agreement_and_votes_met():
    r = majority_vote(["A", "A", "A", "B"], min_agreement=0.5, min_votes=2)
    assert r.agreement == 0.75
    assert r.is_confident is True


def test_min_votes_gate_independent_of_agreement():
    # perfect agreement but only ONE vote → not confident
    r = majority_vote(["only one", "", ""], min_votes=2)
    assert r.agreement == 1.0
    assert r.is_confident is False


def test_custom_min_agreement():
    r = majority_vote(["A", "A", "B", "C"], min_agreement=0.6)
    assert r.agreement == 0.5
    assert r.is_confident is False   # 0.5 < 0.6


# ── determinism / tie-break ───────────────────────────────────────────

def test_tie_broken_by_earliest_appearance_deterministically():
    r1 = majority_vote(["B", "A", "A", "B"])   # tie A/B
    r2 = majority_vote(["B", "A", "A", "B"])
    assert r1.answer == r2.answer              # deterministic
    assert normalize_answer(r1.answer) == "b"  # B appears first


def test_confident_but_wrong_shows_high_agreement():
    # The failure mode from the data: model is confidently wrong. Voting
    # cannot fix it, but the HIGH agreement is exactly the signal the caller
    # needs — this is not "uncertain", it is "consistently (wrongly) sure".
    r = majority_vote(["8 restaurants", "8 restaurants", "8 restaurants"])
    assert r.agreement == 1.0
    assert r.is_confident is True   # confident — correctness is orthogonal
