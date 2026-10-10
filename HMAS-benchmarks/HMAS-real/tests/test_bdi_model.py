"""
Tests for core/bdi_model.py, focused on the two enrichments that live
here: consecutive_conflict tracking (BeliefBase.brf) and the
CONFLICT_DETECTION_ENABLED-gated reconsideration in
BDIState.should_reconsider().
"""
import core.bdi_model as bdi_model
from core.bdi_model import BeliefBase, BDIState, CommitmentStrategy, Desire, DesireStatus


# ── BeliefBase.brf(): consecutive_conflict tracking ─────────────────────

def test_brf_increments_consecutive_conflict_on_disagreement():
    b = BeliefBase()
    # quality != success AND (u_pe - u_v) > 0.3 -> conflict
    b.brf(reward=0.0, u_pe=0.9, u_v=0.4, quality="rejected", feedback="", hint="h1")
    assert b.consecutive_conflict == 1
    b.brf(reward=0.0, u_pe=0.9, u_v=0.4, quality="rejected", feedback="", hint="h2")
    assert b.consecutive_conflict == 2


def test_brf_resets_consecutive_conflict_when_not_disagreeing():
    b = BeliefBase()
    b.brf(reward=0.0, u_pe=0.9, u_v=0.4, quality="rejected", feedback="", hint="h1")
    assert b.consecutive_conflict == 1
    # Small gap between u_pe/u_v -> no conflict this time -> resets
    b.brf(reward=0.0, u_pe=0.5, u_v=0.45, quality="rejected", feedback="", hint="h2")
    assert b.consecutive_conflict == 0


def test_brf_no_conflict_on_success_even_with_large_gap():
    b = BeliefBase()
    b.brf(reward=1.0, u_pe=0.95, u_v=0.6, quality="success", feedback="", hint="h1")
    assert b.consecutive_conflict == 0


# ── BDIState.should_reconsider(): CommitmentStrategy behavior ───────────

def test_should_reconsider_blind_never_reconsiders():
    bdi = BDIState(commitment_strategy=CommitmentStrategy.BLIND)
    bdi.set_task("t1", "goal", "cat")
    bdi.beliefs.brf(0.0, 0.9, 0.9, "rejected", "", "h")  # even a clean success-like signal
    assert bdi.should_reconsider() is False


def test_should_reconsider_open_minded_always_reconsiders():
    bdi = BDIState(commitment_strategy=CommitmentStrategy.OPEN_MINDED)
    bdi.set_task("t1", "goal", "cat")
    assert bdi.should_reconsider() is True


def test_should_reconsider_single_minded_on_success_or_partial():
    bdi = BDIState(commitment_strategy=CommitmentStrategy.SINGLE_MINDED)
    bdi.set_task("t1", "goal", "cat")
    bdi.beliefs.brf(1.0, 0.9, 0.9, "success", "", "h")
    assert bdi.should_reconsider() is True


def test_should_reconsider_single_minded_on_impossible_plan():
    bdi = BDIState(commitment_strategy=CommitmentStrategy.SINGLE_MINDED)
    bdi.set_task("t1", "goal", "cat")
    bdi.beliefs.brf(0.0, 0.1, 0.02, "rejected", "", "h")  # last_u_v < 0.05
    assert bdi.should_reconsider() is True


def test_should_reconsider_single_minded_stays_committed_otherwise():
    bdi = BDIState(commitment_strategy=CommitmentStrategy.SINGLE_MINDED)
    bdi.set_task("t1", "goal", "cat")
    bdi.beliefs.brf(0.0, 0.5, 0.4, "rejected", "", "h")  # not success/partial, u_v not tiny
    assert bdi.should_reconsider() is False


# ── Regression test for the CONFLICT_DETECTION_ENABLED gating bug ──────
#
# Bug found during audit: should_reconsider()'s consecutive_conflict>=2
# branch had no flag guard, so it fired for ANY BDIState consumer in
# SINGLE_MINDED mode even with CONFLICT_DETECTION_ENABLED left at its
# documented default (False) — breaking the "flag off => behavior
# identical to before" invariant the whole enrichment-flag design
# depends on (see config.py's module docstring and
# core/plan_library.py's ResolveConflict comment).

def test_should_reconsider_ignores_conflict_streak_when_flag_disabled(monkeypatch):
    monkeypatch.setattr(bdi_model, "CONFLICT_DETECTION_ENABLED", False)
    bdi = BDIState(commitment_strategy=CommitmentStrategy.SINGLE_MINDED)
    bdi.set_task("t1", "goal", "cat")
    # Build up consecutive_conflict >= 2 via two disagreeing percepts.
    bdi.beliefs.brf(0.0, 0.9, 0.4, "rejected", "", "h1")
    bdi.beliefs.brf(0.0, 0.9, 0.4, "rejected", "", "h2")
    assert bdi.beliefs.consecutive_conflict >= 2
    # With the flag off, this must NOT force reconsideration on its own.
    assert bdi.should_reconsider() is False


def test_should_reconsider_reacts_to_conflict_streak_when_flag_enabled(monkeypatch):
    monkeypatch.setattr(bdi_model, "CONFLICT_DETECTION_ENABLED", True)
    bdi = BDIState(commitment_strategy=CommitmentStrategy.SINGLE_MINDED)
    bdi.set_task("t1", "goal", "cat")
    bdi.beliefs.brf(0.0, 0.9, 0.4, "rejected", "", "h1")
    bdi.beliefs.brf(0.0, 0.9, 0.4, "rejected", "", "h2")
    assert bdi.beliefs.consecutive_conflict >= 2
    assert bdi.should_reconsider() is True


# ── options() / commit() sanity ─────────────────────────────────────────

def test_options_returns_active_desires_sorted_by_priority():
    bdi = BDIState()
    bdi.set_task("t1", "goal", "cat")
    names = [d.name for d in bdi.options()]
    assert names == ["complete_task", "minimize_steps", "specific_answer"]


def test_commit_creates_intention_from_top_option():
    bdi = BDIState()
    bdi.set_task("t1", "goal", "cat")
    bdi.commit("some_plan")
    intention = bdi.current_intention()
    assert intention is not None
    assert intention.plan_name == "some_plan"
    assert intention.source_desire == "complete_task"
    assert intention.step_index == 0
