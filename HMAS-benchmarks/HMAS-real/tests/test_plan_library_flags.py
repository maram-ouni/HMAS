"""
Tests for the 5 enrichment Actions in core/plan_library.py, each gated
behind CONFLICT_DETECTION_ENABLED or DYNAMIC_DESIRES_ENABLED:

  ResolveConflict, EscapePlateau, FinalAttempt, DiagnoseUnknownFailure,
  CategoryHighRisk

For each: build() must return None when its flag is off (REGARDLESS of
whether the triggering condition holds — this is the "self-contained
flag check" the module's comments call out as necessary for consumers
like frering_bdi_baseline.py that don't pass active_desires), and must
return a Step when the flag is on and the condition holds.
"""
import core.plan_library as plan_library
from core.bdi_model import BeliefBase


def belief(**overrides) -> BeliefBase:
    b = BeliefBase(task_id="t1", task_goal="goal", task_type="cat")
    for k, v in overrides.items():
        setattr(b, k, v)
    return b


# ── ResolveConflict (CONFLICT_DETECTION_ENABLED) ────────────────────────

def test_resolve_conflict_none_when_flag_off_even_if_condition_holds(monkeypatch):
    monkeypatch.setattr(plan_library, "CONFLICT_DETECTION_ENABLED", False)
    b = belief(consecutive_conflict=3)
    assert plan_library.ResolveConflict().build(b) is None


def test_resolve_conflict_none_when_flag_on_but_condition_not_met(monkeypatch):
    monkeypatch.setattr(plan_library, "CONFLICT_DETECTION_ENABLED", True)
    b = belief(consecutive_conflict=1)
    assert plan_library.ResolveConflict().build(b) is None


def test_resolve_conflict_fires_when_flag_on_and_condition_met(monkeypatch):
    monkeypatch.setattr(plan_library, "CONFLICT_DETECTION_ENABLED", True)
    b = belief(consecutive_conflict=2)
    step = plan_library.ResolveConflict().build(b)
    assert step is not None
    assert step.action == "resolve_conflict"
    assert "2" in step.hint


# ── EscapePlateau (DYNAMIC_DESIRES_ENABLED) ─────────────────────────────

def test_escape_plateau_none_when_flag_off(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", False)
    b = belief(attempt_count=2, last_u_v=0.3, past_u_v=[0.3, 0.31])
    assert plan_library.EscapePlateau().build(b) is None


def test_escape_plateau_none_when_not_plateaued(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", True)
    b = belief(attempt_count=2, last_u_v=0.3, past_u_v=[0.1, 0.5])  # clearly improving, not plateaued
    assert plan_library.EscapePlateau().build(b) is None


def test_escape_plateau_fires_on_plateau(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", True)
    b = belief(attempt_count=2, last_u_v=0.3, past_u_v=[0.30, 0.31])
    step = plan_library.EscapePlateau().build(b)
    assert step is not None
    assert step.action == "escape_plateau"


# ── FinalAttempt (DYNAMIC_DESIRES_ENABLED) ──────────────────────────────

def test_final_attempt_none_when_flag_off(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", False)
    b = belief(attempt_count=2)
    assert plan_library.FinalAttempt().build(b, safest_hint="do X", max_replans=3) is None


def test_final_attempt_none_without_safest_hint(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", True)
    b = belief(attempt_count=2)
    assert plan_library.FinalAttempt().build(b, safest_hint="", max_replans=3) is None


def test_final_attempt_fires_on_last_attempt(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", True)
    b = belief(attempt_count=2)  # max_replans=3 -> last index is 2
    step = plan_library.FinalAttempt().build(b, safest_hint="do X", max_replans=3)
    assert step is not None
    assert "do X" in step.hint


# ── DiagnoseUnknownFailure (DYNAMIC_DESIRES_ENABLED) ────────────────────

def test_diagnose_unknown_failure_none_when_flag_off(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", False)
    b = belief(consecutive_fallback=3)
    assert plan_library.DiagnoseUnknownFailure().build(b) is None


def test_diagnose_unknown_failure_none_below_threshold(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", True)
    b = belief(consecutive_fallback=1)
    assert plan_library.DiagnoseUnknownFailure().build(b) is None


def test_diagnose_unknown_failure_fires_at_threshold(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", True)
    b = belief(consecutive_fallback=2)
    step = plan_library.DiagnoseUnknownFailure().build(b)
    assert step is not None
    assert step.action == "diagnose_unknown_failure"


# ── CategoryHighRisk (DYNAMIC_DESIRES_ENABLED) ──────────────────────────

def test_category_high_risk_none_when_flag_off(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", False)
    b = belief(attempt_count=0)
    assert plan_library.CategoryHighRisk().build(b, category_high_risk=True) is None


def test_category_high_risk_none_when_not_first_attempt(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", True)
    b = belief(attempt_count=1)
    assert plan_library.CategoryHighRisk().build(b, category_high_risk=True) is None


def test_category_high_risk_fires_on_first_attempt(monkeypatch):
    monkeypatch.setattr(plan_library, "DYNAMIC_DESIRES_ENABLED", True)
    b = belief(attempt_count=0)
    step = plan_library.CategoryHighRisk().build(b, category_high_risk=True)
    assert step is not None
    assert step.action == "category_high_risk"
