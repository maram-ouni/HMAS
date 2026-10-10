"""
Tests for the DeepJSONEval-specific wiring of the ported learned-weights/
calibration mechanism in agents/rule_based_verifier.py
(config.LEARNED_VERIFIER_WEIGHTS / config.LEARNED_PE_WEIGHTS):
_learned_verifier_config, _learned_pe_config, _compute_payoffs.

Exercised on a lightweight stub carrying only the attributes/methods these
touch, so the full threaded agent / message bus isn't needed (same pattern
as test_verifier_agreement.py).
"""
import json

import agents.rule_based_verifier as rbv
from agents.rule_based_verifier import RuleBasedVerifierAgent
from core.payoff_calculator import WEIGHTS_V, WEIGHTS_PE, V_THRESHOLD
from core.verifier_learned_weights import MIN_EXAMPLES_PER_CLASS


class _Stub:
    _FEATURES_V              = RuleBasedVerifierAgent._FEATURES_V
    _PE_FEATURES             = RuleBasedVerifierAgent._PE_FEATURES
    _learned_verifier_config = RuleBasedVerifierAgent._learned_verifier_config
    _learned_pe_config       = RuleBasedVerifierAgent._learned_pe_config
    _compute_payoffs         = RuleBasedVerifierAgent._compute_payoffs

    def __init__(self, trace_dir):
        self._trace_dir = str(trace_dir)
        self._verifier_weights_cache = None
        self._pe_weights_cache = None
        self._verifier_threshold = V_THRESHOLD


def _write_trace(trace_dir, task_id, rows):
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{task_id}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_flag_off_uses_static_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(rbv, "LEARNED_VERIFIER_WEIGHTS", False)
    monkeypatch.setattr(rbv, "LEARNED_PE_WEIGHTS", False)
    s = _Stub(tmp_path / "traces")
    payoffs = s._compute_payoffs("t1", "cat", '{"a":1}', "{}", '{"a":1}', 1)
    assert payoffs["U_V"]["weights_used"] == WEIGHTS_V
    assert payoffs["U_PE"]["weights_used"] == WEIGHTS_PE


def test_degenerates_to_static_with_no_traces(tmp_path):
    s = _Stub(tmp_path / "traces")
    w, t = s._learned_verifier_config()
    assert w == WEIGHTS_V
    assert t == V_THRESHOLD
    pe = s._learned_pe_config()
    assert pe == WEIGHTS_PE


def test_json_valid_stays_pinned_for_both(tmp_path):
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.6, "dim_field_accuracy": 0.8, "dim_depth_match": 0.9,
             "dim_schema_compliance": 0.9, "dim_completeness": 0.8},
            {"reward": 1.0, "dim_field_accuracy": 1.0, "dim_depth_match": 1.0,
             "dim_schema_compliance": 1.0, "dim_completeness": 1.0},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.1, "dim_field_accuracy": 0.1, "dim_depth_match": 0.2,
             "dim_schema_compliance": 0.2, "dim_completeness": 0.1},
        ])
    s = _Stub(trace_dir)
    w, _ = s._learned_verifier_config()
    pe = s._learned_pe_config()
    assert w["json_valid"] == WEIGHTS_V["json_valid"]
    assert pe["json_valid"] == WEIGHTS_PE["json_valid"]
    assert sum(w.values()) == 1.0 or abs(sum(w.values()) - 1.0) < 1e-3
    assert abs(sum(pe.values()) - 1.0) < 1e-3


def test_flag_on_uses_learned_weights_when_data_available(monkeypatch, tmp_path):
    monkeypatch.setattr(rbv, "LEARNED_VERIFIER_WEIGHTS", True)
    monkeypatch.setattr(rbv, "LEARNED_PE_WEIGHTS", True)
    trace_dir = tmp_path / "traces"
    n = MIN_EXAMPLES_PER_CLASS + 5
    for i in range(n):
        _write_trace(trace_dir, f"succ{i}", [
            {"reward": 0.6, "dim_field_accuracy": 0.9, "dim_depth_match": 0.9,
             "dim_schema_compliance": 0.9, "dim_completeness": 0.9},
            {"reward": 1.0, "dim_field_accuracy": 1.0, "dim_depth_match": 1.0,
             "dim_schema_compliance": 1.0, "dim_completeness": 1.0},
        ])
    for i in range(n):
        _write_trace(trace_dir, f"fail{i}", [
            {"reward": 0.1, "dim_field_accuracy": 0.1, "dim_depth_match": 0.1,
             "dim_schema_compliance": 0.1, "dim_completeness": 0.1},
        ])
    s = _Stub(trace_dir)
    payoffs = s._compute_payoffs("t1", "cat", '{"a":1}', "{}", '{"a":1}', 1)
    # weights_used must differ from the static prior once the fit engages
    # (json_valid stays pinned, the other two are redistributed).
    assert payoffs["U_V"]["weights_used"] != WEIGHTS_V or \
           payoffs["U_PE"]["weights_used"] != WEIGHTS_PE
