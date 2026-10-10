"""
Tests for the weight-override kwargs added to core/payoff_calculator.py
(compute_U_V/compute_U_PE/compute_payoffs) to support
config.LEARNED_VERIFIER_WEIGHTS / config.LEARNED_PE_WEIGHTS. `weights=None`
(default) must be byte-identical to the pre-existing static behavior.
"""
import pytest

from core.payoff_calculator import (
    compute_U_V, compute_U_PE, compute_payoffs, WEIGHTS_V, WEIGHTS_PE,
)

GEN = '{"a": 1, "b": {"c": 2}}'
GT  = '{"a": 1, "b": {"c": 2}}'
SCHEMA = '{"type": "object"}'


def test_compute_u_v_default_weights_unchanged():
    r1 = compute_U_V(GEN, GT, 2)
    r2 = compute_U_V(GEN, GT, 2, weights=None)
    assert r1["payoff"] == r2["payoff"]
    assert r1["weights_used"] == WEIGHTS_V


def test_compute_u_pe_default_weights_unchanged():
    r1 = compute_U_PE(GEN, SCHEMA, GT, 2)
    r2 = compute_U_PE(GEN, SCHEMA, GT, 2, weights=None)
    assert r1["payoff"] == r2["payoff"]
    assert r1["weights_used"] == WEIGHTS_PE


def test_compute_payoffs_default_unchanged():
    r1 = compute_payoffs(GEN, SCHEMA, GT, 2)
    r2 = compute_payoffs(GEN, SCHEMA, GT, 2, verifier_weights=None, planner_weights=None)
    assert r1["U_V"]["payoff"] == r2["U_V"]["payoff"]
    assert r1["U_PE"]["payoff"] == r2["U_PE"]["payoff"]


def test_compute_u_v_override_changes_payoff():
    custom = {"json_valid": 0.0, "field_accuracy": 0.0, "depth_match": 1.0}
    r = compute_U_V(GEN, GT, 2, weights=custom)
    assert r["weights_used"] == custom
    assert r["payoff"] == pytest.approx(r["dim_depth_match"], abs=1e-6)


def test_compute_u_pe_override_changes_payoff():
    custom = {"json_valid": 1.0, "schema_compliance": 0.0, "completeness": 0.0}
    r = compute_U_PE(GEN, SCHEMA, GT, 2, weights=custom)
    assert r["weights_used"] == custom
    assert r["payoff"] == pytest.approx(r["dim_json_valid"], abs=1e-6)


def test_compute_payoffs_override_forwarded_to_both():
    v_override = {"json_valid": 1.0, "field_accuracy": 0.0, "depth_match": 0.0}
    pe_override = {"json_valid": 1.0, "schema_compliance": 0.0, "completeness": 0.0}
    r = compute_payoffs(GEN, SCHEMA, GT, 2,
                        verifier_weights=v_override, planner_weights=pe_override)
    assert r["U_V"]["weights_used"] == v_override
    assert r["U_PE"]["weights_used"] == pe_override
