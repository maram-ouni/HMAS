"""
Tests for core/payoff_calculator.py's `weights` override on compute_U_V
(config.LEARNED_VERIFIER_WEIGHTS) — must preserve default behavior
exactly when weights=None, and actually apply custom weights when given.
"""
from core.payoff_calculator import compute_U_V, compute_payoffs, WEIGHTS_V


def test_none_weights_uses_static_defaults():
    a = compute_U_V(reward=1.0, answer="$17.11", task_goal="What is the price?")
    b = compute_U_V(reward=1.0, answer="$17.11", task_goal="What is the price?", weights=None)
    assert a["payoff"] == b["payoff"]


def test_custom_weights_change_the_payoff():
    default = compute_U_V(reward=0.0, answer="42", task_goal="How many?")
    custom_weights = {"safety": 0.10, "specificity": 0.70, "success_v": 0.10, "relevance": 0.10}
    custom = compute_U_V(reward=0.0, answer="42", task_goal="How many?", weights=custom_weights)
    assert default["payoff"] != custom["payoff"]
    # dims themselves (safety/specificity/etc.) are unaffected by weights —
    # only how they're combined into the scalar payoff changes.
    assert default["dim_specificity"] == custom["dim_specificity"]


def test_custom_weights_breakdown_reflects_the_weights_used():
    custom_weights = {"safety": 0.10, "specificity": 0.70, "success_v": 0.10, "relevance": 0.10}
    out = compute_U_V(reward=0.0, answer="42", task_goal="How many?", weights=custom_weights)
    assert "0.7x" in out["breakdown"]


def test_compute_payoffs_forwards_verifier_weights():
    custom_weights = {"safety": 0.10, "specificity": 0.70, "success_v": 0.10, "relevance": 0.10}
    out = compute_payoffs(reward=0.0, steps_used=3, steps_max=10, answer="42",
                          task_goal="How many?", verifier_weights=custom_weights)
    expected = compute_U_V(reward=0.0, answer="42", task_goal="How many?",
                           weights=custom_weights)
    assert out["U_V"]["payoff"] == expected["payoff"]


def test_compute_payoffs_none_verifier_weights_unchanged_from_before():
    out = compute_payoffs(reward=1.0, steps_used=2, steps_max=10, answer="$17.11",
                          task_goal="price?")
    expected = compute_U_V(reward=1.0, answer="$17.11", task_goal="price?", weights=WEIGHTS_V)
    assert out["U_V"]["payoff"] == expected["payoff"]
