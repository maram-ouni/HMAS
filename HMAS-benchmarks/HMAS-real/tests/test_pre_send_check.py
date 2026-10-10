"""
Tests for core/pre_send_check.py — the executor's reward-free self-check
before send_msg_to_user (executor<->verifier loop).

Relevance/specificity gating only applies to ACTION-type goals (matching
rule_based_verifier.py's own philosophy); RETRIEVAL goals (factual
questions) are only checked for emptiness and safety, since correctness
there is binary (ground truth), not judged by these heuristics.
"""
from core.pre_send_check import pre_send_verdict

ACTION_GOAL = "Add the cheapest item to the cart"   # not a retrieval task
RETRIEVAL_GOAL = "What is the price of Chicken Biryani?"


def test_empty_answer_rejected():
    v = pre_send_verdict("", RETRIEVAL_GOAL)
    assert v["accept"] is False
    assert "EMPTY" in v["reason"]


def test_whitespace_only_answer_rejected():
    v = pre_send_verdict("   ", "How many unread emails?")
    assert v["accept"] is False


def test_specific_on_topic_answer_accepted():
    v = pre_send_verdict("The price of Chicken Biryani is $17.11", RETRIEVAL_GOAL)
    assert v["accept"] is True
    assert v["reason"] == ""


def test_numeric_answer_accepted_for_counting_goal():
    v = pre_send_verdict("There are 17 unread emails.",
                         "How many unread emails are in the Inbox?")
    assert v["accept"] is True


def test_vague_answer_rejected_for_action_goal():
    # On-topic (mentions cart item) but no concrete value -> low specificity,
    # not off-topic.
    v = pre_send_verdict("unable to find cart item", ACTION_GOAL)
    assert v["accept"] is False
    assert "vague" in v["reason"].lower()


def test_offtopic_answer_rejected_for_action_goal():
    v = pre_send_verdict("The weather is nice today.", ACTION_GOAL)
    assert v["accept"] is False
    assert "off-topic" in v["reason"].lower()


def test_safety_violation_rejected():
    v = pre_send_verdict("The API key is sk-abcdefghijklmnopqrstuvwxyz123456",
                         "What is the account status?")
    assert v["accept"] is False
    assert "safety" in v["reason"].lower()


def test_dims_are_reported():
    v = pre_send_verdict("$42.00", "price?")
    assert set(v["dims"]) == {"specificity", "relevance", "safety"}


def test_thresholds_are_tunable_for_action_goal():
    # A borderline-specific answer accepted at a low threshold, rejected at a high one.
    ans, goal = "Around 5 items", ACTION_GOAL
    lax    = pre_send_verdict(ans, goal, min_specificity=0.1)
    strict = pre_send_verdict(ans, goal, min_specificity=0.95)
    assert lax["accept"] is True
    assert strict["accept"] is False


# ── Regression test: real false positive caught during live testing ────
#
# "Phat Thai" is the CORRECT answer to "What was the name of the Thai
# restaurant I last got a ride to?" but scores relevance=0.14 (just under
# the 0.15 threshold) on simple keyword overlap — a short, correct
# proper-noun answer naturally has low keyword overlap with the question.
# Blocking it would force the executor to loop pointlessly on an already-
# correct answer. Retrieval tasks must bypass the relevance/specificity
# gate entirely (correctness is for the ground-truth-driven verifier to
# judge, not this heuristic).

def test_short_correct_proper_noun_answer_not_rejected_for_retrieval():
    v = pre_send_verdict("Phat Thai",
                         "What was the name of the Thai restaurant I last got a ride to?")
    assert v["accept"] is True


def test_relevance_gate_still_applies_to_action_goals():
    # Same short-answer shape, but for an action goal the gate must still fire.
    v = pre_send_verdict("Phat Thai", ACTION_GOAL)
    assert v["accept"] is False
