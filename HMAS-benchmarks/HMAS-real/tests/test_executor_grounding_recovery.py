"""
Tests for grounding recovery (config.EXECUTOR_GROUNDING_RECOVERY_ENABLED):
agents/llm_executor_agent.py::GroqDemoAgent._detect_grounding_correction.

The method is a classmethod reading only obs["last_action_error"], so it is
called directly without instantiating the full GroqDemoAgent (which would
pull in the REAL harness).
"""
from agents.llm_executor_agent import GroqDemoAgent


def detect(err) -> str:
    return GroqDemoAgent._detect_grounding_correction({"last_action_error": err})


# ── Fires on the grounding error, extracting the bid ──────────────────

def test_fires_on_grounding_error_with_quoted_bid():
    msg = detect('ValueError: Could not find element with bid "739"')
    assert msg
    assert '"739"' in msg


def test_fires_on_grounding_error_without_quotes():
    msg = detect("Could not find element with bid 843")
    assert msg
    assert "843" in msg


def test_case_insensitive():
    assert detect('could not find element with BID "12"')


def test_bid_named_in_message_is_the_failing_one():
    # The instruction must name the exact stale/invalid bid so the model
    # knows which id NOT to reuse.
    msg = detect('Could not find element with bid "555"')
    assert "555" in msg
    assert "does NOT exist" in msg


# ── Does NOT fire on unrelated errors / no error ──────────────────────

def test_ignores_multi_action_error():
    assert detect("ValueError: Received a multi-action, only single-actions are allowed.") == ""


def test_ignores_unrelated_error():
    assert detect("TimeoutError: page did not load") == ""


def test_empty_error_no_correction():
    assert detect("") == ""


def test_none_error_no_correction():
    assert detect(None) == ""


def test_missing_key_no_correction():
    # obs without the key at all must not raise.
    assert GroqDemoAgent._detect_grounding_correction({}) == ""


# ── Message content is directive about the AXTree ─────────────────────

def test_message_directs_to_current_axtree():
    msg = detect('Could not find element with bid "9"')
    low = msg.lower()
    assert "axtree" in low
    assert "stale" in low or "do not reuse" in low
