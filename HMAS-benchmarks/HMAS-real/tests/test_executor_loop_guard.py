"""
Tests for the within-shot loop guard (config.EXECUTOR_LOOP_GUARD_ENABLED):
agents/llm_executor_agent.py::GroqDemoAgent._action_head and
._detect_loop_correction.

The detection logic reads only self.action_history, so it is exercised on a
lightweight stub instead of instantiating the full GroqDemoAgent (which would
pull in the REAL harness). _action_head is a staticmethod, called directly.
"""
from agents.llm_executor_agent import GroqDemoAgent


class _Stub:
    """Minimal object exposing exactly what _detect_loop_correction reads."""
    _LOOP_IGNORE_VERBS = GroqDemoAgent._LOOP_IGNORE_VERBS
    _action_head = staticmethod(GroqDemoAgent._action_head)

    def __init__(self, history):
        self.action_history = history


def detect(history) -> str:
    return GroqDemoAgent._detect_loop_correction(_Stub(history))


# ── _action_head ──────────────────────────────────────────────────────

def test_action_head_groups_by_verb_and_target():
    assert GroqDemoAgent._action_head("fill('198', '333 Fremont')") == "fill:'198'"
    assert GroqDemoAgent._action_head("fill('198', 'Pacific Catch')") == "fill:'198'"
    assert GroqDemoAgent._action_head("click('100')") == "click:'100'"
    assert GroqDemoAgent._action_head("scroll(0, 200)") == "scroll:0"


# ── REPEAT pattern (same action 3x) ───────────────────────────────────

def test_repeat_fires_on_three_identical():
    msg = detect(["click('100')", "click('100')", "click('100')"])
    assert msg
    assert "click('100')" in msg


def test_repeat_needs_three_not_two():
    assert detect(["click('100')", "click('100')"]) == ""


def test_repeat_broken_by_different_action():
    assert detect(["click('100')", "click('100')", "click('101')"]) == ""


# ── CHURN pattern (2 targets over last 4) ─────────────────────────────

def test_churn_fires_on_click_alternation():
    msg = detect(["click('1')", "click('2')", "click('1')", "click('2')"])
    assert msg
    assert "alternating" in msg.lower()


def test_churn_fires_on_fill_swap_same_two_fields():
    # udriver-3 pattern: re-filling the same two bids with swapped values.
    msg = detect([
        "fill('198', '333 Fremont')",
        "fill('208', 'Pacific Catch')",
        "fill('198', 'Pacific Catch')",
        "fill('208', '333 Fremont')",
    ])
    assert msg
    assert "alternating" in msg.lower()


def test_churn_needs_exactly_two_distinct_targets():
    # Three distinct targets in the window -> genuine progress, no loop.
    assert detect(["click('1')", "click('2')", "click('3')", "click('1')"]) == ""


# ── Ignored verbs / no-loop cases ─────────────────────────────────────

def test_scroll_repeat_is_ignored():
    # Scroll loops are handled by the separate answer-override, not here.
    assert detect(["scroll(0, 200)", "scroll(0, 200)", "scroll(0, 200)"]) == ""


def test_send_msg_repeat_is_ignored():
    assert detect(["send_msg_to_user('x')"] * 3) == ""


def test_normal_progress_no_loop():
    assert detect([
        "click('1')", "fill('2', 'x')", "click('3')", "send_msg_to_user('done')"
    ]) == ""


def test_empty_history_no_loop():
    assert detect([]) == ""


def test_scroll_does_not_count_toward_churn_window():
    # Scrolls are stripped before the window is taken, so a real 2-target
    # churn buried among scrolls is still caught.
    msg = detect([
        "click('1')", "scroll(0, 200)", "click('2')",
        "scroll(0, 400)", "click('1')", "click('2')",
    ])
    assert msg
    assert "alternating" in msg.lower()
