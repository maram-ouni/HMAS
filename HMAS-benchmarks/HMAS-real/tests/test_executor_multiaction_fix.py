"""
Tests for the balanced-paren action parsing fix
(config.EXECUTOR_MULTIACTION_FIX_ENABLED): agents/llm_executor_agent.py's
_action_call_end / _scan_action_calls, and the gated behavior of
_extract_action_call / _keep_first_action_only.

The parsing methods are static/class methods, called directly without
instantiating the full GroqDemoAgent. The flag is patched on the module
(the methods read the module-level global via `from config import ...`).
"""
import agents.llm_executor_agent as ex
from agents.llm_executor_agent import GroqDemoAgent as G


# ── _action_call_end / _scan_action_calls (flag-independent) ──────────

def test_call_end_stops_at_matching_paren():
    t = "fill('1', 'x')rest"
    end = G._action_call_end(t, t.index("("))
    assert t[:end] == "fill('1', 'x')"


def test_call_end_ignores_paren_inside_quotes():
    t = "fill('1', 'Meeting (weekly)')"
    end = G._action_call_end(t, t.index("("))
    assert t[:end] == t  # whole string is one call


def test_call_end_handles_nested_parens_outside_quotes():
    t = "click((1))after"
    end = G._action_call_end(t, t.index("("))
    assert t[:end] == "click((1))"


def test_call_end_unbalanced_returns_minus_one():
    t = "fill('1', 'x'"
    assert G._action_call_end(t, t.index("(")) == -1


def test_scan_finds_single_call_with_paren_in_arg():
    spans = G._scan_action_calls("fill('221', 'Meeting (weekly)')")
    assert len(spans) == 1


def test_scan_finds_two_separate_calls():
    spans = G._scan_action_calls("click('5')\nfill('6', 'hi (there)')")
    assert len(spans) == 2


def test_scan_ignores_verb_inside_quoted_arg():
    # 'click' inside the fill's string arg must NOT be counted as a call.
    spans = G._scan_action_calls("fill('1', 'click here to submit')")
    assert len(spans) == 1


# ── _extract_action_call — gated behavior ─────────────────────────────

def test_extract_keeps_full_action_when_fix_on(monkeypatch):
    monkeypatch.setattr(ex, "EXECUTOR_MULTIACTION_FIX_ENABLED", True)
    t = "fill('221', 'Meeting (weekly)')"
    assert G._extract_action_call(t) == "fill('221', 'Meeting (weekly)')"


def test_extract_truncates_when_fix_off(monkeypatch):
    # Documents the OLD (buggy) behavior is preserved when the flag is off.
    monkeypatch.setattr(ex, "EXECUTOR_MULTIACTION_FIX_ENABLED", False)
    t = "fill('221', 'Meeting (weekly)')"
    assert G._extract_action_call(t) == "fill('221', 'Meeting (weekly)"


def test_extract_from_backtick_block_with_paren(monkeypatch):
    monkeypatch.setattr(ex, "EXECUTOR_MULTIACTION_FIX_ENABLED", True)
    t = "Reasoning...\n```fill('221', 'A (B)')```"
    assert G._extract_action_call(t) == "fill('221', 'A (B)')"


# ── _keep_first_action_only — gated behavior ──────────────────────────

def test_keep_first_single_action_with_paren_unchanged(monkeypatch):
    monkeypatch.setattr(ex, "EXECUTOR_MULTIACTION_FIX_ENABLED", True)
    t = "fill('221', 'Meeting (weekly)')"
    assert G._keep_first_action_only(t) == t  # single action → returned as-is


def test_keep_first_of_two_actions(monkeypatch):
    monkeypatch.setattr(ex, "EXECUTOR_MULTIACTION_FIX_ENABLED", True)
    t = "click('5')\nfill('6', 'hi (there)')"
    assert G._keep_first_action_only(t) == "```click('5')```"


def test_keep_first_off_path_still_works(monkeypatch):
    monkeypatch.setattr(ex, "EXECUTOR_MULTIACTION_FIX_ENABLED", False)
    t = "click('5')\nfill('6', 'x')"
    assert G._keep_first_action_only(t) == "```click('5')```"
