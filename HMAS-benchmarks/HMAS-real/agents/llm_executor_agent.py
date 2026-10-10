# ─────────────────────────────────────────────────────────
# llm_executor_agent.py — LLM AGENT (DemoAgent from REAL)
#
# This agent IS the DemoAgent of the REAL benchmark.
# It plans AND executes web navigation autonomously
# by interpreting the AXTree at each step.
#
# Its only connection to the BDI:
#   The strategic hint (optional) provided by BDIPlannerAgent
#   is injected into the initial observation (step 0 only).
#   After step 0, DemoAgent is fully autonomous.
#
# Internal architecture of DemoAgent (REAL):
#   obs (AXTree) -> LLM (Groq llama-3.3-70b) -> action -> step -> ...
#   Each step: the LLM sees the full AXTree + history
#              and chooses the next action.
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import dataclasses
import os
import queue
import sys
import threading
import time

_AGISDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MAS_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _AGISDK_ROOT)
sys.path.insert(0, _MAS_ROOT)

from openai import OpenAI
from agisdk import REAL
from agisdk.REAL.demo_agent.basic_agent import DemoAgent, DemoAgentArgs

from core.message_bus     import MessageBus, Message, MessageType
from core.history_manager  import HistoryManager
from core.pre_send_check   import pre_send_verdict
from config                import (GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL,
                                   LLM_MODEL_FALLBACK, MAX_STEPS, EXECUTOR_SELF_VERIFY,
                                   EXECUTOR_LOOP_GUARD_ENABLED,
                                   EXECUTOR_GROUNDING_RECOVERY_ENABLED,
                                   EXECUTOR_MULTIACTION_FIX_ENABLED)


# ── GroqDemoAgent: DemoAgent REAL with Groq backend ──────

class GroqDemoAgent(DemoAgent):
    """
    DemoAgent from the REAL benchmark using Groq as the LLM.

    Only difference from the base DemoAgent:
      - query_model points to the Groq API (llama-3.3-70b)
      - A BDI hint can be injected at step 0

    Everything else (observation, AXTree, action parsing, history)
    is strictly identical to the original REAL DemoAgent.
    No benchmark file is modified.
    """

    def __init__(self, hint: str = "", task_goal: str = "", **kwargs):
        kwargs.pop("model_name",         None)
        kwargs.pop("openrouter_api_key", None)
        super().__init__(
            model_name         = "openrouter/dummy",
            openrouter_api_key = "not_used",
            **kwargs,
        )
        self.model_name  = LLM_MODEL
        self.hint        = hint
        self.task_goal   = task_goal
        # executor<->verifier self-verification: how many premature/low-quality
        # send_msg_to_user calls we've blocked this episode (budget-limited to
        # avoid an infinite block loop). Only used when EXECUTOR_SELF_VERIFY.
        self._self_verify_blocks = 0
        self._MAX_SELF_BLOCKS    = 2
        # Within-shot loop guard (opt-in — config.EXECUTOR_LOOP_GUARD_ENABLED).
        # Set at the start of each get_action() from the compacted
        # action_history, read by query_model this same step to inject a
        # corrective instruction into the system prompt. "" = no loop.
        self._loop_correction    = ""
        # Grounding recovery (opt-in — config.EXECUTOR_GROUNDING_RECOVERY_ENABLED).
        # Same mechanism: set from obs["last_action_error"] at the start of
        # get_action() when the previous action referenced a non-existent
        # bid, injected by query_model this same step. "" = no grounding error.
        self._grounding_correction = ""
        self.query_model = self._make_groq_query()

    MAX_AXTREE_CHARS = 25000

    def obs_preprocessor(self, obs: dict) -> dict:
        processed = super().obs_preprocessor(obs)
        axtree = processed.get("axtree_txt", "")
        if len(axtree) > self.MAX_AXTREE_CHARS:
            processed["axtree_txt"] = axtree[:self.MAX_AXTREE_CHARS] + "\n...[AXTree truncated]"
        return processed

    # Rules injected into the system message at EVERY step (persistent).
    # Unlike goal_object[0] which is only visible at step 0,
    # the system message is sent at every LLM call.
    _SYSTEM_APPEND = (
        "\n\n=== ABSOLUTE RULES (MUST BE FOLLOWED AT EVERY STEP) ===\n"
        "RULE 1: Each response must contain EXACTLY ONE ```action(args)``` block. "
        "Never zero, never two.\n"
        "RULE 2: send_msg_to_user(text) IMMEDIATELY TERMINATES the task and sends "
        "the text to the evaluator. Call it ONLY when the task is completed.\n"
        "RULE 3: If you are stuck, click on a visible NAVIGATION LINK in "
        "the AXTree. Never invent a 'goto' action — that action does not exist.\n"
        "RULE 4: Use ONLY the numeric bids visible in the AXTree. "
        "NEVER invent names like 'search_bar_bid', 'button_bid' — those elements "
        "do not exist. Read the AXTree and use the real numbers: fill('100', 'text'), "
        "click('107'). If you do not know the bid of an element, read the AXTree."
    )

    def _make_groq_query(self):
        """
        Builds query_model using Groq.
        Automatic retry on RateLimitError:
          - waits for the delay indicated by the API (max 60s)
          - switches to LLM_MODEL_FALLBACK if the daily quota is exhausted
        """
        client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        active_model = [LLM_MODEL]   # list so it can be mutated inside the closure
        agent_self   = self          # capture to access hint at each step

        _action_re = __import__("re").compile(
            r'(?:click|fill|scroll|hover|select_option|send_msg_to_user|'
            r'report_infeasible|press|clear|drag_and_drop|upload_file|focus|dblclick)'
            r'\s*\('
        )
        _force_action_msg = (
            "Your previous response contained no valid action call.\n"
            "You MUST output EXACTLY ONE action in triple backticks. Examples:\n"
            "  ```send_msg_to_user('your complete answer here')```\n"
            "  ```scroll(0, 500)```\n"
            "  ```click('123')```\n"
            "Output ONLY the action line. No reasoning. No explanation."
        )

        def query_model(system_msgs, user_msgs):
            messages = []
            if system_msgs:
                msg  = system_msgs[0]
                text = msg.get("text", msg) if isinstance(msg, dict) else str(msg)
                # Inject the rules + BDI hint into the system message
                # at EVERY step (not only step 0)
                extra = GroqDemoAgent._SYSTEM_APPEND
                # Task-goal-specific rule: homepage/inbox tasks have the answer
                # already visible in the AXTree — model must NOT scroll.
                goal_lower = agent_self.task_goal.lower()
                if "homepage" in goal_lower or "last message" in goal_lower:
                    extra += (
                        "\n\nRULE 5 (IMMEDIATE ANSWER TASK): The answer is "
                        "already visible in the current AXTree. Do NOT scroll "
                        "or click. Your FIRST action MUST be "
                        "send_msg_to_user('your complete answer on ONE LINE'). "
                        "Read the AXTree now and answer immediately."
                    )
                if agent_self.hint:
                    extra += (
                        "\n\n=== BDI STRATEGIC HINT ===\n"
                        + agent_self.hint
                    )
                # Within-shot loop guard (opt-in): set in get_action from the
                # action history, cleared each step. Placed LAST so it is the
                # most recent, highest-salience instruction when a loop fires.
                if agent_self._loop_correction:
                    extra += (
                        "\n\n=== ⚠ LOOP GUARD (ACT ON THIS NOW) ===\n"
                        + agent_self._loop_correction
                    )
                # Grounding recovery (opt-in): also placed last, most recent
                # instruction when the previous action hit an invalid bid.
                if agent_self._grounding_correction:
                    extra += (
                        "\n\n=== ⚠ GROUNDING RECOVERY (ACT ON THIS NOW) ===\n"
                        + agent_self._grounding_correction
                    )
                messages.append({"role": "system", "content": text + extra})
            content = [
                {"type": "text", "text": m["text"]}
                for m in user_msgs
                if isinstance(m, dict) and m.get("type") == "text"
            ]
            messages.append({"role": "user", "content": content})

            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model       = active_model[0],
                        messages    = messages,
                        temperature = 0.01,
                        # gpt-oss-120b is a REASONING model: it emits reasoning
                        # tokens before the action. 512 was too short
                        # (reasoning consumed everything → action never emitted).
                        # 4096 leaves room for reasoning + final action.
                        max_tokens  = 4096,
                    )
                    out = resp.choices[0].message.content or ""
                    if not out.strip():
                        # Empty response (e.g. entire budget spent on reasoning) → retry
                        print(f"  [GroqQuery] Empty response (finish="
                              f"{resp.choices[0].finish_reason}) — retry {attempt+1}/3")
                        continue
                    # ── Answer-override: send_msg_to_user retry ──────────
                    # Problem: llama-3.3 keeps scrolling on list/summary tasks
                    # instead of calling send_msg_to_user(). Fix: if the model
                    # navigated but did not answer, force an answer call.
                    import re as _re2
                    _answer_now_msg = (
                        "You navigated but did not answer the question. "
                        "The content you need is already in the AXTree above. "
                        "Read it and output ONLY this (single line, no newlines inside):\n"
                        "```send_msg_to_user('your complete answer on ONE LINE')```"
                    )
                    # Always clean newlines inside send_msg_to_user so pyparsing
                    # can parse the action (multi-line args cause silent failures)
                    _m0 = _re2.search(r'send_msg_to_user\s*\((.*)\)', out, _re2.DOTALL)
                    if _m0:
                        _inner0 = _re2.sub(r'\s*\n\s*', ' ', _m0.group(1).strip())
                        out = out[:_m0.start()] + f"send_msg_to_user({_inner0})" + out[_m0.end():]

                    has_answer = bool(_re2.search(r'send_msg_to_user\s*\(', out))
                    has_action = bool(_action_re.search(out))
                    if not has_answer:
                        hist_scrolls = sum(
                            len(_re2.findall(r'\bscroll\s*\(', m.get("text", "")))
                            for m in user_msgs if isinstance(m, dict))
                        hist_clicks = sum(
                            len(_re2.findall(r'\bclick\s*\(', m.get("text", "")))
                            for m in user_msgs if isinstance(m, dict))
                        # Detect scroll loop (2+ scrolls beyond clicks)
                        stuck = hist_scrolls >= hist_clicks + 2

                        # Check if there is prior action history in user_msgs.
                        # If not, this is the initial greeting step ("Hi! I am your
                        # UI assistant…") — do NOT retry, let DemoAgent handle it.
                        # Retrying here caused the model to call report_infeasible
                        # immediately (step 1), giving steps=1 for every task.
                        hist_has_actions = any(
                            any(k in m.get("text", "")
                                for k in ('click(', 'fill(', 'scroll(',
                                          'report_infeasible(', 'send_msg_to_user('))
                            for m in user_msgs if isinstance(m, dict)
                        )

                        # Only retry when:
                        # - stuck in a scroll loop (force answer), OR
                        # - missing action AND we are past the initial greeting step
                        if stuck or (not has_action and hist_has_actions):
                            retry_suffix = _answer_now_msg if (has_action or stuck) else _force_action_msg
                            retry_messages = messages + [
                                {"role": "assistant", "content": out},
                                {"role": "user",      "content": retry_suffix},
                            ]
                            r2 = client.chat.completions.create(
                                model       = active_model[0],
                                messages    = retry_messages,
                                temperature = 0.0,
                                max_tokens  = 512,
                            )
                            out2 = r2.choices[0].message.content or ""
                            if out2.strip() and _action_re.search(out2):
                                m2 = _re2.search(r'send_msg_to_user\s*\((.*)\)', out2, _re2.DOTALL)
                                if m2:
                                    inner = _re2.sub(r'\s*\n\s*', ' ', m2.group(1).strip())
                                    out2  = f"send_msg_to_user({inner})"
                                print(f"  [GroqQuery] Answer-override → {repr(out2[:80])}")
                                out = out2

                    # ── executor<->verifier self-verification (opt-in) ──────
                    # Before committing send_msg_to_user, consult the verifier's
                    # DETERMINISTIC reward-free criteria on the candidate answer.
                    # If it fails (empty / vague / off-topic / unsafe) and we
                    # still have block budget, re-query WITH the verifier's
                    # feedback instead of committing a bad answer. Breaks the
                    # strict pipeline: the verifier is consulted mid-episode.
                    if EXECUTOR_SELF_VERIFY:
                        improved = self._self_verify_and_maybe_retry(
                            out, messages, client, active_model[0], _re2)
                        if improved is not None:
                            out = improved
                    return out

                except Exception as e:
                    err_str = str(e)
                    # Rate limit: extract the delay and wait
                    if "rate_limit_exceeded" in err_str or "429" in err_str:
                        # Daily quota (TPD) exhausted → switch to fallback
                        if "tokens per day" in err_str.lower() or "TPD" in err_str:
                            if active_model[0] != LLM_MODEL_FALLBACK:
                                print(f"\n  [GroqQuery] Daily quota exhausted "
                                      f"({LLM_MODEL}). "
                                      f"Switching to {LLM_MODEL_FALLBACK}.")
                                active_model[0] = LLM_MODEL_FALLBACK
                                continue   # retry immediately with the fallback
                            else:
                                raise   # both models are exhausted
                        # Per-minute rate limit → wait
                        wait = 15
                        import re as _re
                        m = _re.search(r"try again in (\d+)m?([\d.]+)?s", err_str)
                        if m:
                            mins  = int(m.group(1) or 0)
                            secs  = float(m.group(2) or 0)
                            wait  = min(60, mins * 60 + secs)
                        print(f"\n  [GroqQuery] Rate limit (RPM) — "
                              f"waiting {wait:.0f}s before retry {attempt+1}/3...")
                        time.sleep(wait)
                    else:
                        raise
            raise RuntimeError("Groq: 3 attempts failed (rate limit)")

        return query_model

    @staticmethod
    def _extract_sent_answer(out: str, _re2) -> str | None:
        """Return the answer inside send_msg_to_user(...), or None if absent."""
        m = _re2.search(r"send_msg_to_user\s*\(\s*['\"]?(.*?)['\"]?\s*\)", out, _re2.DOTALL)
        if not m:
            return None
        return m.group(1).strip()

    def _self_verify_and_maybe_retry(self, out, messages, client, model, _re2):
        """
        executor<->verifier self-check on a candidate send_msg_to_user answer.

        Returns an IMPROVED `out` string if the answer was rejected and a
        re-query produced a new action; otherwise None (caller keeps `out`).
        No-op (None) when there is no answer to check, when the answer passes,
        or when the per-episode block budget is exhausted.
        """
        answer = self._extract_sent_answer(out, _re2)
        if answer is None:
            return None  # not a send action -> nothing to self-verify
        verdict = pre_send_verdict(answer, self.task_goal)
        if verdict["accept"] or self._self_verify_blocks >= self._MAX_SELF_BLOCKS:
            return None
        self._self_verify_blocks += 1
        print(f"  [SelfVerify] Blocked premature answer {answer[:50]!r} "
              f"({self._self_verify_blocks}/{self._MAX_SELF_BLOCKS}): {verdict['reason'][:60]}")
        retry_messages = messages + [
            {"role": "assistant", "content": out},
            {"role": "user",
             "content": ("Do NOT send that answer yet. " + verdict["reason"]
                         + " Take one more action to gather the exact value, "
                           "then answer.")},
        ]
        try:
            r2 = client.chat.completions.create(
                model=model, messages=retry_messages, temperature=0.0, max_tokens=512)
            out2 = (r2.choices[0].message.content or "").strip()
        except Exception:
            return None  # best-effort: never break the episode on a self-check
        if not out2:
            return None
        m2 = _re2.search(r"send_msg_to_user\s*\((.*)\)", out2, _re2.DOTALL)
        if m2:
            inner = _re2.sub(r"\s*\n\s*", " ", m2.group(1).strip())
            out2 = f"send_msg_to_user({inner})"
        return out2

    MAX_HISTORY_IN_PROMPT = 5  # keep only the last 5 actions

    # List of valid actions (no goto: absent from subsets=["chat","bid","infeas"])
    _VALID_ACTS = (r'click|fill|scroll|hover|select_option|send_msg_to_user|'
                   r'report_infeasible|press|clear|drag_and_drop|upload_file|'
                   r'focus|dblclick')

    @classmethod
    def _extract_action_call(cls, text: str) -> str:
        """
        Extracts ONLY the action call 'name(args)' from the LLM response.

        Why: the REAL DemoAgent stores ALL the reasoning text
        (200+ words) in action_history, then re-injects it into the prompt at
        each step. The LLM then re-reads its own confused reasoning and
        loses track of the state (did I log in? did I click Send?).

        By keeping only 'click("662")' instead of the full paragraph,
        the history becomes clear and traceable — which drastically improves
        multi-step state tracking (root cause of looping on networking).
        """
        import re as _re
        if EXECUTOR_MULTIACTION_FIX_ENABLED:
            # Balanced-paren scan: preserves the WHOLE call even when an
            # argument contains a ')' (e.g. fill('221', 'Meeting (weekly)')),
            # which the [^)]* regex below would truncate and store mangled
            # into action_history, desynchronizing the model on later steps.
            spans = cls._scan_action_calls(text)
            if spans:
                s, e = spans[0]
                return text[s:e].strip()
            return text.strip()[:80]
        # 1) Action inside a ```...``` block
        m = _re.search(
            r'```[^`]*?((?:' + cls._VALID_ACTS + r')\s*\([^)]*\))[^`]*```',
            text, _re.DOTALL)
        if m:
            return m.group(1).strip()
        # 2) Bare call without backticks
        m = _re.search(r'((?:' + cls._VALID_ACTS + r')\s*\([^)]*\))',
                       text, _re.DOTALL)
        if m:
            return m.group(1).strip()
        # 3) Nothing recognized → keep a short excerpt
        return text.strip()[:80]

    # Action verbs recognized by the balanced-paren scanner (superset of
    # _VALID_ACTS: also includes 'goto', which is invalid but the LLM emits
    # it and _keep_first_action_only must still count it as a call).
    _SCAN_ACTS = (r'click|fill|scroll|goto|hover|select_option|send_msg_to_user|'
                  r'report_infeasible|press|clear|drag_and_drop|upload_file|'
                  r'focus|dblclick')

    @staticmethod
    def _action_call_end(text: str, open_idx: int) -> int:
        """
        Given the index of an action call's opening '(', returns the index
        just AFTER its matching ')', or -1 if unbalanced. Paren depth is
        tracked OUTSIDE quoted strings only, so parentheses inside a quoted
        argument (e.g. 'Meeting (weekly)') don't close the call early.
        """
        depth = 0
        quote = None
        i = open_idx
        n = len(text)
        while i < n:
            c = text[i]
            if quote is not None:
                if c == quote and text[i - 1] != "\\":
                    quote = None
            elif c in ("'", '"'):
                quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return -1

    @classmethod
    def _scan_action_calls(cls, text: str) -> list:
        """
        Returns (start, end) spans of every top-level action call in `text`,
        balanced-paren + quote aware. After each call, scanning resumes past
        its end, so action-verb words appearing INSIDE a call's arguments
        (e.g. the word 'click' inside fill('1', 'click here')) are not
        double-counted — more robust than the old [^)]* regex.
        """
        import re as _re
        pat = _re.compile(r'(?:' + cls._SCAN_ACTS + r')\s*\(')
        spans = []
        pos = 0
        while True:
            m = pat.search(text, pos)
            if not m:
                break
            open_idx = m.end() - 1        # index of the '('
            end = cls._action_call_end(text, open_idx)
            if end == -1:
                pos = m.end()             # unbalanced — skip this prefix
                continue
            spans.append((m.start(), end))
            pos = end                     # resume AFTER the whole call
        return spans

    @classmethod
    def _keep_first_action_only(cls, text: str) -> str:
        """
        Ensures there is at most ONE action in the LLM response.
        Covers 3 cases:
          1. Multiple separate ```...``` blocks
          2. A single ```...``` block with multiple calls on separate lines
          3. Bare function calls without backticks
        """
        import re as _re
        if EXECUTOR_MULTIACTION_FIX_ENABLED:
            spans = cls._scan_action_calls(text)
            if len(spans) >= 2:
                s, e = spans[0]
                first = text[s:e]
                print(f"  [GroqAgent] Multi-action ({len(spans)} calls) → 1st: {first[:60]}")
                return f"```{first}```"
            if len(spans) == 0:
                blocks = list(_re.finditer(r'```[^`]*```', text, _re.DOTALL))
                if len(blocks) >= 2:
                    return text[:blocks[0].end()]
            return text
        _ACTS = (r'click|fill|scroll|goto|hover|select_option|send_msg_to_user|'
                 r'report_infeasible|press|clear|drag_and_drop|upload_file|focus')
        # Find ALL action calls in the text, regardless of context
        call_re = _re.compile(r'(?:' + _ACTS + r')\s*\([^)]*\)', _re.DOTALL)
        calls   = list(call_re.finditer(text))

        if len(calls) >= 2:
            # Multiple actions → keep only the first one, wrap it in backticks
            first = calls[0].group(0)
            print(f"  [GroqAgent] Multi-action ({len(calls)} calls) → 1st: {first[:60]}")
            return f"```{first}```"

        if len(calls) == 0:
            # No recognized call → still check for multiple blocks
            blocks = list(_re.finditer(r'```[^`]*```', text, _re.DOTALL))
            if len(blocks) >= 2:
                return text[:blocks[0].end()]

        return text

    @staticmethod
    def _action_head(a: str) -> str:
        """
        (verb, first-arg) signature of a compact action, e.g.
        "fill('198', 'x')" -> "fill:'198'". Groups actions that target the
        SAME element regardless of the value typed — so re-filling one
        field with different text still counts as churning the same target.
        """
        verb = a.split("(", 1)[0].strip()
        arg0 = ""
        if "(" in a:
            inner = a.split("(", 1)[1]
            arg0 = inner.split(",", 1)[0].rstrip(")").strip()
        return f"{verb}:{arg0}"

    # Repeated/oscillating scrolls are handled separately (scroll-loop
    # answer-override in _make_groq_query) and are often legitimate, so the
    # loop guard ignores them and focuses on click/fill/select churn.
    _LOOP_IGNORE_VERBS = ("scroll", "send_msg_to_user", "report_infeasible")

    def _detect_loop_correction(self) -> str:
        """
        Inspects the compacted action_history for a within-shot loop and
        returns a corrective instruction (or "" if none). Two patterns:
          1. REPEAT   — the last 3 actions are byte-identical (same bid,
                        same value) with no effect.
          2. CHURN    — the last 4 actions cycle over only 2 distinct
                        targets (e.g. click A / click B / click A…, or
                        re-filling the same two fields with swapped values).
        Scrolls/answers are ignored (see _LOOP_IGNORE_VERBS).
        """
        hist = [a for a in self.action_history
                if a and a.split("(", 1)[0].strip() not in self._LOOP_IGNORE_VERBS]

        # 1) Exact repeat of the last action, 3x in a row.
        if len(hist) >= 3 and hist[-1] == hist[-2] == hist[-3]:
            act = hist[-1]
            return (
                f"LOOP DETECTED: you have issued the SAME action {act} three "
                f"times and it is NOT working. STOP repeating it. Read the "
                f"AXTree again and do something DIFFERENT — target another "
                f"element, or if the information you need is already visible, "
                f"call send_msg_to_user('your complete answer on ONE LINE') now."
            )

        # 2) Churn over only 2 distinct targets across the last 4 actions.
        if len(hist) >= 4:
            window = hist[-4:]
            heads = [self._action_head(a) for a in window]
            distinct = set(heads)
            if len(distinct) == 2 and not (hist[-1] == hist[-2] == hist[-3] == hist[-4]):
                a, b = sorted(distinct)
                return (
                    f"LOOP DETECTED: you keep alternating between the same two "
                    f"targets ({a} and {b}) without making progress. STOP. "
                    f"Re-read the AXTree carefully, decide the ONE correct action, "
                    f"and if the task's answer is already visible, call "
                    f"send_msg_to_user('your complete answer on ONE LINE') now."
                )

        return ""

    # "Could not find element with bid" is the SDK's grounding error when an
    # action targets a bid that is not in the current page (invented, or
    # stale after the page changed). The bid appears in quotes; capture it if
    # present, otherwise still recover generically.
    _GROUNDING_ERR_RE = __import__("re").compile(
        r'could not find element with bid\s*["\']?(\w+)?', __import__("re").IGNORECASE)

    @classmethod
    def _detect_grounding_correction(cls, obs: dict) -> str:
        """
        Reads obs["last_action_error"] and, if the previous action failed
        because it referenced a non-existent bid, returns a forceful
        instruction to use only a bid literally present in the current
        AXTree (or "" otherwise). The parent DemoAgent already forwards the
        raw error text, but buried among the history and phrased weakly
        ("reflect on any error message"); this makes it the highest-salience
        instruction and names the exact failure mode (invented/stale bid).
        """
        err = str(obs.get("last_action_error", "") or "")
        m = cls._GROUNDING_ERR_RE.search(err)
        if not m:
            return ""
        bid = m.group(1)
        bid_clause = (f'the bid "{bid}" you used does NOT exist on the page'
                      if bid else "your last action used a bid that does NOT exist on the page")
        return (
            f"GROUNDING ERROR: {bid_clause}. The AXTree below is the ONLY source "
            f"of truth for element ids. You MUST pick a bid number that literally "
            f"appears in the CURRENT AXTree. Do NOT reuse that bid, and do NOT "
            f"reuse bids from earlier steps — ids become stale once the page "
            f"changes. Read the AXTree now, find the element you need, and use "
            f"its exact current bid."
        )

    def get_action(self, obs: dict) -> tuple:
        """
        Truncates history to MAX_HISTORY_IN_PROMPT.
        Enforces a single action per response (anti multi-action).
        Compresses history to parsed actions (reliable state tracking).
        Rules + BDI hint are injected into the system message
        at EVERY step via _make_groq_query.
        """
        if len(self.action_history) > self.MAX_HISTORY_IN_PROMPT:
            self.action_history = self.action_history[-self.MAX_HISTORY_IN_PROMPT:]

        # Within-shot loop guard (opt-in): computed from PAST actions
        # BEFORE querying, so query_model injects the correction this step.
        if EXECUTOR_LOOP_GUARD_ENABLED:
            self._loop_correction = self._detect_loop_correction()
            if self._loop_correction:
                print(f"  [GroqAgent] Loop guard fired → injecting correction")
        else:
            self._loop_correction = ""

        # Grounding recovery (opt-in): computed from the previous step's
        # error BEFORE querying, injected by query_model this same step.
        if EXECUTOR_GROUNDING_RECOVERY_ENABLED:
            self._grounding_correction = self._detect_grounding_correction(obs)
            if self._grounding_correction:
                print(f"  [GroqAgent] Grounding recovery fired → injecting correction")
        else:
            self._grounding_correction = ""

        action, info = super().get_action(obs)

        # 1) Keep only the first action if the LLM generated multiple
        clean = self._keep_first_action_only(action)
        if clean != action:
            print(f"  [GroqAgent] Multi-action detected → first action kept")

        # 2) Compress history: replace verbose reasoning with
        #    the parsed action alone.
        compact = self._extract_action_call(clean)
        if self.action_history:
            self.action_history[-1] = compact

        return clean, info


@dataclasses.dataclass
class GroqDemoAgentArgs(DemoAgentArgs):
    """Args to instantiate GroqDemoAgent via REAL.harness()."""
    model_name: str = f"groq/{LLM_MODEL}"
    hint:       str = ""
    task_goal:  str = ""

    def make_agent(self):
        agent = GroqDemoAgent(
            hint                    = self.hint,
            task_goal               = self.task_goal,
            chat_mode               = self.chat_mode,
            demo_mode               = self.demo_mode,
            use_html                = self.use_html,
            use_axtree              = self.use_axtree,
            use_screenshot          = self.use_screenshot,
            system_message_handling = self.system_message_handling,
        )
        self._created_agent = agent   # reference to retrieve action_history after run()
        return agent


# ── LLMExecutorAgent: thread that launches DemoAgent ─────────

class LLMExecutorAgent(threading.Thread):
    """
    LLM Agent — executes web navigation via DemoAgent (REAL).

    Responsibilities:
      1. Receive EXECUTION_REQUESTED (from BDIPlannerAgent)
      2. Instantiate GroqDemoAgent with the BDI hint
      3. Launch REAL.harness() — DemoAgent navigates autonomously
      4. Publish EXECUTION_DONE (to RuleBasedVerifierAgent)

    DemoAgent plans AND executes everything on its own.
    LLMExecutorAgent is simply the "launcher" of the session.
    """

    def __init__(self, bus: MessageBus,
                 history_manager: HistoryManager,
                 stop_event: threading.Event):
        super().__init__(daemon=True, name="LLMExecutorThread")
        self.bus        = bus
        self.stop_event = stop_event

    # ── Main loop ─────────────────────────────────

    def run(self) -> None:
        print("[LLMExecutorThread] Started — REAL DemoAgent ready...")
        while not self.stop_event.is_set():
            try:
                msg = self.bus.get("executor", timeout=2)
            except queue.Empty:
                continue

            if msg.msg_type == MessageType.STOP:
                break
            elif msg.msg_type == MessageType.EXECUTION_REQUESTED:
                self._run(msg)

        print("[LLMExecutorThread] Stopped.")

    # ── Execution ─────────────────────────────────────────

    def _run(self, msg: Message) -> None:
        task_id   = msg.content["task_id"]
        task_goal = msg.content["task_goal"]
        hint      = msg.content.get("hint",      "")
        plan_name = msg.content.get("plan_name", "")
        attempt   = msg.content.get("attempt",   1)
        max_steps = msg.content.get("max_steps", MAX_STEPS)
        headless  = msg.content.get("headless",  True)

        print(f"\n  [LLMExecutor] Launching DemoAgent (attempt={attempt}, "
              f"task={task_id})")
        if hint:
            print(f"  [LLMExecutor] BDI hint: {hint[:80]}")
        else:
            print(f"  [LLMExecutor] No hint (first autonomous attempt)")

        result = self._run_harness(task_id, task_goal, hint, max_steps, headless, attempt)

        print(f"  [LLMExecutor] reward={result['reward']:.1f} "
              f"steps={result['steps']} duration={result['duration']:.1f}s")

        self.bus.publish(Message(
            sender   = "executor",
            receiver = "verifier",
            msg_type = MessageType.EXECUTION_DONE,
            content  = {
                "task_id":       task_id,
                "task_goal":     task_goal,
                "reward":        result["reward"],
                "steps":         result["steps"],
                "answer":        result["answer"],
                "success":       result["success"],
                "duration":      result["duration"],
                "hint_used":     hint,
                "plan_name":     plan_name,
                "attempt":       attempt,
                "max_steps":     max_steps,
                "headless":      headless,
                "action_history": result.get("action_history", []),
                "last_error":    result.get("last_error", ""),
            }
        ))

    @staticmethod
    def _parse_task_id(task_id: str):
        """
        Converts "v1.dashdish-10" → ("dashdish", 10)
        as expected by REAL.harness(task_type=..., task_id=...).
        """
        name = task_id.split(".", 1)[-1]   # "dashdish-10"
        idx  = name.rfind("-")
        return name[:idx], int(name[idx + 1:])

    def _run_harness(self, task_id: str, task_goal: str, hint: str,
                     max_steps: int, headless: bool,
                     attempt: int) -> dict:
        """Launch REAL.harness() with GroqDemoAgent (DemoAgent + Groq + hint)."""
        agent_args = GroqDemoAgentArgs(
            use_screenshot = False,
            use_axtree     = True,
            hint           = hint,
            task_goal      = task_goal,
        )
        task_type_str, task_num = self._parse_task_id(task_id)
        try:
            t0  = time.perf_counter()
            h   = REAL.harness(
                agentargs     = agent_args,
                task_type     = task_type_str,
                task_id       = task_num,
                headless      = headless,
                max_steps     = max_steps,
                num_workers   = 1,
                force_refresh = True,
            )
            raw      = h.run()
            duration = time.perf_counter() - t0

            task_result = {}
            if isinstance(raw, dict):
                for v in raw.values():
                    task_result = v
                    break

            # Retrieve action_history from the agent created by make_agent()
            agent_obj      = getattr(agent_args, "_created_agent", None)
            action_history = list(getattr(agent_obj, "action_history", []))
            last_obs       = getattr(agent_obj, "last_observation", {}) or {}
            last_error     = str(last_obs.get("last_action_error", "") or "")
            return {
                "reward":         float(task_result.get("cum_reward", 0.0)),
                "steps":          int(task_result.get(
                    "stats.cum_steps",
                    task_result.get("n_steps", max_steps),
                )),
                "answer":         str(task_result.get("agent_response", "")),
                "success":        bool(task_result.get("success", False)),
                "duration":       round(duration, 2),
                "action_history": action_history,
                "last_error":     last_error,
            }
        except Exception as e:
            print(f"  [LLMExecutor] REAL.harness error: {e}")
            return {
                "reward":   0.0,
                "steps":    max_steps,
                "answer":   "",
                "success":  False,
                "duration": 0.0,
            }
