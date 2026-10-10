# ─────────────────────────────────────────────────────────
# plan_library.py — BDI PLAN LIBRARY (REAL benchmark) — v2
#
# A BDI Plan (Rao & Georgeff, 1991 / AgentSpeak) is a triplet:
#     trigger  — the desire/goal that activates it
#     context  — applicability condition (boolean over beliefs)
#     body     — a SEQUENCE OF ACTIONS the agent executes (the recipe)
#
# v2 change vs the previous version: a plan's body is an ORDERED LIST
# of Actions, not a single hint-producing rule. The agent COMMITS to a
# plan (Intention) and walks its steps one by one; it only
# re-deliberates once the plan is exhausted or no longer applicable.
# That is means-end BDI reasoning, not an if/elif cascade in disguise.
#
# An ACTION is a primitive step that, from the beliefs, produces an
# execution Step (a hint for the DemoAgent). An action can be NOT
# applicable in the current state (build() returns None) — the agent
# then skips it and moves to the next one in the body. Actions do not
# check `tried_hints` themselves: that dedup is handled once, centrally,
# by `Plan.has_executable_step()` / the planner's step-walking logic —
# see agents/bdi_planner_agent.py.
#
# Priority hierarchy (higher = selected first). Entries marked (*) are
# gated behind a config flag (config.CONFLICT_DETECTION_ENABLED /
# config.DYNAMIC_DESIRES_ENABLED) — inert by default, since their trigger
# desire is only added to active_desires when the flag is on (see
# agents/bdi_planner_agent.py's _compute_context()):
#   100  html_prescan          (classic external information)
#    95  immediate_answer      ("homepage/last message" task -> reinforce RULE 5)
#  * 94  final_attempt         (last attempt before budget exhausted -> safest known approach)
#  * 92  resolve_conflict      (executor/verifier disagree repeatedly -> recheck mechanical criteria)
#    90  warm_start            (learned strategy, 1st attempt)
#  * 87  category_high_risk    (low historical success for this category -> cautious opener)
#    85  first_attempt         (free 1st attempt)
#    80  fix_grounding         (answered without checking -> stop inventing bids)
#    75  fix_completeness      (incomplete answer -> never answered)
#    70  multi_action_error    (protocol error)
#    65  recover_stuck         (action loop -> wrap up before steps run out)
#  * 60  escape_plateau        (score plateaued/degrading -> completely different entry point)
#    55  verifier_feedback     (enriched verifier feedback)
#    45  change_strategy       (degrading trend -> low efficiency)
#  * 42  diagnose_unknown_failure (no specific plan matched twice running -> deep diagnostic)
#    35  zero_specificity      (answer too vague)
#     0  fallback              (last resort)
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFLICT_DETECTION_ENABLED, DYNAMIC_DESIRES_ENABLED


# ─────────────────────────────────────────────────────────
# Action-history analysis helpers (shared across Actions)
# ─────────────────────────────────────────────────────────

def _action_name(response_text: str) -> str:
    m = re.search(r'```[^\n`]*\n?([\w_]+)\s*\(', str(response_text))
    if m:
        return m.group(1)
    for act in ('click', 'fill', 'scroll', 'goto', 'hover',
                'select_option', 'send_msg_to_user', 'report_infeasible',
                'press', 'clear', 'drag_and_drop', 'upload_file'):
        if f'{act}(' in str(response_text):
            return act
    return str(response_text).split("(")[0].strip()[:30]


def _extract_sent_answer(action_list: list) -> str:
    for a in reversed(action_list):
        m = re.search(r"send_msg_to_user\(['\"](.{1,120}?)['\"]", str(a))
        if m:
            return m.group(1).strip()
    return ""


def _nav_actions(action_list: list) -> List[str]:
    return [
        str(a)[:60] for a in action_list
        if any(k in str(a) for k in ('click(', 'scroll(', 'fill(', 'select_option('))
    ]


def _answered_without_navigating(action_list: list) -> bool:
    nav_before_send = False
    for a in action_list:
        if any(k in str(a) for k in ('click(', 'scroll(', 'fill(')):
            nav_before_send = True
        if 'send_msg_to_user(' in str(a):
            return not nav_before_send
    return False


# ═══════════════════════════════════════════════════════════
#  Step produced by an action
# ═══════════════════════════════════════════════════════════

@dataclass
class Step:
    """An executable step: what the agent sends to the LLM executor."""
    action: str   # action name (traceability)
    hint:   str   # instruction for the DemoAgent ("" = free navigation)


# ═══════════════════════════════════════════════════════════
#  Primitive ACTIONS (building blocks of plan bodies)
#
#  Each one is a straight port of the corresponding single-hint Plan
#  from the previous architecture: same context_condition +
#  generate_hint logic, merged into one build().
# ═══════════════════════════════════════════════════════════

class Action:
    """Primitive step of a plan. build() -> Step, or None if not applicable."""
    name: str = "action"

    def build(self, beliefs, **ctx) -> Optional[Step]:
        raise NotImplementedError


class HtmlPrescan(Action):
    """HTML pre-scan (classic, no LLM) found a relevant keyword."""
    name = "html_prescan"

    def build(self, beliefs, prescan_hint="", **ctx):
        if not prescan_hint:
            return None
        return Step(self.name, prescan_hint)


class ImmediateAnswer(Action):
    """
    Homepage / last-message task: reinforce RULE 5 at every attempt.

    LLMExecutorAgent already injects RULE 5 into the system message, but
    after a shot-1 failure, lower-priority actions ("navigate
    differently") would CONTRADICT it. This action always takes
    precedence and produces a hint that CONFIRMS RULE 5.
    """
    name = "immediate_answer"
    _TRIGGERS = ("homepage", "last message")

    def build(self, beliefs, **ctx):
        goal = beliefs.task_goal.lower()
        if not any(t in goal for t in self._TRIGGERS):
            return None
        # Use tried_hints length as the shot counter so each attempt
        # still generates a unique hint.
        shot_num = len(beliefs.tried_hints) + 1
        hint = (
            f"[Shot {shot_num}] IMMEDIATE ANSWER TASK: The answer is already visible "
            "in the current AXTree — no navigation required. "
            "Your FIRST action MUST be send_msg_to_user('exact value from page'). "
            "Do NOT scroll. Do NOT click. Do NOT goto any URL. "
            "Read the AXTree right now and send the answer immediately."
        )
        return Step(self.name, hint)


class WarmStart(Action):
    """Reuses a learned strategy (provided via warm_hint), 1st attempt only."""
    name = "warm_start"

    def build(self, beliefs, warm_hint="", **ctx):
        if beliefs.attempt_count != 0 or not warm_hint:
            return None
        return Step(self.name, warm_hint)


class FirstAttempt(Action):
    """1st attempt, no hint: DemoAgent is free to explore."""
    name = "first_attempt"

    def build(self, beliefs, warm_hint="", prescan_hint="", **ctx):
        if beliefs.attempt_count == 0 and not warm_hint and not prescan_hint:
            return Step(self.name, "")
        return None


class AnsweredWithoutNav(Action):
    """The agent answered without navigating the page."""
    name = "answered_without_nav"

    def build(self, beliefs, action_history=None, **ctx):
        ah = action_history or []
        if not (_answered_without_navigating(ah) and beliefs.attempt_count > 0):
            return None
        wrong = _extract_sent_answer(ah)
        hint = (
            "The previous attempt sent the answer"
            + (f" '{wrong}'" if wrong else "")
            + " WITHOUT verifying on the page — that was incorrect. "
            "You must first navigate to the correct element, "
            "read the actual value in the AXTree, "
            "then send the exact answer you see."
        )
        return Step(self.name, hint)


class FabricatedBid(Action):
    """
    Pattern from log analysis: the agent invents bid numbers
    ("hypothetically bid '367'") instead of reading them from the
    AXTree, then clicks non-existent elements and never answers.
    """
    name = "fabricated_bid"
    _MARKERS = (
        "hypothetical", "let's say", "let us say",
        "bid for", "assuming", "hypothetically",
        "bid might be", "bid is probably",
        "say the bid", "suppose the bid",
        "correct bid", "the bid for",
    )

    def build(self, beliefs, action_history=None, **ctx):
        ah     = action_history or []
        ah_str = " ".join(str(a) for a in ah).lower()
        if not any(m in ah_str for m in self._MARKERS):
            return None
        hint = (
            "CRITICAL ERROR from previous attempt: you GUESSED or INVENTED "
            "bid numbers (e.g. 'hypothetically bid 367') instead of reading "
            "them from the AXTree. BIDs change every page load — you cannot "
            "know them in advance. "
            "STRICT RULE: Only use bid numbers that are VISIBLE in the current "
            "AXTree text. Read the AXTree carefully — each element shows its bid "
            "as a plain number. Never assume, guess, or comment on what a bid "
            "'might be' or 'is probably'."
        )
        return Step(self.name, hint)


class IncompleteAnswer(Action):
    """Navigation succeeded but the answer is incomplete (specificity~0)."""
    name = "incomplete_answer"

    def build(self, beliefs, action_history=None, **ctx):
        ah = action_history or []
        if not (
            beliefs.last_dim_specificity < 0.1
            and beliefs.last_u_v > 0.3
            and bool(_nav_actions(ah))
            and beliefs.attempt_count > 0
        ):
            return None
        wrong = _extract_sent_answer(ah)
        nav   = _nav_actions(ah)
        hint = (
            "You found the target"
            + (f" and sent: '{wrong[:80]}'" if wrong else "")
            + " but the answer was INCOMPLETE. "
            "Navigate to the correct page and send ALL requested information: "
            "include the full description, menu items (with prices if shown), "
            "and reviews — not just the rating. "
            f"Previous navigation: {' → '.join(nav[:3])}. "
            "Follow the same navigation path and collect ALL details."
        )
        return Step(self.name, hint)


class NeverAnswered(Action):
    """
    Pattern from log analysis: the agent navigates (click/scroll/fill)
    but never calls send_msg_to_user(), exhausting steps unanswered.
    """
    name = "never_answered"

    def build(self, beliefs, action_history=None, **ctx):
        ah         = action_history or []
        has_nav    = any(k in str(a) for a in ah for k in ('click(', 'scroll(', 'fill('))
        never_sent = not any('send_msg_to_user' in str(a) for a in ah)
        if not (beliefs.attempt_count >= 1 and has_nav and never_sent and len(ah) >= 2):
            return None
        nav  = _nav_actions(ah)
        hint = (
            "In the previous attempt, you navigated the page"
            + (f" ({' → '.join(nav[:3])})" if nav else "")
            + " but NEVER sent your final answer — you used all steps "
            "without calling send_msg_to_user(). "
            "RULE: As soon as you see the requested value in the AXTree, "
            "call send_msg_to_user('exact value') IMMEDIATELY. "
            "Do not continue navigating after finding the answer."
        )
        return Step(self.name, hint)


class MultiActionError(Action):
    """The agent produced multiple actions in a single response."""
    name = "multi_action_error"

    def build(self, beliefs, action_history=None, last_error="", **ctx):
        ah          = action_history or []
        multi_count = sum(1 for a in ah if "multi-action" in str(a).lower())
        if not ("multi-action" in last_error.lower() or multi_count >= 2):
            return None
        hint = (
            "IMPORTANT: you must generate ONLY ONE action per response. "
            "Write only a single ```action(args)``` block per step. "
            "Never combine multiple actions in the same response."
        )
        return Step(self.name, hint)


class StuckAction(Action):
    """The agent repeats the same action >= 3 times in a row."""
    name = "stuck_action"

    def build(self, beliefs, action_history=None, **ctx):
        ah = action_history or []
        if len(ah) < 3:
            return None
        names = [_action_name(a) for a in ah[-3:]]
        if len(set(names)) != 1:
            return None
        blocked = _action_name(ah[-1])
        if blocked == "scroll":
            unblock = (
                "Instead of scrolling in a loop, click on a visible navigation "
                "LINK in the AXTree. "
                "Do not use goto() — that action does not exist."
            )
        elif blocked == "click":
            unblock = (
                "Instead of clicking the same element again, "
                "use scroll(0, 300) to discover other elements "
                "or click a different navigation link."
            )
        else:
            unblock = (
                "Try scroll(0, 300) to discover new elements, "
                "or click a navigation link in the top menu."
            )
        hint = (
            f"You are stuck on '{blocked}' for "
            f"{min(len(ah), 5)} steps — the page is not changing. "
            + unblock
        )
        return Step(self.name, hint)


class StepsExhausted(Action):
    """Close to the goal but out of steps."""
    name = "steps_exhausted"

    def build(self, beliefs, action_history=None, n_steps=0, max_steps=15, **ctx):
        if not (n_steps >= max_steps - 2 and beliefs.last_u_v > 0.3):
            return None
        ah     = action_history or []
        recent = ah[-5:] if len(ah) >= 5 else ah
        recent_summary = " | ".join(str(a)[:55] for a in recent)
        hint = (
            f"You were close to the goal (U_V={beliefs.last_u_v:.2f}) "
            f"after {n_steps} steps but ran out of steps. "
            f"Last actions: {recent_summary}. "
            "Continue from this state and complete the final step."
        )
        return Step(self.name, hint)


class VerifierFeedback(Action):
    """Verifier feedback enriched with action-history context."""
    name = "verifier_feedback"

    def build(self, beliefs, action_history=None, **ctx):
        if not (bool(beliefs.last_feedback) and beliefs.attempt_count > 0):
            return None
        if beliefs.last_feedback in beliefs.tried_hints:
            return None
        ah    = action_history or []
        nav   = _nav_actions(ah)
        wrong = _extract_sent_answer(ah)
        parts = []
        if wrong:
            parts.append(f"answer given: '{wrong}'")
        if nav:
            parts.append(f"navigation performed: {' -> '.join(nav[:3])}")

        # Always prepend a navigation instruction so the agent explores
        # before trying to answer (prevents an immediate report_infeasible/empty send)
        nav_prefix = (
            "NAVIGATE FIRST: open the page, find the requested information "
            "in the AXTree, then call send_msg_to_user with the exact value. "
        )
        if parts:
            context = " | ".join(parts)
            hint = nav_prefix + f"{beliefs.last_feedback} [Previous context — {context}]"
        else:
            hint = nav_prefix + beliefs.last_feedback
        return Step(self.name, hint)


class LowEfficiency(Action):
    """Too many wasted steps (dim_efficiency very low)."""
    name = "low_efficiency"

    def build(self, beliefs, action_history=None, **ctx):
        if not (beliefs.last_dim_efficiency < 0.25 and beliefs.attempt_count >= 1):
            return None
        ah          = action_history or []
        nav         = _nav_actions(ah)
        nav_summary = " | ".join(nav[-3:]) if nav else "no navigation detected"
        hint = (
            f"The previous attempt ({len(ah)} steps) was inefficient. "
            f"Navigation actions: {nav_summary}. "
            "Check whether the answer is directly visible in the AXTree "
            "without additional navigation."
        )
        return Step(self.name, hint)


class DegradingTrend(Action):
    """Degrading trend over past scores."""
    name = "degrading_trend"

    def build(self, beliefs, **ctx):
        if beliefs.trend() != "DEGRADING":
            return None
        scores = " -> ".join(f"{u:.2f}" for u in beliefs.past_u_v[-3:])
        hint = (
            f"The {len(beliefs.past_u_v)} previous attempts are degrading "
            f"(U_V: {scores}). "
            "Completely change your approach — try a different navigation path."
        )
        return Step(self.name, hint)


class ZeroSpecificity(Action):
    """Answer too vague, value incorrect or missing."""
    name = "zero_specificity"

    def build(self, beliefs, action_history=None, **ctx):
        if not (beliefs.last_dim_specificity < 0.3 and beliefs.attempt_count >= 1):
            return None
        ah    = action_history or []
        wrong = _extract_sent_answer(ah)
        hint = (
            "Navigate to the target page and find the exact value in the AXTree. "
            "The previous answer"
            + (f" ('{wrong}')" if wrong else "")
            + " was incorrect or too vague. "
            "Once you see the value on screen, call send_msg_to_user with the "
            "EXACT content visible: full names, precise numbers, text as it appears. "
            "Do not infer or guess — copy the actual values from the AXTree."
        )
        return Step(self.name, hint)


class ResolveConflict(Action):
    """
    [Conflict detection] The executor keeps behaving as if it succeeded
    (high U_PE) while the verifier keeps rejecting it (low U_V), two
    attempts running (BeliefBase.consecutive_conflict >= 2 — see brf()).
    Recadres the executor onto the verifier's exact, mechanical criteria
    instead of its own sense of completeness.
    """
    name = "resolve_conflict"

    def build(self, beliefs, **ctx):
        # Self-contained flag check (not just active_desires gating in
        # bdi_planner_agent.py): consecutive_conflict is maintained by
        # BeliefBase.brf() for ANY caller, so other consumers of
        # PlanLibrary that don't pass active_desires (e.g. the frering_bdi
        # ablation baseline) must not accidentally pick this plan up.
        if not CONFLICT_DETECTION_ENABLED or beliefs.consecutive_conflict < 2:
            return None
        hint = (
            f"CRITICAL: you have judged this task complete {beliefs.consecutive_conflict} "
            "times in a row, but the verifier rejected it every time. Stop trusting your own "
            "sense of completeness. Before calling send_msg_to_user, mechanically check: "
            "(1) does the value you are about to send EXACTLY match what is visible in the "
            "current AXTree, character for character? (2) did the goal ask for MULTIPLE pieces "
            "of information (e.g. price AND rating AND description) that you have not all "
            "included? (3) is the value grounded on THIS page, not assumed from memory or a "
            "previous step? Only send the answer once all three checks pass."
        )
        return Step(self.name, hint)


class EscapePlateau(Action):
    """
    [Dynamic desire] Score has plateaued (BeliefBase.is_plateau()) or is
    degrading, and still below the success threshold. Additive to
    DegradingTrend (priority 45): fires earlier (priority 60) and demands
    a fully different entry point, not just "try again differently".
    """
    name = "escape_plateau"

    def build(self, beliefs, **ctx):
        # Self-contained flag check — see ResolveConflict.build() comment:
        # attempt_count/last_u_v/past_u_v are generic BeliefBase fields
        # maintained for any caller, so this must not fire for a consumer
        # of PlanLibrary that doesn't pass active_desires.
        if not DYNAMIC_DESIRES_ENABLED:
            return None
        if not (beliefs.attempt_count >= 1 and beliefs.last_u_v < 0.7
                and (beliefs.is_plateau(2, 0.05) or beliefs.trend() == "DEGRADING")):
            return None
        scores = " -> ".join(f"{u:.2f}" for u in beliefs.past_u_v[-3:])
        hint = (
            f"You are stuck on a plateau (U_V: {scores} — not improving). "
            "Do not repeat variations of the same navigation path, it is not working. "
            "Pick a COMPLETELY different entry point: use the search bar instead of browsing "
            "categories, or browse categories instead of searching, or start from a different "
            "section of the site entirely."
        )
        return Step(self.name, hint)


class FinalAttempt(Action):
    """
    [Dynamic desire] Last attempt before the deliberation budget runs
    out (attempt_count == max_replans - 1). Falls back to the most
    reliable known hint for this category rather than exploring further —
    unlike WarmStart, which only fires on attempt_count == 0.
    """
    name = "final_attempt"

    def build(self, beliefs, safest_hint="", max_replans=0, **ctx):
        if not DYNAMIC_DESIRES_ENABLED:
            return None
        if not safest_hint or beliefs.attempt_count != max(0, max_replans - 1):
            return None
        return Step(self.name,
                     f"FINAL ATTEMPT — use the most reliable known approach: {safest_hint}")


class DiagnoseUnknownFailure(Action):
    """
    [Dynamic desire] consecutive_fallback >= 2: no specific plan has
    recognized this failure mode for two attempts running (only the
    generic Fallback matched). Provide a deeper diagnostic prompt instead
    of another generic retry.
    """
    name = "diagnose_unknown_failure"

    def build(self, beliefs, action_history=None, **ctx):
        if not DYNAMIC_DESIRES_ENABLED or beliefs.consecutive_fallback < 2:
            return None
        ah = action_history or []
        recent = ah[-5:] if len(ah) >= 5 else ah
        recent_summary = " | ".join(str(a)[:60] for a in recent) or "no actions recorded"
        hint = (
            f"DIAGNOSTIC: {beliefs.consecutive_fallback} attempts in a row failed in a way no "
            f"known pattern recognizes. Recent actions: {recent_summary}. "
            "Before acting, explicitly state in one sentence WHAT specifically went wrong last "
            "time (e.g. 'the element I need is not on this page', 'I misread the goal', "
            "'the site structure is different than expected'), then take an action that "
            "directly addresses that diagnosis rather than repeating the same navigation."
        )
        return Step(self.name, hint)


class CategoryHighRisk(Action):
    """
    [Dynamic desire, ORDINARY not preemptive] This category has a
    persistently low average score across many past tasks (WorldModel).
    Injects a cautious, preventive scaffold on the first attempt only —
    priority sits below warm_start, so it only fires when no successful
    warm-start hint exists for this category.
    """
    name = "category_high_risk"

    def build(self, beliefs, category_high_risk=False, **ctx):
        if not DYNAMIC_DESIRES_ENABLED:
            return None
        if beliefs.attempt_count != 0 or not category_high_risk:
            return None
        hint = (
            "NOTE: tasks on this site have historically been difficult to complete "
            "accurately. Proceed carefully: read the full AXTree before acting, "
            "verify the exact page/section the goal refers to before extracting any "
            "value, and double-check the requested value is complete (not partially "
            "read) before sending your answer."
        )
        return Step(self.name, hint)


class Fallback(Action):
    """Last resort — hint text varies with the attempt number (never 'already tried')."""
    name = "fallback"

    def build(self, beliefs, action_history=None, **ctx):
        ah    = action_history or []
        nav   = _nav_actions(ah)
        wrong = _extract_sent_answer(ah)
        parts = [f"Attempt {beliefs.attempt_count + 1} — a different approach is required."]
        if wrong:
            parts.append(f"The previous answer '{wrong}' was incorrect.")
        if nav:
            parts.append(
                f"Previous navigation: {' -> '.join(nav[:3])}."
                " Try a different path."
            )
        parts.append(f"Best U_V achieved: {beliefs.best_u_v():.2f}.")
        return Step(self.name, " ".join(parts))


# ═══════════════════════════════════════════════════════════
#  PLAN = trigger + context + body (sequence of actions)
# ═══════════════════════════════════════════════════════════

@dataclass
class Plan:
    name:           str
    description:    str
    trigger_desire: str   # "complete_task" | "minimize_steps" | "specific_answer"
    priority:       int
    body:           List[Action] = field(default_factory=list)
    _context:       Optional[Callable] = None
    allows_empty_hint: bool = False

    def context_condition(self, beliefs, **ctx) -> bool:
        return self._context(beliefs, **ctx) if self._context else True

    def has_executable_step(self, beliefs, **ctx) -> bool:
        """True if at least one action in the body yields a still-unused hint."""
        for action in self.body:
            step = action.build(beliefs, **ctx)
            if step is None:
                continue
            if step.hint and step.hint in beliefs.tried_hints:
                continue
            return True
        return False


# ═══════════════════════════════════════════════════════════
#  LIBRARY
# ═══════════════════════════════════════════════════════════

class PlanLibrary:
    """
    Library of BDI Plans (with action bodies).
    select_plan() = filter() of the BDI cycle: returns the
    highest-priority applicable plan with at least one executable step.
    """

    def __init__(self):
        self._plans: List[Plan] = self._build_library()

    def _build_library(self) -> List[Plan]:
        plans = [
            Plan("html_prescan", "HTML pre-scan found a relevant keyword",
                 "complete_task", 100,
                 body=[HtmlPrescan()]),

            Plan("immediate_answer", "Homepage/last-message task: reinforce RULE 5",
                 "complete_task", 95,
                 body=[ImmediateAnswer()]),

            # [Dynamic desire] Last attempt before the budget runs out —
            # falls back to the safest known approach. Gated behind
            # config.DYNAMIC_DESIRES_ENABLED (see bdi_planner_agent.py's
            # _compute_context()); inert otherwise since its trigger
            # desire is never added to active_desires.
            Plan("final_attempt", "Last attempt before budget exhausted -> safest known approach",
                 "final_attempt", 94,
                 body=[FinalAttempt()]),

            # [Conflict detection] Executor/verifier disagree repeatedly.
            # Gated behind config.CONFLICT_DETECTION_ENABLED.
            Plan("resolve_conflict", "Executor/verifier disagree repeatedly -> recheck mechanical criteria",
                 "resolve_conflict", 92,
                 body=[ResolveConflict()]),

            Plan("warm_start", "Reuse successful strategy from history",
                 "complete_task", 90,
                 body=[WarmStart()],
                 allows_empty_hint=True),

            # [Dynamic desire, ordinary] Category has a low historical
            # average score — cautious first-attempt scaffold. Sits below
            # warm_start so a proven hint always wins when one exists.
            # Gated behind config.DYNAMIC_DESIRES_ENABLED.
            Plan("category_high_risk", "Category has low historical success -> cautious opening scaffold",
                 "category_high_risk", 87,
                 body=[CategoryHighRisk()]),

            Plan("first_attempt", "First attempt — DemoAgent is free",
                 "complete_task", 85,
                 body=[FirstAttempt()],
                 allows_empty_hint=True),

            # Grounding recipe: force navigation before answering, then —
            # if that alone was already tried — stop inventing bid numbers.
            Plan("fix_grounding", "Answered without checking -> stop inventing bids",
                 "complete_task", 80,
                 body=[AnsweredWithoutNav(), FabricatedBid()]),

            # Completeness recipe: collect all requested details, escalating
            # to "you never answered at all" if that hint is exhausted.
            # Kept under complete_task (not specific_answer / minimize_steps)
            # so it can't be starved if a lower-priority desire gets
            # suspended by brf() on a hard rejection.
            Plan("fix_completeness", "Incomplete answer -> never answered",
                 "complete_task", 75,
                 body=[IncompleteAnswer(), NeverAnswered()]),

            Plan("multi_action_error", "Force single action per response",
                 "complete_task", 70,
                 body=[MultiActionError()]),

            # Recovery recipe: break the action loop, then — if still stuck
            # near the step budget — wrap up from the current state. Kept
            # under complete_task rather than minimize_steps: brf() suspends
            # minimize_steps on a hard rejection (U_PE < 0.2), which is
            # exactly when this recovery is needed most.
            Plan("recover_stuck", "Break action loop -> wrap up before steps run out",
                 "complete_task", 65,
                 body=[StuckAction(), StepsExhausted()]),

            # [Dynamic desire] Score plateaued or degrading — escalate to a
            # fully different entry point. Additive to change_strategy
            # below (does not replace DegradingTrend there). Gated behind
            # config.DYNAMIC_DESIRES_ENABLED.
            Plan("escape_plateau", "Score plateaued or degrading -> completely different entry point",
                 "escape_plateau", 60,
                 body=[EscapePlateau()]),

            Plan("verifier_feedback", "Enrich verifier feedback with action context",
                 "complete_task", 55,
                 body=[VerifierFeedback()]),

            # Strategy-change recipe: same reasoning as recover_stuck — kept
            # under complete_task so it survives a minimize_steps suspension.
            Plan("change_strategy", "Different approach -> more direct navigation",
                 "complete_task", 45,
                 body=[DegradingTrend(), LowEfficiency()]),

            # [Dynamic desire] No specific plan matched two attempts running
            # (only the generic fallback did) — deeper diagnostic than
            # Fallback itself. Gated behind config.DYNAMIC_DESIRES_ENABLED.
            Plan("diagnose_unknown_failure", "No specific plan matched twice in a row -> deep diagnostic",
                 "diagnose_unknown_failure", 42,
                 body=[DiagnoseUnknownFailure()]),

            Plan("zero_specificity", "Provide exact value, not a vague answer",
                 "specific_answer", 35,
                 body=[ZeroSpecificity()]),

            Plan("fallback", "Last resort contextual hint",
                 "complete_task", 0,
                 body=[Fallback()]),
        ]
        return sorted(plans, key=lambda p: p.priority, reverse=True)

    def select_plan(self, beliefs, active_desires=None,
                     executor_reliability_fn=None, **ctx) -> Plan:
        """
        filter() of the BDI cycle — returns the highest-priority plan
        whose (1) trigger desire is active, (2) context is true, and
        (3) has at least one executable step (unused hint).

        `active_desires` (set of names — see BDIState.options()) makes
        Desires actually decisional rather than mere labels: a plan
        whose desire was suspended or achieved is skipped regardless of
        priority.

        `executor_reliability_fn(task_type, plan_name) -> float` carries
        a belief about the executor: a plan believed unreliable for this
        site (<0.1) is skipped in favor of the next one.
        """
        for plan in self._plans:
            if active_desires is not None and plan.trigger_desire not in active_desires:
                continue
            if not plan.context_condition(beliefs, **ctx):
                continue
            if not plan.has_executable_step(beliefs, **ctx):
                continue
            if executor_reliability_fn and plan.priority > 0 and beliefs.task_type:
                reliability = executor_reliability_fn(beliefs.task_type, plan.name)
                if reliability < 0.1:
                    print(f"  [BDI] Skipping plan '{plan.name}' — executor believed unreliable "
                          f"for '{beliefs.task_type}' (success rate={reliability:.2f})")
                    continue
            return plan
        return self._plans[-1]  # fallback (always applicable)

    def get_plan(self, name: str) -> Optional[Plan]:
        for p in self._plans:
            if p.name == name:
                return p
        return None

    def __repr__(self) -> str:
        return ("PlanLibrary(" + ", ".join(
            f"{p.name}[{'→'.join(a.name for a in p.body)}]" for p in self._plans) + ")")
