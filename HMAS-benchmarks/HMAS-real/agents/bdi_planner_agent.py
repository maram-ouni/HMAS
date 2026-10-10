# ─────────────────────────────────────────────────────────
# bdi_planner_agent.py — REAL BDI PLANNER AGENT — v2
#
# Implements the full BDI cycle (Rao & Georgeff, 1991):
#
#   perceive   → receives TASK_ASSIGNED or VERIFICATION_RESULT
#   brf        → updates Beliefs via BeliefBase.brf()
#   reconsider → should the current intention be dropped?
#                (driven by CommitmentStrategy)
#   options()  → active, not-yet-achieved desires, by priority
#   filter()   → PlanLibrary.select_plan() picks the best applicable Plan
#   execute    → publishes EXECUTION_REQUESTED to LLMExecutorAgent
#
# v2 changes vs the previous version:
#   - Plans have a BODY (sequence of Actions) instead of one hint.
#     While the agent holds its commitment (SINGLE_MINDED, no
#     reconsideration needed), it walks the body one step per
#     internal cycle via _emit_next_step()/_dispatch() instead of
#     re-selecting a plan (means-end reasoning, not re-deliberation).
#   - The deliberation loop (up to `max_replans` attempts) is now
#     fully INTERNAL to this agent: it never gets recreated between
#     attempts, so its memory (Beliefs/Desires/Intentions) persists
#     for the whole task. `max_replans` is overridable per task via
#     TASK_ASSIGNED's `max_attempts` (wired to --shots).
#   - WorldModel: cross-task/cross-run beliefs about category
#     difficulty and per-(category, plan) executor reliability feed
#     into PlanLibrary.select_plan() (see core/world_model.py).
#   - Anti-regression guard: the agent tracks `_best_content`, the
#     best attempt seen for the current task (by U_V). If the budget
#     is exhausted without success, that best attempt is reported —
#     not the last one — so the MAS never regresses.
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import queue
import re as _re_prescan
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.message_bus     import MessageBus, Message, MessageType
from core.history_manager  import HistoryManager
from core.bdi_model        import BDIState, CommitmentStrategy
from core.plan_library     import PlanLibrary, Step
from core.utils            import extract_task_type
from core.strategy_learner import StrategyLearner
from core.world_model      import WorldModel
from config                import (
    MAX_STEPS, MAX_REPLANS, RESULTS_DIR,
    TRACE_ENABLED, CONFLICT_DETECTION_ENABLED, DYNAMIC_DESIRES_ENABLED,
    UCB1_ENABLED,
)


class BDIPlannerAgent(threading.Thread):
    """
    Real BDI planner agent — strategic, LLM-free.

    The PlanLibrary replaces the old if/elif cascade. Each Plan is a
    trigger/context/body triplet; the agent commits to one (Intention)
    and walks its body across internal replans instead of
    re-deliberating on every cycle.
    """

    _TASK_BASE_URLS = {
        "opendining": "https://evals-opendining.vercel.app",
        "dashdish":   "https://evals-dashdish.vercel.app",
        "omnizon":    "https://evals-omnizon.vercel.app",
        "staynb":     "https://evals-staynb.vercel.app",
        "gomail":     "https://evals-gomail.vercel.app",
    }

    def __init__(self, bus: MessageBus,
                 history_manager: HistoryManager,
                 stop_event: threading.Event,
                 commitment_strategy: CommitmentStrategy = CommitmentStrategy.SINGLE_MINDED,
                 memory_dir: str = None):
        super().__init__(daemon=True, name="BDIPlannerThread")
        self.bus              = bus
        self.hm               = history_manager
        self.stop_event       = stop_event
        self.bdi              = BDIState(commitment_strategy=commitment_strategy)
        self.plan_library     = PlanLibrary()
        self._max_steps       = MAX_STEPS
        self._headless        = True
        # Isolated-memory override (None = today's hardcoded RESULTS_DIR
        # paths, shared across runs).
        self._memory_dir      = memory_dir or RESULTS_DIR
        self.learner          = StrategyLearner(results_dir=self._memory_dir)
        # Internal deliberation budget (overridable per task via
        # TASK_ASSIGNED's max_attempts). The agent iterates up to this
        # many attempts WITHOUT being recreated — its memory persists
        # for the whole task.
        self.max_replans      = MAX_REPLANS
        # Best attempt seen for the current task — the MAS must never
        # regress: once the budget is exhausted, the best attempt is
        # reported, not the last one.
        self._best_content: dict = {}
        # Cross-task/cross-run beliefs: category difficulty
        # (environment) + executor reliability per (category, plan).
        self.world            = WorldModel(results_dir=self._memory_dir)

        print(f"[BDIPlannerThread] Plan library: {self.plan_library}")

    # ── Main loop ─────────────────────────────────────────

    def run(self) -> None:
        print("[BDIPlannerThread] Started — real BDI cycle (no LLM)...")
        print(f"[BDIPlannerThread] Commitment strategy: "
              f"{self.bdi.commitment_strategy.value}")
        while not self.stop_event.is_set():
            try:
                msg = self.bus.get("planner", timeout=2)
            except queue.Empty:
                continue

            if msg.msg_type == MessageType.STOP:
                break
            elif msg.msg_type == MessageType.TASK_ASSIGNED:
                self._on_task_assigned(msg)
            elif msg.msg_type == MessageType.VERIFICATION_RESULT:
                self._on_verification_result(msg)

        print("[BDIPlannerThread] Stopped.")

    # ── Message handlers ──────────────────────────────────

    def _on_task_assigned(self, msg: Message) -> None:
        """
        Initial percept: new task. Initializes beliefs and starts the
        BDI cycle — its own internal loop drives every replan from
        here on, until success or budget exhaustion.
        """
        task_id      = msg.content["task_id"]
        task_goal    = msg.content["task_goal"]
        prev_result  = msg.content.get("prev_result", {})
        self._max_steps = msg.content.get("max_steps", MAX_STEPS)
        self._headless  = msg.content.get("headless", True)
        max_attempts = msg.content.get("max_attempts", None)

        # Deliberation budget for THIS task (wired to --shots).
        if max_attempts is not None:
            self.max_replans = max(1, int(max_attempts))

        # BRF: initializes beliefs. Unlike the old shots architecture,
        # this state PERSISTS across every internal replan below —
        # the agent is never recreated mid-task.
        self.bdi.set_task(task_id, task_goal, extract_task_type(task_id))
        self._best_content = {}

        # Optional cross-session seed (used by --demo to inject a
        # realistic prior failure before the first internal attempt).
        if prev_result:
            self.bdi.brf(
                reward          = prev_result.get("reward",          0.0),
                u_pe            = prev_result.get("U_PE",            0.0),
                u_v             = prev_result.get("U_V",             0.0),
                quality         = prev_result.get("quality",         "rejected"),
                feedback        = prev_result.get("feedback",        ""),
                hint            = prev_result.get("strategy",        ""),
                dim_efficiency  = prev_result.get("dim_efficiency",  0.0),
                dim_specificity = prev_result.get("dim_specificity", 0.0),
                action_history  = prev_result.get("action_history",  []),
                last_error      = prev_result.get("last_error",      ""),
            )
            for h in prev_result.get("tried_hints", []):
                if h and h not in self.bdi.beliefs.tried_hints:
                    self.bdi.beliefs.tried_hints.append(h)
            print(f"\n  [BDI] Seeded prior result: "
                  f"reward={prev_result.get('reward',0):.1f} "
                  f"U_V={prev_result.get('U_V',0):.3f} "
                  f"quality={prev_result.get('quality','?')} "
                  f"tried_hints={len(self.bdi.beliefs.tried_hints)}")

        print(f"\n  [BDI] New task: {task_id} (budget={self.max_replans} attempt(s))")
        print(f"  [BDI] {self.bdi.summary()}")

        # First turn of the internal deliberation loop. Subsequent
        # turns are triggered by _on_verification_result (percept →
        # brf → reconsider → replan) until the budget runs out.
        self._bdi_cycle(reconsider=True)

    def _on_verification_result(self, msg: Message) -> None:
        """
        Percept: verification result.

        Full BDI cycle:
          1. brf()         → belief update
          2. reconsider()  → drop the intention, or advance its step
          3. should_replan → internal replan, or report best attempt
        """
        reward          = msg.content.get("reward",          0.0)
        u_pe            = msg.content.get("U_PE",            0.0)
        u_v             = msg.content.get("U_V",             0.0)
        quality         = msg.content.get("quality",         "rejected")
        feedback        = msg.content.get("feedback",        "")
        hint_used       = msg.content.get("hint_used", "") or msg.content.get("strategy", "")
        dim_efficiency  = msg.content.get("dim_efficiency",  0.0)
        dim_specificity = msg.content.get("dim_specificity", 0.0)
        dim_quality     = msg.content.get("dim_quality",     0.0)
        action_history  = msg.content.get("action_history",  [])
        last_error      = msg.content.get("last_error",      "")

        # Plan that produced THIS result — captured before brf()/the
        # next commit() advances the intention stack.
        plan_used = self.bdi.current_intention().plan_name \
            if self.bdi.current_intention() else "unknown"

        # 1. BRF (also updates consecutive_conflict — see BeliefBase.brf())
        self.bdi.brf(reward, u_pe, u_v, quality, feedback, hint_used,
                     dim_efficiency, dim_specificity, action_history, last_error)

        # Fallback-streak tracking: brf() cannot do this itself since it
        # never receives plan_name. Reset whenever a SPECIFIC plan matched.
        if plan_used == "fallback":
            self.bdi.beliefs.consecutive_fallback += 1
        else:
            self.bdi.beliefs.consecutive_fallback = 0

        # Persistent beliefs: the environment (category difficulty) and
        # the executor agent (reliability of this plan for this category).
        self.world.record_task_outcome(self.bdi.beliefs.task_type, u_v)
        self.world.record_plan_outcome(self.bdi.beliefs.task_type, plan_used,
                                        success=(quality == "success"))

        print(f"\n  [BDI] Beliefs updated: quality={quality}, U_V={u_v:.3f}, "
              f"trend={self.bdi.beliefs.trend()}")
        print(f"  [BDI] {self.bdi.summary()}")

        # Anti-regression guard: remember the BEST attempt of the task
        # (by U_V). If the budget runs out without success, this is
        # what gets reported to the system, not the last attempt.
        report = {
            "task_id":         self.bdi.beliefs.task_id,
            "attempts":        self.bdi.beliefs.attempt_count,
            "U_PE":            u_pe,
            "U_V":             u_v,
            "reward":          reward,
            "steps":           msg.content.get("steps",  0),
            "answer":          msg.content.get("answer", ""),
            "quality":         quality,
            "verdict":         msg.content.get("verdict", ""),
            "hint_used":       hint_used,
            "plan_name":       plan_used,
            "bdi_trend":       self.bdi.beliefs.trend(),
            "bdi_best_u_v":    self.bdi.beliefs.best_u_v(),
            "feedback":        self.bdi.beliefs.last_feedback,
            "dim_efficiency":  dim_efficiency,
            "dim_specificity": dim_specificity,
            "dim_quality":     dim_quality,
            "dim_safety":      msg.content.get("dim_safety",    0.0),
            "dim_relevance":   msg.content.get("dim_relevance", 0.0),
            "action_history":  action_history,
            "last_error":      last_error,
            "violations":      msg.content.get("violations", []),
            "is_retrieval":    msg.content.get("is_retrieval"),
        }
        if not self._best_content or u_v > self._best_content.get("U_V", -1.0):
            self._best_content = report

        # Per-attempt structured trace (opt-in — see config.TRACE_ENABLED).
        # Written here rather than by the verifier: this is the only place
        # that has both the verifier's payoff detail (forwarded in
        # msg.content) AND the planner's own desire/plan state.
        if TRACE_ENABLED:
            self._write_trace({
                **report,
                "active_desires":       sorted(d.name for d in self.bdi.options()),
                "consecutive_fallback": self.bdi.beliefs.consecutive_fallback,
                "consecutive_conflict": self.bdi.beliefs.consecutive_conflict,
                "verifier_weights":     msg.content.get("verifier_weights", {}),
                "planner_weights":      msg.content.get("planner_weights", {}),
            })

        if quality in ("success", "partial"):
            # RuleVerifier already published RESULT_ACCEPTED.
            return

        # 2. Reconsider — under SINGLE_MINDED, the intention is only
        # dropped if the plan clearly failed (BDIState.should_reconsider()).
        # If the commitment is kept, ADVANCE one step in the plan's body
        # so _bdi_cycle executes the recipe's next action.
        reconsider = self.bdi.should_reconsider()
        if reconsider:
            self.bdi.drop_intention()
            print(f"  [BDI] Intention dropped "
                  f"(strategy={self.bdi.commitment_strategy.value})")
        else:
            intention = self.bdi.current_intention()
            if intention:
                intention.step_index += 1

        # 3. Replan or give up — decision fully INTERNAL to the agent.
        if self.bdi.should_replan(self.max_replans):
            print(f"  [BDI] Replan (attempt {self.bdi.beliefs.attempt_count + 1}/{self.max_replans})")
            self._bdi_cycle(reconsider=reconsider)
        else:
            print(f"  [BDI] Budget exhausted ({self.max_replans}). "
                  f"Reporting best attempt (U_V={self._best_content.get('U_V', 0.0):.3f}).")
            content = dict(self._best_content)
            content["reason"] = f"BDI: {self.max_replans} replans without success"
            self.bus.publish(Message(
                sender   = "planner",
                receiver = "system",
                msg_type = MessageType.ERROR_REPORTED,
                content  = content,
            ))

    def _write_trace(self, entry: dict) -> None:
        """
        Appends one JSON line per internal attempt to
        <memory_dir>/traces/<task_id>.jsonl (config.TRACE_ENABLED).

        This is the piece needed for any per-attempt game-theoretic
        analysis (regret curve, conflict resolution rate, convergence
        speed) — the CSV in mas_orchestrator.py only ever holds one row
        per TASK (whichever attempt was ultimately reported), not one
        per internal replan.

        Best-effort: a disk error here must never stall the planner
        thread, which is the sole consumer of the bus's "planner" queue.
        """
        try:
            trace_dir = os.path.join(self._memory_dir, "traces")
            os.makedirs(trace_dir, exist_ok=True)
            path = os.path.join(trace_dir, f"{entry['task_id']}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [BDI] Trace write failed (non-fatal): {e}")

    # ── BDI cycle ─────────────────────────────────────────

    def _bdi_cycle(self, reconsider: bool = True) -> None:
        """
        One turn: options() → (continuation OR filter()) → execute().

        Real commitment: if the agent holds its intention
        (reconsider=False), it WALKS the next step of the committed
        plan's BODY instead of re-selecting a plan. It only
        re-deliberates (select_plan) once the plan is exhausted, no
        longer applicable, or reconsideration was decided.
        """
        b = self.bdi.beliefs
        active = self.bdi.options()
        print(f"  [BDI] Active desires: {[d.name for d in active]}")
        ctx = self._compute_context()

        # 1. Continuation of the committed plan (means-end: follow the recipe).
        if not reconsider:
            intention = self.bdi.current_intention()
            if intention:
                plan = self.plan_library.get_plan(intention.plan_name)
                if plan and (plan.priority == 0 or plan.context_condition(b, **ctx)):
                    step = self._emit_next_step(plan, intention, b, ctx)
                    if step is not None:
                        print(f"  [BDI] Commitment held — plan '{plan.name}' "
                              f"step {intention.step_index + 1}/{len(plan.body)}")
                        self._dispatch(plan, intention, step, b)
                        return
                # plan exhausted / out of context -> drop it and re-deliberate
                self.bdi.drop_intention()

        # 2. Deliberation: select a NEW plan (filter), commit, then
        #    execute its first applicable step.
        plan = self.plan_library.select_plan(b, **ctx)
        self.bdi.commit(plan_name=plan.name)
        intention = self.bdi.current_intention()
        step = self._emit_next_step(plan, intention, b, ctx)
        if step is None:
            # Safety net: should not happen (fallback always has a step).
            step = Step("fallback", f"Attempt {b.attempt_count + 1} — try a different approach.")
        print(f"  [BDI] Deliberate → plan '{plan.name}' "
              f"(priority={plan.priority}, desire={intention.source_desire})")
        self._dispatch(plan, intention, step, b)

    def _emit_next_step(self, plan, intention, b, ctx):
        """
        Advances through the plan's BODY from step_index, skipping
        non-applicable actions (build() -> None) or ones whose hint was
        already tried, and returns the first executable step (without
        permanently advancing the index — that happens after
        verification, in _on_verification_result).
        """
        while intention.step_index < len(plan.body):
            action = plan.body[intention.step_index]
            step = action.build(b, **ctx)
            if step is None or (step.hint and step.hint in b.tried_hints):
                intention.step_index += 1
                continue
            return step
        return None

    def _dispatch(self, plan, intention, step, b) -> None:
        """execute() — publishes the EXECUTION_REQUESTED for this step."""
        hint = step.hint
        intention.hint = hint
        print(f"  [BDI] Action: {step.action}")
        print(f"  [BDI] Hint: {hint[:90] or '(none — free generation)'}")
        self.bus.publish(Message(
            sender   = "planner",
            receiver = "executor",
            msg_type = MessageType.EXECUTION_REQUESTED,
            content  = {
                "task_id":   b.task_id,
                "task_goal": b.task_goal,
                "hint":      hint,
                "plan_name": plan.name,
                "attempt":   b.attempt_count + 1,
                "max_steps": self._max_steps,
                "headless":  self._headless,
            }
        ))

    # ── Context computation (fed to the Plans) ────────────

    def _compute_context(self) -> dict:
        """
        Pre-computes every variable the Actions need to evaluate their
        build(), centralized here once per cycle rather than duplicated
        in each Action.
        """
        b  = self.bdi.beliefs
        ah = b.last_action_history
        n  = len(ah)

        # options() = active desires (neither suspended nor achieved) —
        # passed to select_plan() so Desires actually gate plan choice,
        # not just get logged above. New desires from the enrichment
        # flags are added ad-hoc here (never mutated into bdi.desires):
        # whether they ACTUALLY fire this cycle is still gated by their
        # Action.build() condition (has_executable_step()) — membership
        # here only says the desire/plan pair exists at all, exactly like
        # the 3 static desires already work.
        active_desires = {d.name for d in self.bdi.options()}
        if CONFLICT_DETECTION_ENABLED:
            active_desires.add("resolve_conflict")
        category_is_high_risk = False
        safest_hint = ""
        if DYNAMIC_DESIRES_ENABLED:
            active_desires.update({
                "escape_plateau", "final_attempt",
                "diagnose_unknown_failure", "category_high_risk",
            })
            category_is_high_risk = (
                self.world.category_avg_u_v(b.task_type) < 0.5
                and self.world.category_samples(b.task_type) >= 10
            )
            safest_hint = self._get_safest_hint()

        return {
            "action_history":          ah,
            "last_error":               b.last_error,
            "n_steps":                  n,
            "max_steps":                self._max_steps,
            "max_replans":              self.max_replans,
            "prescan_hint":             self._html_prescan(),
            "warm_hint":                self._get_warm_hint(),
            "safest_hint":              safest_hint,
            "category_high_risk":       category_is_high_risk,
            "executor_reliability_fn":  self.world.executor_reliability,
            "active_desires":           active_desires,
        }

    # ── HTML pre-scan (classic, no LLM) ───────────────────

    def _html_prescan(self) -> str:
        """
        Fetches the page HTML (no JS, no LLM) and looks for keywords
        extracted from the goal. Returns a hint with a direct URL if
        found, "" otherwise.
        """
        b        = self.bdi.beliefs
        base_url = self._TASK_BASE_URLS.get(b.task_type, "")
        if not base_url:
            return ""

        keywords = _re_prescan.findall(r'"([^"]{3,})"', b.task_goal)
        if not keywords:
            keywords = _re_prescan.findall(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', b.task_goal
            )
        keywords = [k.strip() for k in keywords if len(k.strip()) > 3]
        if not keywords:
            return ""

        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page    = browser.new_page()
                page.goto(base_url, timeout=30000)
                page.wait_for_load_state("networkidle")
                html = page.content()
                browser.close()
        except Exception as e:
            print(f"  [BDI-Prescan] Playwright error: {e}")
            return ""

        for keyword in keywords:
            if keyword.lower() not in html.lower():
                continue

            pos    = html.lower().find(keyword.lower())
            window = html[max(0, pos - 300): pos + 300]
            href_m = _re_prescan.search(r'href=["\']([^"\'#][^"\']*)["\']', window)

            if href_m:
                href = href_m.group(1)
                if not href.startswith("http"):
                    href = base_url.rstrip("/") + "/" + href.lstrip("/")
                print(f"  [BDI-Prescan] '{keyword}' → {href}")
                return (
                    f"HTML pre-scan: '{keyword}' is present on the page "
                    f"with a direct link at {href}. "
                    f"Find and click the link corresponding to '{keyword}' "
                    f"in the AXTree. If not immediately visible, scroll down "
                    f"until it appears, then click it."
                )
            else:
                print(f"  [BDI-Prescan] '{keyword}' in HTML but no direct link.")
                return (
                    f"HTML pre-scan: '{keyword}' exists in the page content "
                    f"but is NOT visible in the AXTree at initial load. "
                    f"It may be hidden below the fold. "
                    f"Scroll down (scroll(0, 400)) several times until "
                    f"'{keyword}' appears in the AXTree, then interact with it."
                )

        print(f"  [BDI-Prescan] Keywords {keywords} not found in HTML.")
        return ""

    # ── Warm-start (classic, no LLM) ──────────────────────

    def _get_warm_hint(self) -> str:
        """
        Queries the strategy memory for a hint that led to success on
        this task type (classic, no LLM).

        A learned hint is reused only if it is:
          - generic (no reference to another task instance)
          - not self-referential ("previous attempt"...)
          - not already tried
        """
        b = self.bdi.beliefs
        if b.attempt_count != 0:
            return ""

        learned = (
            self.learner.get_best_hint_ucb1(b.task_type, exclude=b.tried_hints)
            if UCB1_ENABLED else
            self.learner.get_best_hint(b.task_type, exclude=b.tried_hints, require_success=True)
        )
        if (learned
                and self._hint_is_applicable(learned, b.task_goal)
                and self._is_reusable_opener(learned)):
            print(f"  [BDI] Warm-start available (task_type={b.task_type})")
            return learned
        return ""

    def _get_safest_hint(self) -> str:
        """
        Best known hint for this category, regardless of attempt number —
        used by the final_attempt dynamic desire. Unlike _get_warm_hint(),
        not restricted to attempt_count==0 or to "opener"-safe hints
        (self-referential phrasing is fine mid-task).
        """
        b = self.bdi.beliefs
        learned = (
            self.learner.get_best_hint_ucb1(b.task_type, exclude=b.tried_hints)
            if UCB1_ENABLED else
            self.learner.get_best_hint(b.task_type, exclude=b.tried_hints, require_success=True)
        )
        if learned and self._hint_is_applicable(learned, b.task_goal):
            return learned
        return ""

    # ── Applicability filters for learned hints ───────────

    def _hint_is_applicable(self, hint: str, goal: str) -> bool:
        """
        A learned hint applies to the current task only if it does not
        reference entities specific to ANOTHER task instance.
        """
        goal_l   = (goal or "").lower()
        entities = _re_prescan.findall(r'"([^"]{3,})"', hint)
        entities += _re_prescan.findall(r"'([^']{3,})'", hint)
        entities += _re_prescan.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', hint
        )
        for e in entities:
            e = e.strip().lower()
            if len(e) > 3 and e not in goal_l:
                return False
        return True

    def _is_reusable_opener(self, hint: str) -> bool:
        """
        True if the hint can serve as an opener (before any attempt).
        Corrective hints that mention "previous attempt" make no sense
        before the 1st attempt.
        """
        h = hint.lower()
        self_referential = (
            "previous attempt", "previous answer", "previous navigation",
            "you sent", "you found", "you are stuck", "was incorrect",
            "was incomplete", "was inefficient", "ran out of steps",
            "navigation actions:", "navigation performed:", "attempt",
            "the answer is incorrect", "different approach",
        )
        return not any(s in h for s in self_referential)
