# ─────────────────────────────────────────────────────────
# rule_based_verifier.py — CLASSIC VERIFIER AGENT
#
# Role in the hybrid architecture:
#   Deterministic judge — 100% algorithmic evaluation.
#   No LLM calls. Fast and reproducible decisions.
#
# Difference from VerifierAgent (LLM):
#   VerifierAgent  : uses compute_payoffs (regex heuristics)
#                    but the DECISION is plain Python code
#                    -> already classic in practice, but not explicit
#   RuleBasedVerifierAgent : same + publishes VERIFICATION_RESULT
#                    to the BDIPlanner for belief updates
#                    -> the FINAL DECISION belongs to the BDI
#
# Message flow:
#   EXECUTION_DONE (from LLMExecutor)
#     -> computes U_PE and U_V by rules
#     -> publishes VERIFICATION_RESULT -> BDIPlanner (always)
#     -> publishes RESULT_ACCEPTED     -> system     (if quality OK)
#
# Why separate the decision from the computation?
#   The Verifier computes payoffs (factual).
#   The BDI decides whether that is good enough (strategic).
#   This separation allows the BDI to have a complete view
#   (history, number of replans, trend) before deciding.
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import os
import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.message_bus     import MessageBus, Message, MessageType
from core.history_manager  import HistoryManager
from core.payoff_calculator import (
    compute_payoffs, is_retrieval_task, WEIGHTS_V, WEIGHTS_PE,
    V_THRESHOLD, V_SUCCESS_THRESHOLD,
    PE_THRESHOLD, PE_SUCCESS_THRESHOLD, REL_THRESHOLD,
)
from core.strategy_learner import StrategyLearner, KEEP_THRESHOLD
from core.verifier_learned_weights import (
    compute_learned_weights, compute_learned_pe_weights,
    calibrate_threshold, PE_FEATURES,
)
from core.verifier_calibration import precision_calibrated_threshold

# Target precision (P(correct | accepted)) the verifier's acceptance
# threshold is calibrated toward — see core/verifier_calibration.py and
# experiments/verifier_calibration_report.py (measured: static 0.4 gave
# only 65% precision; ~0.50 gives 80% at no recall cost).
VERIFIER_TARGET_PRECISION = 0.8
from core.interaction_metrics import InteractionMetrics
from core.self_consistency import majority_vote
from core.utils            import extract_task_type
from config                import (MAX_STEPS,
                                    LEARNED_VERIFIER_WEIGHTS,
                                    LEARNED_PE_WEIGHTS,
                                    VERIFIER_AGREEMENT_ENABLED,
                                    INTERACTION_METRICS_ENABLED, RESULTS_DIR)


class RuleBasedVerifierAgent(threading.Thread):
    """
    Classic Verifier agent — verification without LLM.

    Computes payoffs (U_PE, U_V) using deterministic heuristics:
      - dim_success     : reward >= 1.0? (binary)
      - dim_efficiency  : steps used vs max (step-wise function)
      - dim_quality     : regex on the answer (digits, length...)
      - dim_safety      : regex for leaks (API keys, emails, phone)
      - dim_specificity : precision of the value in the answer

    Quality decision by configured thresholds:
      U_V >= 0.7 -> SUCCESS   (stop shots)
      U_V >= 0.4 -> PARTIAL   (acceptable)
      U_V <  0.4 -> REJECTED  (replan)

    Always publishes VERIFICATION_RESULT to the BDI Planner
    so that it can update its beliefs.
    """

    def __init__(self, bus: MessageBus,
                 history_manager: HistoryManager,
                 stop_event: threading.Event,
                 memory_dir: str = None):
        super().__init__(daemon=True, name="RuleVerifierThread")
        self.bus              = bus
        self.hm               = history_manager
        self.history          = history_manager.get_history("verifier")
        self.stop_event       = stop_event
        # Isolated-memory override (None = today's hardcoded RESULTS_DIR,
        # shared across runs — see config.MEMORY_DIR).
        self._memory_dir      = memory_dir or RESULTS_DIR
        # Logistic-regression-fitted weights + calibrated thresholds from
        # historical traces (opt-in — see core/verifier_learned_weights.py).
        # Scoped by RETRIEVAL vs ACTION (is_retrieval_task), not by site
        # family or globally: task_goal isn't known until a VERIFICATION
        # message arrives, so the fit happens lazily in _verify() via
        # _learned_verifier_config()/_learned_pe_config(), memoized per
        # retrieval bool below. A single cross-everything fit let one
        # task's failure profile push weights around for a completely
        # unrelated one; site family turned out not to be the right axis
        # either (a factual-question task and a booking task on the same
        # site have nothing in common verifier-wise) — see AVANCEMENT.md.
        #
        # U_V (LEARNED_VERIFIER_WEIGHTS): reweighting alone rarely crosses
        # the fixed static threshold, so calibrate_threshold is the
        # complementary lever that actually moves accept/reject decisions.
        self._trace_dir           = os.path.join(self._memory_dir, "traces")
        self._weights_cache       = {}   # is_retrieval -> (learned_weights, verifier_threshold)
        self._pe_weights_cache    = {}   # is_retrieval -> learned_pe_weights
        self._verifier_threshold  = V_THRESHOLD  # default when LEARNED_VERIFIER_WEIGHTS is off
        # U_PE (LEARNED_PE_WEIGHTS): same method, applied to the
        # PlannerExecuter's utility. Its calibrated threshold feeds
        # StrategyLearner's KEEP_THRESHOLD (EXPLOITATION vs EXPLORATION —
        # see core/strategy_learner.py). NOTE: get_strategy(), the only
        # method that branches on that threshold, is not called anywhere
        # in the live agent pipeline today (the BDI/Executor use
        # get_best_hint[_ucb1] instead) — so this currently only changes
        # the KEEP/CHANGE label in the Learner's logs and the
        # "exploitables" count in get_learning_summary(), not live
        # behavior. Left globally-scoped (computed once, unlike U_V/U_PE
        # above) since it doesn't gate anything live either way.
        pe_keep_threshold = None
        if LEARNED_PE_WEIGHTS:
            global_pe_weights = compute_learned_pe_weights(self._trace_dir, WEIGHTS_PE)
            pe_keep_threshold = calibrate_threshold(
                self._trace_dir, global_pe_weights, KEEP_THRESHOLD,
                feature_keys=PE_FEATURES)
        self.strategy_learner = StrategyLearner(results_dir=self._memory_dir,
                                                keep_threshold=pe_keep_threshold)
        # Live inter-agent interaction tracking (opt-in — see
        # core/interaction_metrics.py). Purely observational: never
        # affects any decision, only what gets printed/persisted.
        self.interaction_metrics = None
        if INTERACTION_METRICS_ENABLED:
            self.interaction_metrics = InteractionMetrics(results_dir=self._memory_dir)
        # Cross-attempt answer agreement (opt-in — config.VERIFIER_AGREEMENT_ENABLED).
        # Accumulates the answers this verifier has seen per task (across the
        # retry loop's attempts) and derives a self-consistency confidence:
        # convergent answers → high confidence, divergent → abstain. On the
        # benchmark (ground-truth reward decides) this is recorded but does
        # NOT change the accept decision; in a deployment WITHOUT reward it
        # is the trustworthiness signal — especially for RETRIEVAL tasks,
        # where U_V is not consulted for the decision at all.
        self._task_answers: dict = {}

    def _learned_verifier_config(self, retrieval: bool) -> tuple:
        """
        Lazily fits + memoizes (learned_weights, verifier_threshold) for
        U_V, scoped to `retrieval` (True=factual question, False=action
        task) alone. See __init__.
        """
        if retrieval not in self._weights_cache:
            weights = compute_learned_weights(self._trace_dir, WEIGHTS_V, retrieval=retrieval)
            # Threshold calibrated for a TARGET PRECISION (accepted→correct)
            # from historical (U_V, reward) traces — see
            # core/verifier_calibration.py. Replaces the old Youden's-J
            # objective, which on this data lowered the threshold and hurt
            # precision. Degenerates to the static V_THRESHOLD when there
            # isn't enough data of both classes yet.
            threshold = precision_calibrated_threshold(
                self._trace_dir, static_fallback=V_THRESHOLD,
                target_precision=VERIFIER_TARGET_PRECISION, retrieval=retrieval)
            self._weights_cache[retrieval] = (weights, threshold)
        return self._weights_cache[retrieval]

    def _learned_pe_config(self, retrieval: bool) -> dict:
        """Lazily fits + memoizes learned U_PE weights, scoped to `retrieval`."""
        if retrieval not in self._pe_weights_cache:
            self._pe_weights_cache[retrieval] = compute_learned_pe_weights(
                self._trace_dir, WEIGHTS_PE, retrieval=retrieval)
        return self._pe_weights_cache[retrieval]

    def _record_agreement(self, task_id: str, answer: str):
        """
        Accumulates `answer` for `task_id` across the attempts this verifier
        sees, and returns (agreement, abstain) from a self-consistency vote
        over them — (None, None) when the feature is off. Recorded/forwarded
        only; it never overrides the ground-truth accept decision here, but
        is the trustworthiness signal in a deployment without reward (and the
        ONLY one for retrieval tasks, where U_V is not consulted).
        """
        if not VERIFIER_AGREEMENT_ENABLED:
            return None, None
        self._task_answers.setdefault(task_id, []).append(answer)
        vote = majority_vote(self._task_answers[task_id])
        print(f"  [RuleVerifier] agreement={vote.agreement:.0%} over "
              f"{vote.n_votes}/{vote.n_total} answers → "
              f"{'ABSTAIN (low confidence)' if vote.abstain else 'confident'}")
        return vote.agreement, vote.abstain

    # ── Main loop ─────────────────────────────────────────

    def run(self) -> None:
        print("[RuleVerifierThread] Started — "
              "algorithmic verification (no LLM)...")
        while not self.stop_event.is_set():
            try:
                msg = self.bus.get("verifier", timeout=2)
            except queue.Empty:
                continue

            if msg.msg_type == MessageType.STOP:
                break
            elif msg.msg_type == MessageType.EXECUTION_DONE:
                self._verify(msg)

        print("[RuleVerifierThread] Stopped.")

    # ── Verification ──────────────────────────────────────

    def _verify(self, msg: Message) -> None:
        """
        Computes payoffs by rules and publishes the results.

        Publication order:
          1. VERIFICATION_RESULT -> BDIPlanner  (always, for brf)
          2. RESULT_ACCEPTED     -> system      (only if quality OK)

        The BDI planner receives VERIFICATION_RESULT and decides
        with full context (history, trend, replans).
        """
        task_id   = msg.content["task_id"]
        task_goal = msg.content.get("task_goal", "")
        reward    = msg.content.get("reward",    0.0)
        steps     = msg.content.get("steps",     MAX_STEPS)
        answer    = msg.content.get("answer",    "")
        hint_used      = msg.content.get("hint_used",      "")
        plan_name      = msg.content.get("plan_name",      "")
        attempt        = msg.content.get("attempt",        1)
        max_steps      = msg.content.get("max_steps",      MAX_STEPS)
        headless       = msg.content.get("headless",       True)
        action_history = msg.content.get("action_history", [])
        last_error     = msg.content.get("last_error",     "")

        print(f"\n  [RuleVerifier] Evaluating {task_id} (attempt={attempt}) "
              f"[CLASSIC — no LLM]")

        # ── Dynamic / learned verifier weights (opt-in) ───────────────
        # verifier_weights = None → compute_payoffs falls back to the
        # static WEIGHTS_V. When LEARNED_VERIFIER_WEIGHTS is on, the
        # reward-fitted weights (scoped by RETRIEVAL vs ACTION — the axis
        # that actually explains why the dimensions matter, since retrieval
        # tasks are graded on reward alone and action tasks on the U_V
        # threshold) are used directly, degenerating to the static prior
        # when there isn't enough data. (The old self-referential
        # DYNAMIC_VERIFIER_WEIGHTS heuristic — a dimension whose own scores
        # were low gained weight, without checking whether it predicted the
        # ground truth — was removed: superseded by these learned weights.)
        retrieval = is_retrieval_task(task_goal)
        verifier_weights   = None
        learned_pe_weights = None
        if LEARNED_VERIFIER_WEIGHTS:
            verifier_weights, self._verifier_threshold = self._learned_verifier_config(retrieval)
        if LEARNED_PE_WEIGHTS:
            learned_pe_weights = self._learned_pe_config(retrieval)

        # ── Payoff computation (deterministic) ────────────
        payoffs   = compute_payoffs(reward, steps, max_steps, answer, task_goal,
                                    verifier_weights=verifier_weights,
                                    planner_weights=learned_pe_weights)
        u_pe      = payoffs["U_PE"]
        u_v       = payoffs["U_V"]
        u_v_score = u_v["payoff"]
        u_pe_score = u_pe["payoff"]

        # Comparison with history (non-Markovian)
        avg_past    = self.history.get_average_payoff()
        improvement = self._compare_to_history(u_v_score, avg_past)

        # ── Recording ─────────────────────────────────────
        self._save_to_history(task_id, hint_used, reward, steps, u_pe, u_v)
        self._save_strategy(task_id, hint_used, u_pe, reward, steps, plan_name)

        # ── Quality classification (thresholds) ───────────
        is_crash  = (steps <= 1 and reward == 0.0)
        relevance = u_v["dim_relevance"]

        # Classification logic — correctness-aware:
        #
        #   "success"  : reward>=1 AND réponse non-vide
        #                → on fait confiance au ground truth du benchmark.
        #                La heuristique U_V ne peut pas rejeter un vrai succès.
        #
        #                Ancienne condition : reward>=1 AND U_V>=0.7
        #                Problème : U_V pénalise les listes de noms (relevance=0,
        #                specificity faible) alors que l'agent a bien répondu.
        #                Exemple concret observé :
        #                  answer="The Royal Dine, Vintage Bites..." reward=1.0
        #                  → U_V=0.640 < 0.7 → rejeté alors que c'est correct.
        #
        #   "partial"  : ONLY for ACTION/transaction tasks, where partial
        #                completion is meaningful. Requires U_V>=0.4 AND the
        #                answer is on-topic (relevance gate).
        #
        #   For RETRIEVAL tasks (factual questions), correctness is BINARY:
        #   it is determined by ground truth (reward), not by answer format.
        #   A well-formatted but wrong/unverified answer (reward=0) is REJECTED
        #   — it can no longer slip through as "partial" on format alone.
        #
        #   "rejected" : everything else, or a crash.
        if is_crash:
            quality = "rejected"
        elif reward >= 1.0 and answer.strip():
            # Ground truth confirms success — on fait confiance au benchmark.
            # U_V reste calculé pour le feedback BDI mais ne bloque pas.
            quality = "success"
        elif retrieval:
            # Factual question → no partial credit without ground truth.
            quality = "rejected"
        elif u_v_score >= self._verifier_threshold and relevance >= REL_THRESHOLD:
            # Action task with on-topic, non-trivial progress. Threshold
            # is calibrated from historical data when LEARNED_VERIFIER_
            # WEIGHTS is on (core/verifier_learned_weights.py), else the
            # static V_THRESHOLD.
            quality = "partial"
        else:
            quality = "rejected"

        accepted = quality in ("success", "partial")
        feedback = self._build_feedback(u_pe, u_v, retrieval, reward)
        verdict  = self._build_verdict(quality, u_pe_score, u_v_score,
                                       improvement, is_crash, reward)

        print(f"  [RuleVerifier] {verdict}")
        print(f"  [RuleVerifier] {u_pe['breakdown']}")
        print(f"  [RuleVerifier] {u_v['breakdown']}")

        if self.interaction_metrics is not None:
            self.interaction_metrics.record(u_pe_score, u_v_score, reward, quality)
            print(self.interaction_metrics.summary_line())

        # ── Cross-attempt agreement (opt-in, production trustworthiness) ──
        agreement, agreement_abstain = self._record_agreement(task_id, answer)

        # ── Publication 1: VERIFICATION_RESULT → BDIPlanner ──
        # Always published, even if accepted.
        # Allows the BDI to update its beliefs (brf).
        self.bus.publish(Message(
            sender   = "verifier",
            receiver = "planner",
            msg_type = MessageType.VERIFICATION_RESULT,
            content  = {
                "task_id":         task_id,
                "reward":          reward,
                "steps":           steps,
                "answer":          answer,
                "quality":         quality,
                "feedback":        feedback,
                "verdict":         verdict,
                "is_retrieval":    retrieval,
                "agreement":       agreement,
                "agreement_abstain": agreement_abstain,
                "U_PE":            u_pe_score,
                "U_V":             u_v_score,
                "dim_success":     u_pe["dim_success"],
                "dim_efficiency":  u_pe["dim_efficiency"],
                "dim_quality":     u_pe["dim_quality"],
                "dim_safety":      u_v["dim_safety"],
                "dim_specificity": u_v["dim_specificity"],
                "dim_relevance":   u_v["dim_relevance"],
                "violations":      u_v["violations"],
                "improvement":     improvement,
                "hint_used":       hint_used,
                "plan_name":       plan_name,
                "max_steps":       max_steps,
                "headless":        headless,
                "action_history":  action_history,
                "last_error":      last_error,
                "verifier_weights": verifier_weights or {},
                "planner_weights":  learned_pe_weights or {},
            }
        ))

        # ── Publication 2: RESULT_ACCEPTED → system (if OK) ──
        if accepted:
            self.bus.publish(Message(
                sender   = "verifier",
                receiver = "system",
                msg_type = MessageType.RESULT_ACCEPTED,
                content  = {
                    "task_id":         task_id,
                    "task_goal":       task_goal,
                    "reward":          reward,
                    "steps":           steps,
                    "answer":          answer,
                    "attempt":         attempt,
                    "hint_used":       hint_used,
                    "plan_name":       plan_name,
                    "U_PE":            u_pe_score,
                    "U_planner":       u_pe_score,
                    "U_executer":      u_pe_score,
                    "U_verifier":      u_v_score,
                    "U_V":             u_v_score,
                    "dim_success":     u_pe["dim_success"],
                    "dim_efficiency":  u_pe["dim_efficiency"],
                    "dim_safety":      u_v["dim_safety"],
                    "dim_specificity": u_v["dim_specificity"],
                    "dim_relevance":   u_v["dim_relevance"],
                    "violations":      u_v["violations"],
                    "improvement":     improvement,
                    "accepted":        True,
                    "quality":         quality,
                    "verdict":         verdict,
                    "feedback":        feedback,
                    "action_history":  action_history,
                    "last_error":      last_error,
                }
            ))

    # ── Helpers ───────────────────────────────────────────

    def _compare_to_history(self, u_v_score: float, avg_past: float) -> str:
        if avg_past <= 0:
            return "FIRST EXECUTION"
        if u_v_score > avg_past + 0.1:
            return "IMPROVEMENT vs history"
        if u_v_score < avg_past - 0.1:
            return "DEGRADATION vs history"
        return "STABLE vs history"

    def _build_feedback(self, u_pe: dict, u_v: dict,
                        retrieval: bool = True, reward: float = 0.0) -> str:
        """Explicit feedback to guide the BDI during replan."""
        # Failed factual question: the answer is wrong or was not verified
        # on the page. This is the most important corrective signal.
        if retrieval and reward < 1.0:
            return ("The answer is incorrect for this factual question. "
                    "Navigate to the exact element on the page, READ the "
                    "real value in the AXTree, and send that exact value — "
                    "do not rely on prior assumptions or injected context.")
        if u_v["violations"]:
            return f"Security violations: {u_v['violations']} — correct the answer."
        if u_v.get("dim_relevance", 1.0) < REL_THRESHOLD:
            return ("Answer is off-topic — it does not address what the goal "
                    "asks. Re-read the goal and answer the requested entity.")
        if u_pe["dim_success"] == 0.0:
            return "Task failed — try a completely different approach."
        if u_pe["dim_efficiency"] == 0.0:
            return "Too many steps — use a more direct approach."
        if u_v["dim_specificity"] < 0.3:
            return ("Answer too vague — produce a specific value "
                    "(number, exact price, name).")
        return (f"U_V={u_v['payoff']:.3f} < threshold {V_THRESHOLD} "
                f"— improve the quality of the answer.")

    def _build_verdict(self, quality: str, u_pe: float, u_v: float,
                       improvement: str, is_crash: bool,
                       reward: float = 0.0) -> str:
        crash_note   = " [CRASH]"    if is_crash              else ""
        reward_note  = " [reward=0]" if reward == 0.0 else ""
        label = {
            "success":  "COMPLETE SUCCESS",
            "partial":  "ACCEPTED PARTIAL",
            "rejected": "REJECTED",
        }[quality]
        # Displays both thresholds: partial (>=self._verifier_threshold,
        # calibrated when LEARNED_VERIFIER_WEIGHTS is on) and success
        # (>={V_SUCCESS_THRESHOLD})
        return (f"{label}{crash_note}{reward_note} | "
                f"U_PE={u_pe:.3f} | "
                f"U_V={u_v:.3f} "
                f"[partial>={self._verifier_threshold} | success>={V_SUCCESS_THRESHOLD}] | "
                f"{improvement}")

    def _save_to_history(self, task_id: str, hint_used: str, reward: float,
                         steps: int, u_pe: dict, u_v: dict) -> None:
        accepted = u_v["payoff"] >= V_THRESHOLD
        strategy = hint_used[:60] if hint_used else "no_hint"
        for role, (payoff, d_s, d_e, notes) in {
            "planner": (
                u_pe["payoff"],
                u_pe["dim_success"], u_pe["dim_efficiency"],
                "OK" if accepted else "Failed — change strategy",
            ),
            "verifier": (
                u_v["payoff"],
                u_pe["dim_success"], u_v["dim_safety"],
                f"{'Accepted' if accepted else 'Rejected'} "
                f"U_V={u_v['payoff']:.3f}",
            ),
        }.items():
            self.hm.record_run(
                agent_role=role, task_id=task_id, strategy=strategy,
                reward=reward, steps=steps, payoff=payoff,
                dim_success=d_s, dim_efficiency=d_e,
                dim_safety=u_v["dim_safety"], notes=notes,
            )

    def _save_strategy(self, task_id: str, hint_used: str,
                       u_pe: dict, reward: float, steps: int,
                       plan_name: str = "") -> None:
        task_type  = extract_task_type(task_id)
        run_number = (len(self.strategy_learner
                         .get_strategies_for_task(task_type)) + 1)
        # Record the hint as the "strategy" (free text)
        hint_as_steps = [hint_used] if hint_used else ["no_hint"]
        self.strategy_learner.record_run(
            task_type  = task_type,
            steps      = hint_as_steps,
            payoff     = u_pe["payoff"],
            reward     = reward,
            steps_used = steps,
            run_number = run_number,
            plan_name  = plan_name,
        )
