# ─────────────────────────────────────────────────────────
# rule_based_verifier.py — VÉRIFICATEUR DÉTERMINISTE
#
# Spécifique à DeepJSONEval.
#
# Rôle dans l'architecture hybride :
#   Juge déterministe — 100% algorithmique, aucun appel LLM.
#
# Dimensions évaluées :
#   dim_json_valid       : le JSON est-il parsable ?
#   dim_schema_compliance: conforme à la structure du schéma ?
#   dim_field_accuracy   : F1 score vs ground truth
#   dim_depth_match      : profondeur d'imbrication correcte ?
#
# Décision par seuils :
#   U_V >= 0.7 -> SUCCESS  (arrêter les shots)
#   U_V >= 0.4 -> PARTIAL  (acceptable)
#   U_V <  0.4 -> REJECTED (replanifier)
#
# Publications :
#   1. VERIFICATION_RESULT → BDIPlanner (toujours, pour brf)
#   2. RESULT_ACCEPTED     → system     (si qualité OK)
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import os
import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.message_bus       import MessageBus, Message, MessageType
from core.history_manager    import HistoryManager
from core.payoff_calculator  import (
    compute_payoffs, dim_json_valid, get_schema_violations,
    V_THRESHOLD, V_SUCCESS_THRESHOLD,
    PE_SUCCESS_THRESHOLD, WEIGHTS_V, WEIGHTS_PE,
)
from core.payoff_calculator_dynamic import compute_payoffs_dynamic, DynamicWeightTracker
from core.strategy_learner   import StrategyLearner
from core.utils              import extract_task_type
# Mécanismes Verifier/décision portés depuis agisdk/mas_project_v2 (flags,
# off par défaut — voir config.py). `_compute_learned_weights` (préfixe `_`)
# est importé intentionnellement : le fichier source est gardé identique à
# agisdk pour que les diffs futurs restent propres ; les tuples de features
# et la clé pivot sont spécifiques à DeepJSONEval, passés ici à l'appel.
from core.verifier_learned_weights import _compute_learned_weights
from core.verifier_calibration     import precision_calibrated_threshold
from core.interaction_metrics      import InteractionMetrics
from core.self_consistency         import majority_vote
from config import (
    LEARNED_VERIFIER_WEIGHTS, LEARNED_PE_WEIGHTS,
    VERIFIER_AGREEMENT_ENABLED, INTERACTION_METRICS_ENABLED,
    RESULTS_DIR,
)


class RuleBasedVerifierAgent(threading.Thread):
    """
    Vérificateur classique pour DeepJSONEval — sans LLM.
    """

    # Features apprises (spécifiques DeepJSONEval) — clé pivot toujours
    # "json_valid" : condition nécessaire au reward>0 dans les deux formules
    # (poids 0.40 côté U_V, 0.60 côté U_PE), gardée à son a priori statique ;
    # le budget restant est redistribué entre les deux autres dimensions.
    _FEATURES_V  = ("field_accuracy", "depth_match")
    _PE_FEATURES = ("schema_compliance", "completeness")

    def __init__(self, bus: MessageBus,
                 history_manager: HistoryManager,
                 stop_event: threading.Event,
                 strategies_path: str = None,
                 dynamic_weights_path: str = None,
                 trace_path: str = None,
                 interaction_metrics_path: str = None):
        super().__init__(daemon=True, name="RuleVerifierThread")
        self.bus              = bus
        self.hm               = history_manager
        self.history          = history_manager.get_history("verifier")
        self.stop_event       = stop_event
        # `strategies_path` : même fichier isolé que celui du planner (voir
        # BDIPlannerAgent) — BUG corrigé le 2026-07-10 : ce vérificateur
        # écrivait auparavant TOUJOURS dans le fichier partagé cumulatif
        # (learned_strategies.json, ~2000+ entrées accumulées depuis le début
        # du projet) même en mode --isolated, alors que le planner LIT depuis
        # le fichier isolé pour le warm-start. Le fichier isolé restait donc
        # vide et le warm-start (greedy puis UCB1) ne pouvait JAMAIS trouver
        # de candidat en mode isolé — découvert via le test d'intégration du
        # job #891871 (0 warm-start observé sur 30 tâches isolées).
        self.strategy_learner = StrategyLearner(path=strategies_path)
        # Poids Hedge/AdaBoost DIAGNOSTIQUES (V2.1, #5) : calculés TOUJOURS en
        # parallèle du vérificateur statique, jamais utilisés pour le gating
        # (weights_v_used reste celui du calcul statique ci-dessous). Permet
        # de comparer statique vs adaptatif SUR LE MÊME run au lieu de devoir
        # choisir --dynamic-verifier en exclusif. Désactivé par la sous-classe
        # RuleBasedVerifierAgentDynamic (redondant : ses poids réels SONT déjà
        # les dynamiques).
        self._compute_diagnostic_weights = True
        self.diagnostic_weight_tracker   = DynamicWeightTracker(path=dynamic_weights_path)

        # Mécanismes Verifier/décision portés depuis agisdk (flags, off par
        # défaut). `_trace_dir` doit pointer vers le même répertoire que le
        # tracer du BDI Planner (core/audit_trace.py) pour lire les mêmes
        # tentatives historiques.
        self._trace_dir       = trace_path or os.path.join(RESULTS_DIR, "traces")
        self._verifier_weights_cache = None
        self._pe_weights_cache       = None
        self._verifier_threshold     = V_THRESHOLD
        self.interaction_metrics = (
            InteractionMetrics(results_dir=interaction_metrics_path)
            if INTERACTION_METRICS_ENABLED else None
        )
        self._task_answers: dict = {}

    # ── Boucle principale ─────────────────────────────────

    def run(self) -> None:
        print("[RuleVerifierThread] Started — "
              "algorithmic JSON verification (no LLM)...")
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

    # ── Vérification ──────────────────────────────────────

    def _verify(self, msg: Message) -> None:
        task_id           = msg.content["task_id"]
        schema            = msg.content.get("schema",            "")
        ground_truth_json = msg.content.get("ground_truth_json", "")
        generated_json    = msg.content.get("generated_json",    "")
        true_depth        = msg.content.get("true_depth",        0)
        hint_used         = msg.content.get("hint_used",         "")
        plan_name         = msg.content.get("plan_name",         "")
        attempt           = msg.content.get("attempt",           1)
        last_error        = msg.content.get("last_error",        "")

        category = msg.content.get("category", "")

        print(f"\n  [RuleVerifier] Evaluating {task_id} "
              f"(attempt={attempt}) [CLASSIC — no LLM]")

        # ── Calcul des payoffs (déterministe) ─────────────
        payoffs = self._compute_payoffs(task_id, category, generated_json, schema,
                                        ground_truth_json, true_depth)
        u_pe    = payoffs["U_PE"]
        u_v     = payoffs["U_V"]

        u_pe_score = u_pe["payoff"]
        u_v_score  = u_v["payoff"]

        is_valid       = u_pe["dim_json_valid"] >= 1.0
        schema_comp    = u_pe["dim_schema_compliance"]
        field_acc      = u_v["dim_field_accuracy"]
        depth_score    = u_v["dim_depth_match"]
        actual_depth   = u_v["actual_depth"]
        depth_match    = abs(actual_depth - true_depth) == 0

        # Comparaison avec l'historique
        avg_past    = self.history.get_average_payoff()
        improvement = self._compare_to_history(u_v_score, avg_past)

        # ── Classification de la qualité ──────────────────
        if not is_valid:
            quality = "rejected"
        elif u_v_score >= V_SUCCESS_THRESHOLD:
            quality = "success"
        elif u_v_score >= self._verifier_threshold:
            quality = "partial"
        else:
            quality = "rejected"

        accepted = quality == "success"  # U_V >= 0.7 only, aligned with baseline threshold
        feedback = self._build_feedback(u_pe, u_v, is_valid)
        verdict  = self._build_verdict(quality, u_pe_score, u_v_score,
                                       improvement, is_valid)

        # Calcul de la "récompense" synthétique
        reward = field_acc if is_valid else 0.0

        print(f"  [RuleVerifier] {verdict}")
        print(f"  [RuleVerifier] {u_pe['breakdown']}")
        print(f"  [RuleVerifier] {u_v['breakdown']}")

        # Enregistrement
        self._save_to_history(task_id, hint_used, reward, attempt, u_pe, u_v)
        self._save_strategy(task_id, hint_used, u_pe, reward, attempt)

        failed_fields = u_v.get("failed_fields", [])

        # Rapport structuré des violations de schéma — purement diagnostique,
        # n'affecte ni U_PE/U_V ni quality/accepted (V2.1, changement #4).
        schema_violations = get_schema_violations(u_pe.get("parsed_obj"), schema)

        # Métriques d'interaction PE/V et vote self-consistency — portés
        # depuis agisdk, purement diagnostiques/additifs (opt-in, off par
        # défaut). N'affectent ni U_PE/U_V ni quality/accepted.
        if self.interaction_metrics is not None:
            self.interaction_metrics.record(u_pe_score, u_v_score, reward, quality)
            print(self.interaction_metrics.summary_line())
        agreement, agreement_abstain = self._record_agreement(task_id, generated_json)

        # Poids Hedge/AdaBoost diagnostiques (V2.1, #5) — calculés en plus des
        # poids statiques réellement utilisés, jamais à leur place. None si ce
        # vérificateur EST déjà la variante dynamique (RuleBasedVerifierAgentDynamic).
        weights_v_diagnostic = None
        if self._compute_diagnostic_weights:
            try:
                diag = compute_payoffs_dynamic(
                    generated_json, schema, ground_truth_json, true_depth,
                    category=category, task_id=task_id,
                    tracker=self.diagnostic_weight_tracker,
                )
                weights_v_diagnostic = diag["U_V"].get("weights_used")
            except Exception as e:
                print(f"  [RuleVerifier] Diagnostic weights failed (non-blocking): {e}")

        # ── Publication 1 : VERIFICATION_RESULT → BDIPlanner ──
        self.bus.publish(Message(
            sender   = "verifier",
            receiver = "planner",
            msg_type = MessageType.VERIFICATION_RESULT,
            content  = {
                "task_id":              task_id,
                "reward":               reward,
                "quality":              quality,
                "feedback":             feedback,
                "verdict":              verdict,
                "U_PE":                 u_pe_score,
                "U_V":                  u_v_score,
                "is_valid_json":        is_valid,
                "dim_json_valid":       u_pe["dim_json_valid"],
                "dim_schema_compliance": schema_comp,
                "dim_field_accuracy":   field_acc,
                "dim_depth_match":      depth_score,
                "dim_completeness":     u_pe.get("dim_completeness"),
                "actual_depth":         actual_depth,
                "depth_match":          depth_match,
                "improvement":          improvement,
                "hint_used":            hint_used,
                "plan_name":            plan_name,
                "generated_json":       generated_json,
                "last_error":           last_error,
                "failed_fields":        failed_fields,
                "schema_violations":    schema_violations,
                "weights_v_used":       u_v.get("weights_used"),
                "weights_pe_used":      u_pe.get("weights_used"),
                "weights_v_diagnostic": weights_v_diagnostic,
                "agreement":            agreement,
                "agreement_abstain":    agreement_abstain,
            }
        ))

        # ── Publication 2 : RESULT_ACCEPTED → system ──────────
        if accepted:
            self.bus.publish(Message(
                sender   = "verifier",
                receiver = "system",
                msg_type = MessageType.RESULT_ACCEPTED,
                content  = {
                    "task_id":              task_id,
                    "reward":               reward,
                    "quality":              quality,
                    "feedback":             feedback,
                    "verdict":              verdict,
                    "attempt":              attempt,
                    "hint_used":            hint_used,
                    "plan_name":            plan_name,
                    "generated_json":       generated_json,
                    "U_PE":                 u_pe_score,
                    "U_V":                  u_v_score,
                    "is_valid_json":        is_valid,
                    "dim_json_valid":       u_pe["dim_json_valid"],
                    "dim_schema_compliance": schema_comp,
                    "dim_field_accuracy":   field_acc,
                    "dim_depth_match":      depth_score,
                    "actual_depth":         actual_depth,
                    "improvement":          improvement,
                    "accepted":             True,
                    "last_error":           last_error,
                    "schema_violations":    schema_violations,
                }
            ))

    # ── Mécanismes Verifier/décision portés depuis agisdk (opt-in) ────

    def _learned_verifier_config(self) -> tuple:
        """(poids appris pour U_V, seuil calibré) — calculés une fois par
        vie de l'agent (une tâche), repli statique si pas assez de traces
        historiques (MIN_EXAMPLES_PER_CLASS). Voir config.LEARNED_VERIFIER_WEIGHTS."""
        if self._verifier_weights_cache is None:
            weights = _compute_learned_weights(
                self._trace_dir, WEIGHTS_V, self._FEATURES_V,
                pinned_key="json_valid", retrieval=None)
            threshold = precision_calibrated_threshold(
                self._trace_dir, static_fallback=V_THRESHOLD,
                target_precision=0.8, retrieval=None)
            self._verifier_weights_cache = (weights, threshold)
        return self._verifier_weights_cache

    def _learned_pe_config(self) -> dict:
        """Poids appris pour U_PE. Voir config.LEARNED_PE_WEIGHTS."""
        if self._pe_weights_cache is None:
            self._pe_weights_cache = _compute_learned_weights(
                self._trace_dir, WEIGHTS_PE, self._PE_FEATURES,
                pinned_key="json_valid", retrieval=None)
        return self._pe_weights_cache

    def _record_agreement(self, task_id: str, answer: str):
        """Vote self-consistency sur les réponses successives d'UNE tâche.
        Voir config.VERIFIER_AGREEMENT_ENABLED. Off par défaut → (None, None)."""
        if not VERIFIER_AGREEMENT_ENABLED:
            return None, None
        self._task_answers.setdefault(task_id, []).append(answer)
        vote = majority_vote(self._task_answers[task_id])
        print(f"  [RuleVerifier] agreement={vote.agreement:.0%} over "
              f"{vote.n_votes}/{vote.n_total} answers -> "
              f"{'ABSTAIN (low confidence)' if vote.abstain else 'confident'}")
        return vote.agreement, vote.abstain

    # ── Point d'extension (variante dynamique) ─────────────

    def _compute_payoffs(self, task_id: str, category: str, generated_json: str,
                         schema: str, ground_truth_json: str, true_depth: int) -> dict:
        """Poids statiques (WEIGHTS_PE/WEIGHTS_V) par défaut — comportement de
        référence inchangé. Si LEARNED_VERIFIER_WEIGHTS/LEARNED_PE_WEIGHTS
        sont actifs, surcharge par des poids appris sur l'historique de
        traces (opt-in, repli statique si data insuffisante). Voir
        RuleBasedVerifierAgentDynamic pour la variante à poids adaptatifs
        (payoff_calculator_dynamic.py, mécanisme différent, job-scopé)."""
        verifier_weights = planner_weights = None
        if LEARNED_VERIFIER_WEIGHTS:
            verifier_weights, self._verifier_threshold = self._learned_verifier_config()
        if LEARNED_PE_WEIGHTS:
            planner_weights = self._learned_pe_config()
        return compute_payoffs(generated_json, schema, ground_truth_json, true_depth,
                               verifier_weights=verifier_weights,
                               planner_weights=planner_weights)

    # ── Helpers ───────────────────────────────────────────

    def _compare_to_history(self, u_v_score: float, avg_past: float) -> str:
        if avg_past <= 0:
            return "FIRST EXECUTION"
        if u_v_score > avg_past + 0.1:
            return "IMPROVEMENT vs history"
        if u_v_score < avg_past - 0.1:
            return "DEGRADATION vs history"
        return "STABLE vs history"

    def _build_feedback(self, u_pe: dict, u_v: dict, is_valid: bool) -> str:
        if not is_valid:
            return ("The output is not valid JSON. "
                    "Return ONLY a raw JSON object starting with '{'.")
        if u_pe["dim_schema_compliance"] < 0.5:
            return ("The JSON structure does not match the schema. "
                    "Follow the schema key names and nesting exactly.")
        if u_v["dim_field_accuracy"] < 0.4:
            return ("Field values are mostly incorrect. "
                    "Re-read the text and extract exact values.")
        if u_v["dim_depth_match"] < 0.75:
            return ("The nesting depth is wrong. "
                    "Adjust the nesting to match the expected depth.")
        return (f"U_V={u_v['payoff']:.3f} — "
                f"improve schema compliance and field accuracy.")

    def _build_verdict(self, quality: str, u_pe: float, u_v: float,
                       improvement: str, is_valid: bool) -> str:
        invalid_note = " [INVALID JSON]" if not is_valid else ""
        label = {
            "success":  "COMPLETE SUCCESS",
            "partial":  "ACCEPTED PARTIAL",
            "rejected": "REJECTED",
        }[quality]
        return (f"{label}{invalid_note} | "
                f"U_PE={u_pe:.3f} | "
                f"U_V={u_v:.3f} "
                f"[partial>={V_THRESHOLD} | success>={V_SUCCESS_THRESHOLD}] | "
                f"{improvement}")

    def _save_to_history(self, task_id: str, hint_used: str, reward: float,
                         steps: int, u_pe: dict, u_v: dict) -> None:
        accepted = u_v["payoff"] >= V_THRESHOLD
        strategy = hint_used[:60] if hint_used else "no_hint"
        for role, (payoff, d_s, d_e, notes) in {
            "planner": (
                u_pe["payoff"],
                u_pe["dim_json_valid"], u_pe["dim_schema_compliance"],
                "OK" if accepted else "Failed — change strategy",
            ),
            "verifier": (
                u_v["payoff"],
                u_pe["dim_json_valid"], u_v["dim_field_accuracy"],
                f"{'Accepted' if accepted else 'Rejected'} U_V={u_v['payoff']:.3f}",
            ),
        }.items():
            self.hm.record_run(
                agent_role=role, task_id=task_id, strategy=strategy,
                reward=reward, steps=steps, payoff=payoff,
                dim_success=d_s, dim_efficiency=d_e,
                dim_safety=u_v["dim_field_accuracy"], notes=notes,
            )

    def _save_strategy(self, task_id: str, hint_used: str,
                       u_pe: dict, reward: float, steps: int) -> None:
        task_type  = extract_task_type(task_id)
        run_number = len(self.strategy_learner.get_strategies_for_task(task_type)) + 1
        hint_as_steps = [hint_used] if hint_used else ["no_hint"]
        self.strategy_learner.record_run(
            task_type  = task_type,
            steps      = hint_as_steps,
            payoff     = u_pe["payoff"],
            reward     = reward,
            steps_used = steps,
            run_number = run_number,
        )
