# ─────────────────────────────────────────────────────────
# rule_based_verifier_frering.py — VÉRIFICATEUR BINARISÉ (ablation Frering BDI)
#
# Même principe que l'ablation Frering BDI de v1 : le BDI ne reçoit qu'un
# signal binaire (JSON valide ou non, succès ou rejet) au lieu des métriques
# détaillées (schema_compliance, field_accuracy, depth_match, failed_fields).
# Isole la contribution du feedback structuré du RuleBasedVerifier — mais
# construit ici sur les VRAIS agents v2 (corps d'actions, désirs dynamiques,
# fix de contexte), pas un script standalone séparé.
#
# Découplage clé (fidèle à v1) : le SCORE RAPPORTÉ (U_V/U_PE/dims dans le
# message, utilisé pour RESULT_ACCEPTED et le rapport final) reste RÉEL —
# seule la CROYANCE transmise au raisonnement BDI (champs `belief_*`, lus
# par BDIPlannerAgentFrering au lieu des champs réels) est dégradée. `quality`
# et `is_valid_json` restent réels : la classification succès/échec doit
# rester une mesure honnête, seul le diagnostic détaillé est retiré.
# ─────────────────────────────────────────────────────────

from __future__ import annotations

from core.message_bus      import Message, MessageType
from core.payoff_calculator import (
    compute_payoffs, get_schema_violations, V_THRESHOLD, V_SUCCESS_THRESHOLD,
)
from agents.rule_based_verifier import RuleBasedVerifierAgent


class RuleBasedVerifierAgentFrering(RuleBasedVerifierAgent):
    """Ablation : calcule les payoffs réels (rapport fidèle) mais transmet au
    planner une croyance binarisée (`belief_*`) au lieu des dimensions détaillées."""

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

        print(f"\n  [FreringVerifier] Evaluating {task_id} "
              f"(attempt={attempt}) [ablation — binary belief]")

        payoffs = compute_payoffs(generated_json, schema, ground_truth_json, true_depth)
        u_pe, u_v = payoffs["U_PE"], payoffs["U_V"]
        u_pe_score, u_v_score = u_pe["payoff"], u_v["payoff"]

        is_valid     = u_pe["dim_json_valid"] >= 1.0
        schema_comp  = u_pe["dim_schema_compliance"]
        field_acc    = u_v["dim_field_accuracy"]
        depth_score  = u_v["dim_depth_match"]
        actual_depth = u_v["actual_depth"]
        depth_match  = abs(actual_depth - true_depth) == 0

        avg_past    = self.history.get_average_payoff()
        improvement = self._compare_to_history(u_v_score, avg_past)

        if not is_valid:
            quality = "rejected"
        elif u_v_score >= V_SUCCESS_THRESHOLD:
            quality = "success"
        elif u_v_score >= V_THRESHOLD:
            quality = "partial"
        else:
            quality = "rejected"

        accepted = quality == "success"
        feedback = self._build_feedback(u_pe, u_v, is_valid)
        verdict  = self._build_verdict(quality, u_pe_score, u_v_score, improvement, is_valid)
        reward   = field_acc if is_valid else 0.0
        failed_fields = u_v.get("failed_fields", [])
        schema_violations = get_schema_violations(u_pe.get("parsed_obj"), schema)

        print(f"  [FreringVerifier] {verdict}  (croyance transmise au BDI : binaire)")
        print(f"  [FreringVerifier] {u_pe['breakdown']}")
        print(f"  [FreringVerifier] {u_v['breakdown']}")

        self._save_to_history(task_id, hint_used, reward, attempt, u_pe, u_v)
        self._save_strategy(task_id, hint_used, u_pe, reward, attempt)

        # ── Croyance dégradée transmise au BDI (miroir exact de v1) ────────
        # reward : 3 paliers (1.0 succès / 0.2 valide-mais-rejeté / 0.0 invalide)
        # belief_u_v : STRICTEMENT binaire (pas de palier "partial")
        # dims/failed_fields : à zéro — le BDI ne "voit" aucun diagnostic détaillé
        belief_reward = 1.0 if accepted else (0.2 if is_valid else 0.0)
        belief_u_v    = 1.0 if accepted else 0.0
        belief_feedback = ("success" if accepted
                            else ("invalid JSON" if not is_valid else "rejected"))

        # ── Publication 1 : VERIFICATION_RESULT → BDIPlanner ──
        # U_V/U_PE/dims RÉELS conservés (rapport final fidèle) ; belief_*
        # dégradés en plus, lus par BDIPlannerAgentFrering à la place.
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
                "actual_depth":         actual_depth,
                "depth_match":          depth_match,
                "improvement":          improvement,
                "hint_used":            hint_used,
                "plan_name":            plan_name,
                "generated_json":       generated_json,
                "last_error":           last_error,
                "failed_fields":        failed_fields,
                "schema_violations":    schema_violations,
                # Croyance binarisée (ablation) :
                "belief_reward":        belief_reward,
                "belief_u_v":           belief_u_v,
                "belief_feedback":      belief_feedback,
            }
        ))

        # ── Publication 2 : RESULT_ACCEPTED → system (inchangé, valeurs réelles) ──
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
