# ─────────────────────────────────────────────────────────
# bdi_planner_agent_frering.py — PLANNER BDI, ablation Frering BDI
#
# Réutilise BDIPlannerAgent EN ENTIER (corps d'actions, désirs dynamiques,
# commitment, budget interne — rien ne change dans le raisonnement lui-même).
# Seul _on_verification_result est réécrit : la CROYANCE mise à jour via
# brf() est dégradée (binaire, dims à zéro), reproduisant la logique de
# l'ablation v1 — mais le RAPPORT final (_best_content, utilisé pour le
# score sauvegardé) reste basé sur les U_V/U_PE/dims RÉELS transmis par
# RuleBasedVerifierAgentFrering. C'est ce découplage qui permet de mesurer
# honnêtement "quelle est la vraie qualité obtenue quand le BDI raisonne à
# l'aveugle", au lieu de biaiser le score par la dégradation elle-même.
# ─────────────────────────────────────────────────────────

from __future__ import annotations

from core.message_bus         import Message, MessageType
from agents.bdi_planner_agent import BDIPlannerAgent


class BDIPlannerAgentFrering(BDIPlannerAgent):
    """Ablation : brf() reçoit une croyance binarisée, le rapport final reste réel."""

    def _on_verification_result(self, msg: Message) -> None:
        # Valeurs RÉELLES — pour le rapport final (_best_content) uniquement.
        u_pe              = msg.content.get("U_PE",              0.0)
        u_v               = msg.content.get("U_V",               0.0)
        quality           = msg.content.get("quality",           "rejected")
        is_valid_json     = msg.content.get("is_valid_json",     False)
        schema_compliance = msg.content.get("dim_schema_compliance", 0.0)
        field_accuracy    = msg.content.get("dim_field_accuracy", 0.0)
        depth_match       = msg.content.get("dim_depth_match",   False)
        actual_depth      = msg.content.get("actual_depth",      0)
        generated_json    = msg.content.get("generated_json",    "")
        last_error        = msg.content.get("last_error",        "")
        failed_fields     = msg.content.get("failed_fields",     [])
        hint_used         = msg.content.get("hint_used",         "")

        # Croyance DÉGRADÉE — c'est CECI qui alimente brf() (ablation).
        # reward/u_v binarisés (fournis par RuleBasedVerifierAgentFrering) ;
        # dims/failed_fields mis à zéro ici même (aucun diagnostic détaillé
        # transmis au raisonnement, quel que soit le vérificateur utilisé) ;
        # `quality`/`is_valid_json` restent réels — la classification
        # succès/échec doit rester honnête, seul le diagnostic est retiré.
        belief_reward   = msg.content.get("belief_reward",   0.0)
        belief_u_v      = msg.content.get("belief_u_v",       0.0)
        belief_feedback = msg.content.get("belief_feedback",  "")
        schema_violations = msg.content.get("schema_violations", [])

        plan_used = self.bdi.current_intention().plan_name \
            if self.bdi.current_intention() else "unknown"

        # 1. BRF — avec la croyance DÉGRADÉE, pas les valeurs réelles.
        # actual_depth=0 (comme v1) : aucune indication de profondeur transmise.
        self.bdi.brf(belief_reward, 0.0, belief_u_v, quality, belief_feedback, hint_used,
                     is_valid_json, 0.0, 0.0,
                     False, 0, generated_json, last_error,
                     [], plan_used)

        self.world.record_task_outcome(self.bdi.beliefs.category, belief_u_v)
        self.world.record_plan_outcome(self.bdi.beliefs.category, plan_used,
                                        success=(quality == "success"))

        # Trace d'audit (V2.1, #1) — sur cette ablation, on trace les valeurs
        # RÉELLES (u_pe/u_v/dims), pas la croyance dégradée binaire : la trace
        # sert à ANALYSER l'ablation de l'extérieur, elle ne doit donc pas
        # elle-même être aveuglée.
        plan_obj = self.plan_library.get_plan(plan_used)
        trigger_desire = plan_obj.trigger_desire if plan_obj else ""
        self.tracer.record_attempt(self.bdi.beliefs.task_id, {
            "attempt_index":        self.bdi.beliefs.attempt_count,
            "plan_name":            plan_used,
            "trigger_desire":       trigger_desire,
            "preempting":           trigger_desire in type(self.bdi)._PREEMPTING_DESIRE_NAMES,
            "temperature":          self._last_dispatch_temperature,
            "U_PE":                 u_pe,
            "U_V":                  u_v,
            "dim_schema_compliance": schema_compliance,
            "dim_field_accuracy":   field_accuracy,
            "dim_depth_match":      depth_match,
            "quality":              quality,
            "is_valid_json":        is_valid_json,
            "actual_depth":         actual_depth,
            "true_depth":           self.bdi.beliefs.true_depth,
            "failed_fields_count":  len(failed_fields),
            "schema_violations":    schema_violations,
            "consecutive_fallback": self.bdi.beliefs.consecutive_fallback,
            "consecutive_conflict": self.bdi.beliefs.consecutive_conflict,
            "belief_u_v_ablation":  belief_u_v,
            "world_model_reads": {
                "category_n":         self.world.category_samples(self.bdi.beliefs.category),
                "category_avg_u_v":   self.world.category_avg_u_v(self.bdi.beliefs.category),
                "plan_reliability":   self.world.executor_reliability(
                    self.bdi.beliefs.category, plan_used),
            },
        })

        print(f"\n  [BDI-Frering] Beliefs updated (dégradées): quality={quality}, "
              f"belief_U_V={belief_u_v:.3f} (réel U_V={u_v:.3f}), "
              f"trend={self.bdi.beliefs.trend()}")
        print(f"  [BDI-Frering] {self.bdi.summary()}")

        # Garde anti-régression — sur le U_V RÉEL (fidèle à v1 : le rapport
        # final n'est jamais dégradé, seul le raisonnement interne l'est).
        report = {
            "task_id":               self.bdi.beliefs.task_id,
            "attempts":              self.bdi.beliefs.attempt_count,
            "U_PE":                  u_pe,
            "U_V":                   u_v,
            "reward":                belief_reward,
            "quality":               quality,
            "hint_used":             hint_used,
            "plan_name":             plan_used,
            "generated_json":        generated_json,
            "is_valid_json":         is_valid_json,
            "dim_json_valid":        1.0 if is_valid_json else 0.0,
            "dim_schema_compliance": schema_compliance,
            "dim_field_accuracy":    field_accuracy,
            "dim_depth_match":       depth_match,
            "actual_depth":          actual_depth,
            "feedback":              belief_feedback,
            "last_error":            last_error,
            "failed_fields":         failed_fields,
            "schema_violations":     schema_violations,
        }
        if not self._best_content or u_v > self._best_content.get("U_V", -1.0):
            self._best_content = report

        if quality == "success":
            return

        reconsider = self.bdi.should_reconsider()
        if reconsider:
            self.bdi.drop_intention()
        else:
            intention = self.bdi.current_intention()
            if intention:
                intention.step_index += 1

        if self.bdi.should_replan(self.max_replans):
            print(f"  [BDI-Frering] Replan (attempt {self.bdi.beliefs.attempt_count + 1}/{self.max_replans})")
            self._bdi_cycle(reconsider=reconsider)
        else:
            print(f"  [BDI-Frering] Budget exhausted ({self.max_replans}). "
                  f"Reporting best attempt (réel U_V={self._best_content.get('U_V', 0.0):.3f}).")
            content = dict(self._best_content)
            content["reason"] = f"BDI: {self.max_replans} replans without success"
            self.bus.publish(Message(
                sender   = "planner",
                receiver = "system",
                msg_type = MessageType.ERROR_REPORTED,
                content  = content,
            ))
