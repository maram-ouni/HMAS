# ─────────────────────────────────────────────────────────
# rule_based_verifier_dynamic.py — VÉRIFICATEUR À POIDS ADAPTATIFS
#
# Variante de RuleBasedVerifierAgent : au lieu de poids fixes
# (WEIGHTS_PE/WEIGHTS_V dans payoff_calculator.py), les poids de chaque
# dimension sont recalculés à partir de l'historique DE CE JOB
# uniquement — jamais cross-job (voir payoff_calculator_dynamic.py pour
# le détail du principe et des deux niveaux d'historique combinés).
#
# N'existe qu'en PARALLÈLE de RuleBasedVerifierAgent — celui-ci reste
# la référence gelée pour les architectures déjà benchmarquées.
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.message_bus import MessageBus, Message, MessageType
from core.history_manager import HistoryManager
from core.payoff_calculator_dynamic import compute_payoffs_dynamic, DynamicWeightTracker
from agents.rule_based_verifier import RuleBasedVerifierAgent


class RuleBasedVerifierAgentDynamic(RuleBasedVerifierAgent):
    """Vérificateur déterministe à poids adaptatifs (toujours sans LLM)."""

    def __init__(self, bus: MessageBus, history_manager: HistoryManager,
                 stop_event, strategies_path: str = None,
                 dynamic_weights_path: str = None,
                 trace_path: str = None,
                 interaction_metrics_path: str = None) -> None:
        # Accepte/transmet les nouveaux kwargs (mécanismes Verifier/décision
        # portés depuis agisdk) sans les utiliser : cette variante override
        # entièrement _compute_payoffs avec ses propres poids adaptatifs,
        # donc LEARNED_VERIFIER_WEIGHTS/LEARNED_PE_WEIGHTS n'ont pas de sens
        # ici. Juste pour que MASCoordinator puisse instancier ce verifier
        # sans TypeError quand ces kwargs sont passés.
        super().__init__(bus, history_manager, stop_event, strategies_path=strategies_path,
                          dynamic_weights_path=dynamic_weights_path,
                          trace_path=trace_path,
                          interaction_metrics_path=interaction_metrics_path)
        self.name = "RuleVerifierDynamicThread"
        # En mémoire, jamais persisté sur disque -> meurt avec le job,
        # donc job-scopé sans besoin d'un flag --isolated.
        self.weight_tracker = DynamicWeightTracker()
        # Ces poids dynamiques sont déjà les poids RÉELLEMENT utilisés
        # (weights_v_used) — un second calcul "diagnostique" en parallèle
        # (V2.1 #5) serait redondant, désactivé pour cette variante.
        self._compute_diagnostic_weights = False

    def _compute_payoffs(self, task_id: str, category: str, generated_json: str,
                         schema: str, ground_truth_json: str, true_depth: int) -> dict:
        return compute_payoffs_dynamic(generated_json, schema, ground_truth_json,
                                       true_depth, category=category, task_id=task_id,
                                       tracker=self.weight_tracker)
