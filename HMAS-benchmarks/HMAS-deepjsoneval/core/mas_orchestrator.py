# ─────────────────────────────────────────────────────────
# mas_orchestrator.py — COORDINATEUR PASSIF (architecture hybride)
#
# Adapté de agisdk/mas_project pour DeepJSONEval.
#
# 3 agents en threads OS :
#   BDIPlannerAgent       — planificateur BDI (classique)
#   LLMExecutorAgent      — génération JSON via Groq
#   RuleBasedVerifierAgent — vérificateur déterministe
#
# Flux de messages :
#   system   → planner  : TASK_ASSIGNED
#   planner  → executor : EXECUTION_REQUESTED  (hint BDI)
#   executor → verifier : EXECUTION_DONE       (JSON généré)
#   verifier → planner  : VERIFICATION_RESULT  (payoffs, qualité)
#   verifier → system   : RESULT_ACCEPTED      (si qualité OK)
#   planner  → system   : ERROR_REPORTED       (si budget épuisé)
# ─────────────────────────────────────────────────────────

import csv
import json
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.history_manager import HistoryManager
from core.message_bus     import MessageBus, Message, MessageType
from config               import MAX_REPLANS, RESULTS_FILE


class MASCoordinator:
    """
    Coordinateur léger pour l'architecture hybride DeepJSONEval.

    Le coordinateur est PASSIF :
      1. Crée le bus + les threads
      2. Injecte TASK_ASSIGNED
      3. Attend RESULT_ACCEPTED ou ERROR_REPORTED
      4. Arrête les threads et retourne le résultat
    """

    def __init__(self, world_model_path: str = None, strategies_path: str = None,
                 trace_path: str = None, planner_cls=None, verifier_cls=None,
                 run_id: str = None, reliability_mode: str = "frequentist",
                 use_diagnostic_band: bool = False,
                 dynamic_weights_path: str = None,
                 interaction_metrics_path: str = None):
        self.history_manager = HistoryManager()
        # Isolation d'un run comparatif : voir WorldModel/StrategyLearner.
        # None (défaut) = fichiers partagés cumulatifs, comme en usage normal.
        self.world_model_path = world_model_path
        self.strategies_path  = strategies_path
        # Répertoire de trace d'audit par tentative (V2.1, #1) — même
        # principe d'isolation : None = répertoire partagé results/traces/.
        self.trace_path       = trace_path
        # Identifiant du run courant — propagé au WorldModel pour son
        # snapshot de provenance (V2.1, #8). None = pas de snapshot écrit.
        self.run_id           = run_id
        # V2.1 #2 : "frequentist" (défaut) = comportement historique exact.
        self.reliability_mode = reliability_mode
        # V2.1 #3, opt-in (défaut False = comportement historique inchangé).
        self.use_diagnostic_band = use_diagnostic_band
        # V2.1 #5 : None (défaut) = poids diagnostiques en mémoire seulement,
        # jamais persistés — comme avant. Un path explicite les persiste.
        self.dynamic_weights_path = dynamic_weights_path
        # Mécanismes Verifier/décision portés depuis agisdk (V2 récent) —
        # None (défaut) = fichier partagé cumulatif, comme trace_path/
        # dynamic_weights_path ci-dessus.
        self.interaction_metrics_path = interaction_metrics_path
        # Classes d'agents injectables — None (défaut) = HMAS normal
        # (BDIPlannerAgent, RuleBasedVerifierAgent). Permet de réutiliser
        # TOUT le reste de l'architecture v2 (corps d'actions, désirs
        # dynamiques, fix de contexte) pour une ablation, en substituant
        # seulement le planner/verifier — ex. Frering BDI (verifier binarisé).
        self.planner_cls  = planner_cls
        self.verifier_cls = verifier_cls
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    def run_task(self, task: dict, prev_result: dict = None,
                 max_attempts: int = 1) -> dict:
        """
        Exécute une tâche DeepJSONEval avec l'architecture MAS.

        Le coordinateur crée les agents UNE fois, injecte UN seul
        TASK_ASSIGNED, puis attend le message terminal. Toute la délibération
        (jusqu'à `max_attempts` tentatives) se déroule DANS l'agent BDI, dont
        la mémoire persiste sur toute la durée de la tâche — pas de recréation
        d'agent ni de sérialisation d'état entre les tentatives.

        `prev_result` est conservé pour compatibilité mais n'est plus utilisé :
        la boucle de raisonnement est désormais interne à l'agent.

        task = {
          "id": str,
          "schema": str,
          "text": str,
          "ground_truth_json": str,
          "category": str,
          "true_depth": int,
        }
        """
        from agents.bdi_planner_agent    import BDIPlannerAgent
        from agents.llm_executor_agent   import LLMExecutorAgent
        from agents.rule_based_verifier  import RuleBasedVerifierAgent

        PlannerCls  = self.planner_cls  or BDIPlannerAgent
        VerifierCls = self.verifier_cls or RuleBasedVerifierAgent

        task_id  = task["id"]
        category = task.get("category", "unknown")
        depth    = task.get("true_depth", 0)

        print(f"\n{'='*60}")
        print(f"  HYBRID MAS | {task_id}")
        print(f"  Category  : {category} | Depth: {depth}")
        print(f"  Agents    : {PlannerCls.__name__} | LLMExecutor (Groq) | {VerifierCls.__name__}")
        print(f"{'='*60}")

        bus        = MessageBus()
        stop_event = threading.Event()

        bdi_planner   = PlannerCls(bus, self.history_manager, stop_event,
                                    world_model_path=self.world_model_path,
                                    strategies_path=self.strategies_path,
                                    trace_path=self.trace_path,
                                    run_id=self.run_id,
                                    reliability_mode=self.reliability_mode,
                                    use_diagnostic_band=self.use_diagnostic_band)
        llm_executor  = LLMExecutorAgent(bus, self.history_manager, stop_event)
        # Même fichier isolé que le planner (voir fix du 2026-07-10) — sans
        # ça, le vérificateur écrit ses stratégies dans le fichier partagé
        # cumulatif même en mode isolé, et le warm-start du planner (qui LIT
        # depuis le fichier isolé) ne trouve jamais rien.
        rule_verifier = VerifierCls(bus, self.history_manager, stop_event,
                                     strategies_path=self.strategies_path,
                                     dynamic_weights_path=self.dynamic_weights_path,
                                     trace_path=self.trace_path,
                                     interaction_metrics_path=self.interaction_metrics_path)

        bdi_planner.start()
        llm_executor.start()
        rule_verifier.start()

        t_start = time.perf_counter()

        bus.publish(Message(
            sender   = "system",
            receiver = "planner",
            msg_type = MessageType.TASK_ASSIGNED,
            content  = {
                "task_id":           task_id,
                "schema":            task.get("schema", ""),
                "text":              task.get("text", ""),
                "ground_truth_json": task.get("ground_truth_json", ""),
                "category":          category,
                "true_depth":        depth,
                "max_attempts":      max_attempts,
            }
        ))

        # La délibération interne peut enchaîner jusqu'à `max_attempts` appels
        # LLM ; on dimensionne le timeout en conséquence (60 s / tentative).
        timeout = max(120, 60 * max_attempts)
        result  = None

        try:
            msg            = bus.get("system", timeout=timeout)
            total_duration = time.perf_counter() - t_start

            if msg.msg_type == MessageType.RESULT_ACCEPTED:
                result = self._build_result(msg.content, task, total_duration)
                print(f"\n  [Coordinator] ACCEPTED in {total_duration:.1f}s")

            elif msg.msg_type == MessageType.ERROR_REPORTED:
                reason = msg.content.get("reason", "Unknown error")
                print(f"\n  [Coordinator] Error: {reason}")
                result = self._build_error_result(task, reason, total_duration,
                                                   content=msg.content)

        except queue.Empty:
            total_duration = time.perf_counter() - t_start
            print(f"\n  [Coordinator] Timeout after {total_duration:.1f}s")
            result = self._build_error_result(task, "Timeout", total_duration)

        stop_event.set()
        bus.publish(Message(sender="system", receiver="all",
                            msg_type=MessageType.STOP, content={}))

        bdi_planner.join(timeout=5)
        llm_executor.join(timeout=5)
        rule_verifier.join(timeout=5)

        print(f"\n{bus.get_conversation_log()}")
        self._save_result(result)
        return result

    # ── Construction des résultats ───────────────────────

    def _build_result(self, content: dict, task: dict, duration: float) -> dict:
        return {
            "task_id":              task["id"],
            "category":             task.get("category", ""),
            "true_depth":           task.get("true_depth", 0),
            "attempt":              content.get("attempt", 1),
            "strategy":             content.get("hint_used", ""),
            "hint_used":            content.get("hint_used", ""),
            "plan_name":            content.get("plan_name", ""),
            "generated_json":       content.get("generated_json", ""),
            "reward":               content.get("reward", 0.0),
            "steps":                content.get("attempt", 1),
            "success":              content.get("reward", 0.0) >= 0.7,
            "U_planner":            content.get("U_PE", 0.0),
            "U_PE":                 content.get("U_PE", 0.0),
            "U_verifier":           content.get("U_V", 0.0),
            "U_V":                  content.get("U_V", 0.0),
            "dim_json_valid":       content.get("dim_json_valid",       0.0),
            "dim_schema_compliance": content.get("dim_schema_compliance", 0.0),
            "dim_field_accuracy":   content.get("dim_field_accuracy",   0.0),
            "dim_depth_match":      content.get("dim_depth_match",      0.0),
            "actual_depth":         content.get("actual_depth",         0),
            "is_valid_json":        content.get("is_valid_json",        False),
            "accepted":             True,
            "quality":              content.get("quality", "rejected"),
            "verdict":              content.get("verdict", ""),
            "feedback":             content.get("feedback", ""),
            "improvement":          content.get("improvement", ""),
            "total_duration":       round(duration, 2),
            "total_attempts":       content.get("attempt", 1),
            "error":                None,
            "schema_violations":    content.get("schema_violations", []),
        }

    def _build_error_result(self, task: dict, reason: str,
                             duration: float, content: dict = None) -> dict:
        c = content or {}
        n_attempts = c.get("attempts", 1)
        return {
            "task_id":              task["id"],
            "category":             task.get("category", ""),
            "true_depth":           task.get("true_depth", 0),
            "attempt":              n_attempts,
            "strategy":             c.get("hint_used", ""),
            "hint_used":            c.get("hint_used", ""),
            "plan_name":            c.get("plan_name", ""),
            "generated_json":       c.get("generated_json", ""),
            "reward":               c.get("reward", 0.0),
            "steps":                1,
            "success":              False,
            "U_planner":            c.get("U_PE", 0.0),
            "U_PE":                 c.get("U_PE", 0.0),
            "U_verifier":           c.get("U_V", 0.0),
            "U_V":                  c.get("U_V", 0.0),
            "dim_json_valid":       c.get("dim_json_valid",       0.0),
            "dim_schema_compliance": c.get("dim_schema_compliance", 0.0),
            "dim_field_accuracy":   c.get("dim_field_accuracy",   0.0),
            "dim_depth_match":      c.get("dim_depth_match",      0.0),
            "actual_depth":         c.get("actual_depth",         0),
            "is_valid_json":        c.get("is_valid_json",        False),
            "accepted":             False,
            "quality":              c.get("quality", "rejected"),
            "verdict":              c.get("verdict", f"ERROR: {reason}"),
            "feedback":             c.get("feedback", ""),
            "improvement":          "",
            "total_duration":       round(duration, 2),
            "total_attempts":       n_attempts,
            "error":                reason,
            "failed_fields":        c.get("failed_fields", []),
            "schema_violations":    c.get("schema_violations", []),
        }

    def _save_result(self, result: dict) -> None:
        exists = os.path.exists(RESULTS_FILE)
        fields = ["task_id", "category", "true_depth", "attempt", "strategy",
                  "reward", "steps", "U_planner", "U_verifier",
                  "dim_json_valid", "dim_schema_compliance",
                  "dim_field_accuracy", "dim_depth_match",
                  "actual_depth", "is_valid_json",
                  "accepted", "quality", "improvement",
                  "total_attempts", "total_duration"]
        with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(result)

    def print_summary(self, results: list) -> None:
        print(f"\n{'='*62}")
        print(f"  SUMMARY — DeepJSONEval Hybrid MAS")
        print(f"{'='*62}")
        print(f"  {'Task':<28} {'Cat':<10} {'U_PE':>6} {'U_V':>6} {'OK':>4}")
        print("  " + "─" * 56)
        for r in results:
            print(f"  {r['task_id']:<28} "
                  f"{r['category']:<10} "
                  f"{r['U_PE']:>6.3f} "
                  f"{r['U_V']:>6.3f} "
                  f"{'YES' if r['accepted'] else 'NO':>4}")
        print("  " + "─" * 56)
        n_ok  = sum(1 for r in results if r["accepted"])
        avg_v = sum(r["U_V"] for r in results) / len(results) if results else 0
        print(f"  Accepted : {n_ok}/{len(results)}")
        print(f"  Avg U_V  : {avg_v:.3f}")
        print(f"  CSV      : {RESULTS_FILE}")
