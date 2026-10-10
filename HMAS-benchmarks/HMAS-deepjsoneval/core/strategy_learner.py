# ─────────────────────────────────────────────────────────
# strategy_learner.py — APPRENTISSAGE AUTONOME DE STRATÉGIES
#
# Adapté de agisdk/mas_project pour DeepJSONEval.
# Bandit multi-bras sans catégories prédéfinies :
#   payoff >= KEEP_THRESHOLD -> EXPLOITATION (réutiliser)
#   payoff <  KEEP_THRESHOLD -> EXPLORATION  (LLM génère un nouveau plan)
# ─────────────────────────────────────────────────────────

import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL, RESULTS_DIR
from core.payoff_calculator import PE_SUCCESS_THRESHOLD

KEEP_THRESHOLD = PE_SUCCESS_THRESHOLD


@dataclass
class ActionStrategy:
    """
    Une stratégie = un hint textuel + son payoff U_PE.
    Pas de catégorie prédéfinie — le hint est en texte libre.
    """
    steps:      List[str]
    payoff:     float
    reward:     float
    task_type:  str
    run_number: int
    steps_used: int
    timestamp:  float = field(default_factory=time.time)


class StrategyLearner:
    """
    Apprentissage autonome de stratégies de génération JSON.

    Cycle d'apprentissage :
      1. Premier run    : LLM génère un plan initial
      2. Bon payoff     : EXPLOITATION — réutilise le même plan
      3. Mauvais payoff : EXPLORATION  — LLM génère un plan différent
      4. Convergence    : le plan avec le meilleur payoff domine
    """

    def __init__(self, path: str = None):
        self.client          = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        # `path` isole un run comparatif du fichier partagé cumulatif — voir
        # WorldModel pour le même mécanisme et sa justification (reproductibilité).
        self.strategies_file = path or os.path.join(RESULTS_DIR, "learned_strategies.json")
        self.strategies: dict = {}
        self._lock = threading.Lock()
        self._load_from_disk()

    def get_nohint_success_rate(self, task_type: str, min_samples: int = 5) -> float:
        """
        Taux de succès (reward >= 0.7) des runs sans hint pour ce type de tâche.
        Retourne 0.0 si moins de min_samples runs disponibles (données insuffisantes).
        """
        strats = self.get_strategies_for_task(task_type)
        no_hint_runs = [s for s in strats
                        if not s.steps or s.steps[0] in ("no_hint", "")]
        if len(no_hint_runs) < min_samples:
            return 0.0
        return sum(1 for s in no_hint_runs if s.reward >= 0.7) / len(no_hint_runs)

    def get_best_hint(self, task_type: str,
                      exclude: Optional[List[str]] = None,
                      require_success: bool = True) -> str:
        """
        Choix du hint de warm-start par UCB1 (bandit multi-bras), pas du pur
        greedy. Chaque hint déjà essayé pour ce task_type est un BRAS ; l'ancien
        code choisissait le hint dont le MEILLEUR essai isolé était le plus
        haut (`max(reward, payoff)` sur un seul run) — un hint testé une fois
        par chance pouvait ainsi dominer un hint testé 10 fois avec une
        moyenne plus fiable, et rien n'encourageait jamais à retester un hint
        peu exploré. UCB1 score = moyenne des récompenses observées pour ce
        hint + bonus d'exploration qui grandit quand ce hint a été peu tiré
        (sqrt(2*ln(N)/n)) — converge vers le meilleur hint à long terme
        (comme le greedy) mais explore davantage tôt dans le run.
        Ne considère que les hints ayant réussi au moins une fois
        (require_success) — garde produit : ne jamais warm-start avec un
        hint jamais gagnant.
        """
        self._load_from_disk()
        exclude_set = set(exclude or [])

        by_hint: dict = defaultdict(list)
        for s in self.get_strategies_for_task(task_type):
            hint = s.steps[0] if s.steps else ""
            if not hint or hint == "no_hint" or hint in exclude_set:
                continue
            by_hint[hint].append(s)

        eligible = {h: runs for h, runs in by_hint.items()
                    if not require_success or any(s.reward >= 0.7 for s in runs)}
        if not eligible:
            return ""

        total_pulls = sum(len(runs) for runs in eligible.values())

        def ucb1_score(runs: list) -> float:
            mean_reward = sum(s.reward for s in runs) / len(runs)
            bonus = math.sqrt(2 * math.log(total_pulls + 1) / len(runs))
            return mean_reward + bonus

        return max(eligible, key=lambda h: ucb1_score(eligible[h]))

    def record_run(self, task_type: str, steps: List[str],
                   payoff: float, reward: float,
                   steps_used: int, run_number: int) -> None:
        """Enregistre le résultat d'un run."""
        strategy = ActionStrategy(
            steps=steps, payoff=payoff, reward=reward,
            task_type=task_type, run_number=run_number, steps_used=steps_used,
        )
        with self._lock:
            if task_type not in self.strategies:
                self.strategies[task_type] = []
            self.strategies[task_type].append(strategy)
        self._save_to_disk()

        status   = "OK  " if reward >= 0.7 else "FAIL"
        decision = "KEEP" if payoff >= KEEP_THRESHOLD else "CHANGE"
        print(f"\n  [Learner] [{status}] U_PE={payoff:.3f} -> {decision}")

    def get_strategies_for_task(self, task_type: str) -> List[ActionStrategy]:
        return self.strategies.get(task_type, [])

    def get_best_strategy(self, task_type: str) -> Optional[ActionStrategy]:
        strats = self.strategies.get(task_type, [])
        return max(strats, key=lambda s: s.payoff) if strats else None

    def get_learning_summary(self, task_type: str) -> str:
        strats = self.strategies.get(task_type, [])
        if not strats:
            return f"No strategy learned for {task_type}"
        payoffs = [s.payoff for s in strats]
        avg  = sum(payoffs) / len(payoffs)
        best = max(payoffs)
        n_ok = sum(1 for p in payoffs if p >= KEEP_THRESHOLD)
        return (f"{task_type} : {len(strats)} runs | "
                f"U_PE_avg={avg:.3f} | best={best:.3f} | "
                f"{n_ok}/{len(strats)} exploitable")

    # ── Persistence ───────────────────────────────────────

    def _save_to_disk(self) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with self._lock:
            data = {}
            for task_type, strats in self.strategies.items():
                data[task_type] = [
                    {"steps":      s.steps,
                     "payoff":     s.payoff,
                     "reward":     s.reward,
                     "task_type":  s.task_type,
                     "run_number": s.run_number,
                     "steps_used": s.steps_used,
                     "timestamp":  s.timestamp}
                    for s in strats
                ]
            with open(self.strategies_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_from_disk(self) -> None:
        if not os.path.exists(self.strategies_file):
            return
        try:
            with open(self.strategies_file, encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self.strategies = {}
                for task_type, strats in data.items():
                    self.strategies[task_type] = [
                        ActionStrategy(
                            steps=s["steps"], payoff=s["payoff"], reward=s["reward"],
                            task_type=s["task_type"], run_number=s["run_number"],
                            steps_used=s["steps_used"], timestamp=s.get("timestamp", 0.0),
                        )
                        for s in strats
                    ]
            total = sum(len(v) for v in self.strategies.values())
            if total:
                print(f"  [Learner] {total} strategies loaded from disk")
        except Exception as e:
            print(f"  [Learner] Load failed : {e}")
