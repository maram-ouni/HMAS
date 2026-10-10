# ─────────────────────────────────────────────────────────
# world_model.py — CROYANCES PERSISTANTES SUR L'ENVIRONNEMENT
# ET SUR L'AGENT EXECUTOR (v2 — amélioration architecture BDI)
#
# Contrairement au BeliefBase (croyances sur la tâche courante,
# réinitialisées à chaque set_task()), le WorldModel persiste sur
# disque et survit entre les tâches ET entre les runs. Il porte
# deux types de croyances qui manquaient à la v1 :
#
#   1. Croyances sur l'environnement : difficulté réelle de chaque
#      catégorie de tâche, mesurée sur l'historique complet.
#   2. Croyances sur un AUTRE agent (l'executor LLM) : fiabilité
#      empirique de chaque plan pour chaque catégorie — un proxy
#      de théorie de l'esprit ("je crois que l'executor échouera
#      si je lui envoie ce type de hint pour cette catégorie").
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

_WORLD_MODEL_FILE = os.path.join(RESULTS_DIR, "world_model.json")

# Force du prior face à peu de données, en mode bayésien (V2.1, #2) — voir
# executor_reliability(). Comparable à l'ancien min_samples=4 : à n=KAPPA
# observations, la croyance est à mi-chemin entre le prior optimiste et le
# taux empirique, mais SANS la falaise dure de l'ancien mode fréquentiste.
BAYESIAN_KAPPA = 4.0


class WorldModel:
    """
    Croyances partagées entre tâches et runs — le véritable 'environnement' BDI.

    `path` permet d'isoler un run comparatif : par défaut, toutes les instances
    lisent/écrivent le même fichier partagé (`_WORLD_MODEL_FILE`), ce qui rend
    deux runs successifs NON indépendants (l'historique cumulé d'un run influence
    la sélection de plans du suivant, via `executor_reliability`). Passer un
    `path` dédié démarre avec des croyances vierges, sans jamais toucher le
    fichier partagé — aucun risque de perte de l'apprentissage cumulatif normal.
    """

    def __init__(self, path: str | None = None, run_id: str | None = None,
                 reliability_mode: str = "frequentist"):
        self._path = path or _WORLD_MODEL_FILE
        # `run_id` : identifiant du run courant — si fourni, une copie de
        # provenance des croyances chargées est écrite au démarrage (V2.1 #8).
        self._run_id = run_id
        # `reliability_mode` (V2.1, #2) : "frequentist" (défaut) = comportement
        # HISTORIQUE EXACT (successes/n, falaise dure à min_samples) — pour
        # préserver la comparabilité avec tous les runs déjà mesurés.
        # "bayesian" = lissage Beta-Bernoulli, opt-in explicite (voir
        # --reliability-mode). C'est le SEUL des 8 changements V2.1 qui peut
        # modifier la sélection de plan (executor_reliability alimente
        # PlanLibrary._plan_eligible) — jamais le mode par défaut.
        self._reliability_mode = reliability_mode
        self._lock = threading.Lock()
        self.category_stats: dict = {}   # category -> {"n": int, "sum_u_v": float}
        self.plan_stats: dict = {}       # "category:plan_name" -> {"n": int, "successes": int}
        self._load()
        if self._run_id:
            self._snapshot()

    # ── Croyances sur l'environnement (par catégorie) ──────

    def record_task_outcome(self, category: str, u_v: float) -> None:
        if not category:
            return
        with self._lock:
            s = self.category_stats.setdefault(category, {"n": 0, "sum_u_v": 0.0})
            s["n"] += 1
            s["sum_u_v"] += u_v
        self._save()

    def category_avg_u_v(self, category: str) -> float:
        s = self.category_stats.get(category)
        return (s["sum_u_v"] / s["n"]) if s and s["n"] else 0.0

    def category_samples(self, category: str) -> int:
        return self.category_stats.get(category, {}).get("n", 0)

    # ── Croyances sur l'executor (par catégorie + plan) ────

    def record_plan_outcome(self, category: str, plan_name: str, success: bool) -> None:
        if not category or not plan_name:
            return
        key = f"{category}:{plan_name}"
        with self._lock:
            s = self.plan_stats.setdefault(key, {"n": 0, "successes": 0})
            s["n"] += 1
            if success:
                s["successes"] += 1
        self._save()

    def executor_reliability(self, category: str, plan_name: str,
                             min_samples: int = 4, default: float = 1.0) -> float:
        """
        Croyance sur la fiabilité de l'executor pour ce plan sur cette catégorie.

        Mode "frequentist" (défaut, INCHANGÉ) : tant que l'échantillon est
        insuffisant (`n < min_samples`), croyance optimiste par défaut plutôt
        que de pénaliser un plan sans preuve ; au-dessus, taux brut
        `successes/n` — falaise dure entre les deux régimes (V2.1 #2 : bug
        documenté, un seul échantillon supplémentaire peut faire basculer un
        plan de "fiable" à "exclu").

        Mode "bayesian" (opt-in) : moyenne postérieure Beta-Bernoulli
        `(successes + KAPPA*default) / (n + KAPPA)` — même prior optimiste à
        n=0, mais dégrade PROGRESSIVEMENT avec les échecs au lieu de basculer
        d'un coup à min_samples.
        """
        key = f"{category}:{plan_name}"
        s = self.plan_stats.get(key)
        if self._reliability_mode == "bayesian":
            n         = s["n"] if s else 0
            successes = s["successes"] if s else 0
            return (successes + BAYESIAN_KAPPA * default) / (n + BAYESIAN_KAPPA)
        if not s or s["n"] < min_samples:
            return default
        return s["successes"] / s["n"]

    # ── Persistence ─────────────────────────────────────────

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with self._lock:
            data = {"category_stats": self.category_stats, "plan_stats": self.plan_stats}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self.category_stats = data.get("category_stats", {})
            self.plan_stats     = data.get("plan_stats", {})
            print(f"  [WorldModel] Loaded beliefs from {self._path} — "
                  f"{len(self.category_stats)} categories, "
                  f"{len(self.plan_stats)} (category, plan) pairs")
        except Exception as e:
            print(f"  [WorldModel] Load failed: {e}")

    def _snapshot(self) -> None:
        """
        Copie read-only horodatée des croyances chargées au démarrage de CE
        run (V2.1, changement #8) — pure provenance, jamais relue par
        WorldModel. Permet de savoir exactement ce que l'agent croyait au
        moment d'un run donné, même après que `_WORLD_MODEL_FILE` partagé a
        continué d'évoluer avec des runs ultérieurs. Non-bloquant : un échec
        d'écriture n'interrompt jamais le run.
        """
        try:
            snap_path = os.path.join(RESULTS_DIR, f"world_model_snapshot_{self._run_id}.json")
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump({
                    "run_id":         self._run_id,
                    "source_path":    self._path,
                    "category_stats": self.category_stats,
                    "plan_stats":     self.plan_stats,
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [WorldModel] Snapshot failed (non-blocking): {e}")
