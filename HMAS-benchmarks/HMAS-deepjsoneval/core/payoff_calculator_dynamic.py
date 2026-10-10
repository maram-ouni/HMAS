# ─────────────────────────────────────────────────────────
# payoff_calculator_dynamic.py — PAYOFFS DYNAMIQUES (variante)
#
# N'existe qu'en PARALLÈLE de payoff_calculator.py — celui-ci reste
# intact (référence gelée, utilisée par toutes les architectures déjà
# benchmarquées : HMAS v2, Frering BDI, ReAct, etc.).
#
# Principe : au lieu de poids fixes (WEIGHTS_PE/WEIGHTS_V), chaque
# dimension reçoit un poids qui reflète à quel point elle a été le
# vrai point de friction jusqu'ici — une dimension toujours proche de
# 1.0 (jamais discriminante) perd du poids, une dimension qui échoue
# souvent (le vrai goulot d'étranglement) en gagne. Inspiré du principe
# de pondération adaptative d'experts (Hedge / AdaBoost), simplifié en
# une pondération par taux d'erreur lissée par un prior statique.
#
# Portée de l'historique :
#   Niveau 1 — shots précédents de LA MÊME tâche (poids double, plus
#              spécifique)
#   Niveau 2 — autres tâches de LA MÊME catégorie, dans CE job
#   Niveau 3 — cumulatif entre jobs (V2.1, #5) : désormais possible mais
#   STRICTEMENT opt-in via un `path` explicite (voir DynamicWeightTracker),
#   jamais un fichier partagé par défaut — exactement le mécanisme
#   d'isolation de WorldModel/StrategyLearner, pour ne pas reproduire le
#   bug de contamination inter-runs déjà rencontré sur world_model.json.
#   Sans `path` (défaut), le tracker reste en mémoire, job-scopé, comme avant.
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from core.payoff_calculator import (
    dim_json_valid, dim_schema_compliance, dim_field_accuracy,
    dim_completeness, dim_depth_match, get_failed_fields,
    WEIGHTS_PE, WEIGHTS_V,
)

# Poids du niveau 1 (shots de la même tâche) vs niveau 2 (catégorie,
# dans ce job) dans le pool d'évidence — le niveau 1 est plus
# spécifique à la tâche en cours, donc compte double.
TASK_LEVEL_MULTIPLIER = 2

# Constante de lissage (force du prior statique face à peu de données) :
# avec n observations, confidence = n / (n + SMOOTHING_K). À n=0,
# confidence=0 -> poids strictement identiques aux poids statiques.
SMOOTHING_K = 5.0

# Poids minimum après renormalisation : aucune dimension ne doit
# disparaître complètement, même si elle a toujours été parfaite.
MIN_WEIGHT = 0.05


class DynamicWeightTracker:
    """
    Historique des scores par dimension.

    `path=None` (défaut) : en mémoire seulement, job-scopé, comportement
    HISTORIQUE INCHANGÉ — meurt avec le process. `path` explicite (V2.1, #5) :
    persiste l'historique PAR CATÉGORIE sur disque (l'historique par tâche
    reste job-scopé, jamais persisté — il n'a aucun sens d'un run à l'autre).
    """

    def __init__(self, path: Optional[str] = None):
        self._path = path
        self._by_task: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._by_category: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        if self._path:
            self._load()

    def record(self, category: str, task_id: str, dim_scores: Dict[str, float]) -> None:
        for dim, score in dim_scores.items():
            self._by_task[task_id][dim].append(score)
            self._by_category[category][dim].append(score)
        if self._path:
            self._save()

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            data = {c: {d: list(v) for d, v in dims.items()}
                    for c, dims in self._by_category.items()}
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({"by_category": data}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [DynamicWeightTracker] Save failed (non-blocking): {e}")

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            for cat, dims in data.get("by_category", {}).items():
                for dim, scores in dims.items():
                    self._by_category[cat][dim] = list(scores)
            print(f"  [DynamicWeightTracker] Loaded {len(self._by_category)} "
                  f"categories from {self._path}")
        except Exception as e:
            print(f"  [DynamicWeightTracker] Load failed: {e}")

    def get_weights(self, category: str, task_id: str,
                    static_weights: Dict[str, float]) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        for dim, w_static in static_weights.items():
            task_scores = self._by_task[task_id].get(dim, [])
            cat_scores  = self._by_category[category].get(dim, [])

            n = len(task_scores) * TASK_LEVEL_MULTIPLIER + len(cat_scores)
            if n == 0:
                raw[dim] = w_static
                continue

            total = sum(task_scores) * TASK_LEVEL_MULTIPLIER + sum(cat_scores)
            avg_score = total / n
            error     = 1.0 - avg_score  # à quel point cette dimension a déçu jusqu'ici

            confidence = n / (n + SMOOTHING_K)
            raw[dim] = confidence * error + (1.0 - confidence) * w_static

        total_raw = sum(raw.values()) or 1.0
        normalized = {d: v / total_raw for d, v in raw.items()}

        # Plancher, puis renormalisation finale pour sommer à 1.
        floored = {d: max(v, MIN_WEIGHT) for d, v in normalized.items()}
        total_floored = sum(floored.values())
        return {d: round(v / total_floored, 4) for d, v in floored.items()}


def compute_U_PE_dynamic(generated_json: str, schema_str: str,
                         ground_truth_str: str, true_depth: int,
                         category: str, task_id: str,
                         tracker: DynamicWeightTracker,
                         parsed_obj=None, valid_score: Optional[float] = None) -> dict:
    if parsed_obj is None or valid_score is None:
        valid_score, parsed_obj = dim_json_valid(generated_json)
    schema_score   = dim_schema_compliance(parsed_obj, schema_str) if parsed_obj is not None else 0.0
    complete_score = dim_completeness(parsed_obj, ground_truth_str) if parsed_obj is not None else 0.0

    dims = {"json_valid": valid_score, "schema_compliance": schema_score, "completeness": complete_score}
    w = tracker.get_weights(category, task_id, WEIGHTS_PE)
    u = sum(w[d] * dims[d] for d in dims)

    return {
        "payoff":               round(u, 4),
        "agent_role":           "planner_executor",
        "dim_json_valid":       valid_score,
        "dim_schema_compliance": schema_score,
        "dim_completeness":     complete_score,
        "parsed_obj":           parsed_obj,
        "weights_used":         w,
        "breakdown": (
            f"U_PE = {w['json_valid']}x{valid_score:.2f} + "
            f"{w['schema_compliance']}x{schema_score:.2f} + "
            f"{w['completeness']}x{complete_score:.2f} = {u:.4f} [dynamic]"
        ),
    }


def compute_U_V_dynamic(generated_json: str, ground_truth_str: str,
                        true_depth: int, category: str, task_id: str,
                        tracker: DynamicWeightTracker, parsed_obj=None) -> dict:
    if parsed_obj is None:
        valid_score, parsed_obj = dim_json_valid(generated_json)
    else:
        valid_score = 1.0

    accuracy_score            = dim_field_accuracy(parsed_obj, ground_truth_str)
    depth_score, actual_depth = dim_depth_match(parsed_obj, true_depth)
    failed_fields             = get_failed_fields(parsed_obj, ground_truth_str)

    dims = {"json_valid": valid_score, "field_accuracy": accuracy_score, "depth_match": depth_score}
    w = tracker.get_weights(category, task_id, WEIGHTS_V)
    u = sum(w[d] * dims[d] for d in dims)

    return {
        "payoff":            round(u, 4),
        "agent_role":        "verifier",
        "dim_json_valid":    valid_score,
        "dim_field_accuracy": accuracy_score,
        "dim_depth_match":   depth_score,
        "actual_depth":      actual_depth,
        "failed_fields":     failed_fields,
        "weights_used":      w,
        "breakdown": (
            f"U_V = {w['json_valid']}x{valid_score:.2f} + "
            f"{w['field_accuracy']}x{accuracy_score:.2f} + "
            f"{w['depth_match']}x{depth_score:.2f} = {u:.4f} [dynamic]"
        ),
    }


def compute_payoffs_dynamic(generated_json: str, schema_str: str,
                            ground_truth_str: str, true_depth: int,
                            category: str, task_id: str,
                            tracker: DynamicWeightTracker) -> dict:
    """Équivalent dynamique de compute_payoffs() — même forme de retour,
    + weights_used dans U_PE/U_V pour transparence/traçabilité."""
    u_pe = compute_U_PE_dynamic(generated_json, schema_str, ground_truth_str, true_depth,
                                category, task_id, tracker)
    u_v  = compute_U_V_dynamic(generated_json, ground_truth_str, true_depth,
                               category, task_id, tracker, parsed_obj=u_pe["parsed_obj"])

    # Alimente l'historique pour les prochaines tâches/shots de ce job.
    tracker.record(category, task_id, {
        "json_valid":         u_pe["dim_json_valid"],
        "schema_compliance":  u_pe["dim_schema_compliance"],
        "completeness":       u_pe["dim_completeness"],
        "field_accuracy":     u_v["dim_field_accuracy"],
        "depth_match":        u_v["dim_depth_match"],
    })

    return {
        "U_PE":       u_pe,
        "U_V":        u_v,
        "payoff":     u_pe["payoff"],
        "U_planner":  u_pe["payoff"],
        "U_verifier": u_v["payoff"],
    }
