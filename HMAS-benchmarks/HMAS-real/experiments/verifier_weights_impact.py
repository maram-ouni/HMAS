"""
verifier_weights_impact.py — Mesure rétrospective de l'apport des poids
ET du seuil appris du Verifier (core/verifier_learned_weights.py), en
rejouant la logique de décision réelle de RuleBasedVerifierAgent sur les
traces déjà collectées (aucun run, aucune dépendance Groq) :
  - "statique"    : poids WEIGHTS_V + seuil V_THRESHOLD (comportement
                    d'avant les 2 changements)
  - "appris"      : poids appris + seuil calibré (comportement réel
                    quand config.LEARNED_VERIFIER_WEIGHTS est actif)

Ne mesure PAS un changement de reward (le ground truth ne bouge pas,
et les tâches en succès/retrieval-rejeté sont gated par reward, pas par
U_V — voir agents/rule_based_verifier.py). Mesure ce que les poids et le
seuil affectent réellement : la classification "partial vs rejected"
pour les tâches d'action (reward<1), et le score U_V utilisé par le BDI
pour le suivi de tendance et le choix du meilleur essai.

Usage:
    python mas_project_v2/experiments/verifier_weights_impact.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAS  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_MAS)
sys.path.insert(0, _MAS)

from core.payoff_calculator import WEIGHTS_V, compute_U_V, is_retrieval_task, V_THRESHOLD, REL_THRESHOLD
from core.verifier_learned_weights import compute_learned_weights, calibrate_threshold

TRACE_DIR = os.path.join(_MAS, "results", "traces")


def _get_task_goal(task_id: str) -> str:
    parts   = task_id.split(".", 1)
    version = parts[0] if len(parts) > 1 else "v1"
    name    = parts[1] if len(parts) > 1 else parts[0]
    path = os.path.join(_ROOT, "src", "agisdk", "REAL", "browsergym",
                        "webclones", version, "tasks", f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("goal", "")
    return ""


def _classify(reward: float, answer: str, u_v_score: float, threshold: float,
             relevance: float, retrieval: bool, is_crash: bool) -> str:
    """Mirrors RuleBasedVerifierAgent._verify's classification logic
    (agents/rule_based_verifier.py) exactly, so the replay is faithful."""
    if is_crash:
        return "rejected"
    if reward >= 1.0 and answer.strip():
        return "success"
    if retrieval:
        return "rejected"
    if u_v_score >= threshold and relevance >= REL_THRESHOLD:
        return "partial"
    return "rejected"


def _load_all_rows() -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(TRACE_DIR, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    learned   = compute_learned_weights(TRACE_DIR, WEIGHTS_V)
    threshold = calibrate_threshold(TRACE_DIR, learned, V_THRESHOLD)

    print("=" * 78)
    print("  APPORT DES POIDS + SEUIL APPRIS DU VERIFIER — rejeu sur traces réelles")
    print("=" * 78)
    print(f"  Poids statiques   : {WEIGHTS_V}")
    print(f"  Poids appris      : {learned}")
    print(f"  Seuil statique    : {V_THRESHOLD}")
    print(f"  Seuil calibré     : {threshold}")
    print()

    rows = _load_all_rows()
    goal_cache = {}
    flips = []
    delta_sum, delta_n = 0.0, 0

    for r in rows:
        task_id = r.get("task_id", "")
        if task_id not in goal_cache:
            goal_cache[task_id] = _get_task_goal(task_id)
        goal = goal_cache[task_id]

        reward   = r.get("reward", 0.0)
        answer   = r.get("answer", "") or ""
        steps    = r.get("steps", 0)
        is_crash = (steps <= 1 and reward == 0.0)
        retrieval = is_retrieval_task(goal)

        safety_score      = r.get("dim_safety", 0.0)
        specificity_score = r.get("dim_specificity", 0.0)
        relevance_score   = r.get("dim_relevance", 0.0)
        success_score     = 1.0 if reward >= 1.0 else 0.0

        def _u_v(weights):
            w = weights
            return (w["safety"] * safety_score
                    + w["specificity"] * specificity_score
                    + w["success_v"] * success_score
                    + w["relevance"] * relevance_score)

        u_v_static  = _u_v(WEIGHTS_V)
        u_v_learned = _u_v(learned)

        q_static  = _classify(reward, answer, u_v_static, V_THRESHOLD,
                              relevance_score, retrieval, is_crash)
        q_learned = _classify(reward, answer, u_v_learned, threshold,
                              relevance_score, retrieval, is_crash)

        delta_sum += abs(u_v_learned - u_v_static)
        delta_n   += 1

        if q_static != q_learned:
            flips.append({
                "task_id": task_id, "reward": reward,
                "U_V_static": round(u_v_static, 3), "U_V_learned": round(u_v_learned, 3),
                "quality_static": q_static, "quality_learned": q_learned,
                "answer": answer[:70],
            })

    print(f"  Lignes rejouées : {delta_n}")
    print(f"  Delta moyen |U_V_appris - U_V_statique| : {delta_sum / delta_n:.4f}" if delta_n else "")
    print(f"  Décisions 'partial vs rejected' qui changent : {len(flips)} / {delta_n}")
    print()

    if flips:
        print("  Détail des bascules :")
        print(f"  {'Tâche':<20} {'reward':>6} {'U_V stat':>9} {'U_V appr':>9} {'stat→appr':<20}")
        print("  " + "─" * 74)
        for f in flips:
            transition = f"{f['quality_static']}→{f['quality_learned']}"
            print(f"  {f['task_id']:<20} {f['reward']:>6.1f} {f['U_V_static']:>9.3f} "
                  f"{f['U_V_learned']:>9.3f} {transition:<20}")
    else:
        print("  Aucune bascule de classification sur les données actuelles — "
              "l'effet des poids appris reste, pour l'instant, cantonné au "
              "score U_V continu (feedback BDI / choix du meilleur essai), "
              "pas à la décision finale accepté/rejeté.")

    print(f"\n{'='*78}\n")


if __name__ == "__main__":
    main()
