# ─────────────────────────────────────────────────────────
# gt_analyzer.py — ANALYSE GAME-THEORIQUE DU MAS a 2 AGENTS
#
# Analyse les interactions strategiques entre :
#   Agent PE (PlannerExecuter) : U_PE = 0.7*S + 0.2*E + 0.1*Sa
#   Agent V  (Verifier)        : U_V  = 0.4*S + 0.1*E + 0.5*Sa
#
# Metriques calculees :
#   1. Regret cumulatif de PE  → PE apprend-il de ses echecs ?
#   2. Evolution des strategies → convergence vers strategie optimale ?
#   3. Taux de conflit          → quand PE est content mais V rejette ?
#   4. Equilibre de Nash        → stabilite des strategies ?
#   5. Valeur du jeu            → payoff moyen a l equilibre
#
# ─────────────────────────────────────────────────────────

import csv
import json
import os
import sys
from collections import defaultdict
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_FILE, RESULTS_DIR


def load_results(results_file: str = RESULTS_FILE) -> List[dict]:
    """Charge les resultats depuis le CSV."""
    if not os.path.exists(results_file):
        return []
    results = []
    with open(results_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "task_id":        row.get("task_id", ""),
                "attempt":        int(row.get("attempt", 1)),
                "strategy":       row.get("strategy", ""),
                "reward":         float(row.get("reward", 0)),
                "steps":          int(row.get("steps", 0)),
                "U_PE":           float(row.get("U_planner", 0)),
                "U_V":            float(row.get("U_verifier", 0)),
                "accepted":       row.get("accepted", "False") == "True",
                "improvement":    row.get("improvement", ""),
                "total_attempts": int(row.get("total_attempts", 1)),
            })
    return results


def compute_regret(results: List[dict]) -> Dict:
    """
    Calcule le regret cumulatif de PE par tache.

    Regret(t) = U_PE_optimal - U_PE_t
    U_PE_optimal = max(U_PE) sur tous les runs de cette tache

    Si le regret diminue avec t → PE apprend.
    Si stable → PE ne converge pas.
    """
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    regret_data = {}
    for task_id, runs in by_task.items():
        U_PE_vals = [r["U_PE"] for r in runs]
        U_optimal = max(U_PE_vals) if U_PE_vals else 1.0

        regrets = []
        cumulative = 0.0
        for i, r in enumerate(runs, 1):
            regret_t   = U_optimal - r["U_PE"]
            cumulative += regret_t
            regrets.append({
                "run":         i,
                "U_PE":        r["U_PE"],
                "regret_t":    round(regret_t, 4),
                "regret_cum":  round(cumulative / i, 4),
                "strategy":    r["strategy"],
                "accepted":    r["accepted"],
            })

        slope = _compute_slope([r["regret_cum"] for r in regrets])
        regret_data[task_id] = {
            "runs":     regrets,
            "slope":    round(slope, 4),
            "learning": "OUI" if slope < -0.02 else "NON" if slope > 0.02 else "STABLE",
        }

    return regret_data


def compute_strategy_evolution(results: List[dict]) -> Dict:
    """
    Analyse l evolution des strategies de PE.

    Pour chaque tache : quelles categories ont ete utilisees ?
    Convergent-elles vers une categorie dominante ?
    """
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    evolution = {}
    for task_id, runs in by_task.items():
        strategy_counts  = defaultdict(int)
        strategy_payoffs = defaultdict(list)

        for r in runs:
            cat = r["strategy"]
            strategy_counts[cat]  += 1
            strategy_payoffs[cat].append(r["U_PE"])

        best_cat    = max(strategy_payoffs, key=lambda c: sum(strategy_payoffs[c]) / len(strategy_payoffs[c]))
        best_payoff = sum(strategy_payoffs[best_cat]) / len(strategy_payoffs[best_cat])

        evolution[task_id] = {
            "strategies_used": dict(strategy_counts),
            "avg_payoff_by_strategy": {
                cat: round(sum(vals) / len(vals), 3)
                for cat, vals in strategy_payoffs.items()
            },
            "dominant_strategy": best_cat,
            "dominant_payoff":   round(best_payoff, 3),
            "converged": len(set(r["strategy"] for r in runs[-3:])) == 1 if len(runs) >= 3 else False,
        }

    return evolution


def compute_conflicts(results: List[dict]) -> Dict:
    """
    Detecte les conflits entre PE et V.

    Conflit = PE serait content (U_PE >= 0.4) mais V rejette
    OU V accepte mais U_PE est faible.

    Ces conflits revelent la tension strategique du jeu.
    """
    conflicts = []
    for r in results:
        conflict_type = None
        if r["U_PE"] >= 0.4 and not r["accepted"]:
            conflict_type = "PE_content_V_rejette"
        elif r["accepted"] and r["U_PE"] < 0.4:
            conflict_type = "V_accepte_PE_insatisfait"

        if conflict_type:
            conflicts.append({
                "task_id":       r["task_id"],
                "attempt":       r["attempt"],
                "conflict_type": conflict_type,
                "U_PE":          r["U_PE"],
                "U_V":           r["U_V"],
                "strategy":      r["strategy"],
            })

    n_total    = len(results)
    n_conflict = len(conflicts)

    return {
        "conflict_rate": round(n_conflict / n_total, 3) if n_total > 0 else 0,
        "total_runs":    n_total,
        "total_conflicts": n_conflict,
        "conflicts":     conflicts,
    }


def detect_nash_equilibrium(results: List[dict]) -> Dict:
    """
    Detection heuristique d un equilibre de Nash.

    Equilibre approche si :
    - PE utilise toujours la meme strategie sur les 3 derniers runs
    - V accepte cette strategie
    - Les payoffs sont stables

    Lien GT : sigma* = (sigma_PE*, sigma_V*) tels que
    aucun agent ne peut ameliorer son payoff unilateralement.
    """
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    nash_data = {}
    for task_id, runs in by_task.items():
        if len(runs) < 3:
            nash_data[task_id] = {"status": "PAS ASSEZ DE DONNEES"}
            continue

        recent = runs[-3:]
        strategies_stable = len(set(r["strategy"] for r in recent)) == 1
        all_accepted      = all(r["accepted"] for r in recent)
        U_PE_stable       = max(r["U_PE"] for r in recent) - min(r["U_PE"] for r in recent) < 0.1
        U_V_stable        = max(r["U_V"]  for r in recent) - min(r["U_V"]  for r in recent) < 0.1

        at_equilibrium = strategies_stable and all_accepted and U_PE_stable and U_V_stable

        nash_data[task_id] = {
            "at_equilibrium":     at_equilibrium,
            "strategy_stable":    strategies_stable,
            "all_accepted":       all_accepted,
            "U_PE_stable":        U_PE_stable,
            "U_V_stable":         U_V_stable,
            "equilibrium_strategy": recent[-1]["strategy"] if at_equilibrium else None,
            "equilibrium_U_PE":   round(sum(r["U_PE"] for r in recent) / 3, 3),
            "equilibrium_U_V":    round(sum(r["U_V"]  for r in recent) / 3, 3),
        }

    return nash_data


def full_analysis(results_file: str = RESULTS_FILE) -> dict:
    """
    Analyse complete du jeu.
    Retourne toutes les metriques GT.
    """
    results = load_results(results_file)
    if not results:
        return {"error": "Aucun resultat disponible"}

    regret    = compute_regret(results)
    evolution = compute_strategy_evolution(results)
    conflicts = compute_conflicts(results)
    nash      = detect_nash_equilibrium(results)

    # Resume global
    n_runs     = len(results)
    n_success  = sum(1 for r in results if r["reward"] >= 1.0)
    avg_U_PE   = sum(r["U_PE"] for r in results) / n_runs
    avg_U_V    = sum(r["U_V"]  for r in results) / n_runs

    return {
        "summary": {
            "total_runs":      n_runs,
            "success_rate":    round(n_success / n_runs, 3),
            "avg_U_PE":        round(avg_U_PE, 3),
            "avg_U_V":         round(avg_U_V, 3),
            "conflict_rate":   conflicts["conflict_rate"],
        },
        "regret":    regret,
        "evolution": evolution,
        "conflicts": conflicts,
        "nash":      nash,
    }


def print_analysis(results_file: str = RESULTS_FILE) -> None:
    """Affiche l analyse GT dans le terminal."""
    analysis = full_analysis(results_file)

    if "error" in analysis:
        print(f"\n  {analysis['error']}")
        return

    s = analysis["summary"]
    print(f"\n{'='*62}")
    print(f"  ANALYSE GAME-THEORIQUE — MAS 2 AGENTS")
    print(f"{'='*62}")
    print(f"  Runs total     : {s['total_runs']}")
    print(f"  Taux succes    : {s['success_rate']:.1%}")
    print(f"  U_PE moyen     : {s['avg_U_PE']:.3f}")
    print(f"  U_V moyen      : {s['avg_U_V']:.3f}")
    print(f"  Taux conflit   : {s['conflict_rate']:.1%}")

    print(f"\n  REGRET (PE apprend-il ?)")
    for task_id, data in analysis["regret"].items():
        print(f"  {task_id} : pente={data['slope']:+.4f} → {data['learning']}")
        for r in data["runs"][-3:]:
            print(f"    run={r['run']} U_PE={r['U_PE']:.3f} "
                  f"regret={r['regret_cum']:.3f} cat={r['strategy']}")

    print(f"\n  EVOLUTION DES STRATEGIES")
    for task_id, data in analysis["evolution"].items():
        print(f"  {task_id} :")
        for cat, avg in data["avg_payoff_by_strategy"].items():
            print(f"    {cat:10s} : U_PE_moy={avg:.3f} "
                  f"(n={data['strategies_used'].get(cat,0)})")
        print(f"    Dominante : {data['dominant_strategy']} "
              f"(payoff={data['dominant_payoff']:.3f})")
        print(f"    Convergee : {'OUI' if data['converged'] else 'NON'}")

    print(f"\n  CONFLITS PE vs V")
    c = analysis["conflicts"]
    print(f"  Total : {c['total_conflicts']}/{c['total_runs']} "
          f"({c['conflict_rate']:.1%})")
    for conf in c["conflicts"][:5]:
        print(f"  [{conf['conflict_type']}] "
              f"task={conf['task_id']} "
              f"U_PE={conf['U_PE']:.3f} U_V={conf['U_V']:.3f} "
              f"cat={conf['strategy']}")

    print(f"\n  EQUILIBRE DE NASH")
    for task_id, data in analysis["nash"].items():
        if data.get("at_equilibrium"):
            print(f"  {task_id} : EQUILIBRE ATTEINT "
                  f"(strat={data['equilibrium_strategy']}, "
                  f"U_PE={data['equilibrium_U_PE']:.3f}, "
                  f"U_V={data['equilibrium_U_V']:.3f})")
        else:
            print(f"  {task_id} : pas encore d equilibre "
                  f"({data.get('status', 'en cours')})")

    print(f"\n{'='*62}")

    # Sauvegarder JSON
    out = os.path.join(RESULTS_DIR, "gt_analysis.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"  JSON GT : {out}")


def _compute_slope(values: List[float]) -> float:
    """Regression lineaire — pente negative = amelioration."""
    n = len(values)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2
    ym = sum(values) / n
    num = sum((i - xm) * (v - ym) for i, v in enumerate(values))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0
