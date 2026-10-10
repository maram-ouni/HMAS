# ─────────────────────────────────────────────────────────
# gt_analyzer.py — ANALYSE GAME-THÉORIQUE DU MAS DeepJSONEval
#
# Adapté de agisdk/mas_project :
#   Agent PE (PlannerExecutor) : U_PE = f(json_valid, schema, completeness)
#   Agent V  (Verifier)        : U_V  = f(json_valid, field_accuracy, depth)
#
# Métriques :
#   1. Regret cumulatif de PE
#   2. Évolution des stratégies
#   3. Taux de conflit PE vs V
#   4. Équilibre de Nash heuristique
# ─────────────────────────────────────────────────────────

import csv
import json
import os
import sys
from collections import defaultdict
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_FILE, RESULTS_DIR


def load_trace_results(trace_dir: str = None) -> List[dict]:
    """
    Charge les traces JSONL par tentative (core/audit_trace.py, V2.1 #1) et
    les met en forme dans le MÊME schéma de ligne que load_results() (CSV) —
    ainsi compute_regret/compute_strategy_evolution/detect_nash_equilibrium
    fonctionnent SANS aucune modification sur les deux sources.

    Contrairement au CSV (une ligne par TÂCHE — la tentative finale/meilleure
    seulement), une trace contient une ligne par TENTATIVE : c'est ce qui
    manquait pour que regret/Nash (qui exigent plusieurs runs par task_id)
    aient des données exploitables plutôt que systématiquement
    "NOT ENOUGH DATA".
    """
    trace_dir = trace_dir or os.path.join(RESULTS_DIR, "traces")
    if not os.path.isdir(trace_dir):
        return []
    results = []
    for fname in sorted(os.listdir(trace_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(trace_dir, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                attempt = rec.get("attempt_index", 1)
                results.append({
                    "task_id":             rec.get("task_id", ""),
                    "attempt":             attempt,
                    "strategy":            rec.get("plan_name", ""),
                    "reward":              rec.get("U_V", 0.0),
                    "steps":               attempt,
                    "U_PE":                rec.get("U_PE", 0.0),
                    "U_V":                 rec.get("U_V", 0.0),
                    "accepted":            rec.get("quality") == "success",
                    "improvement":         "",
                    "total_attempts":      attempt,
                    # Champs supplémentaires (absents du CSV) — utilisés par
                    # compute_conflict_resolution(), ignorés des autres.
                    "plan_name":           rec.get("plan_name", ""),
                    "consecutive_conflict": rec.get("consecutive_conflict", 0),
                })
    return results


def load_results(results_file: str = RESULTS_FILE) -> List[dict]:
    """Charge les résultats depuis le CSV."""
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
                "steps":          int(row.get("steps", 1)),
                "U_PE":           float(row.get("U_planner", 0)),
                "U_V":            float(row.get("U_verifier", 0)),
                "accepted":       row.get("accepted", "False") == "True",
                "improvement":    row.get("improvement", ""),
                "total_attempts": int(row.get("total_attempts", 1)),
            })
    return results


def compute_regret(results: List[dict]) -> Dict:
    """Regret cumulatif de PE par tâche."""
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    regret_data = {}
    for task_id, runs in by_task.items():
        U_optimal = max(r["U_PE"] for r in runs) if runs else 1.0
        regrets   = []
        cumulative = 0.0
        for i, r in enumerate(runs, 1):
            regret_t   = U_optimal - r["U_PE"]
            cumulative += regret_t
            regrets.append({
                "run":        i,
                "U_PE":       r["U_PE"],
                "regret_t":   round(regret_t, 4),
                "regret_cum": round(cumulative / i, 4),
                "strategy":   r["strategy"],
                "accepted":   r["accepted"],
            })
        slope = _compute_slope([r["regret_cum"] for r in regrets])
        regret_data[task_id] = {
            "runs":     regrets,
            "slope":    round(slope, 4),
            "learning": "YES" if slope < -0.02 else "NO" if slope > 0.02 else "STABLE",
        }
    return regret_data


def compute_strategy_evolution(results: List[dict]) -> Dict:
    """Analyse l'évolution des stratégies de PE."""
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
    """Détecte les conflits PE vs V."""
    conflicts = []
    for r in results:
        conflict_type = None
        if r["U_PE"] >= 0.4 and not r["accepted"]:
            conflict_type = "PE_satisfied_V_rejects"
        elif r["accepted"] and r["U_PE"] < 0.4:
            conflict_type = "V_accepts_PE_unsatisfied"
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
        "conflict_rate":   round(n_conflict / n_total, 3) if n_total > 0 else 0,
        "total_runs":      n_total,
        "total_conflicts": n_conflict,
        "conflicts":       conflicts,
    }


def detect_nash_equilibrium(results: List[dict]) -> Dict:
    """Détection heuristique d'un équilibre de Nash."""
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    nash_data = {}
    for task_id, runs in by_task.items():
        if len(runs) < 3:
            nash_data[task_id] = {"status": "NOT ENOUGH DATA"}
            continue
        recent            = runs[-3:]
        strategies_stable = len(set(r["strategy"] for r in recent)) == 1
        all_accepted      = all(r["accepted"] for r in recent)
        U_PE_stable       = max(r["U_PE"] for r in recent) - min(r["U_PE"] for r in recent) < 0.1
        U_V_stable        = max(r["U_V"]  for r in recent) - min(r["U_V"]  for r in recent) < 0.1
        at_equilibrium    = strategies_stable and all_accepted and U_PE_stable and U_V_stable

        nash_data[task_id] = {
            "at_equilibrium":       at_equilibrium,
            "strategy_stable":      strategies_stable,
            "all_accepted":         all_accepted,
            "U_PE_stable":          U_PE_stable,
            "U_V_stable":           U_V_stable,
            "equilibrium_strategy": recent[-1]["strategy"] if at_equilibrium else None,
            "equilibrium_U_PE":     round(sum(r["U_PE"] for r in recent) / 3, 3),
            "equilibrium_U_V":      round(sum(r["U_V"]  for r in recent) / 3, 3),
        }
    return nash_data


def compute_conflict_resolution(results: List[dict]) -> Dict:
    """
    Qualité de résolution de désaccord PE/V (V2.1, changement #6).

    Le plan `nash_compromise` (plan_library.py, trigger_desire=resolve_conflict)
    se déclenche EN DIRECT pendant la délibération quand consecutive_conflict
    >= 2 (désaccord persistant : l'exécuteur se croit performant, le
    vérificateur rejette). Ici, mesure POST-HOC si l'intervention a
    effectivement cassé la série : consecutive_conflict retombé à 0 sur CETTE
    MÊME tentative. Nécessite les traces JSONL par tentative
    (core/audit_trace.py) — absent du CSV historique (une ligne par tâche),
    donc `n_interventions` sera 0 sur des résultats chargés depuis le CSV.
    """
    interventions = [r for r in results if r.get("plan_name") == "nash_compromise"]
    if not interventions:
        return {"n_interventions": 0, "resolution_rate": None, "interventions": []}
    resolved = [r for r in interventions if r.get("consecutive_conflict", 1) == 0]
    return {
        "n_interventions":  len(interventions),
        "n_resolved":       len(resolved),
        "resolution_rate":  round(len(resolved) / len(interventions), 3),
        "interventions": [
            {
                "task_id":  r["task_id"],
                "attempt":  r["attempt"],
                "resolved": r.get("consecutive_conflict", 1) == 0,
            }
            for r in interventions
        ],
    }


def full_analysis(results_file: str = RESULTS_FILE, prefer_traces: bool = True,
                  trace_dir: str = None) -> dict:
    """
    Analyse complète du jeu. Si des traces par tentative existent
    (core/audit_trace.py, V2.1 #1), elles sont utilisées en priorité — bien
    plus de granularité que le CSV (une ligne par tâche) — sinon fallback sur
    le CSV historique, pour rester compatible avec les runs antérieurs à
    l'introduction des traces.

    `trace_dir` (opt-in) pointe vers un répertoire de traces isolé (voir
    mas_experiment.py --isolated) au lieu du répertoire partagé par défaut —
    nécessaire pour analyser UN run isolé précis sans mélanger son historique
    avec celui d'autres runs. None (défaut) = comportement inchangé.
    """
    data_source = "csv"
    results = load_trace_results(trace_dir) if prefer_traces else []
    if results:
        data_source = "traces"
    else:
        results = load_results(results_file)

    if not results:
        return {"error": "No results available"}

    regret    = compute_regret(results)
    evolution = compute_strategy_evolution(results)
    conflicts = compute_conflicts(results)
    nash      = detect_nash_equilibrium(results)
    conflict_resolution = compute_conflict_resolution(results)

    n_runs    = len(results)
    n_success = sum(1 for r in results if r["reward"] >= 0.7)
    avg_U_PE  = sum(r["U_PE"] for r in results) / n_runs
    avg_U_V   = sum(r["U_V"]  for r in results) / n_runs

    return {
        "summary": {
            "data_source":   data_source,
            "total_runs":    n_runs,
            "success_rate":  round(n_success / n_runs, 3),
            "avg_U_PE":      round(avg_U_PE, 3),
            "avg_U_V":       round(avg_U_V, 3),
            "conflict_rate": conflicts["conflict_rate"],
        },
        "regret":              regret,
        "evolution":           evolution,
        "conflicts":           conflicts,
        "nash":                nash,
        "conflict_resolution": conflict_resolution,
    }


def print_analysis(results_file: str = RESULTS_FILE, trace_dir: str = None) -> None:
    """Affiche l'analyse GT dans le terminal. `trace_dir` — voir full_analysis()."""
    analysis = full_analysis(results_file, trace_dir=trace_dir)

    if "error" in analysis:
        print(f"\n  {analysis['error']}")
        return

    s = analysis["summary"]
    print(f"\n{'='*62}")
    print(f"  GAME-THEORETIC ANALYSIS — DeepJSONEval MAS")
    print(f"{'='*62}")
    print(f"  Data source    : {s.get('data_source', 'csv')}")
    print(f"  Total runs     : {s['total_runs']}")
    print(f"  Success rate   : {s['success_rate']:.1%}")
    print(f"  Avg U_PE       : {s['avg_U_PE']:.3f}")
    print(f"  Avg U_V        : {s['avg_U_V']:.3f}")
    print(f"  Conflict rate  : {s['conflict_rate']:.1%}")

    print(f"\n  REGRET (is PE learning?)")
    for task_id, data in list(analysis["regret"].items())[:10]:
        print(f"  {task_id} : slope={data['slope']:+.4f} → {data['learning']}")

    print(f"\n  CONFLICTS PE vs V")
    c = analysis["conflicts"]
    print(f"  Total : {c['total_conflicts']}/{c['total_runs']} ({c['conflict_rate']:.1%})")
    for conf in c["conflicts"][:5]:
        print(f"  [{conf['conflict_type']}] "
              f"task={conf['task_id']} "
              f"U_PE={conf['U_PE']:.3f} U_V={conf['U_V']:.3f}")

    print(f"\n  NASH EQUILIBRIUM")
    for task_id, data in list(analysis["nash"].items())[:10]:
        if data.get("at_equilibrium"):
            print(f"  {task_id} : EQUILIBRIUM REACHED "
                  f"(U_PE={data['equilibrium_U_PE']:.3f}, "
                  f"U_V={data['equilibrium_U_V']:.3f})")
        else:
            print(f"  {task_id} : not yet ({data.get('status', 'in progress')})")

    cr = analysis.get("conflict_resolution", {})
    print(f"\n  CONFLICT RESOLUTION (nash_compromise interventions)")
    if cr.get("n_interventions"):
        print(f"  {cr['n_resolved']}/{cr['n_interventions']} resolved "
              f"({cr['resolution_rate']:.1%})")
    else:
        print(f"  No intervention recorded (needs per-attempt traces — see core/audit_trace.py)")

    print(f"\n{'='*62}")

    # Sortie nommée d'après le répertoire de traces analysé — évite d'écraser
    # gt_analysis.json partagé quand trace_dir pointe vers un run isolé.
    out_name = "gt_analysis.json"
    if trace_dir:
        out_name = f"gt_analysis_{os.path.basename(os.path.normpath(trace_dir))}.json"
    out = os.path.join(RESULTS_DIR, out_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"  JSON GT : {out}")


def _compute_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2
    ym = sum(values) / n
    num = sum((i - xm) * (v - ym) for i, v in enumerate(values))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0
