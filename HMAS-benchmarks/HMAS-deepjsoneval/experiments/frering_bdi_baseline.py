"""
frering_bdi_baseline.py — Frering BDI (ablation sans vérificateur détaillé), v2

Même principe que l'ablation Frering BDI de v1 : isoler la contribution du
feedback détaillé du RuleBasedVerifier. Le BDI ne reçoit qu'une croyance
binarisée (JSON valide/succès/rejet) au lieu des dimensions détaillées
(schema_compliance, field_accuracy, depth_match, failed_fields).

Contrairement au script v1 (standalone, boucle de shots externe), ceci
réutilise l'architecture v2 RÉELLE — mêmes agents (BDIPlannerAgent, corps
d'actions, désirs dynamiques, boucle de délibération interne, fix de
contexte), seul le planner/vérificateur est substitué par les variantes
"Frering" (agents/bdi_planner_agent_frering.py,
agents/rule_based_verifier_frering.py). Voir ARCHITECTURE_V2.md §2.

Le score RAPPORTÉ (U_V/U_PE) reste RÉEL — seule la croyance interne utilisée
par le raisonnement BDI est dégradée. C'est ce qui permet de mesurer
honnêtement l'effet de l'ablation, sans biaiser le score par la dégradation
elle-même (même principe que le script v1 original).

Usage :
    python mas_project_v2/experiments/frering_bdi_baseline.py --max-samples 200 --shots 8
    python mas_project_v2/experiments/frering_bdi_baseline.py --max-samples 10 --shots 3 --isolated
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_MAS_DIR      = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _MAS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from core.mas_orchestrator import MASCoordinator
from config                import RESULTS_DIR, LLM_MODEL, LLM_BACKEND, LLM_BASE_URL
from agents.bdi_planner_agent_frering   import BDIPlannerAgentFrering
from agents.rule_based_verifier_frering import RuleBasedVerifierAgentFrering

ARCH_NAME = "frering_bdi"


def parse_args():
    p = argparse.ArgumentParser(
        description="DeepJSONEval — Frering BDI (ablation, architecture v2)"
    )
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--shots",       type=int, default=10,
                   help="Budget de délibération interne (max_replans)")
    p.add_argument("--category",    type=str, default=None)
    p.add_argument("--isolated",    action="store_true",
                   help="WorldModel/StrategyLearner isolés (croyances vierges) — "
                        "voir mas_experiment.py --isolated")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*62}")
    print(f"  DeepJSONEval — Frering BDI (ablation, architecture v2)")
    print(f"  BDIPlannerAgentFrering | LLMExecutor | RuleBasedVerifierAgentFrering")
    print(f"{'='*62}")
    print(f"  LLM     : {LLM_MODEL} ({LLM_BACKEND})")
    print(f"  Shots   : {args.shots}  (budget de délibération interne)")
    print(f"{'='*62}\n")

    sys.path.insert(0, _PROJECT_ROOT)
    from data.dataset_loader import load_dataset_hf, iter_tasks

    dataset = load_dataset_hf(max_samples=args.max_samples)
    tasks   = list(iter_tasks(dataset, category=args.category))
    if not tasks:
        print(f"  Aucune tâche (category={args.category}).")
        return
    print(f"  Tasks   : {len(tasks)}\n")

    world_model_path = strategies_path = trace_path = None
    if args.isolated:
        ts = time.strftime("%Y%m%d_%H%M%S")
        pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        world_model_path = str(pathlib.Path(RESULTS_DIR) / f"world_model_isolated_frering_{ts}.json")
        strategies_path  = str(pathlib.Path(RESULTS_DIR) / f"learned_strategies_isolated_frering_{ts}.json")
        trace_path       = str(pathlib.Path(RESULTS_DIR) / f"traces_isolated_frering_{ts}")
        print(f"  Isolated : run indépendant — WorldModel/StrategyLearner vierges")
        print(f"             {world_model_path}\n")

    coordinator = MASCoordinator(
        world_model_path=world_model_path,
        strategies_path=strategies_path,
        trace_path=trace_path,
        planner_cls=BDIPlannerAgentFrering,
        verifier_cls=RuleBasedVerifierAgentFrering,
    )

    t0          = time.perf_counter()
    all_results = []
    for task in tasks:
        print(f"\n{'─'*62}")
        print(f"  Task {task['id']} — Frering BDI ablation (max {args.shots} attempt(s))")
        print(f"{'─'*62}")

        # task['id']/['schema']/etc. suivent le même format que mas_experiment.py
        t_task = {
            "id":                task["id"],
            "schema":             task["schema"],
            "text":               task["text"],
            "ground_truth_json":  task["ground_truth_json"],
            "category":           task["category"],
            "true_depth":         task["true_depth"],
        }
        result = coordinator.run_task(t_task, max_attempts=args.shots)
        all_results.append(result)

        u_v      = result.get("U_V", 0.0)
        quality  = result.get("quality", "rejected")
        attempts = result.get("total_attempts", result.get("attempt", 1))
        label = {"success": "SUCCESS (>=0.7)", "partial": "PARTIAL (0.4-0.7)",
                 "rejected": "REJECTED (<0.4)"}.get(quality, quality)
        print(f"\n  [{task['id']}] {label} — U_V réel={u_v:.3f} en {attempts} tentative(s)")

    elapsed  = time.perf_counter() - t0
    accepted = sum(1 for r in all_results if r.get("accepted"))
    avg_uv   = sum(r.get("U_V", 0.0) for r in all_results) / len(all_results) if all_results else 0.0

    print(f"\n{'='*62}")
    print(f"  SUMMARY — Frering BDI (ablation, v2)")
    print(f"  Accepted : {accepted}/{len(all_results)} ({100*accepted/len(all_results):.1f}%)")
    print(f"  Avg U_V  : {avg_uv:.3f}")
    print(f"  Time     : {elapsed:.1f}s")
    print(f"{'='*62}\n")

    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = pathlib.Path(RESULTS_DIR) / f"agisdk_{ARCH_NAME}_results_{ts}.json"
    path.write_text(json.dumps({
        "experiment":    f"DeepJSONEval {ARCH_NAME} (ablation, architecture v2)",
        "architecture":  "BDIPlannerAgentFrering + LLMExecutorAgent + RuleBasedVerifierAgentFrering "
                          "(mêmes agents v2, croyance binarisée transmise au planner)",
        "model":         LLM_MODEL,
        "shots":         args.shots,
        "isolated":      args.isolated,
        "total_tasks":   len(all_results),
        "total_time_s":  round(elapsed, 2),
        "n_accepted":    accepted,
        "avg_U_V":       round(avg_uv, 4),
        "results":       all_results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON : {path}\n")


if __name__ == "__main__":
    main()
