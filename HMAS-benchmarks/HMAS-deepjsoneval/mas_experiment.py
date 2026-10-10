"""
mas_experiment.py — Point d'entrée du benchmark DeepJSONEval (MAS Hybride)

Architecture :
  BDIPlannerThread (BDI classique)
    -> LLMExecutorThread (Groq — génération JSON)
    -> RuleVerifierThread (vérificateur déterministe)
    -> system (RESULT_ACCEPTED ou ERROR_REPORTED)

Usage :
    python mas_project_v2/mas_experiment.py --index 0
    python mas_project_v2/mas_experiment.py --indices 0 1 2 3 4
    python mas_project_v2/mas_experiment.py --category patient --max-samples 20
    python mas_project_v2/mas_experiment.py --max-samples 100 --shots 2
    python mas_project_v2/mas_experiment.py --analyze
    python mas_project_v2/mas_experiment.py --stats
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from core.mas_orchestrator import MASCoordinator
from core.strategy_learner import StrategyLearner
from core.gt_analyzer      import print_analysis
from config                import RESULTS_DIR, LLM_MODEL, LLM_BACKEND, LLM_BASE_URL


def parse_args():
    p = argparse.ArgumentParser(
        description="DeepJSONEval — Hybrid MAS (BDI + LLMExecutor + RuleVerifier)"
    )
    p.add_argument("--index",       type=int, default=None,
                   help="Single dataset row index")
    p.add_argument("--indices",     nargs="+", type=int,
                   help="Multiple dataset row indices")
    p.add_argument("--category",    type=str, default=None,
                   help="Run all tasks of a specific category")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max number of samples to evaluate")
    p.add_argument("--shots",       type=int, default=10,
                   help="Maximum shots per task (default: 10)")
    p.add_argument("--until-success", action="store_true",
                   help="Keep trying until success (within --shots limit)")
    p.add_argument("--output",      type=str, default=RESULTS_DIR)
    p.add_argument("--from-failed", type=str, default=None,
                   help="Fichier JSON MAS — rejoue HMAS uniquement sur les tâches échouées")
    p.add_argument("--dynamic-verifier", action="store_true",
                   help="Utilise RuleBasedVerifierAgentDynamic (poids adaptatifs "
                        "par dimension, recalculés à partir de l'historique de CE "
                        "job — shots de la tâche + tâches de la même catégorie, "
                        "jamais cross-job). Défaut : poids statiques (référence).")
    p.add_argument("--reliability-mode", choices=["frequentist", "bayesian"],
                   default="frequentist",
                   help="V2.1 #2 — mode de calcul de la fiabilité executor dans "
                        "WorldModel. 'frequentist' (défaut) = comportement "
                        "HISTORIQUE EXACT (successes/n, falaise dure à "
                        "min_samples=4) — préserve la comparabilité avec tous "
                        "les runs déjà mesurés. 'bayesian' = lissage "
                        "Beta-Bernoulli progressif. SEUL changement V2.1 pouvant "
                        "altérer la sélection de plan — à utiliser avec une "
                        "ablation appariée avant de citer un chiffre.")
    p.add_argument("--diagnostic-quality-band", action="store_true",
                   help="V2.1 #3, opt-in (défaut : désactivé, comportement "
                        "historique inchangé). Nuance la détection de plateau : "
                        "un plateau 'fair' (U_V 0.5-0.7) et STABLE n'escalade "
                        "plus vers escape_plateau, laissant le raffinement fin "
                        "continuer. N'affecte jamais quality/accepted.")
    p.add_argument("--isolated",    action="store_true",
                   help="Run comparatif isolé : WorldModel et StrategyLearner démarrent "
                        "vierges dans des fichiers dédiés à ce run, sans lire/écrire le "
                        "fichier partagé cumulatif (results/world_model.json, "
                        "results/learned_strategies.json). Nécessaire pour que deux runs "
                        "successifs soient des essais réellement indépendants — sans cette "
                        "option, l'historique d'un run (y compris un simple test de "
                        "validation) influence la sélection de plans du run suivant.")

    # Commandes utilitaires
    p.add_argument("--analyze",         action="store_true",
                   help="Game-theoretic analysis of results")
    p.add_argument("--trace-dir",       type=str, default=None,
                   help="Avec --analyze : analyse un répertoire de traces "
                        "isolé précis (voir --isolated) au lieu du répertoire "
                        "partagé par défaut.")
    p.add_argument("--stats",           action="store_true",
                   help="Show dataset statistics")
    p.add_argument("--show-strategies", action="store_true",
                   help="Show learned strategies")
    p.add_argument("--clear-strategies", action="store_true",
                   help="Clear learned strategies")

    return p.parse_args()


def print_banner(shots: int = 1, until_success: bool = False):
    mode = (f"until success (max {shots} shots)"
            if until_success else f"{shots} shot(s)")
    print(f"""
{'='*62}
  DeepJSONEval — Hybrid MAS
  BDIPlanner (classic) | LLMExecutor (Groq) | RuleVerifier
{'='*62}
  LLM           : {LLM_MODEL}
  Agents        : BDIPlanner | LLMExecutor | RuleVerifier
  Communication : queue.Queue thread-safe (blocking)
  Payoff        : U_PE (json_valid, schema, completeness)
                  U_V  (json_valid, field_accuracy, depth)
  Mode          : {mode}
{'='*62}""")


def main():
    args          = parse_args()
    shots         = args.shots
    until_success = args.until_success

    # ── Commandes utilitaires ──────────────────────────────

    if args.analyze:
        print_analysis(trace_dir=args.trace_dir)
        return

    if args.show_strategies:
        learner = StrategyLearner()
        print("\n=== LEARNED STRATEGIES ===")
        if not learner.strategies:
            print("  No strategies learned yet.")
        for task_type, strategies in learner.strategies.items():
            print(f"\n  {task_type} ({len(strategies)} runs):")
            print(f"    {learner.get_learning_summary(task_type)}")
            best = learner.get_best_strategy(task_type)
            if best:
                print(f"    Best (payoff={best.payoff:.3f}, reward={best.reward:.2f}):")
                for i, step in enumerate(best.steps[:3], 1):
                    print(f"      {i}. {step[:70]}")
        print()
        return

    if args.clear_strategies:
        learner = StrategyLearner()
        if os.path.exists(learner.strategies_file):
            os.remove(learner.strategies_file)
            print(f"  Strategies cleared: {learner.strategies_file}")
        else:
            print("  No strategies to clear.")
        return

    # ── Chargement du dataset ──────────────────────────────

    try:
        import sys as _sys
        _sys.path.insert(0, _PROJECT_ROOT)
        from data.dataset_loader import (
            load_dataset_hf, iter_tasks, get_task,
            get_categories, dataset_stats,
        )
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("  Loading dataset...")
    dataset = load_dataset_hf(max_samples=args.max_samples)

    if args.stats:
        stats = dataset_stats(dataset)
        print(f"\n=== DATASET STATISTICS ===")
        print(f"  Total samples : {stats['total']}")
        print(f"  Avg depth     : {stats['avg_depth']:.1f}")
        print(f"\n  Categories:")
        for cat, n in sorted(stats["categories"].items()):
            print(f"    {cat:<20} : {n}")
        print(f"\n  Depth distribution:")
        for depth, n in sorted(stats["depth_dist"].items()):
            print(f"    depth={depth} : {n} samples")
        return

    # ── Sélection des tâches ───────────────────────────────

    if args.from_failed:
        with open(args.from_failed) as f:
            prev = json.load(f)
        # Agréger par task_id : garder le meilleur résultat
        best: dict = {}
        for r in prev.get("results", []):
            tid = r["task_id"]
            if tid not in best or r.get("U_V", 0) > best[tid].get("U_V", 0):
                best[tid] = r
        failed_ids = {tid for tid, r in best.items() if not r.get("accepted")}
        all_tasks  = {t["id"]: t for t in iter_tasks(dataset)}
        tasks = [all_tasks[tid] for tid in failed_ids if tid in all_tasks]
        print(f"\n  [2e passe] {len(tasks)} tâches échouées à rejouer depuis {args.from_failed}")
    elif args.index is not None:
        tasks = [get_task(dataset, args.index)]
    elif args.indices:
        tasks = [get_task(dataset, i) for i in args.indices]
    else:
        tasks = list(iter_tasks(dataset, category=args.category))

    if not tasks:
        print(f"  No tasks to run (category={args.category}).")
        return

    # La clé Groq n'est requise que pour le backend Groq. Sous vLLM, le serveur
    # local est interrogé sans clé (VLLM_API_KEY=EMPTY).
    if LLM_BACKEND == "groq":
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            print("\n  ERROR: GROQ_API_KEY not defined in .env")
            sys.exit(1)
        print(f"  GROQ key  : {groq_key[:8]}...{groq_key[-4:]}")
    else:
        print(f"  Backend   : {LLM_BACKEND} @ {LLM_BASE_URL}")

    print_banner(shots, until_success)
    print(f"\n  Tasks     : {len(tasks)}")
    print(f"  Shots     : {shots}")

    world_model_path = strategies_path = trace_path = dynamic_weights_path = None
    interaction_metrics_path = None
    # Toujours calculé (V2.1 #8) : identifiant utilisé pour le snapshot de
    # provenance WorldModel, isolé ou non.
    #
    # BUG corrigé le 2026-07-13 : un timestamp seul (précision à la seconde)
    # n'est PAS unique entre deux jobs SLURM lancés pour une même ablation sur
    # le même nœud — leur démarrage vLLM ayant un profil très similaire, ils
    # peuvent atteindre ce calcul pile à la même seconde. Observé en conditions
    # réelles (jobs #894648/#894649, ablation frequentist vs bayesian à 200
    # tâches) : les deux runs avaient EXACTEMENT le même run_id, donc
    # partageaient en temps réel les mêmes fichiers world_model_isolated/
    # learned_strategies_isolated/traces_isolated/dynamic_weights_isolated —
    # contamination croisée qui invalide la comparaison. Fix : suffixe garanti
    # unique par job (SLURM_JOB_ID si présent, sinon PID du processus).
    _unique_suffix = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{_unique_suffix}"
    if args.isolated:
        ts = run_id
        pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        world_model_path     = str(pathlib.Path(RESULTS_DIR) / f"world_model_isolated_{ts}.json")
        strategies_path      = str(pathlib.Path(RESULTS_DIR) / f"learned_strategies_isolated_{ts}.json")
        trace_path           = str(pathlib.Path(RESULTS_DIR) / f"traces_isolated_{ts}")
        # V2.1 #5 : persistance des poids diagnostiques, isolée comme le reste
        # — jamais un fichier partagé par défaut (voir payoff_calculator_dynamic.py).
        dynamic_weights_path = str(pathlib.Path(RESULTS_DIR) / f"dynamic_weights_isolated_{ts}.json")
        # Mécanismes Verifier/décision portés depuis agisdk (V2 récent) —
        # même isolation par run que ci-dessus.
        interaction_metrics_path = str(pathlib.Path(RESULTS_DIR) / f"interaction_metrics_isolated_{ts}")
        # Marqueur machine-lisible du dernier run isolé — permet à un script
        # appelant (SLURM) de récupérer trace_path pour l'analyse GT après
        # coup (--analyze --trace-dir ...) sans avoir à reproduire le calcul
        # de run_id. Écrasé à chaque nouveau run isolé (dernier run gagnant).
        pathlib.Path(RESULTS_DIR, "last_isolated_run.json").write_text(
            json.dumps({
                "run_id": ts,
                "world_model_path": world_model_path,
                "strategies_path": strategies_path,
                "trace_path": trace_path,
                "dynamic_weights_path": dynamic_weights_path,
                "interaction_metrics_path": interaction_metrics_path,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"  Isolated  : run indépendant — WorldModel/StrategyLearner démarrent vierges")
        print(f"              {world_model_path}")
        print(f"              {strategies_path}")
        print(f"              {trace_path}")
        print(f"              {dynamic_weights_path}")
        print(f"              {interaction_metrics_path}")

    if args.reliability_mode == "bayesian":
        print(f"  Reliability: BAYESIAN (V2.1 #2 — opt-in, altère potentiellement "
              f"la sélection de plan)")
    if args.diagnostic_quality_band:
        print(f"  Diag. band : ACTIVÉE (V2.1 #3 — opt-in, nuance escape_plateau)")

    # ── Exécution ─────────────────────────────────────────

    verifier_cls = None
    if args.dynamic_verifier:
        from agents.rule_based_verifier_dynamic import RuleBasedVerifierAgentDynamic
        verifier_cls = RuleBasedVerifierAgentDynamic
        print(f"  Verifier  : DYNAMIC (poids adaptatifs, historique job-scopé)")

    t0             = time.perf_counter()
    all_results    = []
    coordinator    = MASCoordinator(world_model_path=world_model_path,
                                     strategies_path=strategies_path,
                                     trace_path=trace_path,
                                     verifier_cls=verifier_cls,
                                     run_id=run_id,
                                     reliability_mode=args.reliability_mode,
                                     use_diagnostic_band=args.diagnostic_quality_band,
                                     dynamic_weights_path=dynamic_weights_path,
                                     interaction_metrics_path=interaction_metrics_path)

    # Une tâche = UN appel run_task. La délibération BDI (jusqu'à `shots`
    # tentatives) se déroule DANS l'agent, dont la mémoire persiste sur toute
    # la tâche — plus de boucle de shots externe. run_task renvoie le résultat
    # terminal : succès (U_V >= 0.7) ou meilleure tentative si budget épuisé.
    for task in tasks:
        print(f"\n{'─'*62}")
        print(f"  Task {task['id']} — BDI internal deliberation (max {shots} attempt(s))")
        print(f"{'─'*62}")

        result = coordinator.run_task(task, max_attempts=shots)
        all_results.append(result)

        quality  = result.get("quality", "rejected")
        u_v      = result.get("U_V", 0.0)
        attempts = result.get("total_attempts", result.get("attempt", 1))
        label    = {"success": "SUCCESS (>=0.7)", "partial": "PARTIAL (0.4-0.7)",
                    "rejected": "REJECTED (<0.4)"}.get(quality, quality)
        print(f"\n  [{task['id']}] {label} — U_V={u_v:.3f} in {attempts} attempt(s)")

    # ── Sauvegarde ────────────────────────────────────────

    total_time = time.perf_counter() - t0
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    json_out = pathlib.Path(args.output) / f"mas_results_{ts}.json"
    json_out.write_text(
        json.dumps({
            "experiment":    "DeepJSONEval Hybrid MAS",
            "architecture":  "BDI + LLMExecutor (Groq) + RuleVerifier",
            "model":         LLM_MODEL,
            "shots":         shots,
            "total_tasks":   len(tasks),
            "total_time_s":  round(total_time, 2),
            "n_shots_total": len(all_results),
            "n_accepted":    sum(1 for r in all_results if r["accepted"]),
            "avg_U_V":       (
                sum(r["U_V"] for r in all_results) / len(all_results)
                if all_results else 0.0
            ),
            "results": all_results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    coordinator.print_summary(all_results)
    print(f"\n  Total time : {total_time:.1f}s")
    print(f"  JSON       : {json_out}")
    print(f"\n{'='*62}\n")


def _print_shots_summary(task_id: str, results: list) -> None:
    if len(results) <= 1:
        return
    print(f"\n  SHOTS SUMMARY — {task_id}")
    print(f"  {'Shot':>5} {'U_V':>6} {'Valid':>6} {'FieldAcc':>9} {'Quality':>18}")
    print("  " + "─" * 50)
    for i, r in enumerate(results, 1):
        q     = r.get("quality", "rejected")
        u_v   = r.get("U_V", 0.0)
        valid = "YES" if r.get("is_valid_json") else "NO"
        facc  = r.get("dim_field_accuracy", 0.0)
        label = {"success": "SUCCESS (>=0.7)", "partial": "PARTIAL (0.4-0.7)",
                 "rejected": "REJECTED (<0.4)"}.get(q, q)
        print(f"  {i:>5} {u_v:>6.3f} {valid:>6} {facc:>9.3f} {label:>18}")
    n_ok = sum(1 for r in results if r.get("accepted"))
    print(f"  {'─'*44}")
    print(f"  Accepted: {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
