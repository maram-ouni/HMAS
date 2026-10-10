"""
experiment_runner.py — Orchestrateur des 6 architectures AgI-SDK pour DeepJSONEval

Lance chaque architecture séquentiellement, collecte les résultats et affiche
un tableau comparatif multi-architectures.

Usage:
    python mas_project/experiments/experiment_runner.py --arch all --max-samples 200 --shots 5
    python mas_project/experiments/experiment_runner.py --arch react --max-samples 200 --shots 5
    python mas_project/experiments/experiment_runner.py --skip-run  # compare les fichiers existants
"""
from __future__ import annotations

import argparse, glob, json, os, pathlib, sys, time

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_MAS_DIR      = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _MAS_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from config import LLM_MODEL, RESULTS_DIR
from data.dataset_loader import load_dataset_hf, iter_tasks

import react_baseline
import reflexion_baseline
import frering_bdi_baseline
import chatbdi_baseline
import hierarchical_bdi_baseline
import hri_bdi_baseline

ARCHITECTURES = {
    "react":            (react_baseline,           "ReAct (Thought+Action)"),
    "reflexion":        (reflexion_baseline,        "Reflexion (verbal reinforcement)"),
    "frering_bdi":      (frering_bdi_baseline,      "Frering BDI (ablation, binary feedback)"),
    "chatbdi":          (chatbdi_baseline,           "ChatBDI (LLM planner, AAMAS 2025)"),
    "hierarchical_bdi": (hierarchical_bdi_baseline,  "Hierarchical BDI (Harada & Kano 2025)"),
    "hri_bdi":          (hri_bdi_baseline,           "HRI BDI Dual-LLM (Mind State)"),
}


def _latest_results_file(arch_name: str) -> str | None:
    pattern = str(pathlib.Path(RESULTS_DIR) / f"agisdk_{arch_name}_results_*.json")
    files   = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    n_total    = len(results)
    n_accepted = sum(1 for r in results if r.get("accepted"))
    avg_uv     = sum(r.get("U_V", 0) for r in results) / n_total
    avg_shots  = sum(r.get("n_shots_used", 1) for r in results) / n_total

    # Par catégorie
    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r.get("category", "?"), []).append(r)
    cat_rates = {cat: sum(1 for x in rs if x.get("accepted")) / len(rs)
                 for cat, rs in cats.items()}

    return {
        "n_total":    n_total,
        "n_accepted": n_accepted,
        "success_rate": round(100 * n_accepted / n_total, 1),
        "avg_U_V":    round(avg_uv, 3),
        "avg_shots":  round(avg_shots, 2),
        "by_category": {cat: round(100 * rate, 1) for cat, rate in sorted(cat_rates.items())},
    }


def print_comparison_table(all_metrics: dict[str, dict]) -> None:
    if not all_metrics:
        print("  No results to compare.")
        return

    # En-tête
    arch_names = list(all_metrics.keys())
    col_w      = 22
    sep        = "─" * (20 + col_w * len(arch_names))

    print(f"\n{'='*70}")
    print("  COMPARISON TABLE — DeepJSONEval AgI-SDK Architectures")
    print(f"{'='*70}")

    # Table 1: métriques globales
    print(f"\n  {'Metric':<20}" + "".join(f"{n:>{col_w}}" for n in arch_names))
    print(f"  {sep}")
    for metric, label in [
        ("success_rate", "Success Rate (%)"),
        ("avg_U_V",      "Avg U_V"),
        ("avg_shots",    "Avg Shots Used"),
        ("n_accepted",   "Tasks Accepted"),
    ]:
        row = f"  {label:<20}"
        for name in arch_names:
            val = all_metrics[name].get(metric, "-")
            row += f"{str(val):>{col_w}}"
        print(row)

    # Table 2: par catégorie
    all_cats = sorted({cat for m in all_metrics.values()
                       for cat in m.get("by_category", {})})
    if all_cats:
        print(f"\n  {'Category':20}" + "".join(f"{n:>{col_w}}" for n in arch_names))
        print(f"  {sep}")
        for cat in all_cats:
            row = f"  {cat:<20}"
            for name in arch_names:
                val = all_metrics[name].get("by_category", {}).get(cat, "-")
                row += f"{str(val):>{col_w}}"
            print(row)

    print(f"\n{'='*70}\n")


def run_architecture(name: str, module, tasks: list, shots: int) -> dict:
    print(f"\n{'─'*60}")
    print(f"  RUNNING: {name}")
    print(f"{'─'*60}")
    t0      = time.perf_counter()
    results = module.run(tasks, shots)
    elapsed = time.perf_counter() - t0

    metrics = compute_metrics(results)
    print(f"\n  [{name}] Done — {metrics['success_rate']}% accepted "
          f"| avg U_V={metrics['avg_U_V']} | {elapsed:.0f}s\n")
    return metrics


def main():
    p = argparse.ArgumentParser(description="DeepJSONEval — AgI-SDK experiment runner")
    p.add_argument("--arch",        type=str, default="all",
                   choices=list(ARCHITECTURES.keys()) + ["all"],
                   help="Architecture to run (default: all)")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--shots",       type=int, default=10)
    p.add_argument("--category",    type=str, default=None)
    p.add_argument("--skip-run",    action="store_true",
                   help="Skip execution — compare latest saved result files only")
    args = p.parse_args()

    selected = (list(ARCHITECTURES.keys())
                if args.arch == "all"
                else [args.arch])

    print(f"\n{'='*60}")
    print(f"  DeepJSONEval — AgI-SDK Architectures")
    print(f"  Model    : {LLM_MODEL}")
    print(f"  Archs    : {', '.join(selected)}")
    print(f"  Samples  : {args.max_samples} | Shots : {args.shots}")
    print(f"{'='*60}\n")

    all_metrics: dict[str, dict] = {}

    if args.skip_run:
        # Charger les derniers fichiers de résultats
        for name in selected:
            path = _latest_results_file(name)
            if path:
                data    = _load_results(path)
                metrics = compute_metrics(data.get("results", []))
                all_metrics[name] = metrics
                print(f"  Loaded {name}: {path}")
            else:
                print(f"  No results found for {name}")
    else:
        # Charger le dataset une seule fois
        print("  Loading dataset...")
        dataset = load_dataset_hf(max_samples=args.max_samples)
        tasks   = list(iter_tasks(dataset, category=args.category))
        print(f"  Tasks loaded: {len(tasks)}\n")

        for name in selected:
            module, _ = ARCHITECTURES[name]
            metrics   = run_architecture(name, module, tasks, args.shots)
            all_metrics[name] = metrics

    # Afficher le tableau comparatif
    print_comparison_table(all_metrics)


if __name__ == "__main__":
    main()
