"""
compare_all.py — Comparaison complète : 6 architectures AgI-SDK + HMAS + LLM Seul

Usage:
    python mas_project_v2/experiments/compare_all.py
    python mas_project_v2/experiments/compare_all.py --hmas  mas_project_v2/results/mas_results_*.json
    python mas_project_v2/experiments/compare_all.py --baseline mas_project_v2/results/baseline_results_*.json
"""
from __future__ import annotations

import argparse, glob, json, os, pathlib, sys

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
_RESULTS_DIR  = os.path.join(_PROJECT_ROOT, "mas_project_v2", "results")

# Fichiers de référence (Mistral-7B, 2026-07-01)
_DEFAULT_HMAS     = "mas_results_20260701_132626.json"
_DEFAULT_BASELINE = "baseline_results_20260701_140256.json"

# Noms d'affichage
_ARCH_LABELS = {
    "react":            "ReAct",
    "reflexion":        "Reflexion",
    "frering_bdi":      "Frering BDI",
    "chatbdi":          "ChatBDI",
    "hierarchical_bdi": "Hierarchical BDI",
    "hri_bdi":          "HRI BDI Dual-LLM",
    "hmas":             "HMAS (MAS Hybride) ★",
    "baseline":         "LLM Seul (Mistral-7B)",
}


def _latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _compute(results: list[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    acc = sum(1 for r in results if r.get("accepted"))
    avg_uv  = sum(r.get("U_V",  0) for r in results) / n
    avg_upe = sum(r.get("U_PE", 0) for r in results) / n
    avg_shots = sum(r.get("n_shots_used", 1) for r in results) / n

    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r.get("category", "?"), []).append(r)
    by_cat = {c: round(100 * sum(1 for x in xs if x.get("accepted")) / len(xs), 1)
              for c, xs in cats.items()}

    return {
        "n_total":      n,
        "n_accepted":   acc,
        "success_rate": round(100 * acc / n, 1),
        "avg_U_V":      round(avg_uv,  3),
        "avg_U_PE":     round(avg_upe, 3),
        "avg_shots":    round(avg_shots, 2),
        "by_category":  by_cat,
    }


def _load(path: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    results = d.get("results", d) if isinstance(d, dict) else d
    # Dédupliquer par task_id : garder accepted > non, puis U_V max
    best: dict = {}
    for r in results:
        tid = r.get("task_id", id(r))
        prev = best.get(tid)
        if prev is None:
            best[tid] = r
        elif (r.get("accepted") and not prev.get("accepted")) or \
             (r.get("accepted") == prev.get("accepted") and r.get("U_V", 0) > prev.get("U_V", 0)):
            best[tid] = r
    return list(best.values())


def _banner(title: str, w: int = 70) -> None:
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")


def print_table(all_metrics: dict[str, dict], title: str = "COMPARAISON COMPLÈTE") -> None:
    _banner(f"DeepJSONEval — {title}")

    names = list(all_metrics.keys())
    labels = [_ARCH_LABELS.get(n, n) for n in names]
    col_w = max(24, max(len(l) for l in labels) + 2)
    sep = "─" * (22 + col_w * len(names))

    # En-tête colonnes
    print(f"\n  {'Métrique':<22}" + "".join(f"{l:>{col_w}}" for l in labels))
    print(f"  {sep}")

    for metric, label in [
        ("success_rate", "Success Rate (%)"),
        ("avg_U_V",      "Avg U_V"),
        ("avg_U_PE",     "Avg U_PE"),
        ("avg_shots",    "Avg Shots"),
        ("n_accepted",   "Tâches acceptées"),
        ("n_total",      "Tâches totales"),
    ]:
        row = f"  {label:<22}"
        for name in names:
            val = all_metrics[name].get(metric, "-")
            row += f"{str(val):>{col_w}}"
        print(row)

    # Par catégorie
    all_cats = sorted({c for m in all_metrics.values() for c in m.get("by_category", {})})
    if all_cats:
        print(f"\n  {'Catégorie':<22}" + "".join(f"{l:>{col_w}}" for l in labels))
        print(f"  {sep}")
        for cat in all_cats:
            row = f"  {cat:<22}"
            for name in names:
                val = all_metrics[name].get("by_category", {}).get(cat, "-")
                row += f"{str(val):>{col_w}}"
            print(row)

    print(f"\n{'='*70}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hmas",     type=str, default=None,
                   help="Chemin vers le fichier HMAS (défaut: mas_results_20260701_132626.json)")
    p.add_argument("--baseline", type=str, default=None,
                   help="Chemin vers le fichier baseline (défaut: baseline_results_20260701_140256.json)")
    p.add_argument("--no-hmas",     action="store_true", help="Exclure HMAS du tableau")
    p.add_argument("--no-baseline", action="store_true", help="Exclure LLM Seul du tableau")
    args = p.parse_args()

    all_metrics: dict[str, dict] = {}

    # ── 1. 6 architectures AgI-SDK ────────────────────────────────────────────
    for arch in ["react", "reflexion", "frering_bdi", "chatbdi", "hierarchical_bdi", "hri_bdi"]:
        pattern = str(pathlib.Path(_RESULTS_DIR) / f"agisdk_{arch}_results_*.json")
        path = _latest(pattern)
        if path:
            results = _load(path)
            m = _compute(results)
            all_metrics[arch] = m
            print(f"  [OK] {_ARCH_LABELS.get(arch, arch):<30} {path.split('/')[-1]}")
        else:
            print(f"  [--] {_ARCH_LABELS.get(arch, arch):<30} (pas de résultats)")

    # ── 2. LLM Seul (baseline) ────────────────────────────────────────────────
    # Priorité au fichier le PLUS RÉCENT (glob) — le nom figé _DEFAULT_BASELINE
    # (2026-07-01) n'est qu'un filet de sécurité si aucun run n'a encore été
    # fait. Avant ce fix, comme ce fichier existe déjà sur disque (copié lors
    # de la création de v2), il était TOUJOURS choisi en priorité, masquant
    # silencieusement tous les runs plus récents.
    if not args.no_baseline:
        bpath = (args.baseline
                 or _latest(str(pathlib.Path(_RESULTS_DIR) / "baseline_results_*.json"))
                 or str(pathlib.Path(_RESULTS_DIR) / _DEFAULT_BASELINE))
        if bpath and os.path.exists(bpath):
            all_metrics["baseline"] = _compute(_load(bpath))
            print(f"  [OK] {'LLM Seul':<30} {os.path.basename(bpath)}")
        else:
            print(f"  [--] LLM Seul — fichier introuvable")

    # ── 3. HMAS (MAS Hybride) ─────────────────────────────────────────────────
    if not args.no_hmas:
        hpath = (args.hmas
                 or _latest(str(pathlib.Path(_RESULTS_DIR) / "mas_results_*.json"))
                 or str(pathlib.Path(_RESULTS_DIR) / _DEFAULT_HMAS))
        if hpath and os.path.exists(hpath):
            all_metrics["hmas"] = _compute(_load(hpath))
            print(f"  [OK] {'HMAS (MAS Hybride)':<30} {os.path.basename(hpath)}")
        else:
            print(f"  [--] HMAS — fichier introuvable")

    if not all_metrics:
        print("\n  Aucun résultat trouvé.\n")
        return

    print_table(all_metrics)

    # Delta HMAS vs meilleur AgI-SDK
    agisdk_names = [n for n in all_metrics if n not in ("hmas", "baseline")]
    if agisdk_names and "hmas" in all_metrics:
        best = max(agisdk_names, key=lambda n: all_metrics[n].get("success_rate", 0))
        delta = all_metrics["hmas"]["success_rate"] - all_metrics[best]["success_rate"]
        sign = "+" if delta >= 0 else ""
        print(f"  Delta HMAS vs meilleur AgI-SDK ({_ARCH_LABELS.get(best, best)}): {sign}{delta:.1f}%\n")


if __name__ == "__main__":
    main()
