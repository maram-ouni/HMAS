#!/usr/bin/env python3
"""
Génère un tableau comparatif Baseline LLM vs MAS BDI
à partir des fichiers de résultats les plus récents.
"""

import json
import glob
import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_latest(pattern: str) -> dict:
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern}")
    path = files[-1]
    print(f"  Loading: {path}")
    with open(path) as f:
        return json.load(f)


def aggregate_mas(results: list) -> dict:
    """Groupe les tentatives par task_id et retient le meilleur résultat."""
    tasks: dict = {}
    for r in results:
        tid = r["task_id"]
        if tid not in tasks:
            tasks[tid] = r
        else:
            prev = tasks[tid]
            # Priorité : accepted > meilleur U_V
            if r.get("accepted") and not prev.get("accepted"):
                tasks[tid] = r
            elif r.get("accepted") == prev.get("accepted"):
                if r.get("U_V", 0) > prev.get("U_V", 0):
                    tasks[tid] = r
    return tasks


def stats_by_category(task_map: dict) -> dict:
    cats: dict = {}
    for r in task_map.values():
        c = r.get("category", "unknown")
        if c not in cats:
            cats[c] = {"total": 0, "accepted": 0, "u_v_sum": 0.0, "u_pe_sum": 0.0}
        cats[c]["total"] += 1
        if r.get("accepted"):
            cats[c]["accepted"] += 1
        cats[c]["u_v_sum"] += r.get("U_V", 0.0)
        cats[c]["u_pe_sum"] += r.get("U_PE", 0.0)
    return cats


def problems_from_mas(task_map: dict) -> list:
    problems = []
    for r in task_map.values():
        if not r.get("accepted"):
            err = r.get("error") or r.get("verdict") or "échec sans message"
            problems.append({
                "task_id": r["task_id"],
                "category": r.get("category", "?"),
                "attempts": r.get("total_attempts", r.get("attempt", "?")),
                "best_U_V": round(r.get("U_V", 0), 3),
                "reason": str(err)[:120],
            })
    return problems


def problems_from_baseline(results: list) -> list:
    problems = []
    for r in results:
        if not r.get("accepted"):
            problems.append({
                "task_id": r["task_id"],
                "category": r.get("category", "?"),
                "U_PE": round(r.get("U_PE", 0), 3),
                "U_V": round(r.get("U_V", 0), 3),
                "reason": "U_PE=0 (JSON invalide ou hors schéma)" if r.get("U_PE", 0) == 0 else "U_V < seuil",
            })
    return problems


def print_table(baseline_cats: dict, mas_cats: dict, all_cats: set):
    header = f"{'Catégorie':<16} {'Base-Acc%':>9} {'Base-UV':>8} {'MAS-Acc%':>9} {'MAS-UV':>8} {'MAS-UPE':>8} {'Δ Acc':>7}"
    print(header)
    print("─" * len(header))

    totals = {"b_total": 0, "b_acc": 0, "b_uv": 0, "m_total": 0, "m_acc": 0, "m_uv": 0, "m_upe": 0}

    for cat in sorted(all_cats):
        b = baseline_cats.get(cat, {})
        m = mas_cats.get(cat, {})

        b_total = b.get("total", 0)
        b_acc = b.get("accepted", 0)
        b_uv = b.get("u_v_sum", 0) / b_total if b_total else 0

        m_total = m.get("total", 0)
        m_acc = m.get("accepted", 0)
        m_uv = m.get("u_v_sum", 0) / m_total if m_total else 0
        m_upe = m.get("u_pe_sum", 0) / m_total if m_total else 0

        b_pct = 100 * b_acc / b_total if b_total else 0
        m_pct = 100 * m_acc / m_total if m_total else 0
        delta = m_pct - b_pct
        delta_str = f"{delta:+.1f}%"

        print(f"{cat:<16} {b_pct:>8.1f}% {b_uv:>8.3f} {m_pct:>8.1f}% {m_uv:>8.3f} {m_upe:>8.3f} {delta_str:>7}")

        totals["b_total"] += b_total; totals["b_acc"] += b_acc; totals["b_uv"] += b_uv * b_total
        totals["m_total"] += m_total; totals["m_acc"] += m_acc; totals["m_uv"] += m_uv * m_total; totals["m_upe"] += m_upe * m_total

    print("─" * len(header))
    bt, ba, mt, ma = totals["b_total"], totals["b_acc"], totals["m_total"], totals["m_acc"]
    b_pct_t = 100 * ba / bt if bt else 0
    m_pct_t = 100 * ma / mt if mt else 0
    b_uv_t = totals["b_uv"] / bt if bt else 0
    m_uv_t = totals["m_uv"] / mt if mt else 0
    m_upe_t = totals["m_upe"] / mt if mt else 0
    delta_t = m_pct_t - b_pct_t
    print(f"{'TOTAL':<16} {b_pct_t:>8.1f}% {b_uv_t:>8.3f} {m_pct_t:>8.1f}% {m_uv_t:>8.3f} {m_upe_t:>8.3f} {delta_t:>+7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Nombre de tâches (pour le titre)")
    parser.add_argument("--baseline", default=None, help="Fichier baseline JSON spécifique")
    parser.add_argument("--mas", default=None, help="Fichier MAS JSON spécifique")
    args = parser.parse_args()

    print("\n" + "═" * 70)
    n_str = f"{args.n} tâches" if args.n else "N tâches"
    print(f"  RAPPORT COMPARATIF — Baseline LLM vs MAS BDI ({n_str})")
    print("═" * 70)

    # ── Chargement ────────────────────────────────────────────────
    print("\n[1] Chargement des résultats...")
    if args.baseline:
        with open(args.baseline) as f:
            b_data = json.load(f)
    else:
        b_data = load_latest("baseline_results_*.json")

    if args.mas:
        with open(args.mas) as f:
            m_data = json.load(f)
    else:
        m_data = load_latest("mas_results_*.json")

    b_results = b_data.get("results", [])
    m_task_map = aggregate_mas(m_data.get("results", []))

    # ── Résumé global ─────────────────────────────────────────────
    print("\n[2] RÉSUMÉ GLOBAL")
    print("─" * 50)
    b_total = b_data.get("total", len(b_results))
    b_acc   = b_data.get("accepted", sum(1 for r in b_results if r.get("accepted")))
    b_uv    = b_data.get("avg_uv", 0)
    b_time  = b_data.get("total_time", 0)

    m_total = m_data.get("total_tasks", len(m_task_map))
    m_acc   = sum(1 for r in m_task_map.values() if r.get("accepted"))
    m_shots = m_data.get("n_shots_total", 0)
    m_uv    = m_data.get("avg_U_V", 0)
    m_time  = m_data.get("total_time_s", 0)

    print(f"  {'':20} {'Baseline LLM':>15} {'MAS BDI':>15}")
    print(f"  {'Tâches traitées':20} {b_total:>15} {m_total:>15}")
    print(f"  {'Tâches acceptées':20} {b_acc:>15} {m_acc:>15}")
    print(f"  {'Taux acceptation':20} {100*b_acc/b_total if b_total else 0:>14.1f}% {100*m_acc/m_total if m_total else 0:>14.1f}%")
    print(f"  {'Avg U_V':20} {b_uv:>15.3f} {m_uv:>15.3f}")
    if m_shots:
        avg_shots = m_shots / m_total if m_total else 0
        print(f"  {'Shots totaux':20} {'1 (1-shot)':>15} {m_shots:>14} ({avg_shots:.1f}/tâche)")
    print(f"  {'Temps total (s)':20} {b_time:>15.1f} {m_time:>15.1f}")

    # ── Tableau par catégorie ─────────────────────────────────────
    print("\n[3] TABLEAU PAR CATÉGORIE")
    print("─" * 70)
    b_cats = stats_by_category({r["task_id"]: r for r in b_results})
    m_cats = stats_by_category(m_task_map)
    all_cats = set(b_cats.keys()) | set(m_cats.keys())
    print_table(b_cats, m_cats, all_cats)

    # ── Problèmes rencontrés ───────────────────────────────────────
    print("\n[4] PROBLÈMES RENCONTRÉS")
    print("─" * 70)

    b_problems = problems_from_baseline(b_results)
    m_problems = problems_from_mas(m_task_map)

    print(f"\n  [Baseline] {len(b_problems)} tâches échouées :")
    b_by_cat: dict = {}
    for p in b_problems:
        b_by_cat.setdefault(p["category"], []).append(p)
    for cat, ps in sorted(b_by_cat.items()):
        print(f"    {cat} ({len(ps)} échecs) : {ps[0]['reason']}")

    print(f"\n  [MAS BDI] {len(m_problems)} tâches échouées :")
    m_by_cat: dict = {}
    for p in m_problems:
        m_by_cat.setdefault(p["category"], []).append(p)
    for cat, ps in sorted(m_by_cat.items()):
        avg_uv = sum(x["best_U_V"] for x in ps) / len(ps)
        print(f"    {cat} ({len(ps)} échecs) : avg U_V={avg_uv:.3f} — {ps[0]['reason'][:80]}")

    # ── Problèmes systémiques ──────────────────────────────────────
    print("\n  [Problèmes systémiques]")

    # Tâches échouées dans les deux systèmes
    b_failed_ids = {r["task_id"] for r in b_results if not r.get("accepted")}
    m_failed_ids = {tid for tid, r in m_task_map.items() if not r.get("accepted")}
    both_failed = b_failed_ids & m_failed_ids
    if both_failed:
        print(f"    - {len(both_failed)} tâches échouées par LES DEUX systèmes (difficultés intrinsèques)")
        by_cat: dict = {}
        for tid in both_failed:
            cat = m_task_map.get(tid, {}).get("category", "?")
            by_cat[cat] = by_cat.get(cat, 0) + 1
        print(f"      Répartition: {dict(sorted(by_cat.items(), key=lambda x: -x[1]))}")

    # Tâches MAS pire que baseline
    worse = []
    for r in b_results:
        tid = r["task_id"]
        if tid in m_task_map:
            b_uv_val = r.get("U_V", 0)
            m_uv_val = m_task_map[tid].get("U_V", 0)
            if b_uv_val > m_uv_val + 0.05:
                worse.append((tid, b_uv_val, m_uv_val))
    if worse:
        print(f"    - {len(worse)} tâches où MAS est nettement moins bon que Baseline (Δ>0.05)")

    print("\n" + "═" * 70 + "\n")


if __name__ == "__main__":
    main()
