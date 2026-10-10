"""
contract_net_demo.py — Démo du Contract-Net multi-modèles, pilotée par les
VRAIES données de comparaison de modèles (aucun run, aucune dépendance Groq).

Chaque modèle enchérit sa fiabilité (taux de succès lissé) par catégorie ;
le planner attribue la tâche au plus offrant. Montre concrètement que router
chaque catégorie vers le modèle empiriquement le plus fiable exploite
directement la comparaison de modèles.

Usage :
    python mas_project_v2/experiments/contract_net_demo.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAS  = os.path.dirname(_HERE)
sys.path.insert(0, _MAS)

from core.contract_net import ModelReliability, run_auction

# Sources = fichiers de résultats Hybrid par modèle (données valides seulement ;
# qwen3-32b/hybrid et gpt-oss-20b/react sont écartés — invalidés par le blocage Groq).
SOURCES = {
    "llama-3.3-70b":   ["experiments/results/hybrid_results.json"],
    "llama-4-scout-17b": ["experiments/results/hybrid_results_llama4_scout_17b.json"],
    "llama-3.1-8b":    ["experiments/results/hybrid_results_llama31_8b.json"],
    "gpt-oss-20b":     ["experiments/results/hybrid_results_gptoss_20b.json"],
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rel = ModelReliability(SOURCES)
    models = rel.models()

    print("=" * 84)
    print("  CONTRACT-NET — attribution par catégorie (enchère = fiabilité lissée)")
    print("=" * 84)
    print(f"  {'Catégorie':<14} " + " ".join(f"{m.split('-')[-1]:>7}" for m in models)
          + f"   {'→ ATTRIBUÉ À':>16}")
    print("  " + "─" * 80)

    awards = {}
    for cat in rel.categories():
        res = run_auction(cat, rel, models)
        awards[cat] = res["winner"]
        cells = []
        for m in models:
            succ, n = res["evidence"][m]
            cells.append(f"{res['bids'][m]:>7.2f}")
        winner_short = res["winner"]
        print(f"  {cat:<14} " + " ".join(cells) + f"   {winner_short:>16}")

    print("  " + "─" * 80)
    print("  (chaque cellule = enchère du modèle pour la catégorie ; plus haut = plus fiable)\n")

    # Combien de catégories chaque modèle remporte
    from collections import Counter
    tally = Counter(awards.values())
    print("  Répartition des attributions :")
    for m in models:
        cats_won = sorted(c for c, w in awards.items() if w == m)
        print(f"    {m:<18} {tally.get(m,0):>2} catégorie(s)  {cats_won}")
    print()
    print("  => Un routeur Contract-Net n'utiliserait PAS un seul modèle partout : il")
    print("     affecterait chaque catégorie au modèle qui y est le plus fiable. C'est une")
    print("     vraie allocation négociée, pilotée par les données de la comparaison.")


if __name__ == "__main__":
    main()
