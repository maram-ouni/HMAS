"""
train_verifier_weights.py — Diagnostic : ajuste les poids U_V du Verifier
par régression logistique sur les traces réelles déjà collectées (aucun
run, aucune dépendance Groq), et compare au prior statique.

Voir core/verifier_learned_weights.py pour la méthode (entraînement
restreint aux lignes reward<1.0, cible = "la tâche a-t-elle fini par
réussir" — non-circulaire, success_v jamais appris).

Usage :
    python mas_project_v2/experiments/train_verifier_weights.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAS  = os.path.dirname(_HERE)
sys.path.insert(0, _MAS)

from core.payoff_calculator import WEIGHTS_V
from core.verifier_learned_weights import (
    compute_learned_weights, _load_training_examples,
    MIN_EXAMPLES_PER_CLASS,
)

TRACE_DIR = os.path.join(_MAS, "results", "traces")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    examples = _load_training_examples(TRACE_DIR)
    n_pos = sum(1 for _, y in examples if y == 1)
    n_neg = sum(1 for _, y in examples if y == 0)

    print("=" * 72)
    print("  POIDS VERIFIER APPRIS — régression logistique sur traces réelles")
    print("=" * 72)
    print(f"  Répertoire de traces : {TRACE_DIR}")
    print(f"  Exemples d'entraînement : {len(examples)} lignes "
          f"(reward<1.0 uniquement)")
    print(f"    -> tâche ayant fini par réussir : {n_pos}")
    print(f"    -> tâche jamais réussie          : {n_neg}")
    print(f"  Seuil minimum requis par classe    : {MIN_EXAMPLES_PER_CLASS}")
    print()

    if n_pos < MIN_EXAMPLES_PER_CLASS or n_neg < MIN_EXAMPLES_PER_CLASS:
        print("  Pas assez de données des deux classes — dégénère vers le "
              "prior statique (comportement attendu, pas une erreur).")
        return

    learned = compute_learned_weights(TRACE_DIR, WEIGHTS_V)

    print(f"  {'Dimension':<14} {'Statique':>10} {'Appris':>10} {'Delta':>10}")
    print("  " + "─" * 46)
    for dim in ("safety", "specificity", "success_v", "relevance"):
        s = WEIGHTS_V[dim]
        l = learned[dim]
        print(f"  {dim:<14} {s:>10.4f} {l:>10.4f} {l - s:>+10.4f}")
    print()
    print(f"  (success_v reste fixé au prior — non appris, voir docstring "
          f"du module)")


if __name__ == "__main__":
    main()
