"""
verifier_calibration_report.py — measures whether the Verifier's U_V score
is a TRUSTWORTHY predictor of correctness, using (U_V, reward) pairs already
collected in result files.

Prints:
  - the reliability diagram (P(correct | U_V bucket)) — should rise;
  - precision/recall of the CURRENT static cutoff (config.V_THRESHOLD=0.4);
  - the calibrated threshold for a target precision (default 0.8): the
    lowest U_V cutoff whose accepted answers are correct >= target of the
    time — turning "accepted" into a signal with a guaranteed precision.

No LLM calls, no benchmark runs.

Usage:
    python mas_project_v2/experiments/verifier_calibration_report.py [results.json ...]
    (defaults to the no-early-stop run if no file is given)
"""
from __future__ import annotations

import glob
import json
import os
import sys

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_DIR = os.path.dirname(_EXP_DIR)
_ROOT    = os.path.dirname(_MAS_DIR)
for p in (_ROOT, _MAS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.verifier_calibration import (
    reliability_bins, precision_recall_at, threshold_for_precision,
)
from core.payoff_calculator import V_THRESHOLD

_RESULTS = os.path.join(_EXP_DIR, "results")
_TARGET_PRECISION = 0.8


def _load_pairs(paths: list) -> list:
    pairs = []
    for path in paths:
        try:
            rows = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for x in rows:
            if "U_V" in x and "reward" in x:
                pairs.append((float(x["U_V"]), float(x["reward"])))
    return pairs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    paths = sys.argv[1:]
    if not paths:
        default = os.path.join(_RESULTS, "hmas_21tasks_v2_results_noearly_vote.json")
        paths = [default] if os.path.exists(default) else glob.glob(
            os.path.join(_RESULTS, "hmas_21tasks_v2_results_*.json"))

    pairs = _load_pairs(paths)
    if not pairs:
        print("Aucune paire (U_V, reward) trouvée.")
        return

    n = len(pairs)
    n_correct = sum(1 for _, r in pairs if r >= 1.0)
    print(f"\n{n} attempts | {n_correct} corrects ({n_correct/n:.0%} base rate)")
    print(f"Fichiers : {', '.join(os.path.basename(p) for p in paths)}")

    print("\n── Reliability diagram : P(correct | U_V bucket) ──")
    for b in reliability_bins(pairs, n_bins=5):
        bar = "█" * round(b.p_correct * 20)
        print(f"  U_V [{b.lo:.1f}, {b.hi:.1f}) : "
              f"{b.p_correct:5.0%} correct  (n={b.n:3d})  {bar}")

    print(f"\n── Seuil statique actuel (V_THRESHOLD = {V_THRESHOLD}) ──")
    pr = precision_recall_at(pairs, V_THRESHOLD)
    print(f"  accepte {pr.n_accepted}/{n} | precision (accepté→correct) = "
          f"{pr.precision:.0%} | recall = {pr.recall:.0%}")

    print(f"\n── Seuil calibré pour precision cible {_TARGET_PRECISION:.0%} ──")
    t = threshold_for_precision(pairs, _TARGET_PRECISION, min_accepts=5)
    if t is None:
        print(f"  Aucun seuil n'atteint {_TARGET_PRECISION:.0%} de précision "
              f"sur ces données (le Verifier ne peut PAS garantir cette "
              f"précision — signal important en soi).")
    else:
        pr2 = precision_recall_at(pairs, t)
        print(f"  U_V >= {t:.3f} → precision {pr2.precision:.0%}, "
              f"recall {pr2.recall:.0%} ({pr2.n_accepted} acceptés)")
        arrow = "↑ plus strict" if t > V_THRESHOLD else "↓ plus permissif"
        print(f"  (vs seuil statique {V_THRESHOLD} : {arrow})")


if __name__ == "__main__":
    main()
