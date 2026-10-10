"""
interaction_metrics_report.py — Rejoue les traces déjà collectées à
travers core/interaction_metrics.py pour donner un premier rapport
d'interaction PE/Verifier sans attendre un nouveau run (aucune
dépendance Groq).

Usage:
    python mas_project_v2/experiments/interaction_metrics_report.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAS  = os.path.dirname(_HERE)
sys.path.insert(0, _MAS)

from core.interaction_metrics import InteractionMetrics

TRACE_DIR = os.path.join(_MAS, "results", "traces")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with tempfile.TemporaryDirectory() as tmp:
        im = InteractionMetrics(results_dir=tmp)
        for path in sorted(glob.glob(os.path.join(TRACE_DIR, "*.jsonl"))):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    im.record(r.get("U_PE", 0.0), r.get("U_V", 0.0),
                              r.get("reward", 0.0), r.get("quality", "rejected"))

        s = im.summary()
        print("=" * 70)
        print("  RAPPORT D'INTERACTION PE / VERIFIER — rejeu sur traces réelles")
        print("=" * 70)
        print(f"  Vérifications rejouées        : {s['n']}")
        print(f"  Taux de conflit                : {s['conflict_rate']:.1%}")
        print(f"  Désaccord moyen |U_PE - U_V|   : {s['mean_disagreement']:.4f}")
        print(f"  Écart U_PE (succès - échec)    : {s['pe_reward_gap']:+.4f}")
        print(f"  Écart U_V  (succès - échec)    : {s['v_reward_gap']:+.4f}")
        print()
        print(f"  {im.summary_line().strip()}")
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
