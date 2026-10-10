"""
verifier_improvement_report.py — quantifies the NEW Verifier's contribution
on already-collected data (no LLM calls, no benchmark run), in the two ways
that actually reflect a JUDGE's quality (decision precision), not task count:

  #1 CALIBRATION — precision of the accept decision at the old static
     threshold (0.4) vs the calibrated one (0.5): how many WRONG answers
     stop being falsely accepted, at what cost in recall.

  #2/#3 AGREEMENT — does cross-attempt agreement predict correctness?
     Per task (needs several shots — use a NO-EARLY-STOP run), the majority
     vote's confidence vs whether the majority answer is actually correct:
     confident→correct / confident→wrong / abstain (honest uncertainty).

Regenerate these numbers for the thesis as more data accumulates.

Usage:
    # #1 pools every hmas_21tasks_v2_results_*.json by default:
    python mas_project_v2/experiments/verifier_improvement_report.py
    # #2/#3 needs a no-early-stop file (>=2 answers/task):
    python .../verifier_improvement_report.py --agreement <noearly_results.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_DIR = os.path.dirname(_EXP_DIR)
_ROOT    = os.path.dirname(_MAS_DIR)
for p in (_ROOT, _MAS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.verifier_calibration import precision_recall_at
from core.self_consistency import majority_vote, normalize_answer

_RESULTS = os.path.join(_EXP_DIR, "results")
OLD_THRESHOLD = 0.4   # pre-consolidation static cutoff
NEW_THRESHOLD = 0.5   # calibrated for ~80% precision (see verifier_calibration_report.py)


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


def report_calibration(paths: list) -> None:
    pairs = _load_pairs(paths)
    if not pairs:
        print("  (aucune paire U_V/reward)")
        return
    print(f"\n=== APPORT #1 — précision de l'acceptation ({len(pairs)} attempts) ===")
    print(f"  {'seuil':>6} | {'acceptés':>8} | {'corrects':>8} | {'FAUX':>5} | "
          f"{'précision':>9} | {'recall':>6}")
    rows = {}
    for t in (OLD_THRESHOLD, NEW_THRESHOLD):
        pr = precision_recall_at(pairs, t)
        rows[t] = pr
        wrong = pr.n_accepted - pr.n_correct_accepted
        print(f"  {t:>6.2f} | {pr.n_accepted:>8} | {pr.n_correct_accepted:>8} | "
              f"{wrong:>5} | {pr.precision:>8.0%} | {pr.recall:>5.0%}")
    old, new = rows[OLD_THRESHOLD], rows[NEW_THRESHOLD]
    false_avoided = ((old.n_accepted - old.n_correct_accepted)
                     - (new.n_accepted - new.n_correct_accepted))
    lost = old.n_correct_accepted - new.n_correct_accepted
    print(f"\n  → {OLD_THRESHOLD} → {NEW_THRESHOLD} : {false_avoided} fausses "
          f"acceptations ÉVITÉES pour {lost} succès perdu(s).")


def report_agreement(path: str) -> None:
    try:
        rows = json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"  (fichier illisible : {path})")
        return
    by = defaultdict(list)
    for x in rows:
        by[x["task_id"]].append(x)

    print(f"\n=== APPORT #2/#3 — l'accord prédit-il la correction ? "
          f"({os.path.basename(path)}) ===")
    conf_correct = conf_wrong = abstain = 0
    for t, shots in sorted(by.items()):
        vote = majority_vote([x.get("answer", "") for x in shots])
        win = normalize_answer(vote.answer)
        correct = any(x.get("reward", 0.0) >= 1.0
                      for x in shots if normalize_answer(x.get("answer", "")) == win)
        if vote.abstain:
            abstain += 1; label = "ABSTENTION (incertitude signalée)"
        elif correct:
            conf_correct += 1; label = "confiant → JUSTE"
        else:
            conf_wrong += 1; label = "confiant → FAUX"
        print(f"  {t.replace('v1.', ''):14} accord={vote.agreement:>4.0%}  {label}")

    confident = conf_correct + conf_wrong
    if confident:
        print(f"\n  → Quand CONFIANT : {conf_correct}/{confident} justes "
              f"({conf_correct/confident:.0%} précision) | {abstain} abstentions honnêtes.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*",
                    help="result files for #1 (default: all hmas_21tasks_v2_results_*.json)")
    ap.add_argument("--agreement", metavar="NOEARLY.json",
                    help="a no-early-stop result file for the #2/#3 agreement analysis")
    args = ap.parse_args()

    paths = args.files or glob.glob(os.path.join(_RESULTS, "hmas_21tasks_v2_results_*.json"))
    report_calibration(paths)

    if args.agreement:
        p = args.agreement if os.path.isabs(args.agreement) else os.path.join(_RESULTS, args.agreement)
        report_agreement(p)
    else:
        default = os.path.join(_RESULTS, "hmas_21tasks_v2_results_noearly_vote.json")
        if os.path.exists(default):
            report_agreement(default)
        else:
            print("\n  (#2/#3 : fournir un run --agreement sans early-stop pour l'analyse d'accord)")


if __name__ == "__main__":
    main()
