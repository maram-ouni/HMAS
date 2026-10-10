"""
self_consistency_report.py — applies core.self_consistency.majority_vote to
the already-collected multi-shot result files and reports, per run:

  - any-shot   : succeeds if ANY shot is correct (optimistic ORACLE upper
                 bound — needs ground truth to pick the winning shot).
  - first-shot : the first attempt only (no retry).
  - vote       : majority answer over the shots (verifier-free reliability).

and, crucially for trustworthiness, the confidence breakdown of the voted
answer per task:

  - confident & correct   : high agreement, majority answer is right.
  - confident & WRONG     : high agreement on a WRONG answer (voting cements
                            the error — the dangerous case a raw accuracy
                            number hides).
  - abstain               : low agreement / mostly no-answer → the system
                            should flag uncertainty instead of committing.

No LLM calls, no benchmark runs — pure post-hoc aggregation of existing data.

Usage:
    python mas_project_v2/experiments/self_consistency_report.py
"""
from __future__ import annotations

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

from core.self_consistency import majority_vote, normalize_answer

_RESULTS = os.path.join(_EXP_DIR, "results")
_RUNS = [
    ("baseline",   "hmas_21tasks_v2_results_lg_baseline.json"),
    ("A",          "hmas_21tasks_v2_results_lg_on.json"),
    ("B",          "hmas_21tasks_v2_results_grounding_on.json"),
    ("C",          "hmas_21tasks_v2_results_multiaction_on.json"),
    ("ABC",        "hmas_21tasks_v2_results_all3_levers.json"),
    ("ABC_noucb",  "hmas_21tasks_v2_results_all3_levers_lowvar.json"),
]


def _by_task(path: str) -> dict:
    rows = json.load(open(path, encoding="utf-8"))
    by = defaultdict(list)
    for x in rows:
        by[x["task_id"]].append(x)
    for t in by:
        by[t].sort(key=lambda x: x.get("shot", 0))
    return by


def _majority_is_correct(shots: list) -> tuple:
    """Returns (vote_success, confident, abstain) for one task's shots."""
    answers = [x.get("answer", "") for x in shots]
    res = majority_vote(answers)
    if res.abstain or not res.answer:
        return False, False, True
    win = normalize_answer(res.answer)
    # a shot that produced the winning answer and scored reward>=1 → correct
    correct = any(x.get("reward", 0.0) >= 1.0
                  for x in shots if normalize_answer(x.get("answer", "")) == win)
    return correct, res.is_confident, res.abstain


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Optional: a specific results file (e.g. a no-early-stop run where every
    # task has the full N_SHOTS, so voting actually has k samples).
    runs = _RUNS
    if len(sys.argv) > 1:
        f = sys.argv[1]
        runs = [(os.path.basename(f).replace("hmas_21tasks_v2_results", "").replace(".json", "") or "run", f)]

    print(f"\n{'RUN':16} | any-shot | first-shot | vote | conf&correct | conf&WRONG | abstain")
    print("-" * 86)
    for name, fname in runs:
        path = fname if os.path.isabs(fname) else os.path.join(_RESULTS, fname)
        if not os.path.exists(path):
            print(f"{name:16} | (fichier absent)")
            continue
        by = _by_task(path)
        any_ok = first_ok = vote_ok = 0
        conf_correct = conf_wrong = abstain = 0
        for shots in by.values():
            any_ok   += any(x.get("reward", 0.0) >= 1.0 for x in shots)
            first_ok += shots[0].get("reward", 0.0) >= 1.0
            v_ok, conf, abst = _majority_is_correct(shots)
            vote_ok += v_ok
            if abst:
                abstain += 1
            elif conf and v_ok:
                conf_correct += 1
            elif conf and not v_ok:
                conf_wrong += 1
        n = len(by)
        print(f"{name:16} |  {any_ok:2}/{n}  |   {first_ok:2}/{n}   | {vote_ok:2}/{n} |"
              f"      {conf_correct:2}      |     {conf_wrong:2}     |   {abstain:2}")

    print("\nLecture :")
    print("  - vote > first-shot  → la self-consistency fiabilise vs un seul essai.")
    print("  - vote < any-shot    → 'any-shot' est une borne oracle (ground truth requis).")
    print("  - conf&WRONG         → le modèle est SÛR mais FAUX ; le vote cimente l'erreur.")
    print("  - abstain            → désaccord/pas de réponse → signaler l'incertitude.")


if __name__ == "__main__":
    main()
