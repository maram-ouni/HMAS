"""
hmas_20tasks.py — HMAS sur le jeu de référence FIXE de 20 tâches.

Constitué après exploration (12 tâches originales + 3 lots de 10
nouvelles tâches) :
  - 17 tâches où l'architecture a démontré un succès (7 sites distincts :
    networkin, opendining, dashdish, gomail, topwork, gocalendar, udriver)
  - 3 tâches où elle échoue systématiquement (fly-unified-1, staynb-3,
    zilloft-1 — famille de sites à widgets combobox/date-picker, cause
    racine confirmée dans functions.py du SDK REAL : fill()/select_option()
    ne gèrent pas les widgets JS personnalisés)

Ce jeu sert de référence stable pour mesurer l'effet de futurs correctifs
(ex: consigne combobox dans le prompt Executor) sans redécouvrir le
terrain à chaque fois.

Usage:
    python mas_project_v2/experiments/hmas_20tasks.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_DIR         = os.path.dirname(_EXPERIMENTS_DIR)
_ROOT            = os.path.dirname(_MAS_DIR)
_MAS_SCRIPT      = os.path.join(_MAS_DIR, "mas_experiment.py")

sys.path.insert(0, _ROOT)
sys.path.insert(0, _MAS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from config import LLM_MODEL

TASKS = [
    # ── 17 tâches validées (succès observé lors de l'exploration) ──
    "v1.networkin-2",
    "v1.networkin-4",
    "v1.opendining-1",
    "v1.dashdish-1",
    "v1.dashdish-6",
    "v1.gomail-1",
    "v1.topwork-8",
    "v1.gocalendar-2",
    "v1.udriver-2",
    "v1.dashdish-3",
    "v1.topwork-6",
    "v1.gocalendar-4",
    "v1.gomail-4",
    "v1.gomail-6",
    "v1.networkin-5",
    "v1.opendining-2",
    "v1.udriver-7",
    # ── 3 tâches bloquées (famille combobox/date-picker — repère) ──
    "v1.fly-unified-1",
    "v1.staynb-3",
    "v1.zilloft-1",
]

N_SHOTS             = 10
MAX_STEPS           = 10
SLEEP_BETWEEN_SHOTS = 6
RESULTS_DIR         = os.path.join(_EXPERIMENTS_DIR, "results")
RESULTS_FILE        = os.path.join(RESULTS_DIR, "hmas_20tasks_results.json")


def _get_task_url(task_id: str) -> str:
    parts   = task_id.split(".", 1)
    version = parts[0] if len(parts) > 1 else "v1"
    name    = parts[1] if len(parts) > 1 else parts[0]
    for path in [
        os.path.join(_ROOT, "src", "agisdk", "REAL", "browsergym",
                     "webclones", version, "tasks", f"{name}.json"),
    ]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("website", {}).get("url", "")
    return ""


def _warmup_server(task_id: str) -> None:
    base_url = _get_task_url(task_id)
    if not base_url:
        return
    config_url = f"{base_url.rstrip('/')}/config?run_id=0&task_id={task_id}&latency=0"
    for url, label in [(base_url, "homepage"), (config_url, "config")]:
        print(f"  [Warmup] {label} ...", end="", flush=True)
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "HMAS-warmup/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status < 500:
                        print(f" OK ({resp.status})")
                        break
            except Exception:
                print(".", end="", flush=True)
                if attempt < 2:
                    time.sleep(6)
        else:
            print(" FAILED")
    time.sleep(3)


def _run_hmas_shot(task_id: str, prev_result: dict | None = None) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix="_res.json", delete=False, mode="w") as f:
        result_file = f.name

    prev_result_file = ""
    if prev_result:
        with tempfile.NamedTemporaryFile(suffix="_prev.json", delete=False,
                                         mode="w", encoding="utf-8") as pf:
            json.dump(prev_result, pf, ensure_ascii=False)
            prev_result_file = pf.name

    cmd = [
        sys.executable, _MAS_SCRIPT,
        "--single-shot",
        "--task",        task_id,
        "--max-steps",   str(MAX_STEPS),
        "--result-file", result_file,
    ]
    if prev_result_file:
        cmd += ["--prev-result-file", prev_result_file]

    # Enrichment flags (config.py's hardcoded defaults are all False) —
    # enabled here so this script actually exercises the full V2
    # architecture, matching experiments/experiment_runner.py's
    # _run_hybrid_shot_subprocess.
    utf8_env = {
        **os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
        "MAS_TRACE_ENABLED":                  "true",
        "MAS_CONFLICT_DETECTION_ENABLED":     "true",
        "MAS_DYNAMIC_DESIRES_ENABLED":        "true",
        "MAS_UCB1_ENABLED":                   "true",
        "MAS_WORLD_MODEL_BAYESIAN_SMOOTHING": "true",
        "MAS_EXECUTOR_SELF_VERIFY":           "true",
        "MAS_LEARNED_VERIFIER_WEIGHTS":       "true",
        "MAS_LEARNED_PE_WEIGHTS":             "true",
        "MAS_INTERACTION_METRICS_ENABLED":    "true",
    }
    try:
        subprocess.run(cmd, timeout=MAX_STEPS * 120, env=utf8_env)
    except subprocess.TimeoutExpired:
        print("  [HMAS] Timeout")
        return None
    except Exception as e:
        print(f"  [HMAS] Erreur: {e}")
        return None
    finally:
        if prev_result_file and os.path.exists(prev_result_file):
            os.unlink(prev_result_file)

    if os.path.exists(result_file):
        with open(result_file, encoding="utf-8") as f:
            result = json.load(f)
        os.unlink(result_file)
        return result
    if os.path.exists(result_file):
        os.unlink(result_file)
    return None


def _is_server_crash(result: dict) -> bool:
    if result is None:
        return True
    verdict = str(result.get("verdict", "")).upper()
    error   = str(result.get("error",   "")).lower()
    steps   = result.get("steps", 0)
    return (steps <= 1
            and result.get("reward", 0.0) == 0.0
            and ("CRASH" in verdict or "timeout" in error or "network" in error))


def _load_results() -> list:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_results(results: list) -> None:
    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _print_summary(results: list) -> None:
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    print(f"\n{'='*65}")
    print(f"  RÉSULTATS HMAS — jeu de référence fixe (20 tâches)")
    print(f"{'='*65}")
    print(f"  {'Tâche':<28} {'Succès':>8} {'Reward':>8} {'U_V moy':>8}")
    print(f"  {'─'*56}")
    total_ok = 0
    for task_id in TASKS:
        shots = by_task.get(task_id, [])
        if not shots:
            print(f"  {task_id:<28} {'—':>8}")
            continue
        n_ok   = sum(1 for s in shots if s["reward"] >= 1.0)
        avg_r  = sum(s["reward"] for s in shots) / len(shots)
        avg_uv = sum(s.get("U_V", 0) for s in shots) / len(shots)
        total_ok += n_ok
        print(f"  {task_id:<28} {n_ok:>4}/{len(shots):>2}   {avg_r:>6.2f}   {avg_uv:>6.3f}")

    n_tasks = len([t for t in TASKS if by_task.get(t)])
    total_shots = sum(len(shots) for shots in by_task.values())
    print(f"  {'─'*56}")
    print(f"  Total succès : {total_ok} / {total_shots} shots")
    print(f"  Taux moyen   : {total_ok/total_shots*100:.1f}%" if total_shots else "")
    print(f"{'='*65}\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n{'='*65}")
    print(f"  HMAS — jeu de référence fixe (20 tâches)")
    print(f"  Architecture : BDI Planner + LLM Executor + Rule Verifier")
    print(f"  Modèle       : {LLM_MODEL}")
    print(f"  Shots        : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"{'='*65}")

    results   = _load_results()
    done_keys = {(r["task_id"], r["shot"]) for r in results}
    print(f"  Reprise : {len(done_keys)} shots déjà effectués\n")

    for i, task_id in enumerate(TASKS):
        shots_done        = [r for r in results if r["task_id"] == task_id]
        # Success is gated on ground-truth reward only — matching the real
        # verifier's philosophy for retrieval tasks (correctness is binary,
        # not judged by U_V). A U_V>=0.7 fallback here could register a
        # false success on a well-formatted but wrong answer.
        already_succeeded = any(r["reward"] >= 1.0 for r in shots_done)

        if len(shots_done) >= N_SHOTS or already_succeeded:
            n_ok  = sum(1 for r in shots_done if r["reward"] >= 1.0)
            label = f"SUCCÈS shot {next(r['shot'] for r in shots_done if r['reward']>=1.0)}" if already_succeeded else f"{n_ok}/{N_SHOTS}"
            print(f"  [{i+1:>2}/{len(TASKS)}] {task_id:<28} SKIP ({label})")
            continue

        print(f"\n  [{i+1:>2}/{len(TASKS)}] {task_id}")
        last_result = shots_done[-1] if shots_done else None
        cumsum = sum(1 - r["reward"] for r in shots_done)

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results if x["task_id"] == task_id and x["shot"] == shot)
                last_result = r
                cumsum += (1 - r["reward"])
                print(f"  [Shot {shot}] SKIP reward={r['reward']:.1f}")
                if r["reward"] >= 1.0:
                    print(f"  → Succès déjà enregistré — tâche suivante.")
                    break
                continue

            _warmup_server(task_id)
            raw = _run_hmas_shot(task_id, prev_result=last_result)

            retry = 0
            while raw is not None and _is_server_crash(raw) and retry < 2:
                print("  [CRASH] Pause 15s puis retry...")
                time.sleep(15)
                _warmup_server(task_id)
                raw = _run_hmas_shot(task_id, prev_result=last_result)
                retry += 1

            if raw is None:
                raw = {"reward": 0.0, "steps": MAX_STEPS, "answer": "",
                       "total_duration": 0.0, "error": "subprocess_failed",
                       "U_PE": 0.0, "U_V": 0.0, "quality": "rejected"}

            cumsum += (1 - raw.get("reward", 0.0))
            record = {
                "architecture":      "hmas",
                "task_id":           task_id,
                "shot":              shot,
                "reward":            raw.get("reward", 0.0),
                "steps":             raw.get("steps", MAX_STEPS),
                "answer":            raw.get("answer", ""),
                "duration_s":        raw.get("total_duration", 0.0),
                "U_PE":              raw.get("U_PE", 0.0),
                "U_V":               raw.get("U_V", 0.0),
                "quality":           raw.get("quality", "rejected"),
                "strategy":          raw.get("strategy", ""),
                "regret_t":          round(1 - raw.get("reward", 0.0), 4),
                "cumulative_regret": round(cumsum / shot, 4),
                "error":             raw.get("error"),
            }
            results.append(record)
            done_keys.add((task_id, shot))
            last_result = raw
            _save_results(results)

            status = "OK  " if raw.get("reward", 0.0) >= 1.0 else "FAIL"
            print(f"  [Shot {shot}] {status} | reward={raw.get('reward',0):.1f} | "
                  f"U_V={raw.get('U_V',0):.3f} | steps={raw.get('steps',0)} | "
                  f"time={raw.get('total_duration',0):.1f}s")

            if raw.get("reward", 0.0) >= 1.0:
                print(f"  → SUCCÈS au shot {shot} (reward=1.0) ! Passage à la tâche suivante.")
                break

            is_last = (shot == N_SHOTS and task_id == TASKS[-1])
            if not is_last:
                time.sleep(SLEEP_BETWEEN_SHOTS)

    _print_summary(results)


if __name__ == "__main__":
    main()
