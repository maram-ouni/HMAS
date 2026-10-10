"""
hmas_scan_all.py — Scan de toutes les nouvelles tâches avec l'architecture HMAS

Lance notre architecture HMAS (3 agents : BDI Planner + LLM Executor + Rule Verifier)
sur les 90 mêmes tâches que react_scan_all.py. But : comparer les résultats et
identifier quelles tâches notre architecture réussit mieux que ReAct.

Classification finale par tâche :
  - ALWAYS_FAIL  : reward=0 sur les 5 shots
  - ALWAYS_PASS  : reward=1 sur les 5 shots
  - DISCRIMINANT : entre 1 et 4 succès sur 5

Usage:
    python mas_project_v2/experiments/hmas_scan_all.py
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

# ── Mêmes 90 tâches que react_scan_all.py ─────────────────────────────────
TASKS = [
    "v1.dashdish-10", "v1.dashdish-11", "v1.dashdish-2", "v1.dashdish-3",
    "v1.dashdish-4",  "v1.dashdish-5",  "v1.dashdish-7", "v1.dashdish-8",
    "v1.dashdish-9",
    "v1.fly-unified-10", "v1.fly-unified-11", "v1.fly-unified-12",
    "v1.fly-unified-13", "v1.fly-unified-14", "v1.fly-unified-2",
    "v1.fly-unified-3",  "v1.fly-unified-4",  "v1.fly-unified-5",
    "v1.fly-unified-6",  "v1.fly-unified-7",  "v1.fly-unified-8",
    "v1.fly-unified-9",
    "v1.gocalendar-1", "v1.gocalendar-10", "v1.gocalendar-3",
    "v1.gocalendar-4", "v1.gocalendar-5",  "v1.gocalendar-6",
    "v1.gocalendar-7", "v1.gocalendar-8",  "v1.gocalendar-9",
    "v1.gomail-2", "v1.gomail-3", "v1.gomail-4", "v1.gomail-5",
    "v1.gomail-6", "v1.gomail-7", "v1.gomail-8",
    "v1.networkin-1",  "v1.networkin-10", "v1.networkin-3",
    "v1.networkin-5",  "v1.networkin-6",  "v1.networkin-7",
    "v1.networkin-8",  "v1.networkin-9",
    "v1.opendining-10", "v1.opendining-2", "v1.opendining-3",
    "v1.opendining-4",  "v1.opendining-5", "v1.opendining-6",
    "v1.opendining-7",  "v1.opendining-8", "v1.opendining-9",
    "v1.staynb-1", "v1.staynb-2", "v1.staynb-4", "v1.staynb-5",
    "v1.staynb-6", "v1.staynb-7", "v1.staynb-8", "v1.staynb-9",
    "v1.topwork-1", "v1.topwork-2", "v1.topwork-3", "v1.topwork-4",
    "v1.topwork-5", "v1.topwork-6", "v1.topwork-7", "v1.topwork-9",
    "v1.udriver-1",  "v1.udriver-10", "v1.udriver-11", "v1.udriver-3",
    "v1.udriver-4",  "v1.udriver-5",  "v1.udriver-6",  "v1.udriver-7",
    "v1.udriver-8",  "v1.udriver-9",
    "v1.zilloft-10", "v1.zilloft-2", "v1.zilloft-3", "v1.zilloft-4",
    "v1.zilloft-5",  "v1.zilloft-6", "v1.zilloft-7", "v1.zilloft-8",
    "v1.zilloft-9",
]

N_SHOTS             = 5
MAX_STEPS           = 10
SLEEP_BETWEEN_SHOTS = 6
RESULTS_DIR         = os.path.join(_EXPERIMENTS_DIR, "results")
RESULTS_FILE        = os.path.join(RESULTS_DIR, "hmas_scan_results.json")


# ── Warmup serveur Vercel (évite timeout Playwright) ──────────────────────

def _get_task_url(task_id: str) -> str:
    parts   = task_id.split(".", 1)
    version = parts[0] if len(parts) > 1 else "v1"
    name    = parts[1] if len(parts) > 1 else parts[0]
    for path in [
        os.path.join(_ROOT, "src", "agisdk", "REAL", "browsergym",
                     "webclones", version, "tasks", f"{name}.json"),
        os.path.join(_ROOT, "src", "agisdk", "REAL", "browsergym",
                     "webclones", "v1", "tasks", f"{name}.json"),
    ]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("website", {}).get("url", "")
    return ""


def _warmup_server(task_id: str, max_retries: int = 3, delay: int = 6) -> None:
    base_url = _get_task_url(task_id)
    if not base_url:
        return
    config_url = f"{base_url.rstrip('/')}/config?run_id=0&task_id={task_id}&latency=0"
    for url, label in [(base_url, "homepage"), (config_url, "config")]:
        print(f"  [Warmup] {label} ...", end="", flush=True)
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "HMAS-warmup/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status < 500:
                        print(f" OK ({resp.status})")
                        break
            except Exception:
                print(".", end="", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(delay)
        else:
            print(" FAILED")
    time.sleep(3)


# ── Lancement d'un shot HMAS en subprocess isolé ──────────────────────────

def _run_hmas_shot(task_id: str, prev_result: dict | None = None) -> dict | None:
    """
    Appelle mas_experiment.py --single-shot dans un subprocess isolé.
    Playwright sync API ne peut pas être redémarré dans le même processus.
    prev_result active le cross-shot learning du BDI Planner.
    """
    with tempfile.NamedTemporaryFile(suffix="_res.json", delete=False,
                                     mode="w") as f:
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

    utf8_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        subprocess.run(cmd, timeout=MAX_STEPS * 120, env=utf8_env)
    except subprocess.TimeoutExpired:
        print("  [HMAS] Timeout")
        return None
    except Exception as e:
        print(f"  [HMAS] Erreur subprocess: {e}")
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


# ── Persistance ────────────────────────────────────────────────────────────

def _load_results() -> list:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_results(results: list) -> None:
    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Classification finale ──────────────────────────────────────────────────

def _print_classification(results: list) -> None:
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    always_pass, always_fail, discriminant = [], [], []
    for task_id, shots in by_task.items():
        if len(shots) < N_SHOTS:
            continue
        n_success = sum(1 for s in shots if s["reward"] >= 1.0)
        if n_success == N_SHOTS:
            always_pass.append((task_id, n_success))
        elif n_success == 0:
            always_fail.append((task_id, n_success))
        else:
            discriminant.append((task_id, n_success))

    print(f"\n{'='*65}")
    print(f"  CLASSIFICATION HMAS ({len(by_task)} completes / {len(TASKS)} total)")
    print(f"{'='*65}")

    print(f"\n  DISCRIMINANTES ({len(discriminant)}) — idéales pour comparer les archi :")
    for t, n in sorted(discriminant, key=lambda x: -x[1]):
        print(f"    {t:<28}  {n}/{N_SHOTS} succès")

    print(f"\n  TOUJOURS REUSSIES ({len(always_pass)}) — trop faciles :")
    for t, _ in sorted(always_pass):
        print(f"    {t}")

    print(f"\n  TOUJOURS ECHOUEES ({len(always_fail)}) — impossibles ou cassées :")
    for t, _ in sorted(always_fail):
        print(f"    {t}")

    # Comparaison directe avec ReAct
    react_file = os.path.join(RESULTS_DIR, "react_scan_results.json")
    if os.path.exists(react_file):
        with open(react_file, encoding="utf-8") as f:
            react_results = json.load(f)
        react_by_task = defaultdict(list)
        for r in react_results:
            react_by_task[r["task_id"]].append(r)

        print(f"\n{'='*65}")
        print(f"  COMPARAISON HMAS vs ReAct (tâches avec 5 shots dans les deux)")
        print(f"{'='*65}")
        print(f"  {'Tâche':<28} {'HMAS':>6} {'ReAct':>6} {'Delta':>6}")
        print(f"  {'─'*50}")
        hmas_better, react_better, tied = [], [], []
        for task_id, shots in by_task.items():
            if len(shots) < N_SHOTS:
                continue
            if len(react_by_task[task_id]) < N_SHOTS:
                continue
            n_hmas  = sum(1 for s in shots if s["reward"] >= 1.0)
            n_react = sum(1 for s in react_by_task[task_id] if s["reward"] >= 1.0)
            delta   = n_hmas - n_react
            mark = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"  {task_id:<28} {n_hmas:>5}/5 {n_react:>5}/5 {mark}{abs(delta):>4}")
            if delta > 0:
                hmas_better.append(task_id)
            elif delta < 0:
                react_better.append(task_id)
            else:
                tied.append(task_id)
        print(f"\n  HMAS meilleur   : {len(hmas_better)} tâches")
        print(f"  ReAct meilleur  : {len(react_better)} tâches")
        print(f"  Egalité         : {len(tied)} tâches")

    print(f"\n  RECOMMANDATION : {len(discriminant)} tâches discriminantes HMAS")
    print(f"  + les 12 déjà testées = {len(discriminant)+12} tâches pour la comparaison finale")
    print(f"{'='*65}\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n{'='*65}")
    print(f"  SCAN HMAS — {len(TASKS)} tâches")
    print(f"  Architecture : BDI Planner + LLM Executor + Rule Verifier")
    print(f"  Modèle       : {LLM_MODEL}")
    print(f"  Shots        : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"  Cross-shot learning : activé (BDI apprend entre shots)")
    print(f"{'='*65}")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print("\n  ERREUR : GROQ_API_KEY non défini dans .env")
        sys.exit(1)

    results   = _load_results()
    done_keys = {(r["task_id"], r["shot"]) for r in results}
    n_done    = len({r["task_id"] for r in results
                     if sum(1 for x in results if x["task_id"] == r["task_id"]) >= N_SHOTS})

    print(f"  Reprise : {len(done_keys)} shots déjà effectués "
          f"({n_done}/{len(TASKS)} tâches complètes)\n")

    for i, task_id in enumerate(TASKS):
        shots_done = [r for r in results if r["task_id"] == task_id]

        if len(shots_done) >= N_SHOTS:
            n_ok = sum(1 for r in shots_done if r["reward"] >= 1.0)
            print(f"  [{i+1:>2}/{len(TASKS)}] {task_id:<28} SKIP ({n_ok}/{N_SHOTS})")
            continue

        print(f"\n  [{i+1:>2}/{len(TASKS)}] {task_id}")

        last_result = shots_done[-1] if shots_done else None
        cumsum = sum(1 - r["reward"] for r in shots_done)

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results
                         if x["task_id"] == task_id and x["shot"] == shot)
                last_result = r
                cumsum += (1 - r["reward"])
                print(f"  [Shot {shot}] SKIP reward={r['reward']:.1f}")
                continue

            _warmup_server(task_id)

            raw = _run_hmas_shot(task_id, prev_result=last_result)

            # Retry automatique sur crash serveur (ne compte pas comme shot)
            retry = 0
            while raw is not None and _is_server_crash(raw) and retry < 2:
                print(f"  [CRASH] Pause 15s puis retry...")
                time.sleep(15)
                _warmup_server(task_id)
                raw = _run_hmas_shot(task_id, prev_result=last_result)
                retry += 1

            if raw is None:
                raw = {"reward": 0.0, "steps": MAX_STEPS, "answer": "",
                       "total_duration": 0.0, "error": "subprocess_failed",
                       "U_PE": 0.0, "U_V": 0.0, "quality": "rejected"}

            cumsum += (1 - raw["reward"])
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
            q      = raw.get("quality", "rejected")
            print(f"  [Shot {shot}] {status} | reward={raw.get('reward',0):.1f} | "
                  f"U_V={raw.get('U_V',0):.3f} | quality={q} | "
                  f"steps={raw.get('steps',0)} | time={raw.get('total_duration',0):.1f}s")

            is_last = (shot == N_SHOTS and task_id == TASKS[-1])
            if not is_last:
                time.sleep(SLEEP_BETWEEN_SHOTS)

    _print_classification(results)


if __name__ == "__main__":
    main()
