"""
hmas_12tasks.py — HMAS sur les 12 tâches validées (5 shots chacune)

Compare l'architecture HMAS (BDI Planner + LLM Executor + Rule Verifier)
sur les mêmes 12 tâches utilisées dans experiment_runner.py.

Usage:
    python mas_project_v2/experiments/hmas_12tasks.py
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

from config import LLM_MODEL, RESULTS_SUFFIX

_ALL_TASKS = [
    "v1.networkin-2",
    "v1.networkin-4",
    "v1.opendining-1",
    "v1.dashdish-1",
    "v1.dashdish-6",
    "v1.gomail-1",
    "v1.topwork-8",
    "v1.fly-unified-1",
    "v1.gocalendar-2",
    "v1.staynb-3",
    "v1.udriver-2",
    "v1.zilloft-1",
]

# Diagnostic ablation support: restrict to a subset of tasks (e.g. the two
# that regressed V2 vs V1) without touching the reference 12-task list.
_tasks_override = os.environ.get("MAS_TASKS_OVERRIDE", "").strip()
TASKS = ([t.strip() for t in _tasks_override.split(",") if t.strip()]
         if _tasks_override else _ALL_TASKS)

N_SHOTS             = 10
MAX_STEPS           = 10
SLEEP_BETWEEN_SHOTS = 6
RESULTS_DIR         = os.path.join(_EXPERIMENTS_DIR, "results")
RESULTS_FILE        = os.path.join(RESULTS_DIR, f"hmas_12tasks_results{RESULTS_SUFFIX}.json")

# Isolated memory — mandatory for a clean reference comparison (established
# convention since the V2/V3 executor-lever and LLM-BDI ablations).
MEMORY_DIR = os.path.join(
    RESULTS_DIR,
    f"isolated_memory_v2_12tasks{RESULTS_SUFFIX or '_' + time.strftime('%Y%m%d_%H%M%S')}"
)

NO_EARLY_STOP = os.environ.get("MAS_NO_EARLY_STOP", "false").strip().lower() in ("1", "true", "yes")

# Internal BDI deliberation budget per shot — see the --shots comment in
# _run_hmas_shot(). Must match V1's config.MAX_REPLANS (=3) for the
# comparison to be fair; overridable for ablation.
DELIBERATION_BUDGET = int(os.environ.get("MAS_DELIBERATION_BUDGET", "3"))


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
        # Internal BDI deliberation budget. MUST be passed explicitly: in
        # single-shot mode `--shots` is repurposed as max_attempts, and its
        # default of 1 makes should_replan() always false, so attempt_count
        # never leaves 0 — which silently makes EVERY corrective plan
        # unreachable (VerifierFeedback, AnsweredWithoutNav, FabricatedBid,
        # NeverAnswered, LowEfficiency, ZeroSpecificity,
        # DiagnoseUnknownFailure all require attempt_count >= 1). The whole
        # "verifier diagnoses -> corrective hint -> retry" loop never ran.
        # V1 hardcodes MAX_REPLANS=3 and never had this problem, so every
        # V1-vs-V2 comparison was 3 corrective attempts against 1 blind one.
        # Found 2026-07-29; DELIBERATION_BUDGET matches V1's MAX_REPLANS.
        "--shots",       str(DELIBERATION_BUDGET),
        "--result-file", result_file,
    ]
    if prev_result_file:
        cmd += ["--prev-result-file", prev_result_file]
    if MEMORY_DIR:
        cmd += ["--memory-dir", MEMORY_DIR]

    # Enrichment flags — all default ON (the full V2 architecture as
    # designed), each a passthrough so an ablation run can flip individual
    # ones via env var without editing this file.
    #
    # History note (2026-07-29): five of these (UCB1, dynamic desires,
    # learned verifier/PE weights, world-model smoothing) were briefly
    # defaulted OFF after a diagnostic suggested they hurt. That diagnostic
    # is VOID: it ran under the deliberation-budget bug (see the --shots
    # comment above), where attempt_count never left 0 — UCB1 had no
    # attempts to arbitrate between, dynamic-desire plans could never
    # trigger, and learned weights had no history. They were judged in a
    # regime where they structurally could not work. With the budget fixed
    # they get a fair test, so the defaults are restored to ON.
    def _flag(name, default="true"):
        return os.environ.get(name, default)

    utf8_env = {
        **os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
        "MAS_TRACE_ENABLED":                  _flag("MAS_TRACE_ENABLED"),
        "MAS_CONFLICT_DETECTION_ENABLED":     _flag("MAS_CONFLICT_DETECTION_ENABLED"),
        "MAS_DYNAMIC_DESIRES_ENABLED":        _flag("MAS_DYNAMIC_DESIRES_ENABLED"),
        "MAS_UCB1_ENABLED":                   _flag("MAS_UCB1_ENABLED"),
        "MAS_WORLD_MODEL_BAYESIAN_SMOOTHING": _flag("MAS_WORLD_MODEL_BAYESIAN_SMOOTHING"),
        "MAS_EXECUTOR_SELF_VERIFY":           _flag("MAS_EXECUTOR_SELF_VERIFY"),
        "MAS_LEARNED_VERIFIER_WEIGHTS":       _flag("MAS_LEARNED_VERIFIER_WEIGHTS"),
        "MAS_LEARNED_PE_WEIGHTS":             _flag("MAS_LEARNED_PE_WEIGHTS"),
        "MAS_INTERACTION_METRICS_ENABLED":    _flag("MAS_INTERACTION_METRICS_ENABLED"),
        "MAS_EXECUTOR_LOOP_GUARD_ENABLED":         _flag("MAS_EXECUTOR_LOOP_GUARD_ENABLED"),
        "MAS_EXECUTOR_GROUNDING_RECOVERY_ENABLED": _flag("MAS_EXECUTOR_GROUNDING_RECOVERY_ENABLED"),
        "MAS_EXECUTOR_MULTIACTION_FIX_ENABLED":    _flag("MAS_EXECUTOR_MULTIACTION_FIX_ENABLED"),
        "MAS_VERIFIER_AGREEMENT_ENABLED":          _flag("MAS_VERIFIER_AGREEMENT_ENABLED"),
    }
    try:
        # Must scale with the deliberation budget, and stay ABOVE the
        # orchestrator's own internal timeout (max_steps * 90 * max_attempts)
        # or the subprocess gets killed mid-deliberation.
        subprocess.run(cmd, timeout=MAX_STEPS * 120 * DELIBERATION_BUDGET, env=utf8_env)
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
    print(f"  RÉSULTATS HMAS — 12 tâches validées")
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
    print(f"  HMAS — 12 tâches validées")
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
        already_succeeded = (not NO_EARLY_STOP) and any(r["reward"] >= 1.0 for r in shots_done)

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
                if (not NO_EARLY_STOP) and r["reward"] >= 1.0:
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

            if (not NO_EARLY_STOP) and raw.get("reward", 0.0) >= 1.0:
                print(f"  → SUCCÈS au shot {shot} (reward=1.0) ! Passage à la tâche suivante.")
                break

            is_last = (shot == N_SHOTS and task_id == TASKS[-1])
            if not is_last:
                time.sleep(SLEEP_BETWEEN_SHOTS)

    _print_summary(results)


if __name__ == "__main__":
    main()
