"""
experiment_runner.py — Main orchestration for the 5-architecture comparison

Runs all 5 architectures sequentially on 4 REAL benchmark tasks (5 shots each),
then computes regret-based metrics and prints a comparison table.

Execution order (to avoid Playwright port conflicts):
  1. ReAct            (react_baseline.py)       — Yao et al., NeurIPS 2022
  2. Reflexion        (reflexion_baseline.py)   — Shinn et al., NeurIPS 2023
  3. Frering BDI      (frering_bdi_baseline.py) — Frering et al., Eng.Appl.AI 2025
  4. ChatBDI          (chatbdi_baseline.py)     — Gatti et al., AAMAS 2025
  5. Hybrid BDI+Verifier (mas_experiment.py via subprocess) — this internship

Usage:
    python mas_project_v2/experiments/experiment_runner.py
    python mas_project_v2/experiments/experiment_runner.py --skip-run    # metrics only
    python mas_project_v2/experiments/experiment_runner.py --arch react   # one arch only
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_DIR         = os.path.dirname(_EXPERIMENTS_DIR)
_ROOT            = os.path.dirname(_MAS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _MAS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import LLM_MODEL, RESULTS_SUFFIX

# ── Experiment parameters ──────────────────────────────────────────────────
TASKS = [
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
N_SHOTS             = 10
MAX_STEPS           = 10
SLEEP_BETWEEN_SHOTS = 8
RESULTS_DIR         = os.path.join(_EXPERIMENTS_DIR, "results")
HYBRID_RESULTS_FILE = os.path.join(RESULTS_DIR, f"hybrid_results{RESULTS_SUFFIX}.json")

# Dedicated, non-contaminated memory directory (world model, learned
# strategies, agent histories, verifier weight stats, per-attempt traces)
# for the hybrid architecture's runs in THIS comparison — keeps the
# 2026-07-15 7-architecture / 12-task / 10-shot experiment reproducible
# and isolated from unrelated earlier --demo test runs already sitting in
# the shared mas_project_v2/results/ (see config.MEMORY_DIR).
ISOLATED_MEMORY_DIR = os.path.join(RESULTS_DIR, "isolated_memory_20260715")

TASK_LABELS = {
    "v1.networkin-2":    "networkin-2",
    "v1.networkin-4":    "networkin-4",
    "v1.opendining-1":   "opendining-1",
    "v1.dashdish-1":     "dashdish-1",
    "v1.dashdish-6":     "dashdish-6",
    "v1.gomail-1":       "gomail-1",
    "v1.topwork-8":      "topwork-8",
    "v1.fly-unified-1":  "fly-unified-1",
    "v1.gocalendar-2":   "gocalendar-2",
    "v1.staynb-3":       "staynb-3",
    "v1.udriver-2":      "udriver-2",
    "v1.zilloft-1":      "zilloft-1",
}

ARCH_LABELS = {
    "react":               "ReAct",
    "reflexion":           "Reflexion",
    "frering_bdi":         "BDI w/o Verifier (ablation)",
    "chatbdi":             "ChatBDI (BDI+LLM planner)",
    "hierarchical_bdi":    "Hierarchical BDI (AIWolf)",
    "hri_bdi":             "BDI-LLM Dual (HRI)",
    "hybrid_bdi_verifier": "Hybrid BDI + Verifier",
}


# ── Task goal helper ───────────────────────────────────────────────────────

def _get_task_goal(task_id: str) -> str:
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
                return json.load(f).get("goal", f"Task {task_id}")
    return f"Accomplish task {task_id}"


# ── Hybrid architecture runner ─────────────────────────────────────────────

def _parse_task(task_id: str):
    name = task_id.split(".", 1)[-1]
    idx  = name.rfind("-")
    return name[:idx], int(name[idx + 1:])


def _run_hybrid_shot_subprocess(task_id: str, max_steps: int,
                                 prev_result: dict) -> dict:
    """
    Runs one shot of the full Hybrid BDI+Verifier architecture
    by calling mas_experiment.py --single-shot in a subprocess.
    This reuses the exact existing implementation.
    """
    with tempfile.NamedTemporaryFile(suffix="_res.json", delete=False) as f:
        res_file = f.name

    prev_file = ""
    if prev_result:
        with tempfile.NamedTemporaryFile(suffix="_prev.json", delete=False,
                                          mode="w", encoding="utf-8") as pf:
            json.dump(prev_result, pf, ensure_ascii=False)
            prev_file = pf.name

    cmd = [
        sys.executable,
        os.path.join(_MAS_DIR, "mas_experiment.py"),
        "--single-shot",
        "--task",        task_id,
        "--max-steps",   str(max_steps),
        "--result-file", res_file,
        "--memory-dir",  ISOLATED_MEMORY_DIR,
    ]
    if prev_file:
        cmd += ["--prev-result-file", prev_file]

    # Enrichment flags (config.py's hardcoded defaults are all False) are
    # enabled ONLY in this subprocess's environment — never in the parent
    # experiment_runner.py process's own os.environ — so that
    # frering_bdi_baseline.py (the only OTHER architecture that imports
    # core/, and runs in-process, not via subprocess) stays a clean,
    # unenriched ablation baseline.
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
        # Executor reliability levers — PASSTHROUGH (default "false"),
        # toggled externally per ablation run, not hardcoded on like the 9 above.
        "MAS_EXECUTOR_LOOP_GUARD_ENABLED": os.environ.get("MAS_EXECUTOR_LOOP_GUARD_ENABLED", "false"),
        "MAS_EXECUTOR_GROUNDING_RECOVERY_ENABLED": os.environ.get("MAS_EXECUTOR_GROUNDING_RECOVERY_ENABLED", "false"),
        "MAS_EXECUTOR_MULTIACTION_FIX_ENABLED": os.environ.get("MAS_EXECUTOR_MULTIACTION_FIX_ENABLED", "false"),
        "MAS_VERIFIER_AGREEMENT_ENABLED": os.environ.get("MAS_VERIFIER_AGREEMENT_ENABLED", "false"),
    }
    try:
        proc = subprocess.run(cmd, timeout=max_steps * 120, env=utf8_env)
        if proc.returncode != 0:
            print(f"  [Hybrid] Subprocess returncode={proc.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  [Hybrid] Subprocess timeout")
        return None
    except Exception as e:
        print(f"  [Hybrid] Subprocess error: {e}")
        return None
    finally:
        if prev_file and os.path.exists(prev_file):
            os.unlink(prev_file)

    if os.path.exists(res_file):
        with open(res_file, encoding="utf-8") as f:
            result = json.load(f)
        os.unlink(res_file)
        return result
    if os.path.exists(res_file):
        os.unlink(res_file)
    return None


def _load_hybrid_results() -> list:
    if os.path.exists(HYBRID_RESULTS_FILE):
        with open(HYBRID_RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_hybrid_results(results: list) -> None:
    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(HYBRID_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_hybrid() -> None:
    print(f"\n{'='*62}")
    print(f"  ARCHITECTURE 4 — Hybrid BDI + Verifier (our system)")
    print(f"  Model  : {LLM_MODEL}")
    print(f"  Tasks  : {TASKS}")
    print(f"  Shots  : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"{'='*62}")

    results   = _load_hybrid_results()
    done_keys = {(r["task_id"], r["shot"]) for r in results}

    for task_id in TASKS:
        print(f"\n  Task : {task_id}")

        cumsum_regret = 0.0
        last_result   = None
        tried_hints   = []

        # Replay history to restore cross-shot state for resume
        completed = sorted(
            [r for r in results if r["task_id"] == task_id],
            key=lambda x: x["shot"]
        )
        for r in completed:
            cumsum_regret += (1 - r["reward"])
            last_result    = r
            hint = r.get("hint_used", "")
            if hint and hint not in tried_hints:
                tried_hints.append(hint)

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results
                         if x["task_id"] == task_id and x["shot"] == shot)
                print(f"  [Shot {shot}] Already done — reward={r['reward']:.1f} (skipping)")
                continue

            print(f"\n  {'─'*58}")
            print(f"  [Shot {shot}/{N_SHOTS}] Launching Hybrid subprocess...")

            # Pass cross-shot context: prev_result contains tried_hints and BDI state
            prev_context = None
            if last_result:
                prev_context = dict(last_result)
                prev_context["tried_hints"] = list(tried_hints)

            raw = _run_hybrid_shot_subprocess(task_id, MAX_STEPS, prev_context)
            if raw is None:
                raw = {
                    "reward": 0.0, "steps": MAX_STEPS, "answer": "",
                    "total_duration": 0.0, "error": "subprocess_failed",
                    "U_PE": 0.0, "U_V": 0.0, "quality": "rejected",
                    "hint_used": "", "strategy": "",
                    "dim_success": 0.0, "dim_efficiency": 0.0,
                    "dim_safety": 0.0, "dim_specificity": 0.0,
                    "verdict": "error",
                }

            # Update cross-shot state
            hint_used = raw.get("hint_used", "") or raw.get("strategy", "")
            if hint_used and hint_used not in tried_hints:
                tried_hints.append(hint_used)
            raw["tried_hints"] = list(tried_hints)
            last_result        = raw

            cumsum_regret += (1 - raw["reward"])
            cum_regret     = cumsum_regret / shot

            record = {
                "architecture":     "hybrid_bdi_verifier",
                "task_id":          task_id,
                "shot":             shot,
                "reward":           raw.get("reward", 0.0),
                "steps":            raw.get("steps", MAX_STEPS),
                "answer":           raw.get("answer", ""),
                "duration_s":       raw.get("total_duration", 0.0),
                "regret_t":         round(1 - raw.get("reward", 0.0), 4),
                "cumulative_regret": round(cum_regret, 4),
                "hint_used":        hint_used,
                "U_planner":        raw.get("U_PE", 0.0),
                "U_executer":       raw.get("U_PE", 0.0),
                "U_verifier":       raw.get("U_V", 0.0),
                "dim_success":      raw.get("dim_success",    0.0),
                "dim_efficiency":   raw.get("dim_efficiency", 0.0),
                "dim_safety":       raw.get("dim_safety",     0.0),
                "dim_specificity":  raw.get("dim_specificity", 0.0),
                "verdict":          raw.get("verdict", ""),
                # Pass cross-shot state for next shot's prev_result
                "quality":          raw.get("quality", "rejected"),
                "U_PE":             raw.get("U_PE", 0.0),
                "U_V":              raw.get("U_V", 0.0),
                "tried_hints":      list(tried_hints),
                "strategy":         hint_used,
                "feedback":         raw.get("feedback", ""),
                "last_error":       raw.get("last_error", ""),
                "dim_relevance":    raw.get("dim_relevance", 0.0),
                "action_history":   raw.get("action_history", []),
            }
            results.append(record)
            done_keys.add((task_id, shot))
            _save_hybrid_results(results)

            status = "SUCCESS" if raw.get("reward", 0.0) >= 1.0 else "FAIL"
            print(f"  [Shot {shot}] {status} | reward={raw.get('reward',0):.1f} | "
                  f"U_V={raw.get('U_V',0):.3f} | quality={raw.get('quality','?')} | "
                  f"cum_regret={cum_regret:.3f}")

            is_last = (shot == N_SHOTS and task_id == TASKS[-1])
            if not is_last:
                print(f"  Sleeping {SLEEP_BETWEEN_SHOTS}s (rate-limit safety)...")
                time.sleep(SLEEP_BETWEEN_SHOTS)

    n_success = sum(1 for r in results if r["reward"] >= 1.0)
    print(f"\n  Hybrid done — {n_success}/{len(results)} shots succeeded")
    print(f"  Results: {HYBRID_RESULTS_FILE}")


# ── Metrics computation ────────────────────────────────────────────────────

def compute_metrics(results: list, task_id: str) -> dict:
    """
    Per-task metrics for one architecture.

    Metrics:
      success_rate     : fraction of shots with reward == 1.0
      avg_regret       : mean(1 - reward_i) over all shots
      regret_curve     : [(shot, cumulative_regret_at_shot)]
      avg_steps        : mean steps per shot
      first_success_shot : first shot with reward >= 1.0 (None if never)
      convergence_shot : first shot where cumulative_regret < 0.2 (None if never)
    """
    task_shots = sorted(
        [r for r in results if r["task_id"] == task_id],
        key=lambda r: r["shot"]
    )
    if not task_shots:
        return {
            "success_rate": 0.0, "avg_regret": 1.0, "regret_curve": [],
            "avg_steps": MAX_STEPS, "first_success_shot": None,
            "convergence_shot": None,
        }

    rewards = [r["reward"] for r in task_shots]
    regrets = [1 - rw for rw in rewards]
    n       = len(rewards)

    regret_curve = []
    running_sum  = 0.0
    for i, reg in enumerate(regrets):
        running_sum += reg
        cum_r = running_sum / (i + 1)
        regret_curve.append((task_shots[i]["shot"], round(cum_r, 4)))

    first_success = next(
        (r["shot"] for r in task_shots if r["reward"] >= 1.0), None
    )
    convergence_shot = next(
        (shot for shot, cr in regret_curve if cr < 0.2), None
    )

    return {
        "success_rate":      sum(1 for rw in rewards if rw >= 1.0) / n,
        "avg_regret":        sum(regrets) / n,
        "regret_curve":      regret_curve,
        "avg_steps":         sum(r["steps"] for r in task_shots) / n,
        "first_success_shot": first_success,
        "convergence_shot":  convergence_shot,
    }


# ── Results loading ────────────────────────────────────────────────────────

def load_all_results() -> dict:
    """Load all result JSONs. Returns {arch_key: [records]}."""
    files = {
        "react":               os.path.join(RESULTS_DIR, "react_results.json"),
        "reflexion":           os.path.join(RESULTS_DIR, "reflexion_results.json"),
        "frering_bdi":         os.path.join(RESULTS_DIR, "frering_bdi_results.json"),
        "chatbdi":             os.path.join(RESULTS_DIR, "chatbdi_results.json"),
        "hierarchical_bdi":    os.path.join(RESULTS_DIR, "hierarchical_bdi_results.json"),
        "hri_bdi":             os.path.join(RESULTS_DIR, "hri_bdi_results.json"),
        "hybrid_bdi_verifier": os.path.join(RESULTS_DIR, "hybrid_results.json"),
    }
    all_results = {}
    for arch, path in files.items():
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                all_results[arch] = json.load(f)
        else:
            all_results[arch] = []
    return all_results


# ── Comparison table ───────────────────────────────────────────────────────

def print_comparison_table(all_results: dict) -> None:
    archs      = ["react", "reflexion", "frering_bdi", "chatbdi", "hierarchical_bdi", "hri_bdi", "hybrid_bdi_verifier"]
    col_labels = [TASK_LABELS[t] for t in TASKS]

    print(f"\n{'='*82}")
    print(f"  COMPARATIVE RESULTS — avg_regret per architecture × task")
    print(f"  (avg_regret = mean(1 - reward) over {N_SHOTS} shots, lower is better)")
    print(f"{'='*82}")

    header = f"  {'Architecture':<30}"
    for label in col_labels:
        header += f"  {label:>12}"
    header += f"  {'AVG_REGRET':>12}"
    print(header)
    print("  " + "─" * 78)

    for arch in archs:
        results    = all_results.get(arch, [])
        label      = ARCH_LABELS.get(arch, arch)
        row        = f"  {label:<30}"
        avg_regrets = []

        for task_id in TASKS:
            m = compute_metrics(results, task_id)
            n_done = len([r for r in results if r["task_id"] == task_id])
            if n_done == 0:
                cell = "     —"
            else:
                cell = f"{m['avg_regret']:>6.3f}"
            row += f"  {cell:>12}"
            if n_done > 0:
                avg_regrets.append(m["avg_regret"])

        if avg_regrets:
            overall = sum(avg_regrets) / len(avg_regrets)
            row += f"  {overall:>12.3f}"
        else:
            row += f"  {'—':>12}"

        print(row)

    print("  " + "─" * 78)

    print(f"\n{'='*82}")
    print(f"  FIRST SUCCESS SHOT — first shot where reward == 1.0 (None = never)")
    print(f"{'='*82}")
    header2 = f"  {'Architecture':<30}"
    for label in col_labels:
        header2 += f"  {label:>12}"
    print(header2)
    print("  " + "─" * 78)
    for arch in archs:
        results = all_results.get(arch, [])
        label   = ARCH_LABELS.get(arch, arch)
        row     = f"  {label:<30}"
        for task_id in TASKS:
            m = compute_metrics(results, task_id)
            n_done = len([r for r in results if r["task_id"] == task_id])
            if n_done == 0:
                cell = "—"
            elif m["first_success_shot"] is None:
                cell = "never"
            else:
                cell = f"shot {m['first_success_shot']}"
            row += f"  {cell:>12}"
        print(row)

    print(f"\n{'='*82}")
    print(f"  SUCCESS RATE — fraction of shots with reward == 1.0")
    print(f"{'='*82}")
    print(header)
    print("  " + "─" * 78)
    for arch in archs:
        results    = all_results.get(arch, [])
        label      = ARCH_LABELS.get(arch, arch)
        row        = f"  {label:<30}"
        rates      = []
        for task_id in TASKS:
            m = compute_metrics(results, task_id)
            n_done = len([r for r in results if r["task_id"] == task_id])
            if n_done == 0:
                cell = "     —"
            else:
                cell = f"{m['success_rate']:>6.2f}"
            row += f"  {cell:>12}"
            if n_done > 0:
                rates.append(m["success_rate"])
        if rates:
            row += f"  {sum(rates)/len(rates):>12.2f}"
        else:
            row += f"  {'—':>12}"
        print(row)

    print(f"\n{'='*82}")
    print(f"  CONVERGENCE SHOT — first shot where cumulative_regret < 0.2")
    print(f"{'='*82}")
    print(header2)
    print("  " + "─" * 78)
    for arch in archs:
        results = all_results.get(arch, [])
        label   = ARCH_LABELS.get(arch, arch)
        row     = f"  {label:<30}"
        for task_id in TASKS:
            m = compute_metrics(results, task_id)
            n_done = len([r for r in results if r["task_id"] == task_id])
            if n_done == 0:
                cell = "—"
            elif m["convergence_shot"] is None:
                cell = "never"
            else:
                cell = f"shot {m['convergence_shot']}"
            row += f"  {cell:>12}"
        print(row)

    print(f"\n{'='*82}\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run 4-architecture comparative experiment on REAL benchmark"
    )
    ap.add_argument("--skip-run",  action="store_true",
                    help="Skip execution — only compute metrics from existing results")
    ap.add_argument("--arch",
                    choices=["react", "reflexion", "frering", "chatbdi", "hier_bdi", "hri_bdi", "hybrid"],
                    help="Run only one specific architecture")
    args = ap.parse_args()

    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        # Import and run baselines sequentially
        run_react     = not args.arch or args.arch == "react"
        run_reflexion = not args.arch or args.arch == "reflexion"
        run_frering   = not args.arch or args.arch == "frering"
        run_chatbdi   = not args.arch or args.arch == "chatbdi"
        run_hier_bdi  = not args.arch or args.arch == "hier_bdi"
        run_hri_bdi   = not args.arch or args.arch == "hri_bdi"
        run_hybrid_   = not args.arch or args.arch == "hybrid"

        if run_react:
            import react_baseline
            react_baseline.main()

        if run_reflexion:
            import reflexion_baseline
            reflexion_baseline.main()

        if run_frering:
            import frering_bdi_baseline
            frering_bdi_baseline.main()

        if run_chatbdi:
            import chatbdi_baseline
            chatbdi_baseline.main()

        if run_hier_bdi:
            import hierarchical_bdi_baseline
            hierarchical_bdi_baseline.main()

        if run_hri_bdi:
            import hri_bdi_baseline
            hri_bdi_baseline.main()

        if run_hybrid_:
            run_hybrid()

    # Load all results and print comparison
    all_results = load_all_results()

    # Detailed per-architecture metrics
    print(f"\n{'='*62}")
    print(f"  DETAILED METRICS — 5 ARCHITECTURES")
    print(f"{'='*62}")
    for arch, results in all_results.items():
        if not results:
            print(f"\n  {ARCH_LABELS.get(arch, arch)}: no results found")
            continue
        print(f"\n  {ARCH_LABELS.get(arch, arch)}")
        for task_id in TASKS:
            m      = compute_metrics(results, task_id)
            n_done = len([r for r in results if r["task_id"] == task_id])
            if n_done == 0:
                print(f"    {TASK_LABELS[task_id]}: no data")
                continue
            print(f"    {TASK_LABELS[task_id]}: "
                  f"success_rate={m['success_rate']:.2f} | "
                  f"avg_regret={m['avg_regret']:.3f} | "
                  f"avg_steps={m['avg_steps']:.1f} | "
                  f"first_success={m['first_success_shot']} | "
                  f"convergence={m['convergence_shot']}")
            print(f"      regret_curve: "
                  + " | ".join(f"t{s}:{cr:.3f}" for s, cr in m["regret_curve"]))

    # Summary comparison table
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
