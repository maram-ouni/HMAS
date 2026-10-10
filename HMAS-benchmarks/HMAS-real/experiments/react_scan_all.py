"""
react_scan_all.py — Scan de toutes les nouvelles tâches avec ReAct

Lance ReAct (Architecture 1) sur les 90 tâches non encore testées du benchmark
REAL (hors omnizon HTTP 451, hors 12 déjà utilisées). But : identifier quelles
tâches sont valides, discriminantes ou impossibles pour tous les agents.

Classification finale par tâche :
  - ALWAYS_FAIL  : reward=0 sur les 5 shots → trop difficile ou tâche cassée
  - ALWAYS_PASS  : reward=1 sur les 5 shots → trop facile, peu discriminante
  - DISCRIMINANT : entre 1 et 4 succès sur 5 → idéale pour comparer les archi.

Usage:
    python mas_project_v2/experiments/react_scan_all.py
"""
from __future__ import annotations

import argparse
import dataclasses
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

from openai import OpenAI
from agisdk import REAL
from agisdk.REAL.demo_agent.basic_agent import DemoAgent, DemoAgentArgs
from config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL

# ── 90 nouvelles tâches (hors omnizon + hors 12 déjà testées) ─────────────
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
RESULTS_FILE        = os.path.join(RESULTS_DIR, "react_scan_results.json")
MAX_AXTREE_CHARS    = 25000
_SUBPROCESS_MARKER  = "--_react-scan-shot"


# ── Agent (ReAct pur, identique à react_baseline.py) ──────────────────────

def _make_groq_query():
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    def query_model(system_msgs, user_msgs):
        messages = []
        if system_msgs:
            messages.append({"role": "system", "content": system_msgs[0]["text"]})
        content = [{"type": "text", "text": m["text"]}
                   for m in user_msgs if m["type"] == "text"]
        messages.append({"role": "user", "content": content})
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.01, max_tokens=4096
        )
        return resp.choices[0].message.content or ""
    return query_model


class _GroqDemoAgent(DemoAgent):
    def __init__(self, **kwargs):
        kwargs.pop("model_name", None)
        kwargs.pop("openrouter_api_key", None)
        super().__init__(model_name="openrouter/dummy",
                         openrouter_api_key="not_used", **kwargs)
        self.model_name  = LLM_MODEL
        self.query_model = _make_groq_query()

    def obs_preprocessor(self, obs: dict) -> dict:
        p = super().obs_preprocessor(obs)
        ax = p.get("axtree_txt", "")
        if len(ax) > MAX_AXTREE_CHARS:
            p["axtree_txt"] = ax[:MAX_AXTREE_CHARS] + "\n...[AXTree truncated]"
        return p


@dataclasses.dataclass
class _GroqDemoAgentArgs(DemoAgentArgs):
    model_name: str = f"groq/{LLM_MODEL}"
    def make_agent(self):
        return _GroqDemoAgent(
            chat_mode               = self.chat_mode,
            demo_mode               = self.demo_mode,
            use_html                = self.use_html,
            use_axtree              = self.use_axtree,
            use_screenshot          = self.use_screenshot,
            system_message_handling = self.system_message_handling,
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_task(task_id: str):
    name = task_id.split(".", 1)[-1]
    idx  = name.rfind("-")
    return name[:idx], int(name[idx + 1:])


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


def _load_results() -> list:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_results(results: list) -> None:
    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Subprocess worker ──────────────────────────────────────────────────────

def _subprocess_worker() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-file", required=True)
    ap.add_argument("--result-file",  required=True)
    args, _ = ap.parse_known_args()

    with open(args.context_file, encoding="utf-8") as f:
        ctx = json.load(f)

    task_type, task_num = _parse_task(ctx["task_id"])
    t0 = time.perf_counter()
    try:
        harness = REAL.harness(
            agentargs     = _GroqDemoAgentArgs(
                use_axtree=True, use_html=False, use_screenshot=False),
            headless      = True,
            max_steps     = MAX_STEPS,
            num_workers   = 1,
            task_type     = task_type,
            task_id       = task_num,
            force_refresh = True,
        )
        raw = harness.run()
        res = {}
        if isinstance(raw, dict):
            for v in raw.values():
                res = v; break
        reward = float(res.get("cum_reward", 0.0))
        steps  = int(res.get("stats.cum_steps", res.get("n_steps", MAX_STEPS)))
        answer = str(res.get("answer", ""))
        error  = None
    except Exception as e:
        reward, steps, answer = 0.0, MAX_STEPS, ""
        error = str(e)[:200]

    duration = time.perf_counter() - t0
    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump({"reward": reward, "steps": steps, "answer": answer,
                   "duration_s": round(duration, 1), "error": error}, f)


def _run_shot_subprocess(task_id: str) -> dict | None:
    ctx = {"task_id": task_id}
    with tempfile.NamedTemporaryFile(suffix="_ctx.json", delete=False,
                                     mode="w", encoding="utf-8") as f:
        json.dump(ctx, f); ctx_file = f.name
    with tempfile.NamedTemporaryFile(suffix="_res.json", delete=False) as f:
        res_file = f.name

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, os.path.abspath(__file__),
           _SUBPROCESS_MARKER, "--context-file", ctx_file, "--result-file", res_file]
    try:
        subprocess.run(cmd, timeout=MAX_STEPS * 120, env=env)
    except subprocess.TimeoutExpired:
        print(f"  [SCAN] Timeout"); return None
    except Exception as e:
        print(f"  [SCAN] Error: {e}"); return None
    finally:
        if os.path.exists(ctx_file): os.unlink(ctx_file)

    if os.path.exists(res_file):
        with open(res_file, encoding="utf-8") as f:
            result = json.load(f)
        os.unlink(res_file)
        return result
    return None


# ── Résumé de classification ───────────────────────────────────────────────

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
    print(f"  CLASSIFICATION DES TACHES ({len(by_task)} completes / {len(TASKS)} total)")
    print(f"{'='*65}")

    print(f"\n  DISCRIMINANTES ({len(discriminant)}) — ideales pour comparer les archi :")
    for t, n in sorted(discriminant, key=lambda x: -x[1]):
        print(f"    {t:<28}  {n}/{N_SHOTS} succes")

    print(f"\n  TOUJOURS REUSSIES ({len(always_pass)}) — trop faciles :")
    for t, n in sorted(always_pass):
        print(f"    {t}")

    print(f"\n  TOUJOURS ECHOUEES ({len(always_fail)}) — impossibles ou cassees :")
    for t, n in sorted(always_fail):
        print(f"    {t}")

    print(f"\n  RECOMMANDATION : utiliser les {len(discriminant)} taches discriminantes")
    print(f"  + les 12 deja testees = {len(discriminant)+12} taches pour la comparaison finale")
    print(f"{'='*65}\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n{'='*65}")
    print(f"  SCAN REACT — {len(TASKS)} nouvelles taches")
    print(f"  Model  : {LLM_MODEL}")
    print(f"  Shots  : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"  But    : identifier les taches discriminantes")
    print(f"{'='*65}")

    results   = _load_results()
    done_keys = {(r["task_id"], r["shot"]) for r in results}
    n_done    = len({r["task_id"] for r in results if
                     sum(1 for x in results if x["task_id"]==r["task_id"]) >= N_SHOTS})

    print(f"  Resume : {len(done_keys)} shots deja effectues ({n_done}/{len(TASKS)} taches completes)\n")

    for i, task_id in enumerate(TASKS):
        goal = _get_task_goal(task_id)
        shots_done = [r for r in results if r["task_id"] == task_id]

        if len(shots_done) >= N_SHOTS:
            n_ok = sum(1 for r in shots_done if r["reward"] >= 1.0)
            print(f"  [{i+1:>2}/{len(TASKS)}] {task_id:<28} SKIP ({n_ok}/{N_SHOTS})")
            continue

        print(f"\n  [{i+1:>2}/{len(TASKS)}] {task_id}")
        print(f"  Goal : {goal[:80]}")
        cumsum = sum(1 - r["reward"] for r in shots_done)

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results if x["task_id"]==task_id and x["shot"]==shot)
                cumsum += (1 - r["reward"])
                print(f"  [Shot {shot}] SKIP reward={r['reward']:.1f}")
                continue

            raw = _run_shot_subprocess(task_id)
            if raw is None:
                raw = {"reward": 0.0, "steps": MAX_STEPS, "answer": "",
                       "duration_s": 0.0, "error": "subprocess_failed"}

            cumsum += (1 - raw["reward"])
            record = {
                "architecture": "react_scan",
                "task_id":      task_id,
                "shot":         shot,
                "reward":       raw["reward"],
                "steps":        raw["steps"],
                "answer":       raw["answer"],
                "duration_s":   raw["duration_s"],
                "regret_t":     round(1 - raw["reward"], 4),
                "cumulative_regret": round(cumsum / shot, 4),
                "error":        raw.get("error"),
            }
            results.append(record)
            done_keys.add((task_id, shot))
            _save_results(results)

            status = "OK" if raw["reward"] >= 1.0 else "FAIL"
            print(f"  [Shot {shot}] {status} | reward={raw['reward']:.1f} | "
                  f"steps={raw['steps']} | time={raw['duration_s']:.1f}s")

            is_last = (shot == N_SHOTS and task_id == TASKS[-1])
            if not is_last:
                time.sleep(SLEEP_BETWEEN_SHOTS)

    _print_classification(results)


if __name__ == "__main__":
    if _SUBPROCESS_MARKER in sys.argv:
        _subprocess_worker()
    else:
        main()
