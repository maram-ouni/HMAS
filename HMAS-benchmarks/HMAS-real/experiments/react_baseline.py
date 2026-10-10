"""
react_baseline.py — Architecture 1: ReAct (Yao et al., NeurIPS 2022, arXiv:2210.03629)

Zero-baseline: the LLM alternates Reasoning and Acting at each step.
No memory between shots, no planner, no verifier.
Each of the 5 shots starts from scratch with the original task goal unchanged.

Usage:
    python mas_project_v2/experiments/react_baseline.py
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
from config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL, RESULTS_SUFFIX

# ── Experiment parameters ──────────────────────────────────────────────────
TASKS = [
    "v1.networkin-2",
    "v1.networkin-4",
    "v1.opendining-1",
    "v1.dashdish-1",
    "v1.dashdish-6",
    "v1.gomail-1",
    "v1.topwork-8",
    # nouvelles — retrieval/easy pour maximiser le signal discriminant
    "v1.fly-unified-1",   # retrieval/easy  — vol le plus tôt (lecture résultats)
    "v1.gocalendar-2",    # action/easy     — lister events (navigation + lecture)
    "v1.staynb-3",        # retrieval/easy  — différence de prix (lecture pure)
    "v1.udriver-2",       # retrieval/easy  — dernier restaurant (lecture historique)
    "v1.zilloft-1",       # retrieval/easy  — compter listings (lecture + filtre)
]
N_SHOTS             = 10
MAX_STEPS           = 10
SLEEP_BETWEEN_SHOTS = 8
RESULTS_DIR         = os.path.join(_EXPERIMENTS_DIR, "results")
RESULTS_FILE        = os.path.join(RESULTS_DIR, f"react_results{RESULTS_SUFFIX}.json")
MAX_AXTREE_CHARS    = 25000
_SUBPROCESS_MARKER  = "--_react-run-shot"


# ── Groq DemoAgent (no hint injection for ReAct) ───────────────────────────

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
                res = v
                break
        reward = float(res.get("cum_reward", 0.0))
        steps  = int(res.get("stats.cum_steps", res.get("n_steps", MAX_STEPS)))
        answer = str(res.get("answer", ""))
        error  = None
    except Exception as e:
        reward, steps, answer = 0.0, MAX_STEPS, ""
        error = str(e)[:200]

    duration = time.perf_counter() - t0
    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump({
            "reward": reward, "steps": steps, "answer": answer,
            "duration_s": round(duration, 1), "error": error,
        }, f, ensure_ascii=False)


# ── Subprocess launcher ────────────────────────────────────────────────────

def _run_shot_subprocess(task_id: str) -> dict:
    ctx = {"task_id": task_id}

    with tempfile.NamedTemporaryFile(suffix="_ctx.json", delete=False,
                                     mode="w", encoding="utf-8") as f:
        json.dump(ctx, f)
        ctx_file = f.name

    with tempfile.NamedTemporaryFile(suffix="_res.json", delete=False) as f:
        res_file = f.name

    cmd = [
        sys.executable, os.path.abspath(__file__),
        _SUBPROCESS_MARKER,
        "--context-file", ctx_file,
        "--result-file",  res_file,
    ]
    try:
        proc = subprocess.run(cmd, timeout=MAX_STEPS * 120)
        if proc.returncode != 0:
            print(f"  [ReAct] Subprocess returncode={proc.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  [ReAct] Subprocess timeout")
        return None
    except Exception as e:
        print(f"  [ReAct] Subprocess error: {e}")
        return None
    finally:
        if os.path.exists(ctx_file):
            os.unlink(ctx_file)

    if os.path.exists(res_file):
        with open(res_file, encoding="utf-8") as f:
            result = json.load(f)
        os.unlink(res_file)
        return result
    return None


# ── Main orchestrator ──────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'='*62}")
    print(f"  ARCHITECTURE 1 — ReAct (Yao et al., NeurIPS 2022)")
    print(f"  Model  : {LLM_MODEL}")
    print(f"  Tasks  : {TASKS}")
    print(f"  Shots  : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"  Memory : none (zero-baseline)")
    print(f"{'='*62}")

    results   = _load_results()
    done_keys = {(r["task_id"], r["shot"]) for r in results}

    for task_id in TASKS:
        goal = _get_task_goal(task_id)
        print(f"\n  Task : {task_id}")
        print(f"  Goal : {goal[:90]}")

        cumsum_regret = 0.0
        completed     = [r for r in results if r["task_id"] == task_id]
        for r in sorted(completed, key=lambda x: x["shot"]):
            cumsum_regret += (1 - r["reward"])

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results
                         if x["task_id"] == task_id and x["shot"] == shot)
                cumsum_regret += (1 - r["reward"])
                print(f"  [Shot {shot}] Already done — reward={r['reward']:.1f} (skipping)")
                continue

            print(f"\n  {'─'*58}")
            print(f"  [Shot {shot}/{N_SHOTS}] Launching subprocess...")

            raw = _run_shot_subprocess(task_id)
            if raw is None:
                raw = {"reward": 0.0, "steps": MAX_STEPS, "answer": "",
                       "duration_s": 0.0, "error": "subprocess_failed"}

            cumsum_regret += (1 - raw["reward"])
            cum_regret     = cumsum_regret / shot

            record = {
                "architecture":     "react",
                "task_id":          task_id,
                "shot":             shot,
                "reward":           raw["reward"],
                "steps":            raw["steps"],
                "answer":           raw["answer"],
                "duration_s":       raw["duration_s"],
                "regret_t":         round(1 - raw["reward"], 4),
                "cumulative_regret": round(cum_regret, 4),
                "context_injected": "",
            }
            results.append(record)
            done_keys.add((task_id, shot))
            _save_results(results)

            status = "SUCCESS" if raw["reward"] >= 1.0 else "FAIL"
            print(f"  [Shot {shot}] {status} | reward={raw['reward']:.1f} | "
                  f"steps={raw['steps']} | cum_regret={cum_regret:.3f} | "
                  f"time={raw['duration_s']:.1f}s")

            is_last = (shot == N_SHOTS and task_id == TASKS[-1])
            if not is_last:
                print(f"  Sleeping {SLEEP_BETWEEN_SHOTS}s (rate-limit safety)...")
                time.sleep(SLEEP_BETWEEN_SHOTS)

    n_success = sum(1 for r in results if r["reward"] >= 1.0)
    print(f"\n  ReAct done — {n_success}/{len(results)} shots succeeded")
    print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    if _SUBPROCESS_MARKER in sys.argv:
        _subprocess_worker()
    else:
        main()
