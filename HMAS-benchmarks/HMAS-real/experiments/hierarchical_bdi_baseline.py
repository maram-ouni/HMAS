"""
hierarchical_bdi_baseline.py — Architecture 6: Hierarchical BDI (AIWolf-inspired)
Ref: Harada & Kano (2025) "Intent-driven AIWolf Agents with Hierarchical BDI
     Model and Personality"

Two-level BDI cycle applied to web navigation:

  Macro-BDI (1 call/shot):
      LLM generates a high-level navigation strategy once per shot,
      analogous to AIWolf's daily strategic planning phase.
      The previous shot outcome (reward, answer, strategy) informs
      a different strategy if the previous shot failed.

  Micro-BDI (every step):
      The macro strategy is injected into the system message at EVERY step.
      The agent continuously re-evaluates its immediate action against the
      high-level directive — key difference vs ChatBDI which only injects
      a fixed plan at step 0.

Navigation style (simplified from AIWolf personality model):
  assertiveness : tendency to answer quickly vs explore more    (0..1)
  thoroughness  : tendency to verify across pages vs trust first (0..1)

Usage:
    python mas_project_v2/experiments/hierarchical_bdi_baseline.py
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
RESULTS_FILE        = os.path.join(RESULTS_DIR, "hierarchical_bdi_results.json")
MAX_AXTREE_CHARS    = 25000
_SUBPROCESS_MARKER  = "--_hierbdi-run-shot"

# Navigation style parameters (simplified AIWolf personality)
ASSERTIVENESS = 0.75   # > 0.65 → answer quickly once info visible
THOROUGHNESS  = 0.50   # > 0.65 → verify across multiple pages


# ── Macro-BDI: one LLM call per shot ──────────────────────────────────────

def _generate_macro_strategy(client: OpenAI, goal: str, shot: int,
                              prev_macro: str = "",
                              prev_reward: float = None,
                              prev_answer: str = "") -> str:
    """
    Macro-BDI (Harada & Kano 2025): generates a high-level navigation
    strategy once per shot, analogous to AIWolf daily strategic planning.
    Navigation style parameters bias the generated strategy.
    """
    style_parts = []
    if ASSERTIVENESS > 0.65:
        style_parts.append(
            "Answer as soon as the information is visible — do NOT over-navigate."
        )
    else:
        style_parts.append("Be thorough — navigate until you are certain of the answer.")

    if THOROUGHNESS > 0.65:
        style_parts.append(
            "If the first page is insufficient, check a related page before answering."
        )
    else:
        style_parts.append("Trust the first page that contains the answer.")

    style_hint = " ".join(style_parts)

    prev_section = ""
    if shot > 1 and prev_macro:
        outcome = "succeeded" if (prev_reward is not None and prev_reward >= 1.0) else "failed"
        prev_section = (
            f"\nPrevious shot {shot - 1} {outcome}.\n"
            f"Strategy used: \"{prev_macro}\"\n"
            + (f"Answer given: \"{prev_answer}\"\n" if prev_answer else "")
            + ("You MUST generate a DIFFERENT strategy.\n"
               if outcome == "failed" else "You may refine this strategy.\n")
        )

    prompt = (
        f"You are the Macro-BDI planner for a web navigation agent.\n"
        f"Task: {goal}\n"
        f"{prev_section}\n"
        f"Navigation style: {style_hint}\n\n"
        f"Generate a concise strategic directive (2-3 sentences) specifying:\n"
        f"1. WHERE to navigate (which page or section)\n"
        f"2. WHAT specific information to extract\n"
        f"3. WHEN to call send_msg_to_user (immediately on sight, or after verification)\n\n"
        f"Be specific and actionable. One paragraph only."
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return "Navigate directly to the relevant section and send the answer immediately."


# ── Micro-BDI: inject macro strategy at every step ────────────────────────

def _make_groq_query(macro_strategy: str = ""):
    """
    Micro-BDI: the macro strategy is injected into the system message at
    EVERY step (not just step 0). The agent continuously aligns its
    immediate action with the high-level directive.
    This is the key difference from ChatBDI (fixed plan injected once).
    """
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    def query_model(system_msgs, user_msgs):
        messages = []
        if system_msgs:
            sys_text = system_msgs[0]["text"]
            if macro_strategy:
                sys_text += (
                    "\n\n=== MACRO-BDI DIRECTIVE (apply at every step) ===\n"
                    + macro_strategy
                    + "\nBefore each action, ask: does this action align with the directive above?"
                )
            messages.append({"role": "system", "content": sys_text})
        content = [{"type": "text", "text": m["text"]}
                   for m in user_msgs if m["type"] == "text"]
        messages.append({"role": "user", "content": content})
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.01, max_tokens=4096
        )
        return resp.choices[0].message.content or ""

    return query_model


# ── GroqDemoAgent wired to Micro-BDI ──────────────────────────────────────

class _GroqDemoAgent(DemoAgent):
    def __init__(self, macro_strategy: str = "", **kwargs):
        kwargs.pop("model_name", None)
        kwargs.pop("openrouter_api_key", None)
        super().__init__(model_name="openrouter/dummy",
                         openrouter_api_key="not_used", **kwargs)
        self.model_name  = LLM_MODEL
        self.query_model = _make_groq_query(macro_strategy=macro_strategy)

    def obs_preprocessor(self, obs: dict) -> dict:
        p = super().obs_preprocessor(obs)
        ax = p.get("axtree_txt", "")
        if len(ax) > MAX_AXTREE_CHARS:
            p["axtree_txt"] = ax[:MAX_AXTREE_CHARS] + "\n...[AXTree truncated]"
        return p


@dataclasses.dataclass
class _GroqDemoAgentArgs(DemoAgentArgs):
    model_name:     str = f"groq/{LLM_MODEL}"
    macro_strategy: str = ""

    def make_agent(self):
        return _GroqDemoAgent(
            macro_strategy          = self.macro_strategy,
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
    macro_strategy      = ctx.get("macro_strategy", "")

    t0 = time.perf_counter()
    try:
        harness = REAL.harness(
            agentargs     = _GroqDemoAgentArgs(
                use_axtree=True, use_html=False, use_screenshot=False,
                macro_strategy=macro_strategy,
            ),
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

def _run_shot_subprocess(task_id: str, macro_strategy: str) -> dict:
    ctx = {"task_id": task_id, "macro_strategy": macro_strategy}

    with tempfile.NamedTemporaryFile(suffix="_ctx.json", delete=False,
                                     mode="w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False)
        ctx_file = f.name

    with tempfile.NamedTemporaryFile(suffix="_res.json", delete=False) as f:
        res_file = f.name

    utf8_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    cmd = [
        sys.executable, os.path.abspath(__file__),
        _SUBPROCESS_MARKER,
        "--context-file", ctx_file,
        "--result-file",  res_file,
    ]
    try:
        proc = subprocess.run(cmd, timeout=MAX_STEPS * 120, env=utf8_env)
        if proc.returncode != 0:
            print(f"  [HierBDI] Subprocess returncode={proc.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  [HierBDI] Subprocess timeout")
        return None
    except Exception as e:
        print(f"  [HierBDI] Subprocess error: {e}")
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
    print(f"  ARCHITECTURE 6 — Hierarchical BDI (AIWolf-inspired)")
    print(f"  Ref    : Harada & Kano (2025)")
    print(f"  Model  : {LLM_MODEL}")
    print(f"  Tasks  : {TASKS}")
    print(f"  Shots  : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"  Style  : assertiveness={ASSERTIVENESS} | thoroughness={THOROUGHNESS}")
    print(f"  Macro-BDI : 1 LLM call/shot  (strategic directive)")
    print(f"  Micro-BDI : injected at every step (dynamic re-evaluation)")
    print(f"{'='*62}")

    groq_client  = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    results      = _load_results()
    done_keys    = {(r["task_id"], r["shot"]) for r in results}

    for task_id in TASKS:
        goal          = _get_task_goal(task_id)
        macro_strategy = ""
        prev_reward    = None
        prev_answer    = ""

        print(f"\n  Task : {task_id}")
        print(f"  Goal : {goal[:90]}")

        cumsum_regret = 0.0
        completed = sorted(
            [r for r in results if r["task_id"] == task_id],
            key=lambda x: x["shot"],
        )
        for r in completed:
            cumsum_regret  += (1 - r["reward"])
            macro_strategy  = r.get("macro_strategy", "")
            prev_reward     = r["reward"]
            prev_answer     = r.get("answer", "")

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results
                         if x["task_id"] == task_id and x["shot"] == shot)
                cumsum_regret  += (1 - r["reward"])
                macro_strategy  = r.get("macro_strategy", "")
                prev_reward     = r["reward"]
                prev_answer     = r.get("answer", "")
                print(f"  [Shot {shot}] Already done — reward={r['reward']:.1f} (skipping)")
                continue

            print(f"\n  {'─'*58}")

            # ── Macro-BDI ─────────────────────────────────────────────────
            print(f"  [Shot {shot}/{N_SHOTS}] Macro-BDI: generating strategy...")
            macro_strategy = _generate_macro_strategy(
                groq_client, goal, shot,
                prev_macro=macro_strategy,
                prev_reward=prev_reward,
                prev_answer=prev_answer,
            )
            preview = macro_strategy[:110].replace("\n", " ")
            print(f"  [Macro-BDI] {preview}...")

            # ── Micro-BDI: run shot with macro injected at every step ────
            print(f"  [Micro-BDI] Launching subprocess...")
            raw = _run_shot_subprocess(task_id, macro_strategy)
            if raw is None:
                raw = {"reward": 0.0, "steps": MAX_STEPS, "answer": "",
                       "duration_s": 0.0, "error": "subprocess_failed"}

            cumsum_regret += (1 - raw["reward"])
            cum_regret     = cumsum_regret / shot
            prev_reward    = raw["reward"]
            prev_answer    = raw.get("answer", "")

            record = {
                "architecture":      "hierarchical_bdi",
                "task_id":           task_id,
                "shot":              shot,
                "reward":            raw["reward"],
                "steps":             raw["steps"],
                "answer":            raw.get("answer", ""),
                "duration_s":        raw["duration_s"],
                "regret_t":          round(1 - raw["reward"], 4),
                "cumulative_regret": round(cum_regret, 4),
                "macro_strategy":    macro_strategy,
                "assertiveness":     ASSERTIVENESS,
                "thoroughness":      THOROUGHNESS,
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
    print(f"\n  Hierarchical BDI done — {n_success}/{len(results)} shots succeeded")
    print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    if _SUBPROCESS_MARKER in sys.argv:
        _subprocess_worker()
    else:
        main()
