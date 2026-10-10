"""
chatbdi_baseline.py — Architecture 4: ChatBDI
(Gatti, Mascardi & Ferrando, AAMAS 2025, p2541)

ChatBDI "chattifies" the BDI cycle: instead of a static plan library, the LLM
receives the agent's current Beliefs (actual page state as natural language) +
the Desire (task goal) and dynamically generates a plan. The plan is then
injected into the executor at every subsequent step.

Key fix vs. previous version: the planner now receives the REAL initial axtree
(beliefs) captured at step 0 before any action — not just the abstract goal.

Architecture:
  Step 0  : capture axtree (Beliefs) from the live page
  Step 0  : LLM Planner(Beliefs + Desire) → natural-language plan (Intention)
  Step 1+ : LLM Executor receives plan injected at every step

Usage:
    python mas_project_v2/experiments/chatbdi_baseline.py
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
RESULTS_FILE        = os.path.join(RESULTS_DIR, "chatbdi_results.json")
MAX_AXTREE_CHARS    = 25000
MAX_BELIEFS_CHARS   = 3000   # cap beliefs sent to planner (cost/latency control)
_SUBPROCESS_MARKER  = "--_chatbdi-run-shot"

# Module-level capture so the subprocess worker can retrieve the plan
# generated inside the agent (harness doesn't expose agent internals).
_CAPTURED_PLANS: list[str] = []


# ── BDI Plan generation (called from INSIDE the agent at step 0) ───────────

def _generate_plan_shot1(client: OpenAI, goal: str, beliefs: str) -> str:
    """
    BDI Planner — shot 1.
    Receives Beliefs (initial page state) + Desire (goal) → Intention (plan).
    This is the core ChatBDI mechanism: the LLM replaces the static plan library
    and generates a context-aware plan grounded in actual page content.
    """
    prompt = (
        "You are a BDI planning agent. You have observed the current state of a "
        "web page (your Beliefs) and you know what you need to accomplish (your Desire).\n\n"
        "=== BELIEFS (current page state) ===\n"
        f"{beliefs}\n\n"
        "=== DESIRE (task goal) ===\n"
        f"{goal}\n\n"
        "Based on what you see on the page right now, generate a numbered step-by-step "
        "plan (3 to 7 steps) to accomplish the goal. Each step must reference specific "
        "elements visible in the page state above (links, buttons, fields, text).\n"
        "The FINAL step MUST be:\n"
        "  Final step: send_msg_to_user('REAL_VALUE_FOUND_ON_PAGE')\n"
        "where REAL_VALUE_FOUND_ON_PAGE is the actual value read from the page.\n\n"
        "Respond ONLY with the numbered steps. No explanation, no preamble."
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Step 1: Read the page content carefully\nFinal step: send_msg_to_user('value found on page')\n(Plan generation failed: {e})"


def _generate_plan_retry(client: OpenAI, goal: str, beliefs: str,
                          previous_plan: str, reward: float,
                          steps: int, answer: str) -> str:
    """
    BDI Planner — shot 2+.
    Receives updated Beliefs + failure context → generates a different plan.
    """
    prompt = (
        "You are a BDI planning agent. A previous attempt has failed. You now see "
        "the current page state (updated Beliefs) and must generate a completely "
        "different plan.\n\n"
        "=== BELIEFS (current page state) ===\n"
        f"{beliefs}\n\n"
        "=== DESIRE (task goal) ===\n"
        f"{goal}\n\n"
        "=== PREVIOUS PLAN (failed) ===\n"
        f"{previous_plan}\n\n"
        "=== FAILURE CONTEXT ===\n"
        f"  Reward: {reward} (0=failure, 1=success)\n"
        f"  Steps used: {steps}/{MAX_STEPS}\n"
        f"  Answer submitted: {answer if answer else '(none — agent did not answer)'}\n\n"
        "Looking at the current page, generate a COMPLETELY DIFFERENT plan that avoids "
        "the previous failure. Use elements actually visible in the Beliefs above.\n"
        "Numbered steps (3 to 7), ending with:\n"
        "  Final step: send_msg_to_user('REAL_VALUE_FOUND_ON_PAGE')\n"
        "Respond ONLY with the numbered steps."
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Step 1: Try a different navigation approach\nFinal step: send_msg_to_user('value found on page')\n(Plan generation failed: {e})"


def _build_context_injection(plan: str) -> str:
    return (
        "=== BDI INTENTIONAL PLAN ===\n"
        "The BDI planner has generated the following plan based on the page you "
        "observed at the start. Follow these steps to accomplish the task:\n\n"
        f"{plan}\n\n"
        "RULES:\n"
        "- Follow this plan. Do not deviate unless the page structure forces it.\n"
        "- Do NOT guess values. Submit ONLY what you actually read on the page.\n"
        "- When you reach the final step, call send_msg_to_user with the real value."
    )


def _count_plan_steps(plan: str) -> int:
    lines = [ln.strip() for ln in plan.splitlines() if ln.strip()]
    return max(sum(
        1 for ln in lines
        if ln[0].isdigit() or ln.lower().startswith("step") or ln.lower().startswith("final")
    ), 1)


# ── ChatBDI DemoAgent — belief capture + plan generation at step 0 ─────────

class _ChatBDIDemoAgent(DemoAgent):
    """
    At step 0: captures beliefs (axtree) → calls LLM planner → stores plan.
    At all steps: injects the BDI plan into the system message.
    This makes the planner context-aware (grounded in actual page state).
    """

    def __init__(self, goal: str, prev_plan: str, prev_reward: float,
                 prev_steps: int, prev_answer: str, **kwargs):
        self._goal        = goal
        self._prev_plan   = prev_plan
        self._prev_reward = prev_reward
        self._prev_steps  = prev_steps
        self._prev_answer = prev_answer
        self._plan        = None   # set at first obs_preprocessor call
        self._groq        = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

        kwargs.pop("model_name", None)
        kwargs.pop("openrouter_api_key", None)
        super().__init__(model_name="openrouter/dummy",
                         openrouter_api_key="not_used", **kwargs)
        self.model_name  = LLM_MODEL
        self.query_model = self._build_query_fn()

    def _build_query_fn(self):
        agent = self
        client = self._groq

        def query_model(system_msgs, user_msgs):
            messages = []
            if system_msgs:
                sys_text = system_msgs[0]["text"]
                if agent._plan:
                    sys_text += "\n\n" + _build_context_injection(agent._plan)
                messages.append({"role": "system", "content": sys_text})
            content = [{"type": "text", "text": m["text"]}
                       for m in user_msgs if m["type"] == "text"]
            messages.append({"role": "user", "content": content})
            resp = client.chat.completions.create(
                model=LLM_MODEL, messages=messages,
                temperature=0.01, max_tokens=4096,
            )
            return resp.choices[0].message.content or ""

        return query_model

    def obs_preprocessor(self, obs: dict) -> dict:
        p = super().obs_preprocessor(obs)
        ax = p.get("axtree_txt", "")
        if len(ax) > MAX_AXTREE_CHARS:
            p["axtree_txt"] = ax[:MAX_AXTREE_CHARS] + "\n...[AXTree truncated]"

        # Step 0 only: capture beliefs → call planner → store plan
        if self._plan is None:
            beliefs = ax[:MAX_BELIEFS_CHARS]
            if self._prev_plan:
                self._plan = _generate_plan_retry(
                    self._groq, self._goal, beliefs,
                    self._prev_plan, self._prev_reward,
                    self._prev_steps, self._prev_answer,
                )
            else:
                self._plan = _generate_plan_shot1(
                    self._groq, self._goal, beliefs,
                )
            # Expose plan to subprocess worker via module-level list
            _CAPTURED_PLANS.append(self._plan)
            n = _count_plan_steps(self._plan)
            print(f"  [ChatBDI] Plan generated from beliefs ({len(beliefs)} chars), {n} steps:")
            for ln in self._plan.splitlines()[:4]:
                print(f"    {ln}")

        return p


@dataclasses.dataclass
class _ChatBDIAgentArgs(DemoAgentArgs):
    model_name:  str   = f"groq/{LLM_MODEL}"
    goal:        str   = ""
    prev_plan:   str   = ""
    prev_reward: float = 0.0
    prev_steps:  int   = 0
    prev_answer: str   = ""

    def make_agent(self):
        return _ChatBDIDemoAgent(
            goal        = self.goal,
            prev_plan   = self.prev_plan,
            prev_reward = self.prev_reward,
            prev_steps  = self.prev_steps,
            prev_answer = self.prev_answer,
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
            agentargs = _ChatBDIAgentArgs(
                use_axtree  = True,
                use_html    = False,
                use_screenshot = False,
                goal        = ctx["goal"],
                prev_plan   = ctx.get("prev_plan", ""),
                prev_reward = ctx.get("prev_reward", 0.0),
                prev_steps  = ctx.get("prev_steps", 0),
                prev_answer = ctx.get("prev_answer", ""),
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
    plan_generated = _CAPTURED_PLANS[-1] if _CAPTURED_PLANS else ""
    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump({
            "reward":         reward,
            "steps":          steps,
            "answer":         answer,
            "duration_s":     round(duration, 1),
            "error":          error,
            "plan_generated": plan_generated,
        }, f, ensure_ascii=False)


# ── Subprocess launcher ────────────────────────────────────────────────────

def _run_shot_subprocess(task_id: str, goal: str, prev_plan: str,
                          prev_reward: float, prev_steps: int,
                          prev_answer: str) -> dict | None:
    ctx = {
        "task_id":     task_id,
        "goal":        goal,
        "prev_plan":   prev_plan,
        "prev_reward": prev_reward,
        "prev_steps":  prev_steps,
        "prev_answer": prev_answer,
    }

    with tempfile.NamedTemporaryFile(suffix="_ctx.json", delete=False,
                                     mode="w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False)
        ctx_file = f.name

    with tempfile.NamedTemporaryFile(suffix="_res.json", delete=False) as f:
        res_file = f.name

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable, os.path.abspath(__file__),
        _SUBPROCESS_MARKER,
        "--context-file", ctx_file,
        "--result-file",  res_file,
    ]
    try:
        subprocess.run(cmd, timeout=MAX_STEPS * 120, env=env)
    except subprocess.TimeoutExpired:
        print(f"  [ChatBDI] Subprocess timeout")
        return None
    except Exception as e:
        print(f"  [ChatBDI] Subprocess error: {e}")
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
    print(f"  ARCHITECTURE 4 — ChatBDI (Gatti et al., AAMAS 2025)")
    print(f"  Model   : {LLM_MODEL}")
    print(f"  Tasks   : {len(TASKS)} tasks")
    print(f"  Shots   : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"  Planner : LLM(Beliefs + Desire) at step 0 of each shot")
    print(f"  Beliefs : axtree captured from live page (first {MAX_BELIEFS_CHARS} chars)")
    print(f"{'='*62}")

    results   = _load_results()
    done_keys = {(r["task_id"], r["shot"]) for r in results}

    for task_id in TASKS:
        goal        = _get_task_goal(task_id)
        last_plan   = ""
        last_reward = 0.0
        last_steps  = 0
        last_answer = ""

        print(f"\n  Task : {task_id}")
        print(f"  Goal : {goal[:90]}")

        cumsum_regret = 0.0
        completed = sorted(
            [r for r in results if r["task_id"] == task_id],
            key=lambda x: x["shot"],
        )
        for r in completed:
            cumsum_regret += (1 - r["reward"])
            last_plan   = r.get("plan_generated", "") or last_plan
            last_reward = r["reward"]
            last_steps  = r["steps"]
            last_answer = r.get("answer", "")

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results
                         if x["task_id"] == task_id and x["shot"] == shot)
                print(f"  [Shot {shot}] Already done — reward={r['reward']:.1f} (skipping)")
                continue

            print(f"\n  {'─'*58}")
            print(f"  [Shot {shot}/{N_SHOTS}] Launching subprocess (plan will be generated inside)...")

            prev_plan   = last_plan   if shot > 1 else ""
            prev_reward = last_reward if shot > 1 else 0.0
            prev_steps  = last_steps  if shot > 1 else 0
            prev_answer = last_answer if shot > 1 else ""

            raw = _run_shot_subprocess(
                task_id, goal, prev_plan, prev_reward, prev_steps, prev_answer
            )
            if raw is None:
                raw = {"reward": 0.0, "steps": MAX_STEPS, "answer": "",
                       "duration_s": 0.0, "error": "subprocess_failed"}

            last_reward = raw["reward"]
            last_steps  = raw["steps"]
            last_answer = raw.get("answer", "")
            last_plan   = raw.get("plan_generated", "")  # returned by subprocess

            cumsum_regret += (1 - raw["reward"])
            cum_regret     = cumsum_regret / shot

            record = {
                "architecture":      "chatbdi",
                "task_id":           task_id,
                "shot":              shot,
                "reward":            raw["reward"],
                "steps":             raw["steps"],
                "answer":            raw["answer"],
                "duration_s":        raw["duration_s"],
                "regret_t":          round(1 - raw["reward"], 4),
                "cumulative_regret": round(cum_regret, 4),
                "plan_generated":    last_plan,
                "context_injected":  "beliefs+goal at step 0",
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
    print(f"\n  ChatBDI done — {n_success}/{len(results)} shots succeeded")
    print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    if _SUBPROCESS_MARKER in sys.argv:
        _subprocess_worker()
    else:
        main()
