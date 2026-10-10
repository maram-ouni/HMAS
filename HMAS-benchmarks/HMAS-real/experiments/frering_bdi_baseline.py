"""
frering_bdi_baseline.py — Architecture 3: BDI without separate Verifier

Inspired by Frering, Steinbauer-Wagner & Holzinger (Engineering Applications of AI,
vol. 141, 2025, DOI:10.1016/j.engappai.2024.109771).

The BDI Planner generates deterministic hints between shots. The LLM Executor
navigates the web. There is NO separate Rule-Based Verifier: evaluation uses only
the binary reward (0 or 1) from the REAL benchmark.

This isolates the specific contribution of the Verifier in the full architecture.

Key differences from the full hybrid:
  - No RuleBasedVerifierAgent thread
  - No U_PE, U_V, dim_success, dim_efficiency, dim_safety, dim_specificity
  - BDI receives only reward and steps to update beliefs
  - No threshold decision (reward < 1.0 → replan immediately)
  - StrategyLearner disk persistence DISABLED (fair comparison)
  - No HTML pre-scan (avoids Playwright conflicts between parent/subprocess)

Usage:
    python mas_project_v2/experiments/frering_bdi_baseline.py
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

from core.bdi_model    import BDIState, CommitmentStrategy
from core.plan_library import PlanLibrary
from core.utils        import extract_task_type

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
RESULTS_FILE        = os.path.join(RESULTS_DIR, "frering_bdi_results.json")
MAX_AXTREE_CHARS    = 25000
_SUBPROCESS_MARKER  = "--_frering-run-shot"


# ── Synchronous BDI Coordinator (no threading, no StrategyLearner) ────────

class FreringBDICoordinator:
    """
    Synchronous BDI planner that runs in the parent process between shots.

    Uses BDIState + PlanLibrary from the full architecture, but:
      - No threading (runs synchronously)
      - No StrategyLearner (no disk persistence — fair comparison)
      - No HTML pre-scan (avoids Playwright conflicts with subprocess)
      - Belief update uses only binary reward (no verifier payoffs)
    """

    def __init__(self):
        self._bdi          = None
        self._plan_library = PlanLibrary()

    def reset_for_task(self, task_id: str, task_goal: str) -> None:
        """Reset BDI state for a new task. Called before the first shot of each task."""
        self._bdi = BDIState(commitment_strategy=CommitmentStrategy.OPEN_MINDED)
        self._bdi.set_task(task_id, task_goal, extract_task_type(task_id))
        print(f"  [FreeringBDI] New task: {task_id} | {self._bdi.summary()}")

    def get_hint(self) -> tuple:
        """
        Run options() -> filter() from the BDI cycle.
        Returns (hint_text, plan_name).
        """
        b = self._bdi.beliefs
        ctx = {
            "action_history": b.last_action_history,
            "last_error":     b.last_error,
            "n_steps":        len(b.last_action_history),
            "max_steps":      MAX_STEPS,
            "prescan_hint":   "",   # disabled: avoids Playwright conflicts
            "warm_hint":      "",   # disabled: no StrategyLearner
        }
        try:
            plan = self._plan_library.select_plan(b, **ctx)
            # v2 plans have a body of Actions rather than one hint — this
            # simplified baseline just takes the first executable step
            # (no cross-shot step_index walking; each shot re-deliberates).
            hint = ""
            for action in plan.body:
                step = action.build(b, **ctx)
                if step is None or (step.hint and step.hint in b.tried_hints):
                    continue
                hint = step.hint
                break
            self._bdi.commit(plan_name=plan.name)
            print(f"  [FreeringBDI] Plan: '{plan.name}' (priority={plan.priority})")
            print(f"  [FreeringBDI] Hint: {hint[:90] or '(none — first attempt)'}")
            return hint, plan.name
        except Exception as e:
            print(f"  [FreeringBDI] Plan selection error: {e} — using empty hint")
            return "", "fallback"

    def update(self, reward: float, steps: int, hint_used: str) -> None:
        """
        Update BDI beliefs after a shot completes.
        Uses simplified payoffs (no verifier): u_pe = u_v = reward.
        """
        quality        = "success" if reward >= 1.0 else "rejected"
        dim_efficiency = max(0.0, 1.0 - steps / MAX_STEPS)
        self._bdi.brf(
            reward          = reward,
            u_pe            = float(reward),
            u_v             = float(reward),
            quality         = quality,
            feedback        = "",
            hint            = hint_used,
            dim_efficiency  = dim_efficiency,
            dim_specificity = 0.0,
            action_history  = [],
            last_error      = "",
        )
        print(f"  [FreeringBDI] Beliefs updated: quality={quality} | "
              f"attempt_count={self._bdi.beliefs.attempt_count} | "
              f"tried_hints={len(self._bdi.beliefs.tried_hints)}")


# ── Groq DemoAgent with hint injection ────────────────────────────────────

def _make_groq_query(hint: str = ""):
    """
    query_model for Groq. The BDI hint is appended to the system message
    at every step (persistent strategic guidance).
    """
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    def query_model(system_msgs, user_msgs):
        messages = []
        if system_msgs:
            sys_text = system_msgs[0]["text"]
            if hint:
                sys_text = (sys_text
                            + "\n\n=== STRATEGIC HINT FROM PLANNER ===\n"
                            + hint)
            messages.append({"role": "system", "content": sys_text})
        content = [{"type": "text", "text": m["text"]}
                   for m in user_msgs if m["type"] == "text"]
        messages.append({"role": "user", "content": content})
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.01, max_tokens=4096
        )
        return resp.choices[0].message.content or ""

    return query_model


class _GroqDemoAgent(DemoAgent):
    def __init__(self, hint: str = "", **kwargs):
        kwargs.pop("model_name", None)
        kwargs.pop("openrouter_api_key", None)
        super().__init__(model_name="openrouter/dummy",
                         openrouter_api_key="not_used", **kwargs)
        self.model_name  = LLM_MODEL
        self.query_model = _make_groq_query(hint=hint)

    def obs_preprocessor(self, obs: dict) -> dict:
        p = super().obs_preprocessor(obs)
        ax = p.get("axtree_txt", "")
        if len(ax) > MAX_AXTREE_CHARS:
            p["axtree_txt"] = ax[:MAX_AXTREE_CHARS] + "\n...[AXTree truncated]"
        return p


@dataclasses.dataclass
class _GroqDemoAgentArgs(DemoAgentArgs):
    model_name: str = f"groq/{LLM_MODEL}"
    hint:       str = ""

    def make_agent(self):
        return _GroqDemoAgent(
            hint                    = self.hint,
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
    hint                = ctx.get("hint", "")

    t0 = time.perf_counter()
    try:
        harness = REAL.harness(
            agentargs     = _GroqDemoAgentArgs(
                use_axtree=True, use_html=False, use_screenshot=False,
                hint=hint,
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

def _run_shot_subprocess(task_id: str, hint: str) -> dict:
    ctx = {"task_id": task_id, "hint": hint}

    with tempfile.NamedTemporaryFile(suffix="_ctx.json", delete=False,
                                     mode="w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False)
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
            print(f"  [FreeringBDI] Subprocess returncode={proc.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  [FreeringBDI] Subprocess timeout")
        return None
    except Exception as e:
        print(f"  [FreeringBDI] Subprocess error: {e}")
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
    print(f"  ARCHITECTURE 3 — BDI without Verifier (ablation)")
    print(f"  Ablation: isolates the contribution of the Verifier vs Hybrid")
    print(f"  Model  : {LLM_MODEL}")
    print(f"  Tasks  : {TASKS}")
    print(f"  Shots  : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"  Planner: BDI deterministic | Verifier: disabled")
    print(f"{'='*62}")

    results     = _load_results()
    done_keys   = {(r["task_id"], r["shot"]) for r in results}
    bdi_coord   = FreringBDICoordinator()

    for task_id in TASKS:
        goal = _get_task_goal(task_id)
        bdi_coord.reset_for_task(task_id, goal)

        print(f"\n  Task : {task_id}")
        print(f"  Goal : {goal[:90]}")

        cumsum_regret = 0.0

        # Replay history to restore BDI state for resume
        completed = sorted(
            [r for r in results if r["task_id"] == task_id],
            key=lambda x: x["shot"]
        )
        for r in completed:
            cumsum_regret += (1 - r["reward"])
            bdi_coord.update(r["reward"], r["steps"], r.get("hint_used", ""))

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results
                         if x["task_id"] == task_id and x["shot"] == shot)
                print(f"  [Shot {shot}] Already done — reward={r['reward']:.1f} (skipping)")
                continue

            # BDI generates hint deterministically in the parent process
            hint, plan_name = bdi_coord.get_hint()

            print(f"\n  {'─'*58}")
            print(f"  [Shot {shot}/{N_SHOTS}] BDI plan='{plan_name}' | "
                  f"attempt_count={bdi_coord._bdi.beliefs.attempt_count}")
            print(f"  [Shot {shot}/{N_SHOTS}] Launching subprocess...")

            raw = _run_shot_subprocess(task_id, hint)
            if raw is None:
                raw = {"reward": 0.0, "steps": MAX_STEPS, "answer": "",
                       "duration_s": 0.0, "error": "subprocess_failed"}

            # Update BDI in parent process with the shot result
            bdi_coord.update(raw["reward"], raw["steps"], hint)

            cumsum_regret += (1 - raw["reward"])
            cum_regret     = cumsum_regret / shot

            record = {
                "architecture":     "frering_bdi",
                "task_id":          task_id,
                "shot":             shot,
                "reward":           raw["reward"],
                "steps":            raw["steps"],
                "answer":           raw["answer"],
                "duration_s":       raw["duration_s"],
                "regret_t":         round(1 - raw["reward"], 4),
                "cumulative_regret": round(cum_regret, 4),
                "hint_used":        hint,
                "hint_rule_applied": plan_name,
                "attempt_count":    bdi_coord._bdi.beliefs.attempt_count,
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
    print(f"\n  BDI without Verifier done — {n_success}/{len(results)} shots succeeded")
    print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    if _SUBPROCESS_MARKER in sys.argv:
        _subprocess_worker()
    else:
        main()
