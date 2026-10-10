"""
hri_bdi_baseline.py — Architecture 7: BDI-LLM Dual (HRI paper)
Ref: "Integrating BDI agents with LLMs for reliable HRI and XAI" (2024)

Two-LLM pipeline with Mind State sharing:

  LLM 1 (Interpreter, 1 call/shot, parent process):
      Reads the task goal + previous shot context → produces a structured
      JSON belief state: {page_to_navigate, element_to_find,
                          answer_strategy, answer_format, key_hint}.

  BDI (deterministic, parent process):
      Uses structured beliefs + tried_hints → selects the best hint.
      Priority: immediate > navigate > key_hint > fallback.

  LLM 2 (Executor, DemoAgent, every step):
      At EVERY step the system message is extended with:
        === BDI MIND STATE ===
        {beliefs_summary}
        === BDI DIRECTIVE ===
        {hint}
      This "Mind State" injection is the key HRI contribution: the
      executor always knows the interpreter's reasoning.

Usage:
    python mas_project_v2/experiments/hri_bdi_baseline.py
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
RESULTS_FILE        = os.path.join(RESULTS_DIR, "hri_bdi_results.json")
MAX_AXTREE_CHARS    = 25000
_SUBPROCESS_MARKER  = "--_hribdi-run-shot"

_FALLBACK_BELIEFS = {
    "page_to_navigate":  "",
    "element_to_find":   "",
    "answer_strategy":   "navigate",
    "answer_format":     "text",
    "key_hint":          "",
}


# ── LLM 1 (Interpreter): structured belief extraction ─────────────────────

def _interpret_beliefs(client: OpenAI, goal: str, shot: int,
                       prev_reward: float = None,
                       prev_answer: str = "") -> dict:
    """
    LLM 1 (HRI paper): reads goal + previous context → structured beliefs.
    Returns a dict with page_to_navigate, element_to_find, answer_strategy,
    answer_format, key_hint.
    """
    prev_section = ""
    if shot > 1 and prev_reward is not None:
        outcome = "succeeded" if prev_reward >= 1.0 else "FAILED"
        prev_section = (
            f"\nPrevious attempt {shot - 1} {outcome}."
            + (f" Answer given: \"{prev_answer}\"." if prev_answer else "")
            + (" Try a different approach.\n" if outcome == "FAILED" else "\n")
        )

    system = (
        "You are a goal interpreter for a web navigation agent. "
        "Parse the task goal and output ONLY valid JSON — no markdown, no explanation. "
        "Keys: page_to_navigate (URL path or page name, empty string if answer is on current page), "
        "element_to_find (CSS label / heading / field name to look for), "
        "answer_strategy ('immediate' if answer is on the landing page / homepage, else 'navigate'), "
        "answer_format (e.g. 'number', 'name', 'text', 'email'), "
        "key_hint (one sentence hint for the executor, empty string if none)."
    )
    user = (
        f"Task goal: {goal}\n"
        f"{prev_section}"
        "Produce the JSON belief state."
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=180,
            temperature=0.3,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        beliefs = json.loads(raw)
        # Ensure all keys present
        for k, v in _FALLBACK_BELIEFS.items():
            beliefs.setdefault(k, v)
        return beliefs
    except Exception as e:
        print(f"  [HRI-LLM1] Parse error ({e}), using fallback beliefs")
        return dict(_FALLBACK_BELIEFS)


# ── BDI plan selector (deterministic) ─────────────────────────────────────

def _bdi_select_hint(shot: int, beliefs: dict, tried_hints: list) -> str:
    """
    BDI (deterministic): selects the best hint from structured beliefs.
    Priority: immediate > navigate > key_hint > fallback.
    Rotates wording on each shot using shot number so hint is always fresh.
    """
    strategy = beliefs.get("answer_strategy", "navigate")
    page     = beliefs.get("page_to_navigate", "")
    element  = beliefs.get("element_to_find", "")
    fmt      = beliefs.get("answer_format", "text")
    key_hint = beliefs.get("key_hint", "")

    if strategy == "immediate":
        hint = (
            f"[Shot {shot}] IMMEDIATE ANSWER: The answer is already visible on the "
            f"current page. Do NOT navigate away. Find the {element or 'requested value'} "
            f"on screen right now and call send_msg_to_user with the exact {fmt} value."
        )
    elif page:
        hint = (
            f"[Shot {shot}] Navigate to {page}. "
            + (f"Look for '{element}'. " if element else "")
            + f"Extract the {fmt} value and call send_msg_to_user immediately."
        )
    elif key_hint:
        hint = f"[Shot {shot}] {key_hint}"
    else:
        hint = (
            f"[Shot {shot}] Find the information required by the task goal. "
            f"Call send_msg_to_user with the exact answer as soon as you see it."
        )

    # If this exact hint was already tried, append a variation marker
    if hint in tried_hints:
        hint += f" (Retry {shot} — be more systematic, check ALL visible text.)"

    return hint


# ── LLM 2 (Executor): Mind State injection at every step ──────────────────

def _make_beliefs_summary(beliefs: dict) -> str:
    return (
        f"page_to_navigate : {beliefs.get('page_to_navigate') or '(current page)'}\n"
        f"element_to_find  : {beliefs.get('element_to_find') or '(any matching field)'}\n"
        f"answer_strategy  : {beliefs.get('answer_strategy', 'navigate')}\n"
        f"answer_format    : {beliefs.get('answer_format', 'text')}\n"
        f"key_hint         : {beliefs.get('key_hint') or '(none)'}"
    )


def _make_groq_query(hint: str = "", beliefs_summary: str = ""):
    """
    LLM 2 (HRI paper): executor receives the BDI Mind State at every step.
    This is the 'Mind State sharing' mechanism: the executor always knows
    what the interpreter understood about the goal.
    """
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    def query_model(system_msgs, user_msgs):
        messages = []
        if system_msgs:
            sys_text = system_msgs[0]["text"]
            injection = ""
            if beliefs_summary:
                injection += (
                    "\n\n=== BDI MIND STATE (updated by Interpreter LLM) ===\n"
                    + beliefs_summary
                )
            if hint:
                injection += (
                    "\n\n=== BDI DIRECTIVE (from BDI plan selector) ===\n"
                    + hint
                    + "\nFollow this directive at every step."
                )
            sys_text += injection
            messages.append({"role": "system", "content": sys_text})
        content = [{"type": "text", "text": m["text"]}
                   for m in user_msgs if m["type"] == "text"]
        messages.append({"role": "user", "content": content})
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.01, max_tokens=4096
        )
        return resp.choices[0].message.content or ""

    return query_model


# ── GroqDemoAgent wired to LLM 2 ──────────────────────────────────────────

class _GroqDemoAgent(DemoAgent):
    def __init__(self, hint: str = "", beliefs_summary: str = "", **kwargs):
        kwargs.pop("model_name", None)
        kwargs.pop("openrouter_api_key", None)
        super().__init__(model_name="openrouter/dummy",
                         openrouter_api_key="not_used", **kwargs)
        self.model_name  = LLM_MODEL
        self.query_model = _make_groq_query(hint=hint,
                                            beliefs_summary=beliefs_summary)

    def obs_preprocessor(self, obs: dict) -> dict:
        p = super().obs_preprocessor(obs)
        ax = p.get("axtree_txt", "")
        if len(ax) > MAX_AXTREE_CHARS:
            p["axtree_txt"] = ax[:MAX_AXTREE_CHARS] + "\n...[AXTree truncated]"
        return p


@dataclasses.dataclass
class _GroqDemoAgentArgs(DemoAgentArgs):
    model_name:      str = f"groq/{LLM_MODEL}"
    hint:            str = ""
    beliefs_summary: str = ""

    def make_agent(self):
        return _GroqDemoAgent(
            hint                    = self.hint,
            beliefs_summary         = self.beliefs_summary,
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
    hint            = ctx.get("hint", "")
    beliefs_summary = ctx.get("beliefs_summary", "")

    t0 = time.perf_counter()
    try:
        harness = REAL.harness(
            agentargs     = _GroqDemoAgentArgs(
                use_axtree=True, use_html=False, use_screenshot=False,
                hint=hint, beliefs_summary=beliefs_summary,
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

def _run_shot_subprocess(task_id: str, hint: str, beliefs_summary: str) -> dict:
    ctx = {"task_id": task_id, "hint": hint, "beliefs_summary": beliefs_summary}

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
            print(f"  [HRI-BDI] Subprocess returncode={proc.returncode}")
    except subprocess.TimeoutExpired:
        print("  [HRI-BDI] Subprocess timeout")
        return None
    except Exception as e:
        print(f"  [HRI-BDI] Subprocess error: {e}")
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
    print(f"  ARCHITECTURE 7 — BDI-LLM Dual (HRI paper)")
    print(f"  Ref    : Integrating BDI agents with LLMs for HRI and XAI")
    print(f"  Model  : {LLM_MODEL}")
    print(f"  Tasks  : {TASKS}")
    print(f"  Shots  : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"  LLM 1  : Interpreter (1 call/shot) → structured beliefs")
    print(f"  BDI    : Deterministic hint selector (priority rules)")
    print(f"  LLM 2  : Executor with Mind State injection at every step")
    print(f"{'='*62}")

    groq_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    results     = _load_results()
    done_keys   = {(r["task_id"], r["shot"]) for r in results}

    for task_id in TASKS:
        goal         = _get_task_goal(task_id)
        tried_hints  = []
        prev_reward  = None
        prev_answer  = ""
        prev_beliefs = dict(_FALLBACK_BELIEFS)

        print(f"\n  Task : {task_id}")
        print(f"  Goal : {goal[:90]}")

        cumsum_regret = 0.0
        completed = sorted(
            [r for r in results if r["task_id"] == task_id],
            key=lambda r: r["shot"],
        )
        for r in completed:
            cumsum_regret += (1 - r["reward"])
            tried_hints.append(r.get("hint", ""))
            prev_reward   = r["reward"]
            prev_answer   = r.get("answer", "")
            prev_beliefs  = r.get("beliefs", dict(_FALLBACK_BELIEFS))

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results
                         if x["task_id"] == task_id and x["shot"] == shot)
                cumsum_regret += (1 - r["reward"])
                tried_hints.append(r.get("hint", ""))
                prev_reward   = r["reward"]
                prev_answer   = r.get("answer", "")
                prev_beliefs  = r.get("beliefs", dict(_FALLBACK_BELIEFS))
                print(f"  [Shot {shot}] Already done — reward={r['reward']:.1f} (skipping)")
                continue

            print(f"\n  {'─'*58}")

            # ── LLM 1: Interpreter ────────────────────────────────────────
            print(f"  [Shot {shot}/{N_SHOTS}] LLM 1 (Interpreter): extracting beliefs...")
            beliefs = _interpret_beliefs(
                groq_client, goal, shot,
                prev_reward=prev_reward,
                prev_answer=prev_answer,
            )
            beliefs_summary = _make_beliefs_summary(beliefs)
            print(f"  [Beliefs] strategy={beliefs['answer_strategy']} | "
                  f"page={beliefs['page_to_navigate'] or '(current)'} | "
                  f"element={beliefs['element_to_find'] or '(any)'}")

            # ── BDI: deterministic hint selection ─────────────────────────
            hint = _bdi_select_hint(shot, beliefs, tried_hints)
            tried_hints.append(hint)
            print(f"  [BDI hint] {hint[:100]}")

            # ── LLM 2: Executor with Mind State ───────────────────────────
            print(f"  [LLM 2 Executor] Launching subprocess with Mind State...")
            raw = _run_shot_subprocess(task_id, hint, beliefs_summary)
            if raw is None:
                raw = {"reward": 0.0, "steps": MAX_STEPS, "answer": "",
                       "duration_s": 0.0, "error": "subprocess_failed"}

            cumsum_regret += (1 - raw["reward"])
            cum_regret     = cumsum_regret / shot
            prev_reward    = raw["reward"]
            prev_answer    = raw.get("answer", "")
            prev_beliefs   = beliefs

            record = {
                "architecture":      "hri_bdi",
                "task_id":           task_id,
                "shot":              shot,
                "reward":            raw["reward"],
                "steps":             raw["steps"],
                "answer":            raw.get("answer", ""),
                "duration_s":        raw["duration_s"],
                "regret_t":          round(1 - raw["reward"], 4),
                "cumulative_regret": round(cum_regret, 4),
                "hint":              hint,
                "beliefs":           beliefs,
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
    print(f"\n  HRI-BDI done — {n_success}/{len(results)} shots succeeded")
    print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    if _SUBPROCESS_MARKER in sys.argv:
        _subprocess_worker()
    else:
        main()
