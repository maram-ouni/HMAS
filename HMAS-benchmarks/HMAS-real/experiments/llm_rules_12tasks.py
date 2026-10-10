"""
llm_rules_12tasks.py — LLM seul + règles BDI statiques (étude d'ablation)

Teste si le LLM seul, enrichi de TOUTES les règles BDI de la PlanLibrary
en prompt statique, atteint les mêmes résultats que l'architecture HMAS complète.

Si oui → l'architecture multi-agent est superflue.
Si non → la sélection dynamique de plans BDI apporte une valeur réelle.

Même 12 tâches que experiment_runner.py / hmas_12tasks.py.
5 shots par tâche (60 shots total).

Usage:
    python mas_project_v2/experiments/llm_rules_12tasks.py
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

N_SHOTS             = 5
MAX_STEPS           = 10
SLEEP_BETWEEN_SHOTS = 6
RESULTS_DIR         = os.path.join(_EXPERIMENTS_DIR, "results")
RESULTS_FILE        = os.path.join(RESULTS_DIR, "llm_rules_12tasks_results.json")
MAX_AXTREE_CHARS    = 25000
_SUBPROCESS_MARKER  = "--_llm-rules-shot"


# ── Règles BDI statiques — distillation de la PlanLibrary (16 plans) ──────

_BDI_STATIC_RULES = """
╔══════════════════════════════════════════════════════════════════╗
║        STRATEGIC RULES FOR WEB NAVIGATION (16 RULES)            ║
║  Follow these strictly — they encode proven failure patterns.    ║
╚══════════════════════════════════════════════════════════════════╝

RULE 1 — NAVIGATE FIRST, ANSWER FROM AXTREE:
  Always open the target page before answering. Find the information in the
  AXTree (accessibility tree), then call send_msg_to_user() with the EXACT
  visible value. Never answer from memory or assumptions.

RULE 2 — IMMEDIATE ANSWER FOR HOMEPAGE / LAST-MESSAGE TASKS:
  If the task asks about something on the homepage or the last message/item
  already visible in the current AXTree, your FIRST action MUST be
  send_msg_to_user('exact value'). Do NOT navigate away first.

RULE 3 — ONLY ONE ACTION PER RESPONSE:
  Output exactly ONE action per response. Never combine multiple actions
  (e.g., click + fill) in a single block. One ```action``` block per turn only.

RULE 4 — NEVER INVENT BID, PRICE, OR NUMERIC VALUES:
  BID numbers, prices, ratings, and IDs change every page load.
  NEVER guess or invent values ("hypothetically bid 367", "let's say $500").
  Only use values EXPLICITLY VISIBLE in the current AXTree.

RULE 5 — SEND ANSWER IMMEDIATELY WHEN FOUND:
  The instant you see the requested value in the AXTree, call
  send_msg_to_user('exact value') WITHOUT any further navigation.
  Finding the value AND sending it must happen in the same turn.

RULE 6 — EXACT VALUES ONLY — COPY FROM AXTREE:
  Copy the EXACT text from the AXTree: full names, complete numbers, text
  as it appears. If you see "$1,234.56" — send "$1,234.56", not "about $1,200".
  No paraphrasing, no rounding, no inferring.

RULE 7 — IF STUCK SCROLLING, CLICK A NAVIGATION LINK:
  If you have scrolled 2+ times without finding the target, STOP scrolling.
  Click on a navigation link (menu item, tab, sidebar link) visible in the
  AXTree. Never use goto() — use click() on actual links.

RULE 8 — IF SAME ACTION REPEATED 3+ TIMES, CHANGE APPROACH:
  If you have performed the same action (same click, same scroll) 3+ times
  in a row and the page has not changed, try:
  (1) scroll(0, 300) to reveal hidden elements, or
  (2) click a different navigation link in the top menu.

RULE 9 — COLLECT ALL REQUIRED DETAILS BEFORE ANSWERING:
  If the task asks for multiple fields (name + price + description + reviews),
  gather ALL of them before calling send_msg_to_user(). Do NOT send a partial
  answer — navigate until you have every piece of requested information.

RULE 10 — IF PREVIOUS ANSWER WAS WRONG: NAVIGATE TO VERIFY:
  If a previous attempt sent a wrong answer, it is likely because you answered
  without checking the page. In your next attempt: navigate to the exact page,
  read the AXTree carefully, then send the precise value you see.

RULE 11 — CHANGE APPROACH COMPLETELY WHEN FAILING REPEATEDLY:
  If 2+ consecutive attempts failed with the same navigation path, try a
  completely different approach: different menu item, different URL segment,
  different search query. Repeating a failed path will not work.

RULE 12 — CHECK IF ANSWER IS ALREADY VISIBLE BEFORE NAVIGATING:
  Before any navigation, scan the CURRENT AXTree fully. If the requested
  value is already there, call send_msg_to_user() immediately — no extra
  clicks or scrolls needed.

RULE 13 — send_msg_to_user() IS FINAL: DO NOT CONTINUE AFTER IT:
  Calling send_msg_to_user() ends the task immediately. Do not call it
  until you are confident in your answer. Once called, the interaction is over.

RULE 14 — report_infeasible() IS ONLY FOR TRULY IMPOSSIBLE TASKS:
  Use report_infeasible() ONLY when the task is genuinely impossible: broken
  page, feature does not exist, logically contradictory request. Do NOT call
  it just because you are confused or stuck — explore more carefully first.

RULE 15 — IF LOW ON STEPS, PRIORITIZE SENDING YOUR BEST ANSWER:
  If you have found a candidate answer but are running low on remaining steps,
  call send_msg_to_user() with the best answer you have found RIGHT NOW.
  Do not waste remaining steps on additional exploration.

RULE 16 — READ THE FULL AXTREE BEFORE EVERY ACTION:
  The AXTree contains all interactive elements and text values on the page.
  Before every action, scan the FULL AXTree to find relevant content.
  The answer is almost always already there — you just need to find it.

══════════════════════════════════════════════════════════════════
"""


# ── Agent (LLM + règles statiques) ────────────────────────────────────────

def _make_groq_query():
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    def query_model(system_msgs, user_msgs):
        messages = []
        if system_msgs:
            # Prepend all BDI static rules before the DemoAgent system prompt
            enriched_system = _BDI_STATIC_RULES + "\n\n" + system_msgs[0]["text"]
            messages.append({"role": "system", "content": enriched_system})
        content = [{"type": "text", "text": m["text"]}
                   for m in user_msgs if m["type"] == "text"]
        messages.append({"role": "user", "content": content})
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.01, max_tokens=4096
        )
        return resp.choices[0].message.content or ""

    return query_model


class _GroqRulesAgent(DemoAgent):
    """DemoAgent enrichi avec les règles BDI statiques dans le system prompt."""

    def __init__(self, **kwargs):
        kwargs.pop("model_name", None)
        kwargs.pop("openrouter_api_key", None)
        super().__init__(model_name="openrouter/dummy",
                         openrouter_api_key="not_used", **kwargs)
        self.model_name  = LLM_MODEL
        self.query_model = _make_groq_query()

    def obs_preprocessor(self, obs: dict) -> dict:
        p  = super().obs_preprocessor(obs)
        ax = p.get("axtree_txt", "")
        if len(ax) > MAX_AXTREE_CHARS:
            p["axtree_txt"] = ax[:MAX_AXTREE_CHARS] + "\n...[AXTree truncated]"
        return p


@dataclasses.dataclass
class _GroqRulesAgentArgs(DemoAgentArgs):
    model_name: str = f"groq/{LLM_MODEL}"

    def make_agent(self):
        return _GroqRulesAgent(
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
            agentargs     = _GroqRulesAgentArgs(
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
        print("  [LLM+Rules] Timeout"); return None
    except Exception as e:
        print(f"  [LLM+Rules] Error: {e}"); return None
    finally:
        if os.path.exists(ctx_file): os.unlink(ctx_file)

    if os.path.exists(res_file):
        with open(res_file, encoding="utf-8") as f:
            result = json.load(f)
        os.unlink(res_file)
        return result
    return None


# ── Résumé comparatif ──────────────────────────────────────────────────────

def _print_comparison(results: list) -> None:
    from collections import defaultdict

    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    # Try to load HMAS 12 tasks results for comparison
    hmas_file = os.path.join(RESULTS_DIR, "hmas_12tasks_results.json")
    hmas_by_task: dict = {}
    if os.path.exists(hmas_file):
        with open(hmas_file, encoding="utf-8") as f:
            hmas_data = json.load(f)
        for r in hmas_data:
            hmas_by_task.setdefault(r["task_id"], []).append(r)

    print(f"\n{'='*75}")
    print(f"  ABLATION — LLM+Règles vs HMAS complet")
    print(f"{'='*75}")
    header = f"  {'Tâche':<28} {'LLM+Rules':>12} {'HMAS':>12} {'Δ':>8}"
    print(header)
    print(f"  {'─'*65}")

    total_llm = 0
    total_hmas = 0
    total_shots_llm = 0
    total_shots_hmas = 0

    for task_id in TASKS:
        llm_shots  = by_task.get(task_id, [])
        hmas_shots = hmas_by_task.get(task_id, [])

        if not llm_shots:
            print(f"  {task_id:<28} {'—':>12}")
            continue

        n_llm  = sum(1 for s in llm_shots  if s["reward"] >= 1.0)
        n_hmas = sum(1 for s in hmas_shots if s.get("reward", 0) >= 1.0) if hmas_shots else None
        total_llm  += n_llm
        total_shots_llm += len(llm_shots)
        if n_hmas is not None:
            total_hmas += n_hmas
            total_shots_hmas += len(hmas_shots)

        llm_str  = f"{n_llm}/{len(llm_shots)}"
        hmas_str = f"{n_hmas}/{len(hmas_shots)}" if n_hmas is not None else "—"
        delta    = (f"+{n_llm - n_hmas}" if n_hmas is not None and n_llm > n_hmas
                    else (f"{n_llm - n_hmas}" if n_hmas is not None else "—"))
        print(f"  {task_id:<28} {llm_str:>12} {hmas_str:>12} {delta:>8}")

    print(f"  {'─'*65}")
    llm_rate  = f"{total_llm/total_shots_llm*100:.1f}%" if total_shots_llm else "—"
    hmas_rate = f"{total_hmas/total_shots_hmas*100:.1f}%" if total_shots_hmas else "—"
    print(f"  {'TOTAL':<28} {f'{total_llm}/{total_shots_llm} ({llm_rate})':>12} "
          f"{f'{total_hmas}/{total_shots_hmas} ({hmas_rate})' if total_shots_hmas else '—':>12}")

    print(f"\n  CONCLUSION ABLATION :")
    if total_shots_llm and total_shots_hmas:
        diff = total_llm / total_shots_llm - total_hmas / total_shots_hmas
        if abs(diff) < 0.02:
            print("  → Résultats EQUIVALENTS — l'architecture BDI n'apporte pas de gain.")
            print("    Le LLM avec règles statiques suffit.")
        elif diff > 0:
            print(f"  → LLM+Rules SUPERIEUR de {diff*100:.1f}pp — architecture BDI inutile !")
        else:
            print(f"  → HMAS SUPERIEUR de {-diff*100:.1f}pp — la sélection dynamique de plans")
            print("    apporte une valeur réelle au-delà du simple enrichissement du prompt.")
    print(f"{'='*75}\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n{'='*65}")
    print(f"  ABLATION — LLM seul + règles BDI statiques")
    print(f"  Architecture : DemoAgent (Groq) + system prompt enrichi")
    print(f"  Modèle       : {LLM_MODEL}")
    print(f"  Shots        : {N_SHOTS} | Max steps : {MAX_STEPS}")
    print(f"  But          : tester si l'archi HMAS est nécessaire")
    print(f"{'='*65}")

    results   = _load_results()
    done_keys = {(r["task_id"], r["shot"]) for r in results}
    print(f"  Reprise : {len(done_keys)} shots déjà effectués\n")

    for i, task_id in enumerate(TASKS):
        goal = _get_task_goal(task_id)
        shots_done        = [r for r in results if r["task_id"] == task_id]
        already_succeeded = any(r["reward"] >= 1.0 for r in shots_done)

        if len(shots_done) >= N_SHOTS or already_succeeded:
            n_ok  = sum(1 for r in shots_done if r["reward"] >= 1.0)
            label = f"SUCCÈS shot {next(r['shot'] for r in shots_done if r['reward']>=1.0)}" if already_succeeded else f"{n_ok}/{N_SHOTS}"
            print(f"  [{i+1:>2}/{len(TASKS)}] {task_id:<28} SKIP ({label})")
            continue

        print(f"\n  [{i+1:>2}/{len(TASKS)}] {task_id}")
        print(f"  Goal : {goal[:80]}")
        cumsum = sum(1 - r["reward"] for r in shots_done)

        for shot in range(1, N_SHOTS + 1):
            if (task_id, shot) in done_keys:
                r = next(x for x in results if x["task_id"] == task_id and x["shot"] == shot)
                cumsum += (1 - r["reward"])
                print(f"  [Shot {shot}] SKIP reward={r['reward']:.1f}")
                if r["reward"] >= 1.0:
                    print(f"  → Succès déjà enregistré — tâche suivante.")
                    break
                continue

            raw = _run_shot_subprocess(task_id)
            if raw is None:
                raw = {"reward": 0.0, "steps": MAX_STEPS, "answer": "",
                       "duration_s": 0.0, "error": "subprocess_failed"}

            cumsum += (1 - raw["reward"])
            record = {
                "architecture":      "llm_static_rules",
                "task_id":           task_id,
                "shot":              shot,
                "reward":            raw["reward"],
                "steps":             raw["steps"],
                "answer":            raw["answer"],
                "duration_s":        raw["duration_s"],
                "regret_t":          round(1 - raw["reward"], 4),
                "cumulative_regret": round(cumsum / shot, 4),
                "error":             raw.get("error"),
            }
            results.append(record)
            done_keys.add((task_id, shot))
            _save_results(results)

            status = "OK  " if raw["reward"] >= 1.0 else "FAIL"
            print(f"  [Shot {shot}] {status} | reward={raw['reward']:.1f} | "
                  f"steps={raw['steps']} | time={raw['duration_s']:.1f}s")

            if raw["reward"] >= 1.0:
                print(f"  → SUCCÈS au shot {shot} ! Passage à la tâche suivante.")
                break

            is_last = (shot == N_SHOTS and task_id == TASKS[-1])
            if not is_last:
                time.sleep(SLEEP_BETWEEN_SHOTS)

    _print_comparison(results)


if __name__ == "__main__":
    if _SUBPROCESS_MARKER in sys.argv:
        _subprocess_worker()
    else:
        main()
