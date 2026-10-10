# ─────────────────────────────────────────────────────────
# mas_orchestrator.py — LIGHTWEIGHT COORDINATOR (hybrid architecture) — v2
#
# Architecture : BDIPlanner (classic) + LLMExecutor (DemoAgent)
#                + RuleBasedVerifier (classic)
#
# The coordinator is PASSIVE:
#   1. Creates bus + threads (once — never recreated between attempts)
#   2. Injects a single TASK_ASSIGNED (carrying max_attempts)
#   3. Waits for RESULT_ACCEPTED or ERROR_REPORTED
#   4. Stops threads and returns the result
#
# v2 change: run_task() takes `max_attempts` and forwards it to the
# BDI planner, which drives up to that many internal replans by
# itself (see agents/bdi_planner_agent.py) — no external shot loop.
#
# All decisions are made by the agents.
#
# ─────────────────────────────────────────────────────────

import csv
import json
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.history_manager  import HistoryManager
from core.message_bus      import MessageBus, Message, MessageType
from config                import MAX_STEPS, RESULTS_FILE, RESULTS_DIR

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class MASCoordinator:
    """
    Lightweight coordinator for the hybrid MAS architecture.

    3 agents in true OS threads:
      BDIPlannerAgent       — classic strategic brain (BDI)
      LLMExecutorAgent      — DemoAgent REAL with Groq backend
      RuleBasedVerifierAgent — deterministic judge without LLM

    Message flow:
      system -> planner  : TASK_ASSIGNED
      planner -> executor : EXECUTION_REQUESTED  (BDI hint)
      executor -> verifier : EXECUTION_DONE      (reward, answer)
      verifier -> planner  : VERIFICATION_RESULT (payoffs, quality)
      verifier -> system   : RESULT_ACCEPTED     (if quality OK)
      planner -> system    : ERROR_REPORTED      (if max replans reached)
    """

    def __init__(self, memory_dir: str = None):
        # Isolated-memory override (None = today's hardcoded RESULTS_DIR
        # shared across runs — see config.MEMORY_DIR / --memory-dir in
        # mas_experiment.py). Threaded down to every agent that builds
        # its own persistent-memory object (WorldModel, StrategyLearner);
        # HistoryManager is built once, here.
        self._memory_dir     = memory_dir or RESULTS_DIR
        self.history_manager = HistoryManager(
            history_file=os.path.join(self._memory_dir, "agent_histories.json")
            if memory_dir else None
        )
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    def run_task(self, task_id: str,
                 task_goal: str,
                 max_steps: int = MAX_STEPS,
                 headless: bool = True,
                 prev_result: dict = None,
                 max_attempts: int = 1) -> dict:
        """
        Runs one REAL task through the hybrid MAS.

        The coordinator creates the three agent threads ONCE and injects
        a single TASK_ASSIGNED carrying `max_attempts`. All deliberation
        (up to `max_attempts` internal replans) happens INSIDE the BDI
        planner agent, whose memory persists for the whole task — no
        thread recreation, no state serialization between attempts.

        `prev_result` is still supported to seed beliefs before the
        first attempt (used by --demo mode).
        """
        from agents.bdi_planner_agent    import BDIPlannerAgent
        from agents.llm_executor_agent   import LLMExecutorAgent
        from agents.rule_based_verifier  import RuleBasedVerifierAgent

        print(f"\n{'='*60}")
        print(f"  HYBRID MAS | {task_id}")
        print(f"  Goal      : {task_goal[:70]}")
        print(f"  Agents   : BDIPlanner | DemoAgent (Groq) | RuleVerifier")
        print(f"  Budget    : {max_attempts} internal attempt(s)")
        print(f"{'='*60}")

        bus        = MessageBus()
        stop_event = threading.Event()

        bdi_planner   = BDIPlannerAgent(bus, self.history_manager, stop_event,
                                        memory_dir=self._memory_dir)
        llm_executor  = LLMExecutorAgent(bus, self.history_manager, stop_event)
        rule_verifier = RuleBasedVerifierAgent(bus, self.history_manager, stop_event,
                                                memory_dir=self._memory_dir)

        bdi_planner.start()
        llm_executor.start()
        rule_verifier.start()

        t_start = time.perf_counter()

        bus.publish(Message(
            sender   = "system",
            receiver = "planner",
            msg_type = MessageType.TASK_ASSIGNED,
            content  = {
                "task_id":      task_id,
                "task_goal":    task_goal,
                "max_steps":    max_steps,
                "headless":     headless,
                "prev_result":  prev_result or {},
                "max_attempts": max_attempts,
            }
        ))

        # Each internal attempt is a full DemoAgent episode (up to
        # max_steps browser steps) — size the wait accordingly.
        timeout = max_steps * 90 * max(1, max_attempts)
        result  = None

        try:
            msg            = bus.get("system", timeout=timeout)
            total_duration = time.perf_counter() - t_start

            if msg.msg_type == MessageType.RESULT_ACCEPTED:
                result = self._build_result(
                    msg.content, task_id, task_goal, total_duration
                )
                print(f"\n  [Coordinator] ACCEPTED in {total_duration:.1f}s")

            elif msg.msg_type == MessageType.ERROR_REPORTED:
                reason = msg.content.get("reason", "Unknown error")
                print(f"\n  [Coordinator] Error: {reason}")
                result = self._build_error_result(
                    task_id, task_goal, reason, total_duration,
                    content=msg.content,
                )

        except queue.Empty:
            total_duration = time.perf_counter() - t_start
            print(f"\n  [Coordinator] Timeout after {total_duration:.1f}s")
            result = self._build_error_result(
                task_id, task_goal, "Timeout", total_duration
            )

        stop_event.set()
        bus.publish(Message(sender="system", receiver="all",
                            msg_type=MessageType.STOP, content={}))

        bdi_planner.join(timeout=5)
        llm_executor.join(timeout=5)
        rule_verifier.join(timeout=5)

        print(f"\n{bus.get_conversation_log()}")
        self._save_result(result)
        return result

    def run_experiment(self, task_ids: list,
                       max_steps: int = MAX_STEPS,
                       headless: bool = True,
                       max_attempts: int = 1) -> list:
        all_results = []
        for task_id in task_ids:
            goal   = self._get_task_goal(task_id)
            result = self.run_task(task_id, goal, max_steps, headless,
                                    max_attempts=max_attempts)
            all_results.append(result)
            if task_id != task_ids[-1]:
                time.sleep(2)
        self._print_summary(all_results)
        return all_results

    # ── Building results ────────────────────────────────

    def _build_result(self, content: dict, task_id: str,
                      task_goal: str, duration: float) -> dict:
        return {
            "task_id":         task_id,
            "task_goal":       task_goal,
            "attempt":         content.get("attempt",  1),
            "strategy":        content.get("hint_used", "demoagent"),
            "hint_used":       content.get("hint_used", ""),
            "plan_name":       content.get("plan_name", ""),
            "reward":          content.get("reward",   0.0),
            "steps":           content.get("steps",    0),
            "answer":          content.get("answer",   ""),
            "success":         content.get("reward",   0.0) >= 1.0,
            "U_planner":       content.get("U_PE",     0.0),
            "U_PE":            content.get("U_PE",     0.0),
            "U_executer":      content.get("U_PE",     0.0),
            "U_verifier":      content.get("U_V",      0.0),
            "U_V":             content.get("U_V",      0.0),
            "dim_success":     content.get("dim_success",    0.0),
            "dim_efficiency":  content.get("dim_efficiency", 0.0),
            "dim_safety":      content.get("dim_safety",     0.0),
            "dim_specificity": content.get("dim_specificity", 0.0),
            "action_history":  content.get("action_history", []),
            "violations":      content.get("violations", []),
            "improvement":     content.get("improvement", ""),
            "accepted":        True,
            "quality":         content.get("quality",  "rejected"),
            "verdict":         content.get("verdict",  ""),
            "feedback":        content.get("feedback", ""),
            "last_error":      content.get("last_error", ""),
            "total_duration":  round(duration, 2),
            "total_attempts":  content.get("attempt",  1),
            "error":           None,
        }

    def _build_error_result(self, task_id: str, task_goal: str,
                             reason: str, duration: float,
                             content: dict = None) -> dict:
        c = content or {}
        return {
            "task_id":         task_id,
            "task_goal":       task_goal,
            "attempt":         c.get("attempts", 1),
            "strategy":        c.get("hint_used", "") or "demoagent",
            "hint_used":       c.get("hint_used", ""),
            "plan_name":       c.get("plan_name", ""),
            "reward":          c.get("reward", 0.0),
            "steps":           c.get("steps",  MAX_STEPS),
            "answer":          c.get("answer", ""),
            "success":         c.get("reward", 0.0) >= 1.0,
            "U_planner":       c.get("U_PE",   0.0),
            "U_PE":            c.get("U_PE",   0.0),
            "U_executer":      c.get("U_PE",   0.0),
            "U_verifier":      c.get("U_V",    0.0),
            "U_V":             c.get("U_V",    0.0),
            "dim_success":     c.get("dim_success",    0.0),
            "dim_efficiency":  c.get("dim_efficiency", 0.0),
            "dim_safety":      c.get("dim_safety",     0.0),
            "dim_specificity": c.get("dim_specificity", 0.0),
            "action_history":  c.get("action_history", []),
            "violations":      c.get("violations", []),
            "improvement":     "",
            "accepted":        False,
            "quality":         c.get("quality", "rejected"),
            "verdict":         c.get("verdict", f"ERROR: {reason}"),
            "feedback":        c.get("feedback", ""),
            "last_error":      c.get("last_error", reason),
            "total_duration":  round(duration, 2),
            "total_attempts":  c.get("attempts", 1),
            "error":           reason,
        }

    def _get_task_goal(self, task_id: str) -> str:
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
                with open(path) as f:
                    return json.load(f).get("goal", f"Task {task_id}")
        return f"Accomplish task {task_id}"

    def _save_result(self, result: dict) -> None:
        exists = os.path.exists(RESULTS_FILE)
        fields = ["task_id", "attempt", "strategy", "plan_name", "reward", "steps",
                  "U_planner", "U_executer", "U_verifier",
                  "dim_success", "dim_efficiency", "dim_safety",
                  "accepted", "improvement", "total_attempts",
                  "total_duration", "answer"]
        with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(result)

    def _print_summary(self, results: list) -> None:
        print(f"\n{'='*62}")
        print(f"  SUMMARY HYBRID MAS (BDI + DemoAgent + RuleVerifier)")
        print(f"{'='*62}")
        print(f"  {'Task':<18} {'Rwd':>4} {'U_PE':>6} {'U_V':>6} "
              f"{'Steps':>6} {'OK':>4}")
        print("  " + "─" * 50)
        for r in results:
            print(f"  {r['task_id']:<18} "
                  f"{r['reward']:>4.1f} "
                  f"{r['U_PE']:>6.3f} "
                  f"{r['U_V']:>6.3f} "
                  f"{r['steps']:>6} "
                  f"{'YES' if r['accepted'] else 'NO':>4}")
        print("  " + "─" * 50)
        n_ok = sum(1 for r in results if r["accepted"])
        print(f"  Accepted : {n_ok}/{len(results)}")
        print(f"  CSV      : {RESULTS_FILE}")


# Alias for potential compatibility
HybridMASCoordinator = MASCoordinator
