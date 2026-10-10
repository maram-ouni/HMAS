# ─────────────────────────────────────────────────────────
# message_bus.py — THREAD-SAFE MESSAGE BUS
#
# Why this is now a real MAS-LLM :
#   Before : mailboxes = dict of Python lists (fake)
#            → a single thread reads/writes, no real blocking
#   After  : mailboxes = queue.Queue per agent (real)
#            → each agent BLOCKS on queue.get() until
#              it receives a message, exactly like a
#              real autonomous agent waiting for events
#
# queue.Queue is thread-safe by design.
# When the Planner publishes PLAN_READY, the Executer
# blocking on queue.get() wakes up automatically
# — without anyone calling it explicitly.
# ─────────────────────────────────────────────────────────

import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum
import time, uuid


class MessageType(Enum):
    """
    Types of messages exchanged in the MAS.

    Classic architecture (2 LLM agents) :
      TASK_ASSIGNED → EXECUTION_DONE
                           ↓              ↓
                  RESULT_ACCEPTED   REPLAN_REQUESTED → EXECUTION_DONE ...

    Hybrid architecture (BDI + LLM + Classic) :
      TASK_ASSIGNED → EXECUTION_REQUESTED → EXECUTION_DONE
                                                   ↓
                                          VERIFICATION_RESULT → BDIPlanner
                                                   ↓
                                          RESULT_ACCEPTED → system
    """
    TASK_ASSIGNED        = "task_assigned"        # system        → planner
    EXECUTION_REQUESTED  = "execution_requested"  # BDIPlanner    → LLMExecutor
    EXECUTION_DONE       = "execution_done"        # LLMExecutor   → RuleVerifier
    VERIFICATION_RESULT  = "verification_result"   # RuleVerifier  → BDIPlanner
    REPLAN_REQUESTED     = "replan_requested"      # verifier      → planner (legacy)
    RESULT_ACCEPTED      = "result_accepted"       # verifier      → system
    ERROR_REPORTED       = "error_reported"        # any agent     → system
    STOP                 = "stop"                  # system        → all


@dataclass
class Message:
    """
    Unit of communication between agents.
    All information passes through here — never via direct calls.
    """
    sender:    str
    receiver:  str
    msg_type:  MessageType
    content:   Dict[str, Any] = field(default_factory=dict)
    msg_id:    str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float           = field(default_factory=time.time)

    def __str__(self) -> str:
        return (f"[{self.msg_id}] "
                f"{self.sender:10s} -> {self.receiver:10s} : "
                f"{self.msg_type.value}")


class MessageBus:
    """
    Central message bus — communication channel of the MAS.

    Uses queue.Queue (not a dict) :
      - Thread-safe : multiple threads can publish/read
                      simultaneously without race conditions
      - Blocking    : bus.get("planner") BLOCKS the calling thread
                      until a message arrives
                      → the agent is truly waiting, not polling

    This blocking property gives agents their true autonomy :
    they consume no CPU when there is nothing to do, and wake up
    automatically when a message arrives.
    """

    def __init__(self):
        # One queue.Queue per agent — true blocking queue
        # "executor" : used only by the hybrid architecture
        self.queues: Dict[str, queue.Queue] = {
            "planner":  queue.Queue(),
            "executor": queue.Queue(),
            "verifier": queue.Queue(),
            "system":   queue.Queue(),
        }
        # Full log for traceability (protected by lock)
        self._log: List[Message] = []
        self._lock = threading.Lock()

    def publish(self, message: Message) -> None:
        """
        Publishes a message into the recipient's queue.

        Thread-safe : can be called from any thread.
        The recipient will wake up automatically if
        it is blocking on get().
        """
        with self._lock:
            self._log.append(message)
        print(f"  MSG {message}")

        if message.receiver == "all":
            for q in self.queues.values():
                q.put(message)
        elif message.receiver in self.queues:
            self.queues[message.receiver].put(message)

    def get(self, agent_name: str, timeout: float = 600) -> Message:
        """
        The agent waits for a message — BLOCKING.

        This is where autonomy becomes real :
        the agent's thread is suspended by the OS
        until a message arrives in its queue.
        It consumes no CPU while waiting.

        Raises queue.Empty if the timeout is exceeded.
        """
        return self.queues[agent_name].get(timeout=timeout)

    def get_conversation_log(self) -> str:
        """Formatted log of all exchanges for analysis."""
        with self._lock:
            log_copy = list(self._log)
        if not log_copy:
            return "No messages exchanged."
        lines = ["=== MAS COMMUNICATION LOG ==="]
        for i, msg in enumerate(log_copy, 1):
            lines.append(
                f"  {i:2d}. {msg.sender:10s} -> {msg.receiver:10s} | "
                f"{msg.msg_type.value:22s} | id={msg.msg_id}"
            )
        return "\n".join(lines)
