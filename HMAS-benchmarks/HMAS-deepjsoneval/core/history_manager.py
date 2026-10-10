# ─────────────────────────────────────────────────────────
# history_manager.py — MÉMOIRE NON-MARKOVIENNE DES AGENTS
# Copié de agisdk/mas_project, adapté pour DeepJSONEval.
# ─────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import List
import json, os, time, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_HISTORY_LENGTH, HISTORY_FILE


@dataclass
class RunRecord:
    task_id:            str
    agent_role:         str
    timestamp:          float
    strategy_used:      str
    action_taken:       str
    reward:             float
    steps_used:         int
    payoff:             float
    dim_success:        float = 0.0
    dim_efficiency:     float = 0.0
    dim_safety:         float = 0.0
    observation:        str   = ""
    notes:              str   = ""


class AgentHistory:
    """Mémoire non-Markovienne h_t d'un agent — information privée θ_i."""

    def __init__(self, agent_role: str):
        self.agent_role = agent_role
        self.records: List[RunRecord] = []

    def add_record(self, record: RunRecord) -> None:
        self.records.append(record)
        if len(self.records) > MAX_HISTORY_LENGTH:
            self.records.pop(0)

    def get_recent_records(self, n: int = 5) -> List[RunRecord]:
        return self.records[-n:] if self.records else []

    def get_success_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.reward >= 0.7) / len(self.records)

    def get_average_payoff(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.payoff for r in self.records) / len(self.records)

    def get_trend(self) -> str:
        if len(self.records) < 4:
            return "UNKNOWN"
        recent = sum(r.payoff for r in self.records[-3:]) / 3
        older  = sum(r.payoff for r in self.records[:-3]) / max(1, len(self.records) - 3)
        if recent > older + 0.1:
            return "IMPROVING"
        if recent < older - 0.1:
            return "DEGRADING"
        return "STABLE"

    def build_context_prompt(self) -> str:
        if not self.records:
            return "=== First attempt — no history ==="
        lines = [f"=== HISTORY (view {self.agent_role.upper()}) ==="]
        lines.append(f"Payoff = YOUR utility U_{self.agent_role}.\n")
        for i, rec in enumerate(self.get_recent_records(5), 1):
            status = "SUCCESS" if rec.reward >= 0.7 else "FAILURE"
            lines.append(f"  Run {i} [{rec.task_id}] : {status} | "
                         f"U_{self.agent_role}={rec.payoff:.3f}")
            if rec.notes:
                lines.append(f"    -> {rec.notes}")
        lines.append(f"\n  U_{self.agent_role} trend : {self.get_trend()}")
        lines.append(f"  Success rate : {self.get_success_rate():.1%}")
        lines.append(f"  Average U_{self.agent_role} : {self.get_average_payoff():.3f}")
        lines.append("=" * 45)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "agent_role": self.agent_role,
            "records": [
                {"task_id": r.task_id, "timestamp": r.timestamp,
                 "strategy": r.strategy_used, "reward": r.reward,
                 "steps": r.steps_used, "payoff": r.payoff,
                 "dim_success": r.dim_success, "dim_efficiency": r.dim_efficiency,
                 "dim_safety": r.dim_safety, "notes": r.notes}
                for r in self.records
            ]
        }


class HistoryManager:
    """Gestionnaire centralisé d'historique — partagé entre agents."""

    def __init__(self):
        self.planner_history  = AgentHistory("planner")
        self.verifier_history = AgentHistory("verifier")
        self._load_from_disk()

    def get_history(self, agent_role: str) -> AgentHistory:
        return {"planner": self.planner_history,
                "verifier": self.verifier_history}.get(agent_role, self.planner_history)

    def record_run(self, agent_role: str, task_id: str, strategy: str,
                   reward: float, steps: int, payoff: float,
                   dim_success: float, dim_efficiency: float,
                   dim_safety: float, notes: str = "") -> None:
        record = RunRecord(
            task_id=task_id, agent_role=agent_role, timestamp=time.time(),
            strategy_used=strategy, action_taken="", reward=reward,
            steps_used=steps, payoff=payoff, dim_success=dim_success,
            dim_efficiency=dim_efficiency, dim_safety=dim_safety, notes=notes,
        )
        self.get_history(agent_role).add_record(record)
        self._save_to_disk()

    def _save_to_disk(self) -> None:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"planner":  self.planner_history.to_dict(),
                       "verifier": self.verifier_history.to_dict()}, f, indent=2)

    def _load_from_disk(self) -> None:
        if not os.path.exists(HISTORY_FILE):
            return
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)
            for role, hdata in data.items():
                if role not in ("planner", "verifier"):
                    continue
                hist = self.get_history(role)
                for r in hdata.get("records", []):
                    hist.records.append(RunRecord(
                        task_id=r["task_id"], agent_role=role, timestamp=r["timestamp"],
                        strategy_used=r["strategy"], action_taken="", reward=r["reward"],
                        steps_used=r["steps"], payoff=r["payoff"],
                        dim_success=r["dim_success"], dim_efficiency=r["dim_efficiency"],
                        dim_safety=r["dim_safety"], notes=r["notes"],
                    ))
        except Exception as e:
            print(f"Warning: history not loaded : {e}")
