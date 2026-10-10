# ─────────────────────────────────────────────────────────
# bdi_model.py — REAL BDI MODEL (Belief-Desire-Intention) — v2
#
# Compliant with the Rao & Georgeff (1991) BDI architecture:
#
#   Beliefs    = belief base about the world state (BeliefBase)
#   Desires    = agent goals (Desire with a full lifecycle)
#   Intentions = committed plans (structured Intention, not just a string)
#
# v2 change vs the previous version: an Intention now tracks
# `step_index`, its position within the committed plan's BODY (a
# sequence of Actions — see core/plan_library.py). As long as the
# agent holds its commitment, it walks the plan's steps one by one
# instead of re-selecting a plan on every cycle; `commit()` no longer
# takes a hint (the hint now comes from whichever step is current).
# Default commitment strategy is also SINGLE_MINDED (was OPEN_MINDED):
# replans now happen in a tight internal loop rather than across full
# external shots, so intentions should persist across that loop
# instead of being re-deliberated on every percept.
#
# BDI cycle:
#   perceive → brf → reconsider → options → filter → execute
#
# ─────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFLICT_DETECTION_ENABLED


# ─────────────────────────────────────────────────────────
# Commitment Strategy
# ─────────────────────────────────────────────────────────

class CommitmentStrategy(Enum):
    """
    Intention reconsideration policy.

    BLIND         : the agent keeps its intention no matter what.
                    Robust but can persist on a doomed plan.

    SINGLE_MINDED : the agent keeps its intention as long as the goal
                    is not reached AND the plan is not obviously
                    impossible (U_V < 0.05 after the 1st attempt).
                    Good balance between stability and reactivity —
                    default here, since internal replans already
                    happen in a tight loop within one task.

    OPEN_MINDED   : the agent reconsiders after every percept.
                    Maximal reactivity — was the default when replans
                    spanned full external shots.
    """
    BLIND         = "blind"
    SINGLE_MINDED = "single_minded"
    OPEN_MINDED   = "open_minded"


# ─────────────────────────────────────────────────────────
# Desire lifecycle
# ─────────────────────────────────────────────────────────

class DesireStatus(Enum):
    PENDING   = "pending"    # not yet activated
    ACTIVE    = "active"     # currently pursued
    SUSPENDED = "suspended"  # temporarily deprioritized
    ACHIEVED  = "achieved"   # accomplished
    FAILED    = "failed"     # impossible to reach


# ─────────────────────────────────────────────────────────
# Belief Base
# ─────────────────────────────────────────────────────────

@dataclass
class BeliefBase:
    """
    What the BDI agent believes about the current state.
    Updated after every verifier feedback (brf).
    """
    task_id:       str   = ""
    task_goal:     str   = ""
    task_type:     str   = ""
    attempt_count: int   = 0

    tried_hints:  List[str]   = field(default_factory=list)
    past_u_v:     List[float] = field(default_factory=list)
    past_rewards: List[float] = field(default_factory=list)

    last_feedback:        str   = ""
    last_quality:         str   = ""
    last_u_v:             float = 0.0
    last_u_pe:            float = 0.0
    last_dim_efficiency:  float = 0.0
    last_dim_specificity: float = 0.0
    last_action_history:  List[str] = field(default_factory=list)
    last_error:           str   = ""

    # Conflict tracking (PE self-assessment vs. V's verdict) and
    # fallback-streak tracking (no specific plan recognized the failure
    # mode). Reset whenever the triggering condition is not reproduced.
    # Maintaining these counters has no observable effect unless a
    # Desire/Plan reacts to them (gated at the call site behind
    # config.CONFLICT_DETECTION_ENABLED / config.DYNAMIC_DESIRES_ENABLED).
    consecutive_conflict:  int = 0
    consecutive_fallback:  int = 0

    def brf(self, reward: float, u_pe: float, u_v: float,
            quality: str, feedback: str, hint: str,
            dim_efficiency: float = 0.0,
            dim_specificity: float = 0.0,
            action_history: list = None,
            last_error: str = "") -> None:
        """
        Belief Revision Function — updates the beliefs.
        Minimal-change principle: only the fields affected by the
        percept are modified.
        """
        self.attempt_count  += 1
        self.past_u_v.append(u_v)
        self.past_rewards.append(reward)
        self.last_feedback           = feedback
        self.last_quality            = quality
        self.last_u_v                = u_v
        self.last_u_pe               = u_pe
        self.last_dim_efficiency     = dim_efficiency
        self.last_dim_specificity    = dim_specificity
        self.last_action_history     = list(action_history or [])
        self.last_error              = last_error or ""
        if hint and hint not in self.tried_hints:
            self.tried_hints.append(hint)

        # Conflict = PE (executor) would call itself successful (high U_PE)
        # while V (verifier) disagrees (low U_V) and the task did not
        # actually succeed — a live signal, unlike gt_analyzer's
        # post-hoc conflict_rate which only reports after the fact.
        if quality != "success" and (u_pe - u_v) > 0.3:
            self.consecutive_conflict += 1
        else:
            self.consecutive_conflict = 0

    def best_u_v(self) -> float:
        return max(self.past_u_v) if self.past_u_v else 0.0

    def avg_u_v(self) -> float:
        return sum(self.past_u_v) / len(self.past_u_v) if self.past_u_v else 0.0

    def is_plateau(self, n: int = 2, epsilon: float = 0.05) -> bool:
        """True if the last `n` scores vary by less than `epsilon`."""
        if len(self.past_u_v) < n:
            return False
        recent = self.past_u_v[-n:]
        return (max(recent) - min(recent)) < epsilon

    def trend(self) -> str:
        """
        Performance trend via linear regression over the full history.
        Returns: "IMPROVING" | "DEGRADING" | "STABLE" | "UNKNOWN".
        """
        n = len(self.past_u_v)
        if n < 2:
            return "UNKNOWN"

        xs = list(range(n))
        xm = sum(xs) / n
        ym = sum(self.past_u_v) / n

        num = sum((x - xm) * (y - ym) for x, y in zip(xs, self.past_u_v))
        den = sum((x - xm) ** 2 for x in xs)

        if den == 0:
            return "STABLE"

        slope = num / den
        if slope > 0.02:
            return "IMPROVING"
        if slope < -0.02:
            return "DEGRADING"
        return "STABLE"


# ─────────────────────────────────────────────────────────
# Desire
# ─────────────────────────────────────────────────────────

@dataclass
class Desire:
    """
    Agent goal — with a full lifecycle.

      - DesireStatus (PENDING/ACTIVE/SUSPENDED/ACHIEVED/FAILED)
        replaces a plain `achieved` bool
      - Desires can be temporarily suspended
        (e.g. minimize_steps suspended while the agent is stuck)
    """
    name:        str
    priority:    int
    description: str
    status:      DesireStatus = DesireStatus.PENDING

    def activate(self)  -> None: self.status = DesireStatus.ACTIVE
    def suspend(self)   -> None: self.status = DesireStatus.SUSPENDED
    def achieve(self)   -> None: self.status = DesireStatus.ACHIEVED
    def fail(self)      -> None: self.status = DesireStatus.FAILED
    def resume(self)    -> None: self.status = DesireStatus.ACTIVE

    @property
    def is_active(self) -> bool:
        return self.status in (DesireStatus.PENDING, DesireStatus.ACTIVE)

    @property
    def is_achieved(self) -> bool:
        return self.status == DesireStatus.ACHIEVED


# ─────────────────────────────────────────────────────────
# Intention
# ─────────────────────────────────────────────────────────

@dataclass
class Intention:
    """
    Committed plan — the agent is committed to executing this plan.

      - plan_name    : which Plan from the library was selected
      - source_desire: which desire this plan serves
      - hint         : strategic hint of the CURRENT step (set by
                        the planner as it walks the plan's body)
      - status       : ACTIVE | SUSPENDED | ACHIEVED | FAILED
      - committed_at : attempt_count at commitment time
      - step_index   : current position in the plan's BODY (sequence
                        of Actions) — as long as the agent holds this
                        commitment, it advances through the body one
                        step per internal cycle instead of
                        re-selecting a plan.
    """
    plan_name:     str
    source_desire: str
    hint:          str
    status:        str = "ACTIVE"    # ACTIVE | SUSPENDED | ACHIEVED | FAILED
    committed_at:  int = 0           # attempt_count at commitment time
    step_index:    int = 0

    def suspend(self)  -> None: self.status = "SUSPENDED"
    def achieve(self)  -> None: self.status = "ACHIEVED"
    def fail(self)     -> None: self.status = "FAILED"
    def resume(self)   -> None: self.status = "ACTIVE"

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"


# ─────────────────────────────────────────────────────────
# Full BDI state
# ─────────────────────────────────────────────────────────

class BDIState:
    """
    Aggregate of the agent's beliefs, desires and intentions.

    Implements the full BDI cycle:
      perceive → brf → reconsider → options → filter → execute

    The commitment strategy is configurable and drives the
    reconsideration decision (should_reconsider).
    """

    def __init__(
        self,
        commitment_strategy: CommitmentStrategy = CommitmentStrategy.SINGLE_MINDED
    ) -> None:
        self.beliefs:             BeliefBase         = BeliefBase()
        self.desires:             List[Desire]       = []
        self.intentions:          List[Intention]    = []
        self.commitment_strategy: CommitmentStrategy = commitment_strategy

    # ── Task initialization ──────────────────────────────

    def set_task(self, task_id: str, task_goal: str, task_type: str) -> None:
        self.beliefs = BeliefBase(
            task_id=task_id, task_goal=task_goal, task_type=task_type
        )
        self.desires = [
            Desire("complete_task",   3, f"Accomplish: {task_goal[:80]}",
                   DesireStatus.ACTIVE),
            Desire("minimize_steps",  2, "Use the minimum number of steps",
                   DesireStatus.ACTIVE),
            Desire("specific_answer", 1, "Precise answer with exact value",
                   DesireStatus.ACTIVE),
        ]
        self.intentions = []

    # ── BRF: belief update ───────────────────────────────

    def brf(self, reward: float, u_pe: float, u_v: float,
            quality: str, feedback: str, hint: str,
            dim_efficiency: float = 0.0,
            dim_specificity: float = 0.0,
            action_history: list = None,
            last_error: str = "") -> None:
        """
        Belief Revision Function — updates beliefs AND desires.

        After every percept (verification result):
          - Beliefs are updated (BeliefBase.brf)
          - Desires are updated according to the quality obtained
          - Active intentions may be marked ACHIEVED/FAILED
        """
        self.beliefs.brf(reward, u_pe, u_v, quality, feedback, hint,
                         dim_efficiency, dim_specificity,
                         action_history, last_error)

        if quality == "success":
            for d in self.desires:
                d.achieve()
            for i in self.intentions:
                i.achieve()

        elif quality == "rejected":
            if self.intentions:
                self.intentions[-1].fail()
            # Suspend minimize_steps while the agent is badly stuck
            if u_pe < 0.2:
                for d in self.desires:
                    if d.name == "minimize_steps":
                        d.suspend()

        elif quality == "partial":
            for d in self.desires:
                if d.name == "minimize_steps" and d.status == DesireStatus.SUSPENDED:
                    d.resume()

    # ── options(): active desires by priority ────────────

    def options(self) -> List[Desire]:
        """
        Generates the options (active, not-yet-achieved desires),
        sorted by decreasing priority. Implements options().
        """
        return sorted(
            [d for d in self.desires if d.is_active and not d.is_achieved],
            key=lambda d: d.priority,
            reverse=True,
        )

    # ── reconsider(): should the intention be dropped? ───

    def should_reconsider(self) -> bool:
        """
        Decides whether the agent should drop its current intention
        and select a new one, based on the CommitmentStrategy.

        BLIND         : never reconsiders.
        SINGLE_MINDED : reconsiders if the goal was reached or the
                        plan seems impossible.
        OPEN_MINDED   : reconsiders after every percept.
        """
        if self.commitment_strategy == CommitmentStrategy.BLIND:
            return False

        if self.commitment_strategy == CommitmentStrategy.OPEN_MINDED:
            return True

        # SINGLE_MINDED: reconsider only when necessary
        b = self.beliefs
        if b.last_quality in ("success", "partial"):
            return True   # goal reached -> free the intention
        if b.attempt_count >= 1 and b.last_u_v < 0.05:
            return True   # plan obviously impossible
        if CONFLICT_DETECTION_ENABLED and b.consecutive_conflict >= 2:
            # Executor keeps believing it succeeded while the verifier
            # keeps rejecting it — held commitments never re-check this
            # on their own (the continuation branch skips select_plan()
            # entirely), so force a re-deliberation that can pick up the
            # resolve_conflict desire. Gated behind the same flag as
            # ResolveConflict itself (plan_library.py) — off by default,
            # so should_reconsider() stays byte-identical to pre-enrichment
            # behavior for any BDIState consumer (e.g. the frering_bdi
            # ablation baseline) when the flag is not explicitly enabled.
            return True
        return False

    # ── commit(): register a new intention ───────────────

    def commit(self, plan_name: str = "unknown") -> None:
        """
        Commits to a plan — creates a new Intention at step 0 of its
        body. The hint is not known yet: it will be set on the
        Intention once the planner emits the current step.
        """
        active = self.options()
        source = active[0].name if active else "complete_task"
        self.intentions.append(Intention(
            plan_name     = plan_name,
            source_desire = source,
            hint          = "",
            status        = "ACTIVE",
            committed_at  = self.beliefs.attempt_count,
            step_index    = 0,
        ))

    def drop_intention(self) -> None:
        if self.intentions:
            self.intentions.pop()

    def current_intention(self) -> Optional[Intention]:
        return self.intentions[-1] if self.intentions else None

    # ── should_replan(): another attempt justified? ──────

    def should_replan(self, max_replans: int) -> bool:
        b = self.beliefs
        if b.last_quality in ("success", "partial"):
            return False
        return b.attempt_count < max_replans

    def replan_reason(self) -> str:
        b = self.beliefs
        if b.last_feedback:
            return b.last_feedback
        return f"Insufficient quality (U_V={b.last_u_v:.3f})"

    # ── summary(): readable BDI state ────────────────────

    def summary(self) -> str:
        b  = self.beliefs
        cs = self.commitment_strategy.value
        current = self.current_intention()
        plan_info = f"plan={current.plan_name}" if current else "no_intention"
        return (
            f"BDI[{b.task_id}] attempt={b.attempt_count} | "
            f"strategy={cs} | "
            f"desires={[d.name for d in self.options()]} | "
            f"best_U_V={b.best_u_v():.3f} | trend={b.trend()} | "
            f"quality={b.last_quality or 'none'} | {plan_info}"
        )
