# ─────────────────────────────────────────────────────────
# bdi_model.py — MODÈLE BDI (Belief-Desire-Intention)
#
# Adapté de agisdk/mas_project pour DeepJSONEval :
#   - Beliefs enrichies pour la génération de JSON
#     (schema, text, ground_truth, category, true_depth...)
#   - Même cycle BDI (Rao & Georgeff, 1991)
#   - CommitmentStrategy identique (BLIND / SINGLE_MINDED / OPEN_MINDED)
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.payoff_calculator import V_SUCCESS_THRESHOLD, DIAGNOSTIC_QUALITY_THRESHOLD

# Écart U_PE - U_V au-delà duquel l'exécuteur (confiant) et le vérificateur
# (qui rejette) sont considérés en CONFLIT — cf. Desire "resolve_conflict".
CONFLICT_GAP_THRESHOLD    = 0.3
CONFLICT_STREAK_THRESHOLD = 2


class CommitmentStrategy(Enum):
    """
    BLIND         : garde l'intention quoi qu'il arrive.
    SINGLE_MINDED : abandonne si le plan semble impossible (U_V < 0.1).
    OPEN_MINDED   : reconsidère après chaque percept (défaut).
    """
    BLIND         = "blind"
    SINGLE_MINDED = "single_minded"
    OPEN_MINDED   = "open_minded"


class DesireStatus(Enum):
    PENDING   = "pending"
    ACTIVE    = "active"
    SUSPENDED = "suspended"
    ACHIEVED  = "achieved"
    FAILED    = "failed"


@dataclass
class BeliefBase:
    """
    Croyances de l'agent BDI sur la tâche de génération JSON courante.
    """
    # Identité de la tâche
    task_id:       str = ""
    task_goal:     str = ""
    task_type:     str = ""   # = category (patient, game, vehicle...)
    attempt_count: int = 0

    # Données du dataset
    schema:            str = ""
    text:              str = ""
    ground_truth_json: str = ""
    category:          str = ""
    true_depth:        int = 0

    # Historique des tentatives
    tried_hints:  List[str]   = field(default_factory=list)
    past_u_v:     List[float] = field(default_factory=list)
    past_rewards: List[float] = field(default_factory=list)

    # Résultats de la dernière vérification
    last_feedback:          str   = ""
    last_quality:           str   = ""
    last_u_v:               float = 0.0
    last_u_pe:              float = 0.0
    last_generated_json:    str   = ""
    last_is_valid_json:     bool  = False
    last_schema_compliance: float = 0.0
    last_field_accuracy:    float = 0.0
    last_depth_match:       bool  = False
    last_actual_depth:      int   = 0
    last_error:             str   = ""
    last_failed_fields:     List[str] = field(default_factory=list)
    # Violations de schéma structurées ({path, message, validator}) — V2.1 #4.
    # Alimente DeepFieldDiagnosis avec des champs concrets au lieu d'une
    # liste générique de propriétés du schéma.
    last_schema_violations: List[dict] = field(default_factory=list)
    # Bande diagnostique secondaire (V2.1 #3) — "good"/"fair"/"poor", calculée
    # à partir de DIAGNOSTIC_QUALITY_THRESHOLD (0.5). Purement informative :
    # n'affecte JAMAIS `quality`/`accepted` (V_THRESHOLD reste la seule source
    # de vérité). Utilisée uniquement, en opt-in, pour nuancer la détection de
    # plateau — voir BDIState.sync_dynamic_desires(use_diagnostic_band=...).
    last_diagnostic_quality: str = ""

    # Suivi pour le désir dynamique diagnose_unknown_failure : combien de
    # tentatives CONSÉCUTIVES ont dû recourir au plan générique "fallback"
    # (= aucun plan spécifique n'a reconnu le type d'échec).
    last_plan_name:         str = ""
    consecutive_fallback:   int = 0

    # Croyance sur le JUGEMENT du vérificateur — quelle dimension pèse le
    # plus dans le calcul de U_V à cet instant (vide avec le vérificateur
    # statique ; voir RuleBasedVerifierAgentDynamic). Permet au planner de
    # cibler explicitement ce que le vérificateur valorise MAINTENANT, pas
    # seulement ce qui a échoué la dernière fois.
    last_verifier_weights: Dict[str, float] = field(default_factory=dict)

    # Suivi pour le désir dynamique resolve_conflict : combien de tentatives
    # CONSÉCUTIVES où l'exécuteur se croit performant (U_PE haut) alors que
    # le vérificateur rejette (U_PE - U_V > CONFLICT_GAP_THRESHOLD) — un
    # désaccord persistant entre les deux agents, pas juste un score faible.
    consecutive_conflict:   int = 0

    def brf(self, reward: float, u_pe: float, u_v: float,
            quality: str, feedback: str, hint: str,
            is_valid_json: bool = False,
            schema_compliance: float = 0.0,
            field_accuracy: float = 0.0,
            depth_match: bool = False,
            actual_depth: int = 0,
            generated_json: str = "",
            last_error: str = "",
            failed_fields: list = None,
            plan_used: str = "",
            verifier_weights: dict = None,
            schema_violations: list = None) -> None:
        """Belief Revision Function — met à jour les croyances."""
        self.attempt_count += 1
        self.past_u_v.append(u_v)
        self.past_rewards.append(reward)
        self.last_feedback          = feedback
        self.last_quality           = quality
        self.last_u_v               = u_v
        self.last_u_pe              = u_pe
        self.last_generated_json    = generated_json
        self.last_is_valid_json     = is_valid_json
        self.last_schema_compliance = schema_compliance
        self.last_field_accuracy    = field_accuracy
        self.last_depth_match       = depth_match
        self.last_actual_depth      = actual_depth
        self.last_error             = last_error or ""
        self.last_failed_fields     = failed_fields or []
        self.last_plan_name         = plan_used
        self.consecutive_fallback   = (self.consecutive_fallback + 1) if plan_used == "fallback" else 0
        self.last_verifier_weights  = verifier_weights or {}
        self.last_schema_violations = schema_violations or []
        if u_v >= V_SUCCESS_THRESHOLD:
            self.last_diagnostic_quality = "good"
        elif u_v >= DIAGNOSTIC_QUALITY_THRESHOLD:
            self.last_diagnostic_quality = "fair"
        else:
            self.last_diagnostic_quality = "poor"

        conflict_now = (quality != "success"
                         and (u_pe - u_v) > CONFLICT_GAP_THRESHOLD)
        self.consecutive_conflict = (self.consecutive_conflict + 1) if conflict_now else 0

        if hint and hint not in self.tried_hints:
            self.tried_hints.append(hint)

    def verifier_priority_dim(self) -> Optional[str]:
        """Dimension actuellement la plus pondérée par le vérificateur, ou
        None si aucun signal (vérificateur statique — poids non publiés)."""
        if not self.last_verifier_weights:
            return None
        return max(self.last_verifier_weights, key=self.last_verifier_weights.get)

    def is_plateau(self, n: int = 2, epsilon: float = 0.05) -> bool:
        """True si les n derniers U_V varient de moins de epsilon."""
        if len(self.past_u_v) < n:
            return False
        recent = self.past_u_v[-n:]
        return max(recent) - min(recent) < epsilon

    def best_u_v(self) -> float:
        return max(self.past_u_v) if self.past_u_v else 0.0

    def avg_u_v(self) -> float:
        return sum(self.past_u_v) / len(self.past_u_v) if self.past_u_v else 0.0

    def trend(self) -> str:
        """Régression linéaire sur l'historique de U_V."""
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


@dataclass
class Desire:
    """Objectif de l'agent avec cycle de vie complet."""
    name:        str
    priority:    int
    description: str
    status:      DesireStatus = DesireStatus.PENDING

    def activate(self)  -> None: self.status = DesireStatus.ACTIVE
    def suspend(self)   -> None: self.status = DesireStatus.SUSPENDED
    def achieve(self)   -> None: self.status = DesireStatus.ACHIEVED
    def fail(self)      -> None: self.status = DesireStatus.FAILED

    @property
    def is_active(self) -> bool:
        return self.status in (DesireStatus.PENDING, DesireStatus.ACTIVE)

    @property
    def is_achieved(self) -> bool:
        return self.status == DesireStatus.ACHIEVED


@dataclass
class Intention:
    """
    Plan engagé. `step_index` = position courante dans le CORPS du plan
    (séquence d'actions) : l'agent, tant qu'il tient son commitment, déroule
    les étapes du body une à une au lieu de re-sélectionner un plan.
    """
    plan_name:     str
    source_desire: str
    hint:          str
    status:        str = "ACTIVE"
    committed_at:  int = 0
    step_index:    int = 0

    def suspend(self)  -> None: self.status = "SUSPENDED"
    def achieve(self)  -> None: self.status = "ACHIEVED"
    def fail(self)     -> None: self.status = "FAILED"
    def resume(self)   -> None: self.status = "ACTIVE"

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"


class BDIState:
    """
    Agrégat BDI — cycle complet :
      perceive → brf → reconsider → options → filter → execute
    """

    def __init__(
        self,
        commitment_strategy: CommitmentStrategy = CommitmentStrategy.OPEN_MINDED
    ) -> None:
        self.beliefs:             BeliefBase         = BeliefBase()
        self.desires:             List[Desire]       = []
        self.intentions:          List[Intention]    = []
        self.commitment_strategy: CommitmentStrategy = commitment_strategy

    def set_task(self, task_id: str, task_goal: str, task_type: str,
                 schema: str = "", text: str = "",
                 ground_truth_json: str = "",
                 category: str = "", true_depth: int = 0) -> None:
        self.beliefs = BeliefBase(
            task_id=task_id, task_goal=task_goal, task_type=task_type,
            schema=schema, text=text,
            ground_truth_json=ground_truth_json,
            category=category, true_depth=true_depth,
        )
        self.desires = [
            Desire("generate_valid_json",   3, "Generate valid JSON matching the schema",
                   DesireStatus.ACTIVE),
            Desire("maximize_accuracy",     2, "Maximize field accuracy vs ground truth",
                   DesireStatus.ACTIVE),
            Desire("match_depth",           1, "Match the expected nesting depth",
                   DesireStatus.ACTIVE),
        ]
        self.intentions = []

    def brf(self, reward: float, u_pe: float, u_v: float,
            quality: str, feedback: str, hint: str,
            is_valid_json: bool = False,
            schema_compliance: float = 0.0,
            field_accuracy: float = 0.0,
            depth_match: bool = False,
            actual_depth: int = 0,
            generated_json: str = "",
            last_error: str = "",
            failed_fields: list = None,
            plan_used: str = "",
            verifier_weights: dict = None,
            schema_violations: list = None) -> None:
        self.beliefs.brf(reward, u_pe, u_v, quality, feedback, hint,
                         is_valid_json, schema_compliance, field_accuracy,
                         depth_match, actual_depth, generated_json, last_error,
                         failed_fields, plan_used, verifier_weights,
                         schema_violations)

        if quality == "success":
            for d in self.desires:
                d.achieve()
            for i in self.intentions:
                i.achieve()
            return

        if quality == "rejected":
            if self.intentions:
                self.intentions[-1].fail()
            if not is_valid_json:
                for d in self.desires:
                    if d.name == "maximize_accuracy":
                        d.suspend()

        # Réactivation : dès que le JSON redevient valide, la raison de la
        # suspension a disparu. Indépendant de `quality` — la bande "partial"
        # (0.4-0.7) est mathématiquement inatteignable tant que V_THRESHOLD ==
        # V_SUCCESS_THRESHOLD (payoff_calculator.py, les deux valent 0.7) : un
        # désir suspendu ne se réactivait donc plus JAMAIS pour le reste de la
        # tâche, même après un retour à un JSON parfaitement structuré. Voir
        # deepjsoneval_107_game : maximize_accuracy suspendu à l'attempt 2
        # (JSON invalide), jamais réactivé malgré schema_compliance=1.0 à
        # l'attempt 5 — improve_accuracy resté inaccessible, U_V=0.688 à
        # 0.012 pt du seuil de succès.
        if is_valid_json:
            for d in self.desires:
                if d.status == DesireStatus.SUSPENDED:
                    d.activate()

    def options(self) -> List[Desire]:
        return sorted(
            [d for d in self.desires if d.is_active and not d.is_achieved],
            key=lambda d: d.priority,
            reverse=True,
        )

    # ── Désirs DYNAMIQUES ───────────────────────────────────
    #
    # Les 3 désirs ci-dessus sont statiques : fixés une fois à set_task(),
    # seul leur STATUT change (actif/suspendu/atteint). generateOptions() dans
    # le cycle BDI classique (Rao & Georgeff) permet aussi de faire ÉMERGER de
    # nouveaux désirs à partir des croyances courantes — un signal qui n'existe
    # pas au début de la tâche mais apparaît en cours de route.
    #
    # Deux catégories, pas un seul mécanisme uniforme :
    #   PRÉEMPTIFS   : signal urgent, a le droit de couper un commitment en
    #                   cours (escape_plateau, final_attempt,
    #                   diagnose_unknown_failure, resolve_conflict). Renvoyés
    #                   par sync_dynamic_desires() pour être passés à
    #                   PlanLibrary.select_plan(dynamic_desires=...).
    #   ORDINAIRES   : simple prudence par défaut, rejoint le scan de
    #                   priorité statique normal SANS rien interrompre
    #                   (category_high_risk) — sinon il préempterait
    #                   aveuglément warm_start même quand une stratégie
    #                   apprise et fiable existe déjà pour cette catégorie.

    _PREEMPTING_DESIRE_NAMES = ("escape_plateau", "final_attempt",
                                 "diagnose_unknown_failure", "resolve_conflict")
    _DYNAMIC_DESIRE_NAMES    = _PREEMPTING_DESIRE_NAMES + ("category_high_risk",)

    def sync_dynamic_desires(self, max_replans: int = None,
                              category_high_risk: bool = False,
                              use_diagnostic_band: bool = False) -> set:
        """
        Réévalue les désirs dynamiques à partir des croyances courantes —
        appelé à chaque tour de _bdi_cycle. Un désir dynamique disparaît de
        lui-même dès que son signal déclencheur n'est plus vrai (pas besoin
        de le marquer "atteint" : il n'a jamais été "réussi", juste plus
        pertinent).

        `max_replans` et `category_high_risk` sont des informations que
        BDIState ne porte pas nativement (budget de la tâche, croyance
        WorldModel sur la catégorie) — fournies par l'appelant (BDIPlannerAgent).

        `use_diagnostic_band` (V2.1 #3, opt-in, défaut False = comportement
        historique inchangé) : si True, un plateau "fair" (0.5-0.7) et STABLE
        (pas dégradant) n'escalade PAS vers escape_plateau — on laisse le
        raffinement fin (improve_accuracy) continuer au lieu de sauter tout de
        suite à une escalade de température, réservée aux cas franchement
        "poor" ou en dégradation active.

        Renvoie uniquement le sous-ensemble PRÉEMPTIF actuellement actif — les
        désirs ordinaires (category_high_risk) sont ajoutés à self.desires
        (visibles via options()) mais pas renvoyés ici : ils rejoignent le
        scan statique normal via active_desires, pas via dynamic_desires.
        """
        b = self.beliefs
        plateaued = (
            b.attempt_count >= 1 and b.last_u_v < 0.7
            and (b.is_plateau(n=2, epsilon=0.05) or b.trend() == "DEGRADING")
        )
        if (use_diagnostic_band and plateaued
                and b.last_diagnostic_quality == "fair"
                and b.trend() != "DEGRADING"):
            plateaued = False
        is_final_attempt = (
            max_replans is not None and b.attempt_count >= 1
            and b.attempt_count == max_replans - 1
        )
        unrecognized_failure = b.consecutive_fallback >= 2
        conflict_persistent  = b.consecutive_conflict >= CONFLICT_STREAK_THRESHOLD

        self.desires = [d for d in self.desires if d.name not in self._DYNAMIC_DESIRE_NAMES]
        preempting: set = set()

        if plateaued:
            self.desires.append(Desire("escape_plateau", priority=10,
                description="Rompre un plateau de scores stagnants ou une tendance dégradante",
                status=DesireStatus.ACTIVE))
            preempting.add("escape_plateau")

        if is_final_attempt:
            self.desires.append(Desire("final_attempt", priority=20,
                description="Dernière tentative avant épuisement du budget — sécurité, pas exploration",
                status=DesireStatus.ACTIVE))
            preempting.add("final_attempt")

        if unrecognized_failure:
            self.desires.append(Desire("diagnose_unknown_failure", priority=15,
                description="2+ tentatives consécutives non reconnues par un plan spécifique",
                status=DesireStatus.ACTIVE))
            preempting.add("diagnose_unknown_failure")

        if conflict_persistent:
            self.desires.append(Desire("resolve_conflict", priority=18,
                description="Désaccord persistant PE/V : l'exécuteur se croit performant "
                            "(U_PE haut) mais le vérificateur rejette systématiquement — "
                            "recadrer sur les critères mécaniques du vérificateur",
                status=DesireStatus.ACTIVE))
            preempting.add("resolve_conflict")

        if category_high_risk and b.attempt_count == 0:
            self.desires.append(Desire("category_high_risk", priority=4,
                description="Catégorie historiquement difficile (croyance WorldModel) — amorce prudente",
                status=DesireStatus.ACTIVE))
            # PAS ajouté à `preempting` : voir docstring.

        return preempting

    def should_reconsider(self) -> bool:
        if self.commitment_strategy == CommitmentStrategy.BLIND:
            return False
        if self.commitment_strategy == CommitmentStrategy.OPEN_MINDED:
            return True
        b = self.beliefs
        if b.last_quality in ("success", "partial"):
            return True
        if b.attempt_count >= 1 and b.last_u_v < 0.05:
            return True
        return False

    def commit(self, plan_name: str = "unknown", source_desire: str = None) -> None:
        """
        S'engager sur un plan — nouvelle Intention, étape 0 de son corps.

        `source_desire` doit être le `trigger_desire` du plan RÉELLEMENT
        choisi (fourni par l'appelant, qui a l'objet Plan sous la main).
        Sans ce paramètre, on retombait sur le désir actif de plus haute
        priorité (via options()) — trompeur dès qu'un désir dynamique de
        priorité élevée (ex. escape_plateau=10) est actif mais que le plan
        RÉELLEMENT choisi appartient à un autre désir (ex. le scan statique
        retombe sur repair_structure/generate_valid_json une fois le hint de
        escape_plateau épuisé) : le log affichait alors "desire=escape_plateau"
        pour un plan qui n'a rien à voir avec ce désir.
        """
        if source_desire is None:  # compat : ancien comportement si non fourni
            active = self.options()
            source_desire = active[0].name if active else "generate_valid_json"
        self.intentions.append(Intention(
            plan_name     = plan_name,
            source_desire = source_desire,
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

    def should_replan(self, max_replans: int) -> bool:
        b = self.beliefs
        if b.last_quality == "success":   # U_V >= 0.7 : tâche résolue
            return False
        return b.attempt_count < max_replans

    def replan_reason(self) -> str:
        b = self.beliefs
        if b.last_feedback:
            return b.last_feedback
        return f"Insufficient quality (U_V={b.last_u_v:.3f})"

    def summary(self) -> str:
        b  = self.beliefs
        cs = self.commitment_strategy.value
        current = self.current_intention()
        plan_info = f"plan={current.plan_name}" if current else "no_intention"
        return (
            f"BDI[{b.task_id}] attempt={b.attempt_count} | "
            f"strategy={cs} | "
            f"category={b.category} depth={b.true_depth} | "
            f"best_U_V={b.best_u_v():.3f} | trend={b.trend()} | "
            f"quality={b.last_quality or 'none'} | {plan_info}"
        )
