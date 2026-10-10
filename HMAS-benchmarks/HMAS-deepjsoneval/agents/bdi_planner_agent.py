# ─────────────────────────────────────────────────────────
# bdi_planner_agent.py — AGENT BDI PLANIFICATEUR
#
# Adapté de agisdk/mas_project pour DeepJSONEval.
# Implémente le cycle BDI complet (Rao & Georgeff, 1991) :
#
#   perceive   → reçoit TASK_ASSIGNED ou VERIFICATION_RESULT
#   brf        → met à jour les Croyances
#   reconsider → abandonne ou garde l'intention courante
#   options()  → désirs actifs par priorité
#   filter()   → PlanLibrary.select() — meilleur plan applicable
#   execute    → publie EXECUTION_REQUESTED à LLMExecutorAgent
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import os
import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.message_bus     import MessageBus, Message, MessageType
from core.history_manager  import HistoryManager
from core.bdi_model        import BDIState, CommitmentStrategy
from core.plan_library     import PlanLibrary, Step
from core.utils            import extract_task_type
from core.strategy_learner import StrategyLearner
from core.world_model      import WorldModel
from core.audit_trace      import AuditTraceRecorder
from config                import MAX_REPLANS

# Traduction humaine des dimensions du vérificateur — utilisée pour faire
# savoir au planner CE QUE le vérificateur valorise le plus MAINTENANT (poids
# dynamiques), pas seulement ce qui a échoué à la dernière tentative.
_DIM_PHRASE = {
    "json_valid":         "producing syntactically valid JSON",
    "schema_compliance":  "matching the schema structure exactly",
    "field_accuracy":     "getting exact field values right",
    "depth_match":        "matching the expected nesting depth",
    "completeness":       "not leaving fields empty",
}


class BDIPlannerAgent(threading.Thread):
    """
    Agent BDI planificateur — sans LLM.

    La PlanLibrary remplace toute cascade if/elif.
    Chaque Plan a context_condition() + generate_hint().
    """

    def __init__(self, bus: MessageBus,
                 history_manager: HistoryManager,
                 stop_event: threading.Event,
                 commitment_strategy: CommitmentStrategy = CommitmentStrategy.SINGLE_MINDED,
                 world_model_path: str = None,
                 strategies_path: str = None,
                 trace_path: str = None,
                 run_id: str = None,
                 reliability_mode: str = "frequentist",
                 use_diagnostic_band: bool = False):
        super().__init__(daemon=True, name="BDIPlannerThread")
        self.bus          = bus
        self.hm           = history_manager
        self.stop_event   = stop_event
        self.bdi          = BDIState(commitment_strategy=commitment_strategy)
        self.plan_library = PlanLibrary()
        # `strategies_path`/`world_model_path` : voir WorldModel — permet
        # d'isoler un run comparatif du fichier partagé cumulatif.
        self.learner      = StrategyLearner(path=strategies_path)
        # Budget de délibération interne (surchargeable par tâche). L'agent
        # itère jusqu'à MAX_REPLANS tentatives SANS être recréé — sa mémoire
        # (Beliefs/Desires/Intentions) persiste sur toute la durée de la tâche.
        self.max_replans  = MAX_REPLANS
        # Meilleure tentative vue pour la tâche courante — le MAS ne doit jamais
        # régresser : à l'épuisement du budget, on remonte le meilleur essai et
        # non le dernier (garde anti-régression, ex-boucle externe de shots).
        self._best_content: dict = {}
        # Croyances persistantes cross-tâches/runs : difficulté par catégorie
        # (environnement) + fiabilité de l'executor par (catégorie, plan) —
        # voir core/world_model.py.
        self.world        = WorldModel(path=world_model_path, run_id=run_id,
                                        reliability_mode=reliability_mode)
        # V2.1 #3, opt-in (défaut False = comportement historique inchangé) —
        # voir BDIState.sync_dynamic_desires.
        self.use_diagnostic_band = use_diagnostic_band
        # Trace structurée par tentative (V2.1, changement #1) — additive,
        # ne remplace ni le CSV ni le JSON de résultat par tâche. Débloque
        # les analyses regret/Nash de gt_analyzer.py, qui n'ont jamais eu de
        # données exploitables faute de granularité multi-tentatives.
        self.tracer       = AuditTraceRecorder(path=trace_path)
        # Température effectivement envoyée à la dernière tentative
        # dispatchée — capturée ici (pas ailleurs dans BeliefBase) car elle
        # ne pilote aucun raisonnement, seulement la trace d'audit.
        self._last_dispatch_temperature: float = 0.1
        print(f"[BDIPlannerThread] Plan library: {self.plan_library}")

    # ── Boucle principale ─────────────────────────────────

    def run(self) -> None:
        print("[BDIPlannerThread] Started — real BDI cycle (no LLM)...")
        while not self.stop_event.is_set():
            try:
                msg = self.bus.get("planner", timeout=2)
            except queue.Empty:
                continue

            if msg.msg_type == MessageType.STOP:
                break
            elif msg.msg_type == MessageType.TASK_ASSIGNED:
                self._on_task_assigned(msg)
            elif msg.msg_type == MessageType.VERIFICATION_RESULT:
                self._on_verification_result(msg)

        print("[BDIPlannerThread] Stopped.")

    # ── Handlers ──────────────────────────────────────────

    def _on_task_assigned(self, msg: Message) -> None:
        task_id           = msg.content["task_id"]
        schema            = msg.content.get("schema", "")
        text              = msg.content.get("text", "")
        ground_truth_json = msg.content.get("ground_truth_json", "")
        category          = msg.content.get("category", "")
        true_depth        = msg.content.get("true_depth", 0)
        max_attempts      = msg.content.get("max_attempts", None)

        task_type = extract_task_type(task_id)

        # Budget de délibération pour CETTE tâche (câblé sur --shots).
        if max_attempts is not None:
            self.max_replans = max(1, int(max_attempts))

        # Nouvelle tâche = nouveau percept initial. On repart d'un état mental
        # propre pour la tâche, mais — contrairement à l'ancienne archi par
        # shots — cet état va PERSISTER durant toute la boucle de délibération
        # interne (toutes les tentatives), sans recréation de l'agent.
        self.bdi.set_task(
            task_id=task_id,
            task_goal=f"Generate JSON for category={category}, depth={true_depth}",
            task_type=task_type,
            schema=schema, text=text,
            ground_truth_json=ground_truth_json,
            category=category, true_depth=true_depth,
        )
        self._best_content = {}

        print(f"\n  [BDI] New task: {task_id} (budget={self.max_replans} tentatives)")
        print(f"  [BDI] {self.bdi.summary()}")

        # Première itération du cycle BDI. Les suivantes sont déclenchées par
        # _on_verification_result (percept → brf → reconsider → replan) tant
        # que le budget n'est pas épuisé — la boucle vit DANS l'agent.
        self._bdi_cycle(reconsider=True)

    def _on_verification_result(self, msg: Message) -> None:
        reward            = msg.content.get("reward",            0.0)
        u_pe              = msg.content.get("U_PE",              0.0)
        u_v               = msg.content.get("U_V",               0.0)
        quality           = msg.content.get("quality",           "rejected")
        feedback          = msg.content.get("feedback",          "")
        hint_used         = msg.content.get("hint_used",         "")
        is_valid_json     = msg.content.get("is_valid_json",     False)
        schema_compliance = msg.content.get("dim_schema_compliance", 0.0)
        field_accuracy    = msg.content.get("dim_field_accuracy", 0.0)
        depth_match       = msg.content.get("dim_depth_match",   False)
        actual_depth      = msg.content.get("actual_depth",      0)
        generated_json    = msg.content.get("generated_json",    "")
        last_error        = msg.content.get("last_error",        "")
        failed_fields     = msg.content.get("failed_fields",     [])
        weights_v_used    = msg.content.get("weights_v_used",    None)
        schema_violations = msg.content.get("schema_violations", [])
        weights_v_diagnostic = msg.content.get("weights_v_diagnostic", None)
        dim_completeness  = msg.content.get("dim_completeness",  0.0)

        # Plan qui a produit CE résultat — capturé avant que brf()/le prochain
        # commit() ne fasse avancer la pile d'intentions.
        plan_used = self.bdi.current_intention().plan_name \
            if self.bdi.current_intention() else "unknown"

        # 1. BRF
        self.bdi.brf(reward, u_pe, u_v, quality, feedback, hint_used,
                     is_valid_json, schema_compliance, field_accuracy,
                     depth_match, actual_depth, generated_json, last_error,
                     failed_fields, plan_used, weights_v_used,
                     schema_violations=schema_violations)

        # Croyances persistantes : l'environnement (difficulté de la catégorie)
        # et l'agent executor (fiabilité de ce plan pour cette catégorie).
        self.world.record_task_outcome(self.bdi.beliefs.category, u_v)
        self.world.record_plan_outcome(self.bdi.beliefs.category, plan_used,
                                        success=(quality == "success"))

        # Trace d'audit par tentative (V2.1, #1) — snapshot des croyances
        # APRÈS mise à jour (brf + world model) pour cette tentative. Champs
        # world_model_reads : lecture post-update, pas la valeur qui a
        # effectivement piloté la sélection du plan de cette tentative (celle-là
        # a été lue plus tôt dans _compute_context/_is_category_high_risk) —
        # suffisant pour l'audit de tendance, pas pour rejouer exactement la
        # décision de sélection de plan.
        plan_obj = self.plan_library.get_plan(plan_used)
        trigger_desire = plan_obj.trigger_desire if plan_obj else ""
        self.tracer.record_attempt(self.bdi.beliefs.task_id, {
            "attempt_index":        self.bdi.beliefs.attempt_count,
            "plan_name":            plan_used,
            "trigger_desire":       trigger_desire,
            "preempting":           trigger_desire in BDIState._PREEMPTING_DESIRE_NAMES,
            "temperature":          self._last_dispatch_temperature,
            "reward":               reward,
            "U_PE":                 u_pe,
            "U_V":                  u_v,
            "dim_schema_compliance": schema_compliance,
            "dim_field_accuracy":   field_accuracy,
            "dim_depth_match":      depth_match,
            "dim_completeness":     dim_completeness,
            "quality":              quality,
            "is_valid_json":        is_valid_json,
            "actual_depth":         actual_depth,
            "true_depth":           self.bdi.beliefs.true_depth,
            "failed_fields_count":  len(failed_fields),
            "schema_violations":    schema_violations,
            "consecutive_fallback": self.bdi.beliefs.consecutive_fallback,
            "consecutive_conflict": self.bdi.beliefs.consecutive_conflict,
            "diagnostic_quality":   self.bdi.beliefs.last_diagnostic_quality,
            "weights_v_used":       weights_v_used,
            "weights_v_diagnostic": weights_v_diagnostic,
            "world_model_reads": {
                "category_n":         self.world.category_samples(self.bdi.beliefs.category),
                "category_avg_u_v":   self.world.category_avg_u_v(self.bdi.beliefs.category),
                "plan_reliability":   self.world.executor_reliability(
                    self.bdi.beliefs.category, plan_used),
            },
        })

        print(f"\n  [BDI] Beliefs updated: quality={quality}, U_V={u_v:.3f}, "
              f"trend={self.bdi.beliefs.trend()}")
        print(f"  [BDI] {self.bdi.summary()}")

        # Garde anti-régression : on mémorise la MEILLEURE tentative de la tâche
        # (par U_V). Si le budget s'épuise sans succès, c'est elle qu'on remonte
        # au système, pas la dernière — le MAS ne doit jamais régresser.
        report = {
            "task_id":               self.bdi.beliefs.task_id,
            "attempts":              self.bdi.beliefs.attempt_count,
            "U_PE":                  u_pe,
            "U_V":                   u_v,
            "reward":                reward,
            "quality":               quality,
            "hint_used":             hint_used,
            "plan_name":             plan_used,
            "generated_json":        generated_json,
            "is_valid_json":         is_valid_json,
            "dim_json_valid":        1.0 if is_valid_json else 0.0,
            "dim_schema_compliance": schema_compliance,
            "dim_field_accuracy":    field_accuracy,
            "dim_depth_match":       depth_match,
            "actual_depth":          actual_depth,
            "feedback":              feedback,
            "last_error":            last_error,
            "failed_fields":         failed_fields,
            "schema_violations":     schema_violations,
        }
        if not self._best_content or u_v > self._best_content.get("U_V", -1.0):
            self._best_content = report

        if quality == "success":   # U_V >= 0.7 : tâche résolue, pas de replan
            return

        # 2. Reconsider — sous SINGLE_MINDED, l'intention n'est abandonnée que
        # si le plan échoue clairement (cf. BDIState.should_reconsider()). Si on
        # garde le commitment, on AVANCE d'une étape dans le corps du plan pour
        # que _bdi_cycle exécute l'action suivante de la recette.
        reconsider = self.bdi.should_reconsider()
        if reconsider:
            self.bdi.drop_intention()
        else:
            intention = self.bdi.current_intention()
            if intention:
                intention.step_index += 1

        # 3–5. Replan ou abandon — décision de délibération INTERNE à l'agent.
        if self.bdi.should_replan(self.max_replans):
            print(f"  [BDI] Replan (attempt {self.bdi.beliefs.attempt_count + 1}/{self.max_replans})")
            self._bdi_cycle(reconsider=reconsider)
        else:
            print(f"  [BDI] Budget exhausted ({self.max_replans}). "
                  f"Reporting best attempt (U_V={self._best_content.get('U_V', 0.0):.3f}).")
            content = dict(self._best_content)
            content["reason"] = f"BDI: {self.max_replans} replans without success"
            self.bus.publish(Message(
                sender   = "planner",
                receiver = "system",
                msg_type = MessageType.ERROR_REPORTED,
                content  = content,
            ))

    # ── Cycle BDI ─────────────────────────────────────────

    def _bdi_cycle(self, reconsider: bool = True) -> None:
        """
        Un tour du cycle : options() → (continuation OU filter()) → execute().

        Commitment réel : si l'agent tient son intention (reconsider=False), il
        DÉROULE l'étape suivante du CORPS du plan engagé au lieu de re-choisir un
        plan. Il ne re-délibère (select_plan) que si le plan est terminé, n'est
        plus applicable, ou si l'agent a décidé de reconsidérer.
        """
        b = self.bdi.beliefs

        # Désirs DYNAMIQUES (generateOptions du cycle BDI classique) : émergent
        # des croyances courantes, pas fixés à set_task() — voir
        # BDIState.sync_dynamic_desires (escape_plateau, final_attempt,
        # diagnose_unknown_failure = préemptifs ; category_high_risk = ordinaire,
        # non renvoyé ici, rejoint juste active_desires via options()).
        dynamic_desires = self.bdi.sync_dynamic_desires(
            max_replans=self.max_replans,
            category_high_risk=self._is_category_high_risk(b.category),
            use_diagnostic_band=self.use_diagnostic_band,
        )
        if dynamic_desires:
            print(f"  [BDI] Dynamic desire(s) active: {sorted(dynamic_desires)}")

        active = self.bdi.options()
        print(f"  [BDI] Active desires: {[d.name for d in active]}")
        ctx = self._compute_context(dynamic_desires)

        # Un désir dynamique urgent prime sur un commitment en cours qui ne le
        # sert pas — signal de "changer complètement d'approche", pas de
        # "continuer la recette actuelle". On force la re-délibération.
        if not reconsider and dynamic_desires:
            intention = self.bdi.current_intention()
            cur_plan = self.plan_library.get_plan(intention.plan_name) if intention else None
            cur_desire = cur_plan.trigger_desire if cur_plan else None
            if not cur_plan or cur_desire not in dynamic_desires:
                print(f"  [BDI] Dynamic desire overrides commitment on "
                      f"'{cur_plan.name if cur_plan else None}' — forcing re-deliberation")
                reconsider = True
                self.bdi.drop_intention()
            else:
                # Le désir du plan engagé est TOUJOURS actif (ex. escape_plateau
                # reste vrai plusieurs tours de suite), mais un AUTRE désir plus
                # urgent peut être apparu ENTRE-TEMPS sans jamais déloger le
                # commitment — le test ci-dessus ne détecte que la disparition du
                # désir courant, pas l'apparition d'un désir plus prioritaire.
                # Bug réel diagnostiqué le 2026-08-05 sur deepjsoneval_69_movie
                # (job #923621) : engagé sur break_plateau (désir escape_plateau,
                # priorité 10) dès la tentative 7 ; à la tentative 10,
                # final_attempt (priorité 20, safe_finish/FixFailedFields) devient
                # actif mais escape_plateau reste vrai aussi → l'agent ne
                # re-délibère jamais, reste sur break_plateau, et le seul plan
                # capable de cibler les champs hallucinés précis n'est jamais
                # atteint alors même que c'est la DERNIÈRE tentative du budget.
                def _desire_priority(name: str) -> int:
                    d = next((d for d in self.bdi.desires if d.name == name), None)
                    return d.priority if d else -1
                cur_priority = _desire_priority(cur_desire)
                more_urgent = sorted(
                    n for n in dynamic_desires
                    if n != cur_desire and _desire_priority(n) > cur_priority
                )
                if more_urgent:
                    print(f"  [BDI] More urgent desire(s) {more_urgent} appeared while "
                          f"committed to '{cur_plan.name}' ({cur_desire}) — forcing re-deliberation")
                    reconsider = True
                    self.bdi.drop_intention()

        # 1. Continuation du plan engagé (means-end : on suit la recette).
        if not reconsider:
            intention = self.bdi.current_intention()
            if intention:
                plan = self.plan_library.get_plan(intention.plan_name)
                if plan and (plan.priority == 0 or plan.context_condition(b, **ctx)):
                    step = self._emit_next_step(plan, intention, b, ctx)
                    if step is not None:
                        print(f"  [BDI] Commitment held — plan '{plan.name}' "
                              f"step {intention.step_index+1}/{len(plan.body)}")
                        self._dispatch(plan, intention, step, b)
                        return
                # plan terminé / hors-contexte → on abandonne et re-délibère
                self.bdi.drop_intention()

        # 2. Délibération : sélection d'un NOUVEAU plan (filter), engagement,
        #    puis exécution de sa première étape applicable.
        plan = self.plan_library.select_plan(b, **ctx)
        self.bdi.commit(plan_name=plan.name, source_desire=plan.trigger_desire)
        intention = self.bdi.current_intention()
        step = self._emit_next_step(plan, intention, b, ctx)
        if step is None:
            # Garde-fou : ne devrait pas arriver (fallback a toujours une étape).
            step = Step("fallback", f"Attempt {b.attempt_count + 1} — regenerate the JSON.", 0.1)
        print(f"  [BDI] Deliberate → plan '{plan.name}' "
              f"(priority={plan.priority}, desire={intention.source_desire})")
        self._dispatch(plan, intention, step, b)

    def _emit_next_step(self, plan, intention, b, ctx):
        """
        Avance dans le CORPS du plan à partir de step_index, saute les actions
        non applicables (build()→None) ou dont le hint a déjà été essayé, et
        renvoie la 1re étape exécutable (sans dépasser son index — l'avance
        définitive se fait après vérification, dans _on_verification_result).
        """
        while intention.step_index < len(plan.body):
            action = plan.body[intention.step_index]
            step = action.build(b, **ctx)
            if step is None or (step.hint and step.hint in b.tried_hints):
                intention.step_index += 1
                continue
            return step
        return None

    def _dispatch(self, plan, intention, step, b) -> None:
        """execute() — publie la requête d'exécution correspondant à l'étape."""
        hint = self._add_priority_dim_note(step.hint, b)
        intention.hint = hint
        self._last_dispatch_temperature = step.temperature
        print(f"  [BDI] Action: {step.action} | temp={step.temperature}")
        print(f"  [BDI] Hint: {hint[:90] or '(none — free generation)'}")
        self.bus.publish(Message(
            sender   = "planner",
            receiver = "executor",
            msg_type = MessageType.EXECUTION_REQUESTED,
            content  = {
                "task_id":           b.task_id,
                "schema":            b.schema,
                "text":              b.text,
                "ground_truth_json": b.ground_truth_json,
                "category":          b.category,
                "true_depth":        b.true_depth,
                "hint":              hint,
                "plan_name":         plan.name,
                "action":            step.action,
                "attempt":           b.attempt_count + 1,
                "prev_json":         b.last_generated_json if b.attempt_count >= 1 else "",
                "temperature":       step.temperature,
            }
        ))

    def _add_priority_dim_note(self, hint: str, b) -> str:
        """
        Ferme la boucle avec le vérificateur DYNAMIQUE : si celui-ci a publié
        ses poids courants (voir RuleBasedVerifierAgentDynamic), le planner
        sait quelle dimension pèse le plus dans le jugement ACTUEL — pas
        seulement ce qui a échoué la dernière fois. N'ajoute rien si le hint
        est vide (génération libre, ex. first_attempt) ou si le vérificateur
        est statique (verifier_priority_dim() renvoie None, sans poids publiés).
        """
        if not hint:
            return hint
        dim = b.verifier_priority_dim()
        if not dim:
            return hint
        phrase = _DIM_PHRASE.get(dim, dim)
        return (f"{hint}\n\n[Verifier priority: right now, "
                f"'{dim}' ({phrase}) carries the most weight in scoring — "
                f"make sure this dimension is right.]")

    def _compute_context(self, dynamic_desires: set = None) -> dict:
        b = self.bdi.beliefs
        return {
            "warm_hint":               self._get_warm_hint(),
            "failed_fields":           b.last_failed_fields,
            "executor_reliability_fn": self.world.executor_reliability,
            "max_replans":             self.max_replans,
            # options() = désirs actifs (statiques + dynamique ORDINAIRE
            # category_high_risk) — passé à select_plan() pour le scan normal.
            # On exclut seulement les désirs PRÉEMPTIFS : ceux-là ont déjà leur
            # propre passe prioritaire via `dynamic_desires` ci-dessous, les
            # inclure aussi ici serait redondant (mais sans risque).
            "active_desires":          {d.name for d in self.bdi.options()
                                         if d.name not in BDIState._PREEMPTING_DESIRE_NAMES},
            # Désirs dynamiques PRÉEMPTIFS (cf. sync_dynamic_desires) — priorité
            # absolue dans select_plan(), pas une simple exclusion.
            "dynamic_desires":         dynamic_desires or set(),
        }

    def _is_category_high_risk(self, category: str, min_samples: int = 10,
                                threshold: float = 0.5) -> bool:
        """
        Croyance WorldModel : cette catégorie a-t-elle historiquement un score
        moyen faible sur un échantillon suffisant (cross-tâches, cross-runs) ?
        `min_samples` évite de pénaliser une catégorie à peine rencontrée —
        même logique de prudence que `executor_reliability`.
        """
        if not category or self.world.category_samples(category) < min_samples:
            return False
        return self.world.category_avg_u_v(category) < threshold

    def _get_warm_hint(self) -> str:
        b = self.bdi.beliefs
        if b.attempt_count != 0:
            return ""
        # Seuil basé sur l'historique accumulé : si le LLM libre réussit déjà
        # >= 50% du temps sur cette catégorie, le warm_start nuit → on le saute.
        nohint_rate = self.learner.get_nohint_success_rate(b.task_type)
        if nohint_rate >= 0.5:
            print(f"  [BDI] Warm-start skipped: nohint_rate={nohint_rate:.2f} >= 0.5 "
                  f"for '{b.task_type}'")
            return ""
        learned = self.learner.get_best_hint(
            b.task_type, exclude=b.tried_hints, require_success=True
        )
        if learned and learned not in b.tried_hints:
            print(f"  [BDI] Warm-start: nohint_rate={nohint_rate:.2f} for '{b.task_type}'")
            return learned
        return ""
