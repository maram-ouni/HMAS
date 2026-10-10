# ─────────────────────────────────────────────────────────
# plan_library.py — BIBLIOTHÈQUE DE PLANS BDI pour DeepJSONEval
#
# Un PLAN BDI (Rao & Georgeff 1991 / AgentSpeak) est un triplet :
#     trigger  — le désir/but qui le déclenche
#     context  — condition d'applicabilité (booléen sur les croyances)
#     body     — une SÉQUENCE D'ACTIONS que l'agent exécute (la recette)
#
# ⚠ Différence clé avec une simple RÈGLE de production (condition → action
# unique) : ici le body est une liste ordonnée d'Actions. L'agent s'ENGAGE
# sur un plan (Intention) et déroule ses étapes une à une ; il ne re-délibère
# que lorsque le plan est terminé ou n'est plus applicable. C'est le
# means-end reasoning BDI, pas un if/elif déguisé.
#
# Une ACTION = une étape primitive qui, à partir des croyances, produit une
# requête d'exécution (hint + température). Une action peut être NON applicable
# dans l'état courant (build() renvoie None) → l'agent saute cette étape et
# passe à la suivante du body.
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# ═══════════════════════════════════════════════════════════
#  Étape produite par une action
# ═══════════════════════════════════════════════════════════

@dataclass
class Step:
    """Une étape exécutable : ce que l'agent envoie à l'exécuteur LLM."""
    action:      str            # nom de l'action (traçabilité)
    hint:        str            # instruction pour le LLM ("" = génération libre)
    temperature: float = 0.1


# ═══════════════════════════════════════════════════════════
#  Helpers de parsing schéma (partagés par plusieurs actions)
# ═══════════════════════════════════════════════════════════

def _build_skeleton(schema_str: str, max_depth: int = 6, max_fields: int = 12) -> str:
    """Parse un JSON Schema et retourne un squelette JSON avec valeurs nulles."""
    try:
        schema = _json.loads(schema_str)
    except Exception:
        return ""

    def extract(obj, depth=0):
        if depth > max_depth or not isinstance(obj, dict):
            return None
        t = obj.get("type", "object")
        if t == "object":
            props = obj.get("properties", {})
            if not props:
                return None
            result = {}
            for key, val in list(props.items())[:max_fields]:
                result[key] = extract(val, depth + 1)
            return result
        elif t == "array":
            items = obj.get("items", {})
            child = extract(items, depth + 1)
            return [child] if child is not None else [None]
        else:
            return None  # feuille → placeholder null

    skeleton = extract(schema)
    if not skeleton:
        return ""
    return _json.dumps(skeleton, indent=2)


def _detect_wrapper(beliefs) -> Optional[str]:
    """Renvoie la clé wrapper si le JSON est enveloppé dans le titre du schéma."""
    if not (beliefs.last_is_valid_json
            and beliefs.last_schema_compliance < 0.3
            and beliefs.last_field_accuracy < 0.1):
        return None
    try:
        obj = _json.loads(beliefs.last_generated_json)
        if not isinstance(obj, dict) or len(obj) != 1:
            return None
        schema_obj = _json.loads(beliefs.schema) if beliefs.schema else {}
        title = schema_obj.get("title", "")
        root_key = next(iter(obj))
        if title and root_key.lower() == title.lower():
            return root_key
    except Exception:
        return None
    return None


def _is_schema_as_data(beliefs) -> bool:
    """Le modèle a retourné la définition du schéma au lieu de données."""
    _META = {"type", "properties", "title", "required", "$schema", "items", "definitions"}
    if not (beliefs.last_is_valid_json and beliefs.last_schema_compliance < 0.3):
        return False
    try:
        obj = _json.loads(beliefs.last_generated_json)
        if not isinstance(obj, dict):
            return False
        return len(set(obj.keys()) & _META) >= 2
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
#  ACTIONS primitives (briques des body de plans)
# ═══════════════════════════════════════════════════════════

class Action:
    """Étape primitive d'un plan. build() → Step, ou None si non applicable."""
    name: str = "action"

    def build(self, beliefs, **ctx) -> Optional[Step]:
        raise NotImplementedError


class GenerateFree(Action):
    """Génération libre — aucun hint (identique au baseline)."""
    name = "generate_free"

    def build(self, beliefs, **ctx):
        return Step(self.name, "", 0.1)


class WarmStart(Action):
    """Réutilise une stratégie apprise (fournie via warm_hint)."""
    name = "warm_start"

    def build(self, beliefs, warm_hint="", **ctx):
        if not warm_hint:
            return None
        return Step(self.name, warm_hint, 0.1)


class RepairInvalidJSON(Action):
    """Corrige un JSON non parsable (markdown, texte, commentaires…)."""
    name = "repair_invalid_json"

    def build(self, beliefs, **ctx):
        if beliefs.last_is_valid_json:
            return None
        # Numéro de tentative inclus dans le hint (comme Fallback.build()) —
        # sans ça, ce texte 100% statique est ajouté à tried_hints à sa
        # PREMIÈRE invocation et devient indisponible pour le reste de la
        # tâche : constaté sur deepjsoneval_71_viecle (run 2026-07-11 20h49)
        # où repair_syntax l'utilise à la tentative 3, puis safe_finish
        # (tentative 10, JSON toujours invalide) le trouve déjà dans
        # tried_hints et retombe sur copy_exact_values — exactement le
        # comportement d'avant le correctif safe_finish, sans aucun effet.
        return Step(self.name,
            f"CRITICAL ERROR (attempt {beliefs.attempt_count + 1}): the previous output was not valid JSON. "
            "Return ONLY a raw JSON object — no markdown code blocks (```), "
            "no explanations, no comments. "
            "The output must be directly parseable by json.loads(). "
            "Start your response with '{' and end with '}'.", 0.1)


class StripWrapper(Action):
    """Retire la clé racine parasite (titre du schéma utilisé comme wrapper)."""
    name = "strip_wrapper"

    def build(self, beliefs, **ctx):
        wrapper = _detect_wrapper(beliefs)
        if not wrapper:
            return None
        try:
            schema_obj = _json.loads(beliefs.schema) if beliefs.schema else {}
            required   = schema_obj.get("required", [])
        except Exception:
            required = []
        req_str = ", ".join(f'"{k}"' for k in required[:6]) if required else "(see schema)"
        # Numéro de tentative inclus (même correctif que RepairInvalidJSON,
        # 2026-07-11) : sans ça, ce hint devient indisponible dans
        # tried_hints après son premier essai, alors que escape_plateau peut
        # préempter plusieurs tentatives de suite sur le MÊME diagnostic
        # persistant (voir BreakPlateau/RestartFresh, 2026-08-04).
        return Step(self.name,
            f"WRONG ROOT STRUCTURE (attempt {beliefs.attempt_count + 1}): "
            f"You wrapped your answer inside a '{wrapper}' key. "
            f"Do NOT add any outer wrapper. "
            f"Your JSON must start DIRECTLY with the required top-level keys: {req_str}. "
            f"WRONG:   {{ \"{wrapper}\": {{ ... }} }}\n"
            f"CORRECT: {{ {req_str}: ... }}", 0.1)


class FixSchemaAsData(Action):
    """Le modèle a renvoyé le schéma au lieu des données — corrige le cadrage."""
    name = "fix_schema_as_data"

    def build(self, beliefs, **ctx):
        if not _is_schema_as_data(beliefs):
            return None
        # Numéro de tentative inclus — voir StripWrapper ci-dessus. Diagnostiqué
        # le 2026-08-04 : sur deepjsoneval_7_viecle (job #923027), ce diagnostic
        # restait vrai jusqu'à la tentative 10 (schéma toujours renvoyé comme
        # données), mais n'était corrigé qu'UNE fois (tentative 2, repair_structure)
        # avant qu'escape_plateau ne préempte et route vers break_plateau
        # (température seule, sans jamais repointer le vrai problème) — le hint
        # statique devenait "déjà essayé" et ne pouvait plus se redéclencher.
        return Step(self.name,
            f"CRITICAL ERROR (attempt {beliefs.attempt_count + 1}): "
            "You returned the JSON SCHEMA definition, not JSON data. "
            "Your task is to generate DATA that fits the schema — extract real values from the text. "
            "Do NOT output the schema itself. The schema is only a template; fill it with actual values.",
            0.1)


class FillSkeleton(Action):
    """Injecte un squelette JSON (valeurs nulles) que le modèle doit remplir."""
    name = "fill_skeleton"

    def build(self, beliefs, **ctx):
        if not (beliefs.last_is_valid_json
                and beliefs.last_schema_compliance < 0.3
                and beliefs.last_field_accuracy < 0.1):
            return None
        skeleton = _build_skeleton(beliefs.schema)
        if not skeleton:
            return None
        return Step(self.name,
            f"The JSON structure is completely wrong (attempt {beliefs.attempt_count + 1}). "
            "Fill in the following skeleton with REAL values extracted from the text. "
            "Replace every null with the actual value from the text. "
            "Do NOT copy null — find the real value:\n\n" + skeleton, 0.1)


class EnforceSchema(Action):
    """
    Rappelle la structure exacte du schéma (clés + imbrication).

    Ne réinjecte PAS le texte du schéma : il est déjà présent intégralement
    dans le prompt de base (_build_prompt() — section "JSON SCHEMA:"). Le
    faire ici dupliquait jusqu'à 1000 caractères à chaque appel — c'est cette
    action qui clôture 5 des 12 plans (repair_syntax, repair_structure,
    fix_depth, strict_schema, deep_diagnosis), donc la duplication était quasi
    systématique. Combinée à prev_json (jusqu'à 1200 chars), elle faisait
    déborder le contexte sur les schémas profonds (depth=6) : vu sur
    deepjsoneval_178_device — prompt jusqu'à 3947 tokens, max_tokens réduit à
    128 par la logique anti-overflow, JSON tronqué et invalide à CHAQUE
    tentative (U_V=0.000 sur les 8 tentatives du budget, deux runs de suite).
    """
    name = "enforce_schema"

    def build(self, beliefs, **ctx):
        return Step(self.name,
            f"The previous JSON did not match the schema structure (attempt {beliefs.attempt_count + 1}). "
            "Follow the schema shown above EXACTLY — same key names, same nesting structure, "
            "same field order. Do not add or omit any field.", 0.1)


class FixFailedFields(Action):
    """
    Corrige UNIQUEMENT les champs signalés en échec par le vérificateur.

    N'était utilisée QUE dans safe_finish (dernier essai, priorité 95) — mais
    son hint était 100% statique (mêmes failed_fields d'une tentative à
    l'autre si le contenu reste faux), donc "déjà essayé" dès qu'un autre plan
    l'avait utilisée plus tôt (ex. via improve_accuracy) : safe_finish
    perdait alors sa SEULE étape exécutable, devenait inéligible, et le
    budget retombait sur un plan moins ciblé (nash_compromise) même à la
    toute dernière tentative. Diagnostiqué le 2026-08-05 sur
    deepjsoneval_69_movie (job #923499) : dim_field_accuracy=0.27, tous les
    champs de Crew hallucinés, mais FixFailedFields jamais atteinte au bon
    moment.
    """
    name = "fix_failed_fields"

    def build(self, beliefs, failed_fields=None, **ctx):
        fields = failed_fields or beliefs.last_failed_fields
        if not fields:
            return None
        fields_str = ", ".join(f'"{f}"' for f in fields[:10])
        return Step(self.name,
            f"The following fields were incorrect or missing (attempt {beliefs.attempt_count + 1}): "
            f"{fields_str}. "
            "Re-read the text carefully and correct ONLY these fields. "
            "Keep all other fields exactly as they were.", 0.1)


class ExtractAllFields(Action):
    """Extraction exhaustive de toutes les informations du texte."""
    name = "extract_all_fields"

    def build(self, beliefs, **ctx):
        return Step(self.name,
            f"Many fields are missing or have wrong values (attempt {beliefs.attempt_count + 1}). "
            "Re-read the text CAREFULLY and extract ALL pieces of information: "
            "numbers, dates, names, prices, identifiers. "
            "Every value in the text should appear in the JSON. "
            "Do not leave any field as null if the information is present in the text.", 0.1)


class MatchDepth(Action):
    """Ajuste la profondeur d'imbrication à la profondeur attendue."""
    name = "match_depth"

    def build(self, beliefs, **ctx):
        if not (beliefs.last_is_valid_json
                and not beliefs.last_depth_match
                and beliefs.true_depth > 0):
            return None
        actual, expected = beliefs.last_actual_depth, beliefs.true_depth
        if actual < expected:
            direction = (f"Add more nesting levels — wrap values inside sub-objects "
                         f"to reach depth {expected} (current depth: {actual}).")
        else:
            direction = (f"Reduce nesting — flatten some sub-objects "
                         f"to reach depth {expected} (current depth: {actual}).")
        return Step(self.name,
            f"The nesting depth is wrong (expected {expected}, got {actual}, "
            f"attempt {beliefs.attempt_count + 1}). " + direction, 0.1)


class CopyExactValues(Action):
    """Recopie les valeurs exactes du texte (pas d'inférence)."""
    name = "copy_exact_values"

    def build(self, beliefs, **ctx):
        return Step(self.name,
            f"The field values do not match the text (attempt {beliefs.attempt_count + 1}). "
            "Read the text again very carefully. "
            "Copy exact values: dates in the format mentioned, numbers exactly as they appear, "
            "names exactly as written. Do not infer or round values — use what the text explicitly states.",
            0.1)


class BreakPlateau(Action):
    """
    Brise un plateau de scores.

    Un plateau a deux causes très différentes, qui appellent des remèdes
    différents :
      1) le modèle hésite entre plusieurs lectures plausibles du texte →
         varier la température aide réellement à explorer d'autres sorties.
      2) le modèle est structurellement bloqué (schéma renvoyé comme
         donnée, wrapper parasite) → la température seule ne change RIEN,
         car le décodage contraint (guided_json) régénère une structure
         quasi identique quel que soit l'échantillonnage (vérifié le
         2026-08-04 sur deepjsoneval_7_viecle, job #923027 : U_V identique
         à 0.4625 sur 5 tentatives consécutives malgré temp=0.7). Dans ce
         cas, escape_plateau (priorité préemptive) prenait le pas sur
         repair_structure/FixSchemaAsData — le bon diagnostic existait mais
         n'était jamais reproposé. Ici on le repointe explicitement au lieu
         de se rabattre sur un message générique + température.
    """
    name = "break_plateau"

    def build(self, beliefs, **ctx):
        scores = " → ".join(f"{u:.2f}" for u in beliefs.past_u_v[-3:])
        wrapper = _detect_wrapper(beliefs)
        if _is_schema_as_data(beliefs):
            return Step(self.name,
                f"PLATEAU DETECTED (attempt {beliefs.attempt_count + 1}, recent scores: {scores}) — "
                "root cause: you keep returning the JSON SCHEMA definition (keys like "
                "'title'/'type'/'properties'/'required') instead of DATA. Temperature will not fix this — "
                "the structure itself must change. Output ONLY the actual values, keyed by the schema's "
                "property NAMES directly (e.g. {\"BasicInfo\": {\"Model\": \"...\"}}), with no 'type', "
                "'properties', 'title', 'required' or 'value' wrapper keys anywhere in your answer.",
                0.7)
        if wrapper:
            return Step(self.name,
                f"PLATEAU DETECTED (attempt {beliefs.attempt_count + 1}, recent scores: {scores}) — "
                f"root cause: your JSON is wrapped inside a '{wrapper}' key that should not be there. "
                "Temperature will not fix this — remove the wrapper and start your object directly "
                "with the schema's required top-level keys.", 0.7)
        return Step(self.name,
            f"PLATEAU DETECTED (attempt {beliefs.attempt_count + 1}, recent scores: {scores}). "
            "Your attempts produce similar results. "
            "Take a completely different approach: re-read the schema from scratch, "
            "identify the fields you keep getting wrong, and extract them character-by-character "
            f"from the text. Best U_V so far: {beliefs.best_u_v():.2f} — push beyond.", 0.7)


class RestartFresh(Action):
    """Repart de zéro (utilisé quand les scores se dégradent)."""
    name = "restart_fresh"

    def build(self, beliefs, **ctx):
        scores = " -> ".join(f"{u:.2f}" for u in beliefs.past_u_v[-3:])
        if _is_schema_as_data(beliefs):
            return Step(self.name,
                f"The scores are degrading ({scores}, attempt {beliefs.attempt_count + 1}) — "
                "you are returning the JSON SCHEMA definition instead of DATA. Start fresh: "
                "output ONLY the actual values keyed by the schema's property NAMES directly, "
                "with no 'type'/'properties'/'title'/'required'/'value' keys anywhere.", 0.5)
        return Step(self.name,
            f"The scores are degrading ({scores}, attempt {beliefs.attempt_count + 1}). "
            "Completely change your approach: "
            "start fresh by re-reading the schema top-to-bottom, "
            "then extract each field from the text one by one.", 0.5)


class StrictSchema(Action):
    """Extraction stricte champ par champ selon le schéma."""
    name = "strict_schema"

    def build(self, beliefs, **ctx):
        return Step(self.name,
            f"Focus exclusively on matching the JSON schema (attempt {beliefs.attempt_count + 1}). "
            "For EACH property in the schema: "
            "1) Find its value in the text, 2) Convert to the correct type "
            "(string, number, boolean, array), 3) Place it at the exact path defined by the schema. "
            "Do not add extra fields not in the schema.", 0.1)


class PrimeStructuralGuidance(Action):
    """
    Amorce prudente pour une catégorie jugée historiquement difficile
    (croyance WorldModel, cf. category_high_risk) : injecte le squelette du
    schéma dès le PREMIER essai, sans attendre un échec pour diagnostiquer —
    contrairement à FillSkeleton qui exige un échec structurel déjà observé.
    """
    name = "prime_structural_guidance"

    def build(self, beliefs, **ctx):
        skeleton = _build_skeleton(beliefs.schema)
        if not skeleton:
            return None
        return Step(self.name,
            "This category of task has historically been difficult to get right on the "
            "first try. To maximize your chances, follow this skeleton structure precisely "
            "and fill in REAL values extracted from the text — do not copy null:\n\n" + skeleton,
            0.1)


class DeepFieldDiagnosis(Action):
    """
    Aucun plan spécifique n'a reconnu le type d'échec sur plusieurs tentatives
    consécutives (2+ fallback de suite) : force une vérification champ par
    champ explicite au lieu du hint générique de Fallback.
    """
    name = "deep_field_diagnosis"

    def build(self, beliefs, **ctx):
        # deep_diagnosis (priorité 60) domine break_plateau/nash_compromise (50/55)
        # dans le pool des désirs préemptifs dès que consecutive_fallback>=2 — donc
        # dès que ce seuil est atteint, c'est LUI qui reçoit le budget de tentatives
        # restant, pas break_plateau. Il doit donc porter le diagnostic structurel
        # spécifique en premier, sinon la correction ciblée (2026-08-04,
        # BreakPlateau/RestartFresh) n'est jamais atteinte pour les tâches qui
        # passent par ce chemin. Vérifié sur deepjsoneval_31/33_viecle (job
        # #923113) : bloquées en boucle deep_diagnosis/fallback, U_V=0.4625
        # identique sur 6+ tentatives, sans jamais repasser par la correction
        # schema-as-data.
        if _is_schema_as_data(beliefs):
            return Step(self.name,
                f"No previous strategy has worked so far (attempt {beliefs.attempt_count + 1}). "
                "You keep returning the JSON SCHEMA definition (keys like 'title'/'type'/"
                "'properties'/'required') instead of DATA. Output ONLY the actual values, keyed "
                "by the schema's property NAMES directly (e.g. {\"BasicInfo\": {\"Model\": \"...\"}}), "
                "with no 'type', 'properties', 'title', 'required' or 'value' wrapper keys anywhere.",
                0.3)
        wrapper = _detect_wrapper(beliefs)
        if wrapper:
            return Step(self.name,
                f"No previous strategy has worked so far (attempt {beliefs.attempt_count + 1}). "
                f"Your JSON is wrapped inside a '{wrapper}' key that should not be there — remove "
                "the wrapper and start your object directly with the schema's required top-level keys.",
                0.3)
        # V2.1 #4 : préférer les violations RÉELLES remontées par le vérificateur
        # (jsonschema, chemin+message précis) à la liste générique de propriétés.
        # Numéro de tentative ajouté (2026-08-04) : sans ça, ce hint devient
        # "déjà essayé" dès que les violations ne varient plus d'une tentative à
        # l'autre (cas typique d'un blocage stable), et ne peut plus refire.
        violations = getattr(beliefs, "last_schema_violations", None)
        if violations:
            detail = "; ".join(
                f'{v.get("path", "?")}: {v.get("message", "")}' for v in violations[:10]
            )
            return Step(self.name,
                f"No previous strategy has worked so far (attempt {beliefs.attempt_count + 1}). "
                f"The verifier reported these SPECIFIC schema violations from your last attempt — "
                f"fix each one explicitly: {detail}. Do not skip this reasoning.", 0.3)
        # Repli : aucune violation connue (ex. premier passage par fallback) —
        # comportement historique inchangé (+ numéro de tentative).
        try:
            schema_obj = _json.loads(beliefs.schema) if beliefs.schema else {}
            props = list(schema_obj.get("properties", {}).keys())[:15]
            field_list = ", ".join(f'"{k}"' for k in props) if props else "(see schema)"
        except Exception:
            field_list = "(see schema)"
        return Step(self.name,
            f"No previous strategy has worked so far (attempt {beliefs.attempt_count + 1}). "
            "Perform a SYSTEMATIC field-by-field "
            "check: for each field below, explicitly locate its value in the text (or note "
            f"it is absent) before writing the final JSON. Fields to verify one by one: "
            f"{field_list}. Do not skip this reasoning.", 0.3)


class ReconcileConflict(Action):
    """
    Recadre l'exécuteur sur les critères MÉCANIQUES du vérificateur, quand
    les deux sont en désaccord persistant : l'exécuteur produit un JSON qu'il
    juge complet (U_PE haut) mais le vérificateur le rejette plusieurs fois
    de suite (U_V bas). Contrairement à ExtractAllFields/CopyExactValues
    (qui supposent juste "pas assez précis"), ici le problème identifié est
    un décalage de CRITÈRE entre les deux agents — l'exécuteur doit arrêter
    de se fier à son propre jugement de complétude et appliquer explicitement
    la grille du vérificateur (structure exacte, valeurs exactes, profondeur
    exacte), champ par champ.
    """
    name = "reconcile_conflict"

    def build(self, beliefs, **ctx):
        return Step(self.name,
            f"Your own output looked complete to you (self-assessed U_PE={beliefs.last_u_pe:.2f}), "
            f"but the automatic verifier has rejected it {beliefs.consecutive_conflict} times in a "
            f"row (verifier score U_V={beliefs.last_u_v:.2f}). Stop relying on your own impression "
            f"of completeness. The verifier checks MECHANICALLY: exact schema structure, exact field "
            f"values copied character-for-character from the text, exact nesting depth. Go through "
            f"the schema field by field and verify each value against the text — do not paraphrase, "
            f"round, or infer values that are not explicitly present.", 0.2)


class Fallback(Action):
    """Dernier recours — hint variable selon la tentative (jamais 'déjà essayé')."""
    name = "fallback"

    def build(self, beliefs, **ctx):
        # Même diagnostic ciblé qu'ailleurs (2026-08-04) : fallback est le SEUL
        # plan qui ne s'épuise jamais (hint toujours frais via le numéro de
        # tentative), donc c'est souvent lui qui reçoit la majorité du budget
        # restant une fois les autres plans "déjà essayés" — sans ce rappel, le
        # diagnostic correct se perd derrière un message générique pour le
        # reste des tentatives.
        if _is_schema_as_data(beliefs):
            return Step(self.name,
                f"Attempt {beliefs.attempt_count + 1} — you keep returning the JSON SCHEMA "
                "definition (keys like 'title'/'type'/'properties'/'required') instead of DATA. "
                "Output ONLY the actual values, keyed by the schema's property NAMES directly, "
                "with no 'type', 'properties', 'title', 'required' or 'value' wrapper keys anywhere. "
                f"Best U_V so far: {beliefs.best_u_v():.2f}.", 0.1)
        parts = [f"Attempt {beliefs.attempt_count + 1} — try a different approach."]
        if not beliefs.last_is_valid_json:
            parts.append("Return ONLY raw JSON (no markdown, no text).")
        parts.append(f"Best U_V so far: {beliefs.best_u_v():.2f}. "
                     f"Category: {beliefs.category}. Expected depth: {beliefs.true_depth}.")
        return Step(self.name, " ".join(parts), 0.1)


# ═══════════════════════════════════════════════════════════
#  PLAN = trigger + context + body (séquence d'actions)
# ═══════════════════════════════════════════════════════════

@dataclass
class Plan:
    name:           str
    description:    str
    trigger_desire: str
    priority:       int
    body:           List[Action] = field(default_factory=list)
    _context:       Optional[Callable] = None
    allows_empty_hint: bool = False

    def context_condition(self, beliefs, **ctx) -> bool:
        return self._context(beliefs, **ctx) if self._context else True

    def has_executable_step(self, beliefs, **ctx) -> bool:
        """True si au moins une action du body produit un hint encore inédit."""
        for action in self.body:
            step = action.build(beliefs, **ctx)
            if step is None:
                continue
            if step.hint and step.hint in beliefs.tried_hints:
                continue
            return True
        return False


# ═══════════════════════════════════════════════════════════
#  BIBLIOTHÈQUE
# ═══════════════════════════════════════════════════════════

class PlanLibrary:
    """
    Bibliothèque de PLANS BDI (avec corps d'actions).
    select_plan() = filter() du cycle BDI : renvoie le plan de plus haute
    priorité applicable et disposant d'au moins une étape exécutable.
    """

    def __init__(self):
        self._plans: List[Plan] = self._build_library()

    def _build_library(self) -> List[Plan]:
        plans = [
            Plan("warm_start", "Réutilise une stratégie apprise",
                 "generate_valid_json", 90,
                 body=[WarmStart()],
                 _context=lambda b, warm_hint="", **c: b.attempt_count == 0 and bool(warm_hint),
                 allows_empty_hint=True),

            Plan("first_attempt", "1er essai — LLM libre (= baseline)",
                 "generate_valid_json", 85,
                 body=[GenerateFree()],
                 _context=lambda b, warm_hint="", **c: b.attempt_count == 0 and not warm_hint,
                 allows_empty_hint=True),

            # Recette de réparation SYNTAXIQUE : réparer le JSON puis rappeler le schéma.
            Plan("repair_syntax", "JSON invalide → réparer syntaxe puis schéma",
                 "generate_valid_json", 80,
                 body=[RepairInvalidJSON(), EnforceSchema()],
                 _context=lambda b, **c: b.attempt_count >= 1 and not b.last_is_valid_json),

            # Recette de réparation STRUCTURELLE : retirer wrapper → recadrer
            # schema-as-data → injecter squelette → forcer le schéma. Les étapes
            # non pertinentes (build()→None) sont sautées automatiquement.
            Plan("repair_structure", "Structure fausse → wrapper/skeleton/schéma",
                 "generate_valid_json", 75,
                 body=[StripWrapper(), FixSchemaAsData(), FillSkeleton(), EnforceSchema()],
                 _context=lambda b, **c: (b.attempt_count >= 1 and b.last_is_valid_json
                                          and b.last_schema_compliance < 0.7)),

            # Recette d'AMÉLIORATION DE PRÉCISION : cibler les champs échoués →
            # extraire tout → recopier les valeurs exactes.
            Plan("improve_accuracy", "Champs faux → ciblage/extraction/copie exacte",
                 "maximize_accuracy", 70,
                 body=[FixFailedFields(), ExtractAllFields(), CopyExactValues()],
                 _context=lambda b, **c: (b.attempt_count >= 1 and b.last_is_valid_json
                                          and b.last_schema_compliance >= 0.5
                                          and b.last_field_accuracy < 0.8)),

            Plan("fix_depth", "Profondeur incorrecte → ajuster puis rappeler schéma",
                 "match_depth", 65,
                 body=[MatchDepth(), EnforceSchema()],
                 _context=lambda b, **c: (b.attempt_count >= 1 and b.last_is_valid_json
                                          and not b.last_depth_match and b.true_depth > 0)),

            # Recette de DÉBLOCAGE : température ↑ / approche neuve, puis restart.
            # trigger_desire="escape_plateau" — désir DYNAMIQUE (BDIState.
            # sync_dynamic_desires), pas l'un des 3 désirs statiques. Sans ça,
            # ce plan restait coincé derrière repair_syntax/repair_structure
            # (priorités 80/75) qui continuent à re-matcher sans progresser —
            # exactement le symptôme du plateau qu'il est censé résoudre.
            Plan("break_plateau", "Scores stagnants → escalade + restart",
                 "escape_plateau", 50,
                 body=[BreakPlateau(), RestartFresh()],
                 _context=lambda b, **c: (b.attempt_count >= 1 and b.last_u_v < 0.7
                                          and (b.is_plateau(n=2, epsilon=0.05)
                                               or b.trend() == "DEGRADING"))),

            Plan("strict_schema", "Blocage persistant → extraction stricte",
                 "generate_valid_json", 35,
                 body=[StrictSchema(), EnforceSchema()],
                 _context=lambda b, **c: b.attempt_count >= 2 and b.last_u_v < 0.5),

            # trigger_desire="final_attempt" — désir dynamique PRÉEMPTIF.
            # Priorité 95 : au-dessus de TOUT le reste, y compris les plans
            # structurels (75-80). Sur la dernière tentative avant épuisement
            # du budget, on ne joue plus l'exploration (température ↑, restart)
            # — on choisit les actions les plus sûres connues pour aider sans
            # jamais faire empirer. Motivé par deepjsoneval_107_game : raté le
            # seuil de succès de 0.0005 point à sa dernière tentative.
            # Condition explicite redondante avec le désir (même défense que
            # break_plateau) : si select_plan() était jamais appelé sans
            # gating (active_desires=None), ce plan à priorité 95 ne doit PAS
            # dominer aveuglément — il ne s'applique QUE sur la dernière
            # tentative réelle.
            #
            # RepairInvalidJSON() en tête de corps (V2.1, 2026-07-11) : sur un
            # run 200 tâches/10 shots, safe_finish s'est déclenché 15 fois et a
            # terminé invalide 13 fois (87%) — dont 11/15 (73%) où le JSON était
            # DÉJÀ invalide en ENTRANT dans cette dernière tentative (le corps
            # d'origine — fix_failed_fields/copy_exact_values/enforce_schema —
            # ne répare aucun JSON invalide, donc ne pouvait jamais s'en sortir).
            # RepairInvalidJSON.build() renvoie déjà None si le JSON est valide
            # (skip automatique via _emit_next_step) — aucun effet sur les cas
            # où safe_finish fonctionnait déjà (ex. deepjsoneval_69_movie,
            # deepjsoneval_167_plant, run isolé 2026-07-11 18h10).
            Plan("safe_finish", "Dernière tentative → réparer si invalide, puis action la plus sûre",
                 "final_attempt", 95,
                 body=[RepairInvalidJSON(), FixFailedFields(), CopyExactValues(), EnforceSchema()],
                 _context=lambda b, max_replans=None, **c: (
                     max_replans is not None and b.attempt_count >= 1
                     and b.attempt_count == max_replans - 1)),

            # trigger_desire="category_high_risk" — désir dynamique ORDINAIRE
            # (pas préemptif : rejoint le scan normal à sa propre priorité,
            # entre warm_start=90 et first_attempt=85 — une stratégie apprise
            # fiable doit rester prioritaire sur cette prudence par défaut).
            Plan("cautious_start", "Catégorie historiquement difficile → amorce structurée",
                 "category_high_risk", 87,
                 body=[PrimeStructuralGuidance()],
                 _context=lambda b, **c: b.attempt_count == 0),

            # trigger_desire="diagnose_unknown_failure" — désir dynamique
            # PRÉEMPTIF. Priorité 60 : au-dessus de break_plateau (50), car un
            # diagnostic structuré vaut mieux qu'une pure escalade de
            # température quand même AUCUN plan spécifique n'a reconnu le
            # problème sur 2+ tentatives (uniquement fallback a matché).
            # Idem : condition explicite redondante avec le désir.
            Plan("deep_diagnosis", "Échec non reconnu (2+ fallback) → diagnostic champ par champ",
                 "diagnose_unknown_failure", 60,
                 body=[DeepFieldDiagnosis(), EnforceSchema()],
                 _context=lambda b, **c: b.consecutive_fallback >= 2),

            # trigger_desire="resolve_conflict" — désir dynamique PRÉEMPTIF.
            # Priorité 55 : entre deep_diagnosis (60, "aucun plan n'a reconnu
            # le problème") et break_plateau (50, "le score stagne") — ici le
            # signal est différent des deux : pas un plateau de score, pas un
            # échec non identifié, mais un DÉSACCORD structurel entre les
            # deux agents (l'exécuteur croit réussir, le vérificateur rejette
            # systématiquement) — cf. GT conflict rate dans gt_analyzer.py,
            # ici détecté et exploité EN DIRECT pendant la délibération au
            # lieu d'une mesure post-hoc.
            Plan("nash_compromise", "Conflit PE/V persistant → recadrage sur critères mécaniques",
                 "resolve_conflict", 55,
                 body=[ReconcileConflict(), EnforceSchema()],
                 _context=lambda b, **c: b.consecutive_conflict >= 2),

            Plan("fallback", "Dernier recours",
                 "generate_valid_json", 0,
                 body=[Fallback()]),
        ]
        return sorted(plans, key=lambda p: p.priority, reverse=True)

    def _plan_eligible(self, plan, beliefs, executor_reliability_fn=None, **ctx) -> bool:
        """Contexte vrai + étape exécutable + pas jugé peu fiable pour la catégorie."""
        if not plan.context_condition(beliefs, **ctx):
            return False
        if not plan.has_executable_step(beliefs, **ctx):
            return False
        if executor_reliability_fn and plan.priority > 0 and beliefs.category:
            reliability = executor_reliability_fn(beliefs.category, plan.name)
            if reliability < 0.1:
                print(f"  [BDI] Skipping plan '{plan.name}' — executor believed unreliable "
                      f"for '{beliefs.category}' (success rate={reliability:.2f})")
                return False
        return True

    def select_plan(self, beliefs, active_desires=None, dynamic_desires=None,
                     executor_reliability_fn=None, **ctx) -> Plan:
        """
        filter() du cycle BDI — renvoie le PLAN de plus haute priorité dont
        (1) le désir déclencheur est actif, (2) le contexte est vrai, et
        (3) qui possède au moins une étape exécutable (hint inédit).

        `active_desires` (ensemble de noms — cf. BDIState.options()) est le
        filtre qui rend les Desires réellement décisionnels et non de simples
        étiquettes : un plan dont le désir a été suspendu ou atteint est
        écarté quelle que soit sa priorité. Exemple concret — "maximize_accuracy"
        est suspendu tant que le JSON reste invalide (cf. BeliefBase.brf) :
        les plans de raffinement des champs (improve_accuracy) sont donc
        écartés tant que la structure de base n'est pas réparée, même si leur
        contexte serait par ailleurs satisfait.

        On garde un tri par PRIORITÉ GLOBALE (pas par groupe de désir) : grouper
        d'abord par désir puis par priorité ferait qu'un plan toujours
        applicable du désir prioritaire (ex. 'fallback', désir generate_valid_json)
        préempterait systématiquement les désirs de priorité inférieure — ça
        rendrait improve_accuracy/fix_depth quasiment inatteignables. Le filtre
        desire agit donc comme une EXCLUSION ponctuelle, pas comme une clé de tri.

        `dynamic_desires` (cf. BDIState.sync_dynamic_desires) est un signal
        URGENT — pas une simple exclusion mais une PRÉEMPTION : si un plan
        servant un désir dynamiquement actif (ex. escape_plateau) est éligible,
        il est retenu avant même de considérer le scan de priorité statique.
        Sans ça, un plan comme break_plateau (priorité 50) resterait coincé
        derrière des plans structurels (75-80) qui continuent de re-matcher
        sans jamais progresser — exactement le symptôme qu'il doit résoudre.

        `executor_reliability_fn(category, plan) → float` porte une croyance
        sur l'exécuteur : un plan jugé peu fiable pour la catégorie (<0.1) est
        écarté au profit du suivant.
        """
        if dynamic_desires:
            for plan in self._plans:
                if plan.trigger_desire not in dynamic_desires:
                    continue
                if self._plan_eligible(plan, beliefs, executor_reliability_fn, **ctx):
                    return plan

        for plan in self._plans:
            if active_desires is not None and plan.trigger_desire not in active_desires:
                continue
            if self._plan_eligible(plan, beliefs, executor_reliability_fn, **ctx):
                return plan
        return self._plans[-1]  # fallback (contexte toujours vrai)

    def get_plan(self, name: str) -> Optional[Plan]:
        for p in self._plans:
            if p.name == name:
                return p
        return None

    def __repr__(self) -> str:
        return ("PlanLibrary(" + ", ".join(
            f"{p.name}[{'→'.join(a.name for a in p.body)}]" for p in self._plans) + ")")
