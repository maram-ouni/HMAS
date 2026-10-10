# ─────────────────────────────────────────────────────────
# payoff_calculator.py — PAYOFFS POUR DeepJSONEval
#
# Adapté de agisdk/mas_project pour la génération de JSON.
#
# Agent PE (PlannerExecutor) — veut générer du JSON valide rapidement :
#   U_PE = 0.60 * dim_json_valid
#          + 0.25 * dim_schema_compliance
#          + 0.15 * dim_completeness
#
# Agent V (Verifier) — veut de la précision et la bonne structure :
#   U_V  = 0.40 * dim_json_valid
#          + 0.35 * dim_field_accuracy
#          + 0.25 * dim_depth_match
#
# Tension GT :
#   PE peut générer du JSON valide mais structurellement approximatif
#   → U_PE élevé mais U_V faible → V rejette
#   PE doit apprendre à produire un JSON précis ET structuré
# ─────────────────────────────────────────────────────────

import json
import re
from typing import Any, Dict, Tuple

# Seuils du Vérificateur
V_THRESHOLD         = 0.7   # U_V < 0.7       -> REJECTED
V_SUCCESS_THRESHOLD = 0.7   # U_V >= 0.7      -> SUCCESS

# Seuils du PlannerExecutor
PE_THRESHOLD         = 0.4
PE_SUCCESS_THRESHOLD = 0.7

# Bande diagnostique secondaire (V2.1, changement #3) — NE remplace ni ne
# nuance V_THRESHOLD/V_SUCCESS_THRESHOLD, seule source de vérité pour
# quality/accepted. Sert uniquement à qualifier un échec en "fair" (proche du
# seuil) vs "poor" (loin du seuil) pour des heuristiques internes du planner
# (cf. BeliefBase.last_diagnostic_quality, BDIState.sync_dynamic_desires).
DIAGNOSTIC_QUALITY_THRESHOLD = 0.5

# Poids
WEIGHTS_PE = {"json_valid": 0.60, "schema_compliance": 0.25, "completeness":    0.15}
WEIGHTS_V  = {"json_valid": 0.40, "field_accuracy":    0.35, "depth_match":     0.25}


# ─────────────────────────────────────────────────────────
# DIMENSIONS OBJECTIVES
# ─────────────────────────────────────────────────────────

def dim_json_valid(generated_json: str) -> Tuple[float, Any]:
    """1.0 si le JSON est parsable, 0.0 sinon. Retourne (score, parsed_obj)."""
    if not generated_json or not generated_json.strip():
        return 0.0, None
    # Nettoyer les blocs markdown éventuels
    cleaned = _strip_markdown(generated_json)
    try:
        parsed = json.loads(cleaned)
        return 1.0, parsed
    except (json.JSONDecodeError, ValueError):
        return 0.0, None


def dim_schema_compliance(parsed_obj: Any, schema_str: str) -> float:
    """
    Mesure la conformité structurelle au schéma.
    Utilise jsonschema si disponible, sinon vérifie les clés requises.
    """
    if parsed_obj is None or not schema_str:
        return 0.0
    try:
        schema = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
    except (json.JSONDecodeError, TypeError):
        return 0.0

    # Essayer jsonschema d'abord
    try:
        import jsonschema
        jsonschema.validate(instance=parsed_obj, schema=schema)
        return 1.0
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback : vérification récursive des clés
    return _check_keys_recursive(parsed_obj, schema)


def _check_keys_recursive(obj: Any, schema: dict) -> float:
    """Vérifie la présence des propriétés requises dans obj."""
    if not isinstance(schema, dict):
        return 1.0
    schema_type = schema.get("type", "")
    properties  = schema.get("properties", {})
    required    = schema.get("required", list(properties.keys()))

    if not required:
        return 1.0

    if not isinstance(obj, dict):
        return 0.0

    present = sum(1 for k in required if k in obj)
    base_score = present / len(required)

    # Récursion sur les sous-objets
    nested_scores = []
    for key, sub_schema in properties.items():
        if key in obj and isinstance(sub_schema, dict) and "properties" in sub_schema:
            nested_scores.append(_check_keys_recursive(obj[key], sub_schema))

    if nested_scores:
        return (base_score + sum(nested_scores) / len(nested_scores)) / 2
    return base_score


def dim_field_accuracy(parsed_obj: Any, ground_truth_str: str) -> float:
    """
    F1 score sur les feuilles (leaf values) du JSON généré vs ground truth.
    """
    if parsed_obj is None or not ground_truth_str:
        return 0.0
    try:
        gt = json.loads(ground_truth_str) if isinstance(ground_truth_str, str) else ground_truth_str
    except (json.JSONDecodeError, TypeError):
        return 0.0

    gen_leaves = _extract_leaves(parsed_obj)
    gt_leaves  = _extract_leaves(gt)

    if not gt_leaves:
        return 1.0

    # Exact match sur les valeurs (normalisées)
    matches = sum(
        1 for k, v in gt_leaves.items()
        if gen_leaves.get(k) == v
    )

    precision = matches / len(gen_leaves) if gen_leaves else 0.0
    recall    = matches / len(gt_leaves)

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def dim_completeness(parsed_obj: Any, ground_truth_str: str) -> float:
    """Recall seul : fraction des champs du ground truth présents dans le généré."""
    if parsed_obj is None or not ground_truth_str:
        return 0.0
    try:
        gt = json.loads(ground_truth_str) if isinstance(ground_truth_str, str) else ground_truth_str
    except (json.JSONDecodeError, TypeError):
        return 0.0

    gen_leaves = _extract_leaves(parsed_obj)
    gt_leaves  = _extract_leaves(gt)

    if not gt_leaves:
        return 1.0

    present = sum(1 for k in gt_leaves if k in gen_leaves)
    return round(present / len(gt_leaves), 4)


def dim_depth_match(parsed_obj: Any, true_depth: int) -> Tuple[float, int]:
    """
    1.0 si la profondeur calculée == true_depth, 0.0 sinon.
    Retourne (score, actual_depth).
    """
    if parsed_obj is None:
        return 0.0, 0
    actual = _compute_depth(parsed_obj)
    score  = 1.0 if actual == true_depth else max(0.0, 1.0 - abs(actual - true_depth) * 0.25)
    return round(score, 4), actual


# ─────────────────────────────────────────────────────────
# UTILITIES INDIVIDUELLES
# ─────────────────────────────────────────────────────────

def compute_U_PE(generated_json: str, schema_str: str,
                 ground_truth_str: str, true_depth: int,
                 weights: dict = None) -> dict:
    """
    Utilité du PlannerExecutor.
    U_PE = 0.60*json_valid + 0.25*schema_compliance + 0.15*completeness

    `weights` (opt-in) permet de surcharger WEIGHTS_PE — utilisé par
    config.LEARNED_PE_WEIGHTS. None (défaut) = comportement inchangé.
    """
    valid_score, parsed = dim_json_valid(generated_json)
    schema_score        = dim_schema_compliance(parsed, schema_str) if parsed is not None else 0.0
    complete_score      = dim_completeness(parsed, ground_truth_str) if parsed is not None else 0.0

    w = weights or WEIGHTS_PE
    u = (w["json_valid"]         * valid_score
         + w["schema_compliance"] * schema_score
         + w["completeness"]      * complete_score)

    return {
        "payoff":               round(u, 4),
        "agent_role":           "planner_executor",
        "dim_json_valid":       valid_score,
        "dim_schema_compliance": schema_score,
        "dim_completeness":     complete_score,
        "parsed_obj":           parsed,
        "weights_used":         w,
        "breakdown": (
            f"U_PE = {w['json_valid']}x{valid_score:.2f} + "
            f"{w['schema_compliance']}x{schema_score:.2f} + "
            f"{w['completeness']}x{complete_score:.2f} = {u:.4f}"
        ),
    }


def get_failed_fields(parsed_obj: Any, ground_truth_str: str) -> list:
    """Retourne les chemins de feuilles du ground truth absents ou incorrects (max 20)."""
    if parsed_obj is None or not ground_truth_str:
        return []
    try:
        gt = json.loads(ground_truth_str) if isinstance(ground_truth_str, str) else ground_truth_str
    except (json.JSONDecodeError, TypeError):
        return []
    gen_leaves = _extract_leaves(parsed_obj)
    gt_leaves  = _extract_leaves(gt)
    return [k for k, v in gt_leaves.items() if gen_leaves.get(k) != v][:20]


def get_schema_violations(parsed_obj: Any, schema_str: str, max_violations: int = 20) -> list:
    """
    Rapport STRUCTURÉ des violations de schéma (chemin + règle + message),
    complémentaire à dim_schema_compliance (scalaire). Purement diagnostique —
    n'affecte aucun score ni seuil d'acceptation (V2.1, changement #4).
    """
    if parsed_obj is None or not schema_str:
        return []
    try:
        schema = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
    except (json.JSONDecodeError, TypeError):
        return []
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(schema)
        violations = []
        for err in validator.iter_errors(parsed_obj):
            violations.append({
                "path":      "/".join(str(p) for p in err.absolute_path) or "(root)",
                "message":   err.message,
                "validator": str(err.validator),
            })
            if len(violations) >= max_violations:
                break
        return violations
    except ImportError:
        return []
    except Exception:
        return []


def compute_U_V(generated_json: str, ground_truth_str: str,
                true_depth: int, parsed_obj: Any = None,
                weights: dict = None) -> dict:
    """
    Utilité du Vérificateur.
    U_V = 0.40*json_valid + 0.35*field_accuracy + 0.25*depth_match

    `weights` (opt-in) permet de surcharger WEIGHTS_V — utilisé par
    config.LEARNED_VERIFIER_WEIGHTS. None (défaut) = comportement inchangé.
    """
    if parsed_obj is None:
        valid_score, parsed_obj = dim_json_valid(generated_json)
    else:
        valid_score = 1.0

    accuracy_score               = dim_field_accuracy(parsed_obj, ground_truth_str)
    depth_score, actual_depth    = dim_depth_match(parsed_obj, true_depth)
    failed_fields                = get_failed_fields(parsed_obj, ground_truth_str)

    w = weights or WEIGHTS_V
    u = (w["json_valid"]       * valid_score
         + w["field_accuracy"] * accuracy_score
         + w["depth_match"]    * depth_score)

    return {
        "payoff":           round(u, 4),
        "agent_role":       "verifier",
        "dim_json_valid":   valid_score,
        "dim_field_accuracy": accuracy_score,
        "dim_depth_match":  depth_score,
        "actual_depth":     actual_depth,
        "failed_fields":    failed_fields,
        "weights_used":     w,
        "breakdown": (
            f"U_V = {w['json_valid']}x{valid_score:.2f} + "
            f"{w['field_accuracy']}x{accuracy_score:.2f} + "
            f"{w['depth_match']}x{depth_score:.2f} = {u:.4f}"
        ),
    }


def compute_payoffs(generated_json: str, schema_str: str,
                    ground_truth_str: str, true_depth: int,
                    verifier_weights: dict = None,
                    planner_weights: dict = None) -> dict:
    """Calcule U_PE et U_V en une seule passe. Point d'entrée principal.

    `verifier_weights`/`planner_weights` (opt-in) surchargent WEIGHTS_V/
    WEIGHTS_PE — voir compute_U_V/compute_U_PE. None (défaut) = inchangé.
    """
    u_pe = compute_U_PE(generated_json, schema_str, ground_truth_str, true_depth,
                        weights=planner_weights)
    u_v  = compute_U_V(generated_json, ground_truth_str, true_depth,
                       parsed_obj=u_pe["parsed_obj"], weights=verifier_weights)
    return {
        "U_PE":      u_pe,
        "U_V":       u_v,
        "payoff":    u_pe["payoff"],
        "U_planner": u_pe["payoff"],
        "U_verifier": u_v["payoff"],
    }


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Supprime les blocs markdown ```json ... ``` du texte."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Retirer la première ligne (```json ou ```) et la dernière (```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_leaves(obj: Any, prefix: str = "") -> dict:
    """Extrait tous les couples (chemin, valeur) des feuilles d'un JSON."""
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                result.update(_extract_leaves(v, path))
            else:
                result[path] = _normalize_value(v)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            path = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                result.update(_extract_leaves(item, path))
            else:
                result[path] = _normalize_value(item)
    return result


def _normalize_value(v: Any) -> str:
    """Normalise une valeur feuille pour comparaison."""
    if v is None:
        return "null"
    return str(v).strip().lower()


def _compute_depth(obj: Any, current: int = 0) -> int:
    """Profondeur maximale d'imbrication d'un objet JSON."""
    if isinstance(obj, dict):
        if not obj:
            return current
        return max(_compute_depth(v, current + 1) for v in obj.values())
    if isinstance(obj, list):
        if not obj:
            return current
        return max(_compute_depth(item, current) for item in obj)
    return current
