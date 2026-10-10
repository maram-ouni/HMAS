"""
hierarchical_bdi_baseline.py — Architecture Hierarchical BDI pour DeepJSONEval
Réf : Harada & Kano 2025

Principe : BDI à deux niveaux.
  Macro-BDI : un appel LLM par shot génère une directive stratégique de haut niveau
              (quel type de structure, quels défis d'extraction, quelle approche).
  Micro-BDI : la directive est injectée dans CHAQUE exécution (pas seulement step 0),
              incitant l'exécuteur à réévaluer l'alignement avant chaque action.
  Paramètres de personnalité : assertiveness=0.75, thoroughness=0.50.

Usage:
    python mas_project/experiments/hierarchical_bdi_baseline.py --max-samples 200 --shots 5
"""
from __future__ import annotations

import argparse, json, os, pathlib, sys, time

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_MAS_DIR      = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _MAS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from openai import OpenAI
from config import (LLM_BACKEND, LLM_API_KEY, LLM_BASE_URL,
                    LLM_MODEL, LLM_MODEL_FALLBACK, RESULTS_DIR)
from core.payoff_calculator import compute_payoffs
from data.dataset_loader import load_dataset_hf, iter_tasks

ARCH_NAME         = "hierarchical_bdi"
SUCCESS_THRESHOLD = 0.7
MAX_CTX           = 4096
ASSERTIVENESS     = 0.75   # biais vers les décisions fermes
THOROUGHNESS      = 0.50   # profondeur d'analyse

_SYS_JSON = """You are a JSON generation expert.
Your task is to extract information from a text and format it as a valid JSON object.

RULES:
1. Return ONLY a raw JSON object — no markdown, no code blocks, no explanations.
2. Your response must start with '{' and end with '}'.
3. Follow the provided JSON schema EXACTLY — use the same key names and nesting structure.
4. Extract values directly from the text. Do not invent values.
5. Use null for fields not mentioned in the text.
6. Preserve the nesting depth specified in the schema."""

_SYS_MACRO = (
    "You are a JSON extraction strategist operating at the macro-BDI level. "
    "Generate a 2-3 sentence strategic directive that specifies: "
    "WHERE (which part of the schema to focus on), "
    "WHAT (which extraction challenges to address), "
    "WHEN (at what point to apply corrections). "
    f"Be assertive (assertiveness={ASSERTIVENESS}) and thorough (thoroughness={THOROUGHNESS})."
)


def _client() -> OpenAI:
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _call_free(client: OpenAI, system: str, user: str,
               temperature: float = 0.4, max_tokens: int = 120) -> str:
    for model in [LLM_MODEL, LLM_MODEL_FALLBACK]:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
                temperature=temperature, max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [HierBDI/macro] {model}: {e}")
    return ""


def _call_json(client: OpenAI, schema: str, user: str, temperature: float = 0.1) -> str:
    extra: dict = {}
    if LLM_BACKEND == "vllm":
        try:
            extra["extra_body"] = {"guided_json": json.loads(schema)}
        except json.JSONDecodeError:
            pass
    else:
        extra["response_format"] = {"type": "json_object"}

    max_tokens = 1024
    for model in [LLM_MODEL, LLM_MODEL_FALLBACK]:
        for attempt in range(2):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": _SYS_JSON},
                              {"role": "user",   "content": user}],
                    temperature=temperature, max_tokens=max_tokens, **extra,
                )
                return r.choices[0].message.content.strip()
            except Exception as e:
                import re
                m = re.search(r"prompt contains (\d+) input tokens", str(e))
                if m and attempt == 0:
                    max_tokens = max(128, MAX_CTX - int(m.group(1)) - 64)
                    continue
                print(f"  [HierBDI/json] {model}: {e}")
                break
    return ""


def _macro_prompt_shot1(schema: str, category: str, true_depth: int) -> str:
    # Extrait le type de structure du schéma pour le Macro-BDI
    try:
        s = json.loads(schema)
        n_props = len(s.get("properties", {}))
        s_type  = s.get("type", "object")
        title   = s.get("title", category)
    except Exception:
        n_props, s_type, title = "?", "object", category
    return (f"CATEGORY: {category} | SCHEMA TITLE: {title} | DEPTH: {true_depth}\n"
            f"SCHEMA TYPE: {s_type} with ~{n_props} top-level properties\n\n"
            "Generate a strategic directive for accurate JSON extraction:")


def _macro_prompt_retry(schema: str, category: str, true_depth: int,
                         prev_outcome: str, shot: int) -> str:
    base = _macro_prompt_shot1(schema, category, true_depth)
    return (base + f"\n\nPREVIOUS SHOT {shot-1} OUTCOME: {prev_outcome}\n"
            "Generate an improved strategic directive:")


def _execute_prompt(schema: str, text: str, category: str, true_depth: int,
                    macro: str, prev_json: str = "") -> str:
    parts = [f"CATEGORY: {category}", f"EXPECTED NESTING DEPTH: {true_depth}", "",
             "JSON SCHEMA:", schema, "", "TEXT:", text, ""]
    if prev_json:
        preview = prev_json[:1000] + ("..." if len(prev_json) > 1000 else "")
        parts += ["PREVIOUS ATTEMPT (incorrect):", preview, ""]
    if macro:
        parts += ["=== MACRO-BDI DIRECTIVE (apply at every step) ===", macro, ""]
    parts.append("Generate the corrected JSON object now:" if (prev_json or macro) else "Generate the JSON object now:")
    return "\n".join(parts)


def _evaluate(generated: str, task: dict) -> dict:
    p = compute_payoffs(generated, task["schema"], task["ground_truth_json"], task["true_depth"])
    uv, upe = p["U_V"], p["U_PE"]
    accepted = uv["payoff"] >= SUCCESS_THRESHOLD
    outcome  = (f"SUCCESS U_V={uv['payoff']:.3f}" if accepted
                else f"FAILED U_V={uv['payoff']:.3f} — "
                     + ("invalid JSON" if not upe["dim_json_valid"]
                        else "schema mismatch" if upe["dim_schema_compliance"] < 0.5
                        else "wrong values" if uv["dim_field_accuracy"] < 0.4
                        else "wrong depth"))
    return {
        "task_id":               task["id"],
        "category":              task["category"],
        "U_PE":                  round(upe["payoff"], 4),
        "U_V":                   round(uv["payoff"], 4),
        "accepted":              accepted,
        "quality":               "success" if accepted else ("partial" if uv["payoff"] >= 0.4 else "rejected"),
        "is_valid_json":         bool(upe["dim_json_valid"] >= 1.0),
        "dim_schema_compliance": round(upe["dim_schema_compliance"], 4),
        "dim_field_accuracy":    round(uv["dim_field_accuracy"], 4),
        "dim_depth_match":       round(uv["dim_depth_match"], 4),
        "actual_depth":          uv["actual_depth"],
        "generated_json":        generated,
        "_outcome":              outcome,
    }


def run_task(client: OpenAI, task: dict, shots: int = 5) -> dict:
    best         = None
    prev_json    = ""
    prev_outcome = ""
    n_shots_used = 0

    for shot in range(1, shots + 1):
        n_shots_used = shot

        # Macro-BDI : directive stratégique
        if shot == 1:
            macro = _call_free(client, _SYS_MACRO,
                               _macro_prompt_shot1(task["schema"], task["category"],
                                                   task["true_depth"]))
        else:
            macro = _call_free(client, _SYS_MACRO,
                               _macro_prompt_retry(task["schema"], task["category"],
                                                   task["true_depth"], prev_outcome, shot))

        print(f"  [HierBDI] {task['id']} shot={shot} macro_len={len(macro)}")

        # Micro-BDI : exécution avec directive injectée
        prompt    = _execute_prompt(task["schema"], task["text"], task["category"],
                                    task["true_depth"], macro,
                                    prev_json if shot > 1 else "")
        generated = _call_json(client, task["schema"], prompt)
        result    = _evaluate(generated, task)

        print(f"  [HierBDI] U_V={result['U_V']:.3f} {result['quality']}")

        if best is None or result["U_V"] > best["U_V"]:
            best = result
        prev_json    = generated
        prev_outcome = result["_outcome"]

        if result["accepted"]:
            break
        if shot < shots:
            time.sleep(0.5)

    best.pop("_outcome", None)
    best["n_shots_used"] = n_shots_used
    return best


def run(tasks: list, shots: int = 5) -> list:
    client = _client()
    return [run_task(client, t, shots) for t in tasks]


def main():
    p = argparse.ArgumentParser(description=f"DeepJSONEval — {ARCH_NAME}")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--shots",       type=int, default=10)
    p.add_argument("--category",    type=str, default=None)
    args = p.parse_args()

    print(f"\n{'='*60}\n  DeepJSONEval — Hierarchical BDI (Harada & Kano 2025)\n  Backend: {LLM_BACKEND} | Model: {LLM_MODEL}\n{'='*60}\n")

    dataset = load_dataset_hf(max_samples=args.max_samples)
    tasks   = list(iter_tasks(dataset, category=args.category))
    print(f"  Tasks: {len(tasks)} | Shots: {args.shots}\n")

    t0      = time.perf_counter()
    results = run(tasks, args.shots)
    elapsed = time.perf_counter() - t0

    accepted = sum(1 for r in results if r["accepted"])
    avg_uv   = sum(r["U_V"] for r in results) / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print(f"  SUMMARY — Hierarchical BDI")
    print(f"  Accepted : {accepted}/{len(results)} ({100*accepted/len(results):.1f}%)")
    print(f"  Avg U_V  : {avg_uv:.3f}")
    print(f"  Time     : {elapsed:.1f}s")
    print(f"{'='*60}\n")

    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = pathlib.Path(RESULTS_DIR) / f"agisdk_{ARCH_NAME}_results_{ts}.json"
    path.write_text(json.dumps({
        "experiment":   f"DeepJSONEval {ARCH_NAME}",
        "architecture": f"Hierarchical BDI (Macro+Micro, assertiveness={ASSERTIVENESS}, Harada & Kano 2025)",
        "model":        LLM_MODEL,
        "shots":        args.shots,
        "total_tasks":  len(results),
        "total_time_s": round(elapsed, 2),
        "n_accepted":   accepted,
        "avg_U_V":      round(avg_uv, 4),
        "results":      results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON: {path}\n")


if __name__ == "__main__":
    main()
