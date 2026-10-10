"""
react_baseline.py — Architecture ReAct adaptée pour DeepJSONEval

Principe : alternance Thought + Action sans mémoire cross-shot.
  Thought : appel LLM libre — raisonnement sur les champs à extraire.
  Action  : appel LLM guidé — génération JSON à partir du raisonnement.

Usage:
    python mas_project/experiments/react_baseline.py --max-samples 200 --shots 5
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

ARCH_NAME = "react"
SUCCESS_THRESHOLD = 0.7
MAX_CTX = 4096

_SYS_JSON = """You are a JSON generation expert.
Your task is to extract information from a text and format it as a valid JSON object.

RULES:
1. Return ONLY a raw JSON object — no markdown, no code blocks, no explanations.
2. Your response must start with '{' and end with '}'.
3. Follow the provided JSON schema EXACTLY — use the same key names and nesting structure.
4. Extract values directly from the text. Do not invent values.
5. Use null for fields not mentioned in the text.
6. Preserve the nesting depth specified in the schema."""

_SYS_REASON = ("You are analyzing a text to prepare accurate JSON extraction. "
               "Think step by step about which value each schema field should receive.")


def _client() -> OpenAI:
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _call_free(client: OpenAI, system: str, user: str,
               temperature: float = 0.3, max_tokens: int = 250) -> str:
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
            print(f"  [ReAct/free] {model}: {e}")
    return ""


def _call_json(client: OpenAI, schema: str, user: str,
               temperature: float = 0.1) -> str:
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
                print(f"  [ReAct/json] {model}: {e}")
                break
    return ""


def _thought_prompt(schema: str, text: str, category: str, true_depth: int) -> str:
    return (f"CATEGORY: {category} | EXPECTED DEPTH: {true_depth}\n\n"
            f"JSON SCHEMA:\n{schema}\n\n"
            f"TEXT:\n{text}\n\n"
            "For each field in the schema, identify the exact value to extract from the text. "
            "List field → value pairs briefly. If a field is absent from the text, write null.")


def _action_prompt(schema: str, text: str, category: str, true_depth: int,
                   thought: str, prev_json: str = "") -> str:
    parts = [f"CATEGORY: {category}", f"EXPECTED NESTING DEPTH: {true_depth}", "",
             "JSON SCHEMA:", schema, "", "TEXT:", text, ""]
    if prev_json:
        preview = prev_json[:1000] + ("..." if len(prev_json) > 1000 else "")
        parts += ["PREVIOUS ATTEMPT (incorrect):", preview, ""]
    if thought:
        parts += ["=== REASONING TRACE ===", thought, ""]
    parts.append("Generate the JSON object now:")
    return "\n".join(parts)


def _evaluate(generated: str, task: dict) -> dict:
    p = compute_payoffs(generated, task["schema"], task["ground_truth_json"], task["true_depth"])
    uv, upe = p["U_V"], p["U_PE"]
    accepted = uv["payoff"] >= SUCCESS_THRESHOLD
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
    }


def run_task(client: OpenAI, task: dict, shots: int = 5) -> dict:
    best = None
    prev_json = ""
    n_shots_used = 0

    for shot in range(1, shots + 1):
        n_shots_used = shot
        # Thought step
        thought = _call_free(client, _SYS_REASON,
                             _thought_prompt(task["schema"], task["text"],
                                             task["category"], task["true_depth"]))
        # Action step
        prompt = _action_prompt(task["schema"], task["text"], task["category"],
                                task["true_depth"], thought, prev_json)
        generated = _call_json(client, task["schema"], prompt)

        result = _evaluate(generated, task)
        result["n_shots_used"] = n_shots_used
        print(f"  [ReAct] {task['id']} shot={shot} U_V={result['U_V']:.3f} {result['quality']}")

        if best is None or result["U_V"] > best["U_V"]:
            best = result
        prev_json = generated

        if result["accepted"]:
            break
        if shot < shots:
            time.sleep(0.5)

    best["n_shots_used"] = n_shots_used
    return best


def run(tasks: list, shots: int = 5) -> list:
    client = _client()
    results = []
    for task in tasks:
        results.append(run_task(client, task, shots))
    return results


def main():
    p = argparse.ArgumentParser(description=f"DeepJSONEval — {ARCH_NAME}")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--shots",       type=int, default=10)
    p.add_argument("--category",    type=str, default=None)
    args = p.parse_args()

    print(f"\n{'='*60}\n  DeepJSONEval — ReAct\n  Backend: {LLM_BACKEND} | Model: {LLM_MODEL}\n{'='*60}\n")

    dataset = load_dataset_hf(max_samples=args.max_samples)
    tasks   = list(iter_tasks(dataset, category=args.category))
    print(f"  Tasks: {len(tasks)} | Shots: {args.shots}\n")

    t0      = time.perf_counter()
    results = run(tasks, args.shots)
    elapsed = time.perf_counter() - t0

    accepted = sum(1 for r in results if r["accepted"])
    avg_uv   = sum(r["U_V"] for r in results) / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print(f"  SUMMARY — ReAct")
    print(f"  Accepted : {accepted}/{len(results)} ({100*accepted/len(results):.1f}%)")
    print(f"  Avg U_V  : {avg_uv:.3f}")
    print(f"  Time     : {elapsed:.1f}s")
    print(f"{'='*60}\n")

    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = pathlib.Path(RESULTS_DIR) / f"agisdk_{ARCH_NAME}_results_{ts}.json"
    path.write_text(json.dumps({
        "experiment":   f"DeepJSONEval {ARCH_NAME}",
        "architecture": "ReAct (Thought + Action, no cross-shot memory)",
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
