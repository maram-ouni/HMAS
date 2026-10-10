"""
baseline_llm.py — LLM direct sans agents BDI ni vérificateur

Appelle le LLM une seule fois par tâche avec le même prompt que le MAS,
calcule les mêmes métriques U_V pour comparer avec le MAS hybride.

Usage:
  python3 mas_project_v2/baseline_llm.py --max-samples 20
  python3 mas_project_v2/baseline_llm.py --max-samples 20 --category movie
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import LLM_BACKEND, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_MODEL_FALLBACK, VLLM_GUIDED_DECODING_BACKEND
from core.payoff_calculator import compute_payoffs
from data.dataset_loader import load_dataset_hf, iter_tasks

_SYSTEM_PROMPT = """You are a JSON generation expert.
Your task is to extract information from a text and format it as a valid JSON object.

RULES:
1. Return ONLY a raw JSON object — no markdown, no code blocks (```), no explanations.
2. Your response must start with '{' and end with '}'.
3. Follow the provided JSON schema EXACTLY — use the same key names and nesting structure.
4. Extract values directly from the text. Do not invent values.
5. Use null for fields not mentioned in the text.
6. Preserve the nesting depth specified in the schema."""


def build_prompt(schema: str, text: str, category: str, true_depth: int) -> str:
    return "\n".join([
        f"CATEGORY: {category}",
        f"EXPECTED NESTING DEPTH: {true_depth}",
        "",
        "JSON SCHEMA:",
        schema,
        "",
        "TEXT:",
        text,
        "",
        "Generate the JSON object now:",
    ])


def _parse_prompt_tokens(error_msg: str) -> int:
    """Extrait le nombre de tokens du prompt depuis un message d'erreur vLLM 400."""
    import re
    m = re.search(r"prompt contains (\d+) input tokens", str(error_msg))
    return int(m.group(1)) if m else 0


def call_llm(client: OpenAI, schema: str, text: str, category: str, true_depth: int) -> str:
    prompt = build_prompt(schema, text, category, true_depth)

    extra_kwargs: dict = {}
    if LLM_BACKEND == "vllm":
        try:
            extra_kwargs["extra_body"] = {
                "guided_json": json.loads(schema),
            }
        except json.JSONDecodeError:
            pass
    else:
        extra_kwargs["response_format"] = {"type": "json_object"}

    MAX_CTX  = 4096
    MARGIN   = 64
    max_tokens = 1024

    for model in [LLM_MODEL, LLM_MODEL_FALLBACK]:
        for attempt in range(2):  # 2e tentative avec max_tokens réduit si ctx overflow
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens,
                    **extra_kwargs,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                prompt_tokens = _parse_prompt_tokens(err)
                if prompt_tokens > 0 and attempt == 0:
                    adjusted = max(128, MAX_CTX - prompt_tokens - MARGIN)
                    print(f"  [Baseline] ctx overflow ({prompt_tokens} tokens) → retry max_tokens={adjusted}")
                    max_tokens = adjusted
                    continue
                print(f"  [Baseline] {model} error: {e}")
                break
    return ""


def main():
    parser = argparse.ArgumentParser(description="Baseline LLM sans agents")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--category",    type=str, default=None)
    parser.add_argument("--from-mas",    type=str, default=None,
                        help="Fichier JSON MAS — exécute la baseline sur exactement les mêmes task_ids")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  BASELINE LLM — sans agents BDI ni vérificateur")
    print(f"  Backend : {LLM_BACKEND} | Modèle : {LLM_MODEL}")
    print(f"{'='*60}\n")

    dataset = load_dataset_hf(max_samples=None)
    all_tasks = list(iter_tasks(dataset, category=args.category))

    if args.from_mas:
        with open(args.from_mas) as f:
            mas_data = json.load(f)
        # Extraire les task_ids uniques dans l'ordre du MAS
        seen = set()
        mas_ids = []
        for r in mas_data.get("results", []):
            tid = r["task_id"]
            if tid not in seen:
                seen.add(tid)
                mas_ids.append(tid)
        id_index = {t["id"]: t for t in all_tasks}
        tasks = [id_index[tid] for tid in mas_ids if tid in id_index]
        print(f"  [baseline] Tâches issues du MAS : {len(tasks)} (from {args.from_mas})\n")
    else:
        tasks = all_tasks[:args.max_samples]

    client  = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    results = []
    t_start = time.time()

    for task in tasks:
        tid = task["id"]
        print(f"  [{tid}] generating...", end=" ", flush=True)
        t0 = time.time()

        generated = call_llm(
            client,
            task["schema"],
            task["text"],
            task["category"],
            task["true_depth"],
        )

        payoffs = compute_payoffs(
            generated_json=generated,
            schema_str=task["schema"],
            ground_truth_str=task["ground_truth_json"],
            true_depth=task["true_depth"],
        )

        u_v     = payoffs["U_verifier"]
        u_pe    = payoffs["U_planner"]
        ok      = u_v >= 0.7
        elapsed = time.time() - t0

        print(f"U_V={u_v:.3f} {'YES' if ok else 'NO'} ({elapsed:.1f}s)")

        results.append({
            "task_id":  tid,
            "category": task["category"],
            "U_PE":     round(u_pe, 3),
            "U_V":      round(u_v, 3),
            "accepted": ok,
        })

    total_time = time.time() - t_start
    accepted   = sum(1 for r in results if r["accepted"])
    avg_uv     = sum(r["U_V"] for r in results) / len(results) if results else 0

    print(f"\n{'='*60}")
    print(f"  SUMMARY — Baseline LLM (sans MAS)")
    print(f"{'='*60}")
    print(f"  {'Task':<30} {'Cat':<12} {'U_PE':>6} {'U_V':>6}  {'OK'}")
    print(f"  {'─'*58}")
    for r in results:
        print(f"  {r['task_id']:<30} {r['category']:<12} {r['U_PE']:>6.3f} {r['U_V']:>6.3f}  {'YES' if r['accepted'] else 'NO'}")
    print(f"  {'─'*58}")
    print(f"  Accepted : {accepted}/{len(results)}")
    print(f"  Avg U_V  : {avg_uv:.3f}")
    print(f"  Total    : {total_time:.1f}s")
    print(f"{'='*60}\n")

    # Sauvegarde
    os.makedirs("mas_project_v2/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = f"mas_project_v2/results/baseline_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({"results": results, "accepted": accepted,
                   "total": len(results), "avg_uv": round(avg_uv, 3),
                   "total_time": round(total_time, 1)}, f, indent=2)

    csv_path = f"mas_project_v2/results/baseline_results_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "category", "U_PE", "U_V", "accepted"])
        w.writeheader()
        w.writerows(results)

    print(f"  JSON : {json_path}")
    print(f"  CSV  : {csv_path}")


if __name__ == "__main__":
    main()
