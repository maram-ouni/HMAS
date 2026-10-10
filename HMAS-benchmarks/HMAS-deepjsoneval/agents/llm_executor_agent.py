# ─────────────────────────────────────────────────────────
# llm_executor_agent.py — AGENT EXÉCUTEUR LLM (génération JSON)
#
# Spécifique à DeepJSONEval.
#
# Rôle : recevoir EXECUTION_REQUESTED du BDIPlanner,
#         appeler le LLM (Groq) pour générer un JSON à partir
#         du schéma + texte + hint BDI, puis publier EXECUTION_DONE.
#
# Le LLM reçoit :
#   - Le schéma JSON Schema
#   - Le texte en langage naturel
#   - Un hint stratégique optionnel du BDI
#
# Le LLM doit retourner UNIQUEMENT un objet JSON brut.
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import os
import queue
import sys
import threading

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.message_bus    import MessageBus, Message, MessageType
from core.history_manager import HistoryManager
from config              import (
    LLM_BACKEND, LLM_API_KEY, LLM_BASE_URL,
    LLM_MODEL, LLM_MODEL_FALLBACK,
    VLLM_GUIDED_DECODING_BACKEND,
)

_SYSTEM_PROMPT = """You are a JSON generation expert.
Your task is to extract information from a text and format it as a valid JSON object.

RULES:
1. Return ONLY a raw JSON object — no markdown, no code blocks (```), no explanations.
2. Your response must start with '{' and end with '}'.
3. Follow the provided JSON schema EXACTLY — use the same key names and nesting structure.
4. Extract values directly from the text. Do not invent values.
5. Use null for fields not mentioned in the text.
6. Preserve the nesting depth specified in the schema."""


class LLMExecutorAgent(threading.Thread):
    """
    Agent exécuteur LLM pour DeepJSONEval.
    Reçoit EXECUTION_REQUESTED, appelle Groq, publie EXECUTION_DONE.
    """

    def __init__(self, bus: MessageBus,
                 history_manager: HistoryManager,
                 stop_event: threading.Event):
        super().__init__(daemon=True, name="LLMExecutorThread")
        self.bus        = bus
        self.hm         = history_manager
        self.stop_event = stop_event
        self.client     = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    # ── Boucle principale ─────────────────────────────────

    def run(self) -> None:
        print(f"[LLMExecutorThread] Started — backend={LLM_BACKEND}, model={LLM_MODEL}")
        while not self.stop_event.is_set():
            try:
                msg = self.bus.get("executor", timeout=2)
            except queue.Empty:
                continue

            if msg.msg_type == MessageType.STOP:
                break
            elif msg.msg_type == MessageType.EXECUTION_REQUESTED:
                self._execute(msg)

        print("[LLMExecutorThread] Stopped.")

    # ── Exécution ─────────────────────────────────────────

    def _execute(self, msg: Message) -> None:
        task_id     = msg.content["task_id"]
        schema      = msg.content.get("schema", "")
        text        = msg.content.get("text", "")
        hint        = msg.content.get("hint", "")
        plan_name   = msg.content.get("plan_name", "")
        attempt     = msg.content.get("attempt", 1)
        category    = msg.content.get("category", "")
        true_depth  = msg.content.get("true_depth", 0)
        prev_json   = msg.content.get("prev_json", "")
        temperature = msg.content.get("temperature", 0.1)

        print(f"\n  [LLMExecutor] Generating JSON for {task_id} "
              f"(attempt={attempt}, category={category}, depth={true_depth}, temp={temperature})")

        generated_json, error = self._call_llm(
            schema, text, hint, category, true_depth, prev_json, temperature
        )

        self.bus.publish(Message(
            sender   = "executor",
            receiver = "verifier",
            msg_type = MessageType.EXECUTION_DONE,
            content  = {
                "task_id":           task_id,
                "schema":            schema,
                "ground_truth_json": msg.content.get("ground_truth_json", ""),
                "category":          category,
                "true_depth":        true_depth,
                "generated_json":    generated_json,
                "hint_used":         hint,
                "plan_name":         plan_name,
                "attempt":           attempt,
                "last_error":        error,
            }
        ))

    # ── Appel LLM ─────────────────────────────────────────

    def _call_llm(self, schema: str, text: str, hint: str,
                  category: str, true_depth: int,
                  prev_json: str = "", temperature: float = 0.1) -> tuple:
        """
        Construit le prompt et appelle le LLM.
        Retourne (generated_json: str, error: str).
        """
        import json as _json

        MAX_CTX  = 4096
        MARGIN   = 64
        # En-dessous de ce budget de sortie, un JSON de profondeur non triviale
        # ne PEUT PAS être généré complet — il sera tronqué, donc invalide à
        # coup sûr. Vu sur deepjsoneval_178_device (depth=6) : prompt à 3947
        # tokens → max_tokens réduit à 128 par l'ancienne logique → 8/8
        # tentatives invalides, U_V=0.000 sur tout le budget, deux runs de
        # suite. Sous ce seuil, on allège le PROMPT (retire prev_json/hint)
        # plutôt que de continuer à rogner la sortie.
        MIN_USABLE_TOKENS = 400

        extra_kwargs: dict = {}
        if LLM_BACKEND == "vllm":
            # True constrained decoding: logits of invalid tokens → -inf at each step.
            # Guarantees syntactically AND structurally valid JSON per the schema.
            try:
                extra_kwargs["extra_body"] = {
                    "guided_json": _json.loads(schema),
                }
            except _json.JSONDecodeError:
                pass  # malformed schema — fall back to free generation
        else:
            # Groq JSON mode: guarantees syntactic JSON validity only (not schema compliance).
            extra_kwargs["response_format"] = {"type": "json_object"}

        def _try(prompt: str, max_tokens: int, model: str):
            """Un appel. Renvoie (raw, error, overflow_prompt_tokens|None)."""
            try:
                response = self.client.chat.completions.create(
                    model       = model,
                    messages    = [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature = temperature,
                    max_tokens  = max_tokens,
                    **extra_kwargs,
                )
                raw = response.choices[0].message.content.strip()
                print(f"  [LLMExecutor] Generated {len(raw)} chars with {model}")
                return raw, "", None
            except Exception as e:
                import re
                m = re.search(r"prompt contains (\d+) input tokens", str(e))
                if m:
                    return "", str(e), int(m.group(1))
                print(f"  [LLMExecutor] {model} error: {e}")
                return "", str(e), None

        full_prompt = self._build_prompt(schema, text, hint, category, true_depth, prev_json)
        lean_prompt = None  # construit seulement si besoin

        for model in [LLM_MODEL, LLM_MODEL_FALLBACK]:
            prompt, max_tokens, tried_lean = full_prompt, 1024, False
            for attempt in range(3):
                raw, err, overflow_tokens = _try(prompt, max_tokens, model)
                if not err:
                    return raw, ""
                if overflow_tokens is None:
                    break  # erreur non liée au contexte — inutile de réessayer ce modèle
                adjusted = MAX_CTX - overflow_tokens - MARGIN
                if adjusted >= MIN_USABLE_TOKENS or tried_lean:
                    # Budget de sortie encore exploitable (ou prompt déjà allégé,
                    # rien de plus à retirer) — on rogne juste max_tokens.
                    max_tokens = max(128, adjusted)
                    print(f"  [LLMExecutor] ctx overflow ({overflow_tokens} tokens) → "
                          f"retry max_tokens={max_tokens}")
                else:
                    # Trop peu de budget de sortie pour un JSON complet : alléger
                    # le PROMPT (retire prev_json + hint) plutôt que la sortie.
                    if lean_prompt is None:
                        lean_prompt = self._build_prompt(schema, text, "", category, true_depth, "")
                    print(f"  [LLMExecutor] ctx overflow ({overflow_tokens} tokens, "
                          f"max_tokens serait {max(128, adjusted)}) → prompt allégé "
                          f"(prev_json/hint retirés), max_tokens=768")
                    prompt, max_tokens, tried_lean = lean_prompt, 768, True

        return "", "All LLM backends failed"

    def _build_prompt(self, schema: str, text: str, hint: str,
                      category: str, true_depth: int,
                      prev_json: str = "") -> str:
        parts = []

        parts.append(f"CATEGORY: {category}")
        parts.append(f"EXPECTED NESTING DEPTH: {true_depth}")
        parts.append("")

        parts.append("JSON SCHEMA:")
        parts.append(schema)
        parts.append("")

        parts.append("TEXT:")
        parts.append(text)
        parts.append("")

        if prev_json:
            # Cap à 1200 chars pour ne pas saturer le contexte
            preview = prev_json[:1200] + ("..." if len(prev_json) > 1200 else "")
            parts.append("PREVIOUS ATTEMPT (incorrect — do NOT copy, fix the errors below):")
            parts.append(preview)
            parts.append("")

        if hint:
            parts.append(f"CORRECTIONS REQUIRED: {hint}")
            parts.append("")

        # Prompt identique au baseline sur le premier essai (pas de hint, pas de prev_json).
        if hint or prev_json:
            parts.append("Generate the corrected JSON object now:")
        else:
            parts.append("Generate the JSON object now:")

        return "\n".join(parts)
