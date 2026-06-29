"""
model_manager.py — Open-weight LLM wrapper for facet scoring (Mode 2 & 3).

Supports:
  - Mistral-7B-Instruct-v0.1
  - Llama-2-7b-chat-hf
  - Qwen2-7B-Instruct
  - Any instruction-tuned HuggingFace model

Falls back gracefully when no GPU is available (CPU inference).
Includes response caching, JSON parsing, and confidence estimation.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import clamp_confidence, clamp_score, get_logger, safe_json_loads

logger = get_logger("model_manager")

# ---------------------------------------------------------------------------
# Optional torch / transformers imports
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers/torch not available — LLM scoring disabled. "
        "Install with: pip install transformers torch accelerate"
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert conversation analyst. Your task is to evaluate a conversation \
on a specific behavioral or linguistic facet. Always respond ONLY with valid JSON."""

FACET_PROMPT_TEMPLATE = """\
Evaluate the following conversation on the facet: "{facet_name}"

Definition: {definition}

Conversation:
{conversation_text}

Rate on a scale of 1 to 5:
1 = Not present / Very low
2 = Slightly present
3 = Moderately present
4 = Clearly present
5 = Strongly present

Respond ONLY as valid JSON (no markdown, no extra text):
{{
  "score": <integer 1-5>,
  "confidence": <float 0.0-1.0>,
  "evidence": "<brief quote or observation from the conversation>"
}}"""


def _format_conversation(turns: list[dict[str, str]], max_chars: int = 1500) -> str:
    """Convert turns list to readable text, truncated to max_chars."""
    lines = []
    for t in turns:
        speaker = t.get("speaker", "user").capitalize()
        text = t.get("text", "")
        lines.append(f"{speaker}: {text}")
    full = "\n".join(lines)
    if len(full) > max_chars:
        full = full[:max_chars] + "..."
    return full


def _cache_key(text: str, facet_name: str) -> str:
    raw = f"{facet_name}|||{text}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LLM Scorer class
# ---------------------------------------------------------------------------

class LLMScorer:
    """
    Wraps an open-weight instruction-tuned LLM for facet-level scoring.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device_map : str
        'auto' (recommended), 'cpu', 'cuda', etc.
    torch_dtype : str
        'float16' (GPU), 'float32' (CPU).
    load_in_4bit : bool
        Use 4-bit quantization (requires bitsandbytes).
    cache : bool
        Cache LLM results in memory to avoid redundant calls.
    max_new_tokens : int
        Max tokens for generated response.
    temperature : float
        Sampling temperature (lower = more deterministic).
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        device_map: str = "auto",
        torch_dtype: str = "float16",
        load_in_4bit: bool = False,
        cache: bool = True,
        max_new_tokens: int = 200,
        temperature: float = 0.1,
        hf_token: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype_str = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.use_cache = cache
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.hf_token = hf_token

        self._cache: dict[str, dict[str, Any]] = {}
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed. Run: pip install transformers torch accelerate")

        logger.info("Loading model: %s", self.model_name)
        t0 = time.perf_counter()

        dtype = torch.float16 if self.torch_dtype_str == "float16" else torch.float32
        if not torch.cuda.is_available():
            dtype = torch.float32
            self.device_map = "cpu"
            logger.warning("No GPU detected — using CPU (slow). For production, use a GPU instance.")

        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.hf_token:
            tokenizer_kwargs["token"] = self.hf_token

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if self.hf_token:
            model_kwargs["token"] = self.hf_token
        if self.load_in_4bit:
            try:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            except Exception:
                logger.warning("4-bit loading failed — falling back to full precision")

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self._model.eval()

        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.1fs", elapsed)
        self._loaded = True

    def _build_prompt(self, conversation_text: str, facet_name: str, definition: str) -> str:
        return FACET_PROMPT_TEMPLATE.format(
            facet_name=facet_name,
            definition=definition,
            conversation_text=conversation_text,
        )

    def _generate(self, prompt: str) -> str:
        """Run model inference and return raw decoded text."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3800,
        )
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=1.15,
            )
        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def score_facet(
        self,
        turns: list[dict[str, str]],
        facet_name: str,
        definition: str,
    ) -> dict[str, Any]:
        """
        Score a conversation on a single facet using LLM.

        Returns
        -------
        {score: int, confidence: float, evidence: str, method: str}
        """
        conv_text = _format_conversation(turns)
        key = _cache_key(conv_text, facet_name)

        if self.use_cache and key in self._cache:
            cached = self._cache[key].copy()
            cached["method"] = "llm_cached"
            return cached

        # Ensure model is loaded
        try:
            self._load()
        except Exception as exc:
            logger.warning("Model load failed (%s) — using fallback score", exc)
            return self._fallback_score(facet_name)

        prompt = self._build_prompt(conv_text, facet_name, definition)
        try:
            raw_output = self._generate(prompt)
        except Exception as exc:
            logger.warning("LLM inference error: %s", exc)
            return self._fallback_score(facet_name)

        parsed = safe_json_loads(raw_output)
        if parsed and isinstance(parsed, dict) and "score" in parsed:
            result = {
                "score":      clamp_score(parsed.get("score", 3)),
                "confidence": clamp_confidence(parsed.get("confidence", 0.5)),
                "evidence":   str(parsed.get("evidence", "LLM-generated"))[:300],
                "method":     "llm",
            }
        else:
            # Fallback when JSON parsing fails
            result = self._fallback_score(facet_name)
            logger.debug("JSON parse failed for facet '%s', raw: %s", facet_name, raw_output[:100])

        if self.use_cache:
            self._cache[key] = result.copy()
        return result

    @staticmethod
    def _fallback_score(facet_name: str) -> dict[str, Any]:
        """Default score when LLM fails."""
        return {
            "score":      3,
            "confidence": 0.25,
            "evidence":   f"LLM scoring failed for '{facet_name}' — using default",
            "method":     "llm_fallback",
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.debug("LLM cache cleared")

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Lazy singleton accessor (avoids repeated model loading across modules)
# ---------------------------------------------------------------------------
_scorer_instance: LLMScorer | None = None


def get_llm_scorer(model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", **kwargs: Any) -> LLMScorer:
    """Return (or create) the global LLM scorer instance."""
    global _scorer_instance
    if _scorer_instance is None or _scorer_instance.model_name != model_name:
        _scorer_instance = LLMScorer(model_name=model_name, **kwargs)
    return _scorer_instance
