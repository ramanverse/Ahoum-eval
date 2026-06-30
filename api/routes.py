"""
api/routes.py — FastAPI router for Ahoum Conversation Evaluation.

Defines public endpoint handlers for:
  - GET  /health          — Service health check
  - GET  /facets          — List all 300 facets
  - POST /evaluate        — Score single conversation
  - POST /batch_evaluate  — Score up to 100 conversations
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.utils import get_logger, load_config, set_seed

logger = get_logger("api_routes")
router = APIRouter()

_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Global state (lazy loaded)
# ---------------------------------------------------------------------------
_evaluator = None
_cfg = None


def _get_evaluator(mode: str = "feature"):
    global _evaluator, _cfg
    if _cfg is None:
        _cfg = load_config()
        set_seed(_cfg.get("project", {}).get("seed", 42))

    facets_path = _ROOT / "data" / "processed" / "facets_cleaned.json"
    if not facets_path.exists():
        raise RuntimeError("facets_cleaned.json not found. Run: python src/data_loader.py")

    if _evaluator is None or _evaluator.mode != mode:
        from src.scorer import ConversationEvaluator
        _evaluator = ConversationEvaluator(
            facets_config=str(facets_path),
            mode=mode,
            config=_cfg,
        )
    return _evaluator


# ---------------------------------------------------------------------------
# Schema Models
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    speaker: str = Field(..., example="user")
    text: str    = Field(..., min_length=1, example="Hello, I need help with my order.")

    @field_validator("speaker")
    @classmethod
    def speaker_valid(cls, v: str) -> str:
        if v.lower() not in {"user", "assistant", "agent", "human", "bot"}:
            raise ValueError(f"Unknown speaker role: {v}")
        return v.lower()


class ConversationRequest(BaseModel):
    conversation_id: str | None = None
    domain: str = "general"
    turns: list[Turn] = Field(..., min_length=1)
    mode: str = Field("feature", pattern="^(feature|llm|hybrid)$")


class BatchRequest(BaseModel):
    conversations: list[ConversationRequest] = Field(..., max_length=100)
    mode: str = Field("feature", pattern="^(feature|llm|hybrid)$")


class FacetScore(BaseModel):
    score: int
    confidence: float
    evidence: str
    method: str


class EvaluationResponse(BaseModel):
    conversation_id: str
    domain: str
    overall_score: float
    overall_confidence: float
    num_facets_evaluated: int
    facet_scores: dict[str, FacetScore]
    category_averages: dict[str, float]
    evaluation_mode: str
    latency_seconds: float


class BatchResponse(BaseModel):
    total: int
    results: list[EvaluationResponse]
    total_latency_seconds: float
    avg_latency_per_conversation: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", tags=["System"])
async def health():
    """Health check — returns system status."""
    facets_loaded = (_ROOT / "data" / "processed" / "facets_cleaned.json").exists()
    return {
        "status": "ok",
        "facets_loaded": facets_loaded,
        "version": "1.0.0",
    }


@router.get("/facets", tags=["Facets"])
async def list_facets(category: str | None = None, limit: int = 50):
    """List available facets, optionally filtered by category."""
    facets_path = _ROOT / "data" / "processed" / "facets_cleaned.json"
    if not facets_path.exists():
        raise HTTPException(status_code=404, detail="Facets not loaded.")
    with open(facets_path) as fh:
        data = json.load(fh)
    facets = data.get("facets", [])
    if category:
        facets = [f for f in facets if f.get("category_display", "").lower() == category.lower()]
    return {
        "total": len(facets),
        "returned": min(limit, len(facets)),
        "facets": facets[:limit],
        "categories": data.get("categories", []),
    }


@router.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_single(request: ConversationRequest):
    """Evaluate a single conversation on all 300 facets."""
    t0 = time.perf_counter()
    try:
        evaluator = _get_evaluator(request.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    conv_dict = {
        "conversation_id": request.conversation_id or f"api_conv_{int(t0)}",
        "domain": request.domain,
        "turns": [{"speaker": t.speaker, "text": t.text} for t in request.turns],
    }

    try:
        result = evaluator.evaluate_conversation(conv_dict)
    except Exception as exc:
        logger.error("Evaluation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")

    latency = round(time.perf_counter() - t0, 3)
    return EvaluationResponse(
        conversation_id=result["conversation_id"],
        domain=result.get("domain", ""),
        overall_score=result["overall_score"],
        overall_confidence=result["overall_confidence"],
        num_facets_evaluated=result["num_facets_evaluated"],
        facet_scores={k: FacetScore(**v) for k, v in result["facet_scores"].items()},
        category_averages=result.get("category_averages", {}),
        evaluation_mode=result.get("evaluation_mode", "feature"),
        latency_seconds=latency,
    )


@router.post("/batch_evaluate", response_model=BatchResponse, tags=["Evaluation"])
async def batch_evaluate(request: BatchRequest):
    """Evaluate up to 100 conversations in a single request."""
    if len(request.conversations) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 conversations per batch request")

    t0 = time.perf_counter()
    try:
        evaluator = _get_evaluator(request.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    conv_dicts = [
        {
            "conversation_id": c.conversation_id or f"batch_{i}",
            "domain": c.domain,
            "turns": [{"speaker": t.speaker, "text": t.text} for t in c.turns],
        }
        for i, c in enumerate(request.conversations)
    ]

    try:
        results = evaluator.batch_evaluate(conv_dicts, show_progress=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {exc}")

    total_latency = round(time.perf_counter() - t0, 3)
    responses = [
        EvaluationResponse(
            conversation_id=r["conversation_id"],
            domain=r.get("domain", ""),
            overall_score=r["overall_score"],
            overall_confidence=r["overall_confidence"],
            num_facets_evaluated=r["num_facets_evaluated"],
            facet_scores={k: FacetScore(**v) for k, v in r["facet_scores"].items()},
            category_averages=r.get("category_averages", {}),
            evaluation_mode=r.get("evaluation_mode", "feature"),
            latency_seconds=total_latency / max(len(results), 1),
        )
        for r in results
    ]

    return BatchResponse(
        total=len(responses),
        results=responses,
        total_latency_seconds=total_latency,
        avg_latency_per_conversation=round(total_latency / max(len(responses), 1), 3),
    )
