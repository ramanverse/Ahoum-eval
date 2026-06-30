"""
api/main.py — FastAPI REST endpoint for scalable batch conversation scoring.

Supports 5000+ conversations without redesign.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 4

Endpoints:
  POST /evaluate          — Score a single conversation
  POST /batch_evaluate    — Score up to 100 conversations
  GET  /facets            — List all 300 facets
  GET  /health            — Health check
  GET  /docs              — Auto-generated Swagger UI
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.utils import get_logger, load_config, set_seed

logger = get_logger("api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Ahoum Conversation Evaluation API",
    description="Multi-facet conversation scoring using open-weight LLMs. 300 behavioral facets, confidence scores.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (loaded once on startup)
# ---------------------------------------------------------------------------
_evaluator = None
_facets_config = None
_cfg = None

_ROOT = Path(__file__).resolve().parent.parent


def _get_evaluator(mode: str = "feature"):
    global _evaluator, _facets_config, _cfg
    if _cfg is None:
        _cfg = load_config()
        set_seed(_cfg.get("project", {}).get("seed", 42))

    facets_path = _ROOT / "data" / "processed" / "facets_cleaned.json"
    if not facets_path.exists():
        raise RuntimeError(
            "facets_cleaned.json not found. Run: python src/data_loader.py"
        )

    if _evaluator is None or _evaluator.mode != mode:
        from src.scorer import ConversationEvaluator

        _evaluator = ConversationEvaluator(
            facets_config=str(facets_path),
            mode=mode,
            config=_cfg,
        )
    return _evaluator


# ---------------------------------------------------------------------------
# Pydantic models
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

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_001",
                "domain": "customer_service",
                "turns": [
                    {"speaker": "user", "text": "My order hasn't arrived yet."},
                    {"speaker": "assistant", "text": "I apologize for the delay. Let me look into that."},
                ],
                "mode": "feature",
            }
        }


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
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    """Health check — returns system status."""
    facets_loaded = (_ROOT / "data" / "processed" / "facets_cleaned.json").exists()
    return {
        "status": "ok",
        "facets_loaded": facets_loaded,
        "version": "1.0.0",
    }


@app.get("/facets", tags=["Facets"])
async def list_facets(category: str | None = None, limit: int = 50):
    """List available facets, optionally filtered by category."""
    facets_path = _ROOT / "data" / "processed" / "facets_cleaned.json"
    if not facets_path.exists():
        raise HTTPException(status_code=404, detail="Facets not loaded. Run: python src/data_loader.py")
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


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
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


@app.post("/batch_evaluate", response_model=BatchResponse, tags=["Evaluation"])
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    api_cfg = cfg.get("api", {})
    uvicorn.run(
        "api.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8080),
        workers=1,  # multi-worker via CLI
        reload=False,
    )
