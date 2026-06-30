"""
api/main.py — Entry point for FastAPI REST API scoring service.

Includes routes from api/routes.py and handles server lifecycle.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 4
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes import router as scoring_router
from src.utils import load_config

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

# Include routes from routes.py
app.include_router(scoring_router)

if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    api_cfg = cfg.get("api", {})
    uvicorn.run(
        "api.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8080),
        workers=1,
        reload=False,
    )
