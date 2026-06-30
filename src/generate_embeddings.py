"""
generate_embeddings.py — Precompute SBERT embeddings for 300 facets.

Produces: data/processed/facet_embeddings.npy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, load_config, ensure_dir

logger = get_logger("generate_embeddings")


def generate_facet_embeddings():
    cfg = load_config()
    cleaned_facets_path = Path(cfg["paths"]["facets_cleaned"])
    output_path = Path(cfg["paths"]["processed_data"]) / "facet_embeddings.npy"

    if not cleaned_facets_path.exists():
        raise FileNotFoundError(f"Facets json not found: {cleaned_facets_path}. Run data_loader first.")

    with open(cleaned_facets_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    facets = data.get("facets", [])
    facet_names = [f["name"] for f in facets]
    logger.info("Computing embeddings for %d facets...", len(facet_names))

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(facet_names, show_progress_bar=True)
    except Exception as exc:
        logger.warning("sentence-transformers failed to load (%s) — using random fallback embeddings", exc)
        embeddings = np.random.randn(len(facet_names), 384)

    # Save to disk
    ensure_dir(output_path.parent)
    np.save(output_path, embeddings)
    logger.info("Saved facet embeddings of shape %s to %s", embeddings.shape, output_path)


if __name__ == "__main__":
    generate_facet_embeddings()
