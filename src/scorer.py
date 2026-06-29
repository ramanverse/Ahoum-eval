"""
scorer.py — Central multi-facet scoring engine (Modes 1, 2, 3).

Orchestrates feature-based and LLM-based scoring into a unified interface.

Usage (CLI):
    python src/scorer.py \
        --conversations data/raw/generated_conversations.json \
        --facets        data/processed/facets_cleaned.json \
        --output        data/processed/scores.parquet \
        --mode          feature
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.facet_mapper import score_all_facets
from src.feature_extractor import extract_features_for_turn
from src.model_manager import get_llm_scorer
from src.preprocessing import normalize_text
from src.utils import clamp_confidence, clamp_score, get_logger, load_config, save_json, set_seed, timeit

logger = get_logger("scorer")

# ---------------------------------------------------------------------------
# Aggregate turn features → conversation-level features
# ---------------------------------------------------------------------------

def aggregate_turn_features(turns: list[dict[str, str]]) -> dict[str, Any]:
    """
    Compute aggregate linguistic features over all turns of a conversation.
    Returns a flat dict suitable for feature-based scoring.
    """
    prev_text: str | None = None
    all_features: list[dict[str, Any]] = []
    all_texts = [t.get("text", "") for t in turns]

    for t in turns:
        raw = t.get("text", "")
        processed = normalize_text(raw, lowercase=True, remove_special=True)
        row = {"original_text": raw, "processed_text": processed}
        feats = extract_features_for_turn(row, prev_text, all_texts)
        all_features.append(feats)
        prev_text = raw

    if not all_features:
        return {}

    # Aggregate: mean for most numeric features; max for safety_flags
    aggregated: dict[str, Any] = {}
    numeric_keys = [k for k in all_features[0] if isinstance(all_features[0][k], (int, float))]
    for key in numeric_keys:
        vals = [f[key] for f in all_features if key in f and f[key] is not None]
        if vals:
            aggregated[key] = float(np.mean(vals))

    # Max for safety-related features (any harmful turn matters)
    for key in ["safety_flags", "exclamation_count", "question_count"]:
        vals = [f.get(key, 0) for f in all_features]
        if vals:
            aggregated[f"{key}_max"] = float(max(vals))
            aggregated[key] = aggregated[f"{key}_max"]

    return aggregated


# ---------------------------------------------------------------------------
# Score combining (hybrid mode)
# ---------------------------------------------------------------------------

def combine_scores(
    feature_result: dict[str, Any],
    llm_result: dict[str, Any],
    feature_weight: float = 0.4,
    llm_weight: float = 0.6,
) -> dict[str, Any]:
    """Blend feature-based and LLM scores."""
    f_score = feature_result["score"]
    l_score = llm_result["score"]
    combined_raw = feature_weight * f_score + llm_weight * l_score
    combined_score = clamp_score(combined_raw)
    combined_conf = clamp_confidence(
        feature_result["confidence"] * feature_weight
        + llm_result["confidence"] * llm_weight
    )
    return {
        "score":      combined_score,
        "confidence": combined_conf,
        "evidence":   f"[Feature] {feature_result['evidence']} | [LLM] {llm_result['evidence']}",
        "method":     "hybrid",
        "feature_score": f_score,
        "llm_score":     l_score,
    }


# ---------------------------------------------------------------------------
# Main Evaluator class
# ---------------------------------------------------------------------------

class ConversationEvaluator:
    """
    Multi-facet conversation evaluator supporting three scoring modes.

    Modes
    -----
    feature  — Fast, rule-based scoring using extracted linguistic features.
    llm      — LLM-augmented scoring for all facets (slow, accurate).
    hybrid   — Feature-based baseline + LLM refinement for critical/uncertain facets.
    """

    def __init__(
        self,
        facets_config: str | Path | dict,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        mode: str = "feature",
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            config = {}
        self.cfg = config
        self.mode = mode
        self.scoring_cfg = config.get("scoring", {})

        # Load facets
        if isinstance(facets_config, dict):
            self.facets = facets_config.get("facets", [])
        else:
            with open(facets_config, "r", encoding="utf-8") as fh:
                facets_data = json.load(fh)
            self.facets = facets_data.get("facets", [])

        logger.info("Loaded %d facets | mode=%s", len(self.facets), mode)

        # LLM scorer (lazy — only loaded if mode needs it)
        self._llm: Any = None
        self._model_name = model_name

        # Thresholds
        self.conf_threshold = self.scoring_cfg.get("confidence_threshold", 0.6)
        self.imp_threshold  = self.scoring_cfg.get("importance_threshold", 0.8)
        self.llm_weight     = self.scoring_cfg.get("llm_weight", 0.6)
        self.feature_weight = self.scoring_cfg.get("feature_weight", 0.4)

    @property
    def llm(self):
        """Lazy-load LLM scorer."""
        if self._llm is None:
            self._llm = get_llm_scorer(self._model_name)
        return self._llm

    def evaluate_conversation(self, conv: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate a single conversation on all facets.

        Parameters
        ----------
        conv : dict with keys 'conversation_id', 'turns', 'domain', etc.

        Returns
        -------
        dict with conversation_id, facet_scores, category_averages,
              overall_score, overall_confidence, evaluation_timestamp
        """
        cid = conv.get("conversation_id", "unknown")
        turns = conv.get("turns", [])

        # Aggregate features
        agg_features = aggregate_turn_features(turns)

        facet_scores: dict[str, dict[str, Any]] = {}

        for facet in self.facets:
            fname = facet["name"]
            importance = facet.get("importance", 0.5)

            if self.mode == "feature":
                result = score_all_facets([facet], agg_features)[fname]

            elif self.mode == "llm":
                result = self.llm.score_facet(turns, fname, facet.get("description", ""))

            else:  # hybrid
                feature_result = score_all_facets([facet], agg_features)[fname]
                # LLM if low confidence or high importance
                if feature_result["confidence"] < self.conf_threshold or importance >= self.imp_threshold:
                    llm_result = self.llm.score_facet(turns, fname, facet.get("description", ""))
                    result = combine_scores(feature_result, llm_result, self.feature_weight, self.llm_weight)
                else:
                    result = feature_result

            facet_scores[fname] = result

        # Category averages
        category_scores: dict[str, list[int]] = {}
        for facet in self.facets:
            cat = facet.get("category_display", "Other")
            if cat not in category_scores:
                category_scores[cat] = []
            s = facet_scores.get(facet["name"], {}).get("score", 3)
            category_scores[cat].append(s)

        category_averages = {
            cat: round(float(np.mean(scores)), 3)
            for cat, scores in category_scores.items()
        }

        all_scores = [v["score"] for v in facet_scores.values()]
        all_confs  = [v["confidence"] for v in facet_scores.values()]

        return {
            "conversation_id":       cid,
            "domain":                conv.get("domain", ""),
            "quality_level":         conv.get("quality_level", ""),
            "facet_scores":          facet_scores,
            "category_averages":     category_averages,
            "overall_score":         round(float(np.mean(all_scores)), 3) if all_scores else 3.0,
            "overall_confidence":    round(float(np.mean(all_confs)), 3)  if all_confs  else 0.5,
            "num_facets_evaluated":  len(facet_scores),
            "evaluation_mode":       self.mode,
            "evaluation_timestamp":  datetime.now().isoformat(),
        }

    @timeit
    def batch_evaluate(
        self,
        conversations: list[dict[str, Any]],
        sample_rate: float = 1.0,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Evaluate a list of conversations.

        Parameters
        ----------
        conversations : list of conversation dicts
        sample_rate   : fraction of facets to evaluate (0–1) — useful for speed testing
        show_progress : show tqdm progress bar
        """
        if sample_rate < 1.0:
            n = max(1, int(len(self.facets) * sample_rate))
            orig_facets = self.facets
            self.facets = self.facets[:n]
            logger.info("Sample rate %.2f → evaluating %d/%d facets", sample_rate, n, len(orig_facets))

        results = []
        iterator = tqdm(conversations, desc="Evaluating", unit="conv") if show_progress else conversations
        for conv in iterator:
            try:
                result = self.evaluate_conversation(conv)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to evaluate %s: %s", conv.get("conversation_id"), exc)

        if sample_rate < 1.0:
            self.facets = orig_facets  # restore

        logger.info("Batch evaluation complete: %d conversations", len(results))
        return results

    def results_to_dataframe(self, results: list[dict[str, Any]]) -> pd.DataFrame:
        """Flatten a list of evaluation results into a DataFrame."""
        rows = []
        for res in results:
            row: dict[str, Any] = {
                "conversation_id":     res["conversation_id"],
                "domain":              res.get("domain", ""),
                "quality_level":       res.get("quality_level", ""),
                "overall_score":       res.get("overall_score", 0),
                "overall_confidence":  res.get("overall_confidence", 0),
                "evaluation_mode":     res.get("evaluation_mode", ""),
                "evaluation_timestamp": res.get("evaluation_timestamp", ""),
            }
            # Facet scores as flat columns
            for fname, fdata in res.get("facet_scores", {}).items():
                safe_name = fname.lower().replace(" ", "_").replace("-", "_")[:50]
                row[f"facet_{safe_name}_score"] = fdata.get("score", 3)
                row[f"facet_{safe_name}_conf"]  = fdata.get("confidence", 0.5)
            # Category averages
            for cat, avg in res.get("category_averages", {}).items():
                safe_cat = cat.lower().replace(" ", "_").replace("&", "and")[:40]
                row[f"cat_{safe_cat}_avg"] = avg
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-facet conversation scoring")
    p.add_argument("--conversations", default="data/raw/generated_conversations.json")
    p.add_argument("--facets",        default="data/processed/facets_cleaned.json")
    p.add_argument("--output",        default="data/processed/scores.parquet")
    p.add_argument("--scores-json",   default="data/processed/scores.json")
    p.add_argument("--mode",          default="feature", choices=["feature", "llm", "hybrid"])
    p.add_argument("--model",         default="mistralai/Mistral-7B-Instruct-v0.1")
    p.add_argument("--sample",        type=int, default=None, help="Evaluate only first N conversations")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config()
    set_seed(cfg.get("project", {}).get("seed", 42))

    with open(args.conversations, "r", encoding="utf-8") as fh:
        convs = json.load(fh)

    if args.sample:
        convs = convs[: args.sample]

    evaluator = ConversationEvaluator(
        facets_config=args.facets,
        model_name=args.model,
        mode=args.mode,
        config=cfg,
    )

    t0 = time.perf_counter()
    results = evaluator.batch_evaluate(convs)
    elapsed = time.perf_counter() - t0

    # Save JSON
    save_json(results, args.scores_json)

    # Save Parquet
    df = evaluator.results_to_dataframe(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\n✅ Scored {len(results)} conversations on {len(evaluator.facets)} facets")
    print(f"   Elapsed: {elapsed:.1f}s  ({elapsed/len(results):.2f}s/conv)")
    print(f"   Output:  {args.output}  |  {args.scores_json}")
