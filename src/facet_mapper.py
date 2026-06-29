"""
facet_mapper.py — Map linguistic features → facet-specific scores.

Each of the 300 facets maps to a set of extracted features and a weighting
function that produces a 1–5 integer score and a confidence value.

This module is the FEATURE-BASED scoring backend (Mode 1).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import clamp_confidence, clamp_score, get_logger

logger = get_logger("facet_mapper")

# ---------------------------------------------------------------------------
# Feature → facet mapping registry
# Each entry maps a category keyword pattern to:
#   (feature_names, weights, direction, base_offset)
# direction: +1 means higher feature → higher score, -1 means inverse
# ---------------------------------------------------------------------------

FEATURE_MAPPING: dict[str, dict[str, Any]] = {
    # --- Linguistic Quality facets ---
    "grammar":           {"features": ["grammar_score", "avg_sentence_length"], "weights": [0.7, 0.3], "dir": 1},
    "vocabulary":        {"features": ["vocabulary_level", "lexical_diversity"], "weights": [0.6, 0.4], "dir": 1},
    "clarity":           {"features": ["readability_score", "avg_sentence_length"], "weights": [0.6, 0.4], "dir": -1},  # high readability grade = harder
    "brevity":           {"features": ["avg_sentence_length", "word_count"], "weights": [0.5, 0.5], "dir": -1},
    "structure":         {"features": ["sentence_count", "grammar_score"], "weights": [0.4, 0.6], "dir": 1},
    "readability":       {"features": ["readability_score"], "weights": [1.0], "dir": -1},
    "lexical_diversity": {"features": ["lexical_diversity"], "weights": [1.0], "dir": 1},
    "sentence_structure":{"features": ["avg_sentence_length", "sentence_count"], "weights": [0.5, 0.5], "dir": 1},

    # --- Pragmatics facets ---
    "politeness":        {"features": ["politeness_markers", "sentiment_polarity"], "weights": [0.7, 0.3], "dir": 1},
    "formality":         {"features": ["formality_score", "vocabulary_level"], "weights": [0.6, 0.4], "dir": 1},
    "relevance":         {"features": ["coherence_score", "reasoning_markers"], "weights": [0.7, 0.3], "dir": 1},
    "informativeness":   {"features": ["word_count", "entity_density", "reasoning_markers"], "weights": [0.3, 0.4, 0.3], "dir": 1},
    "directness":        {"features": ["uncertainty_markers", "negation_ratio"], "weights": [0.6, 0.4], "dir": -1},
    "coherence":         {"features": ["coherence_score"], "weights": [1.0], "dir": 1},
    "context_awareness": {"features": ["coherence_score", "second_person_ratio"], "weights": [0.7, 0.3], "dir": 1},
    "talkativeness":     {"features": ["word_count", "sentence_count"], "weights": [0.6, 0.4], "dir": 1},

    # --- Safety facets ---
    "safety":            {"features": ["safety_flags", "sentiment_polarity"], "weights": [0.8, 0.2], "dir": -1},
    "harmfulness":       {"features": ["safety_flags"], "weights": [1.0], "dir": -1},
    "disrespect":        {"features": ["safety_flags", "sentiment_polarity"], "weights": [0.7, 0.3], "dir": -1},
    "hostility":         {"features": ["safety_flags", "negation_ratio"], "weights": [0.6, 0.4], "dir": -1},
    "dishonesty":        {"features": ["subjectivity_score", "uncertainty_markers"], "weights": [0.5, 0.5], "dir": 1},
    "ethical":           {"features": ["safety_flags", "reasoning_markers"], "weights": [0.6, 0.4], "dir": -1},
    "civility":          {"features": ["safety_flags", "politeness_markers"], "weights": [0.5, 0.5], "dir": -1},
    "hatefulness":       {"features": ["safety_flags"], "weights": [1.0], "dir": -1},

    # --- Emotion & Empathy facets ---
    "empathy":           {"features": ["empathy_signals", "sentiment_polarity"], "weights": [0.7, 0.3], "dir": 1},
    "emotionalism":      {"features": ["subjectivity_score", "exclamation_count"], "weights": [0.6, 0.4], "dir": 1},
    "compassion":        {"features": ["empathy_signals", "politeness_markers"], "weights": [0.6, 0.4], "dir": 1},
    "sensitivity":       {"features": ["empathy_signals", "second_person_ratio"], "weights": [0.7, 0.3], "dir": 1},
    "warmth":            {"features": ["sentiment_polarity", "empathy_signals"], "weights": [0.5, 0.5], "dir": 1},
    "joyfulness":        {"features": ["sentiment_polarity", "exclamation_count"], "weights": [0.7, 0.3], "dir": 1},
    "moroseness":        {"features": ["sentiment_polarity", "negation_ratio"], "weights": [0.6, 0.4], "dir": -1},
    "sadness":           {"features": ["sentiment_polarity"], "weights": [1.0], "dir": -1},

    # --- Behavioral Traits facets ---
    "assertiveness":     {"features": ["imperative_ratio", "first_person_ratio"], "weights": [0.6, 0.4], "dir": 1},
    "confidence":        {"features": ["uncertainty_markers", "first_person_ratio"], "weights": [0.5, 0.5], "dir": -1},
    "honesty":           {"features": ["subjectivity_score", "uncertainty_markers"], "weights": [0.4, 0.6], "dir": -1},
    "persistence":       {"features": ["word_count", "reasoning_markers"], "weights": [0.5, 0.5], "dir": 1},
    "adaptability":      {"features": ["coherence_score", "vocabulary_level"], "weights": [0.5, 0.5], "dir": 1},
    "openness":          {"features": ["subjectivity_score", "uncertainty_markers"], "weights": [0.5, 0.5], "dir": 1},
    "self_control":      {"features": ["negation_ratio", "safety_flags"], "weights": [0.3, 0.7], "dir": -1},
    "independence":      {"features": ["first_person_ratio", "reasoning_markers"], "weights": [0.5, 0.5], "dir": 1},

    # --- Intelligence & Reasoning facets ---
    "common_sense":      {"features": ["coherence_score", "reasoning_markers"], "weights": [0.5, 0.5], "dir": 1},
    "logic":             {"features": ["reasoning_markers", "sentence_count"], "weights": [0.7, 0.3], "dir": 1},
    "creativity":        {"features": ["lexical_diversity", "vocabulary_level"], "weights": [0.5, 0.5], "dir": 1},
    "analytical":        {"features": ["reasoning_markers", "entity_density"], "weights": [0.6, 0.4], "dir": 1},
    "critical_reasoning":{"features": ["reasoning_markers", "coherence_score"], "weights": [0.6, 0.4], "dir": 1},
    "statistical":       {"features": ["entity_density", "reasoning_markers"], "weights": [0.5, 0.5], "dir": 1},
    "originality":       {"features": ["lexical_diversity", "subjectivity_score"], "weights": [0.6, 0.4], "dir": 1},
    "synthesis":         {"features": ["reasoning_markers", "coherence_score", "word_count"], "weights": [0.4, 0.4, 0.2], "dir": 1},
}

# ---------------------------------------------------------------------------
# Normalization bounds per feature (empirical 1st/99th percentile estimates)
# ---------------------------------------------------------------------------
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "grammar_score":            (0.0, 1.0),
    "vocabulary_level":         (0.0, 0.6),
    "lexical_diversity":        (0.2, 1.0),
    "readability_score":        (0.0, 18.0),
    "avg_sentence_length":      (3.0, 40.0),
    "sentence_count":           (1.0, 20.0),
    "coherence_score":          (0.0, 1.0),
    "sentiment_polarity":       (-1.0, 1.0),
    "subjectivity_score":       (0.0, 1.0),
    "formality_score":          (0.0, 1.0),
    "politeness_markers":       (0.0, 5.0),
    "uncertainty_markers":      (0.0, 5.0),
    "empathy_signals":          (0.0, 5.0),
    "reasoning_markers":        (0.0, 8.0),
    "negation_ratio":           (0.0, 0.3),
    "safety_flags":             (0.0, 5.0),
    "entity_density":           (0.0, 20.0),
    "first_person_ratio":       (0.0, 0.4),
    "second_person_ratio":      (0.0, 0.4),
    "third_person_ratio":       (0.0, 0.3),
    "imperative_ratio":         (0.0, 1.0),
    "word_count":               (5.0, 200.0),
    "exclamation_count":        (0.0, 5.0),
    "question_ratio":           (0.0, 1.0),
    "turn_length_variance":     (0.0, 50.0),
}


def _normalize_feature(name: str, value: float) -> float:
    """Map a feature value to [0, 1] using known bounds."""
    lo, hi = FEATURE_BOUNDS.get(name, (0.0, 1.0))
    if hi == lo:
        return 0.5
    normed = (value - lo) / (hi - lo)
    return float(np.clip(normed, 0.0, 1.0))


def _best_mapping(facet_name: str) -> dict[str, Any] | None:
    """Find the best feature mapping for a given facet name via keyword matching."""
    lower = facet_name.lower()
    best_key: str | None = None
    best_hits = 0
    for key in FEATURE_MAPPING:
        keyword_parts = key.replace("_", " ").split()
        hits = sum(1 for part in keyword_parts if part in lower)
        if hits > best_hits:
            best_hits = hits
            best_key = key
    if best_key and best_hits > 0:
        return FEATURE_MAPPING[best_key]
    # Default fallback mapping
    return {
        "features": ["coherence_score", "sentiment_polarity", "grammar_score"],
        "weights": [0.4, 0.3, 0.3],
        "dir": 1,
    }


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_facet_feature_based(
    facet: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute feature-based score for a single facet given extracted features.

    Returns
    -------
    {score: int(1-5), confidence: float(0-1), evidence: str, method: "feature"}
    """
    mapping = _best_mapping(facet.get("name", ""))
    if mapping is None:
        return {"score": 3, "confidence": 0.3, "evidence": "No mapping found", "method": "feature"}

    feat_names: list[str] = mapping["features"]
    weights: list[float] = mapping["weights"]
    direction: int = mapping.get("dir", 1)

    # Gather available feature values
    available_values: list[tuple[float, float]] = []  # (normalized_value, weight)
    missing = 0
    for feat_name, w in zip(feat_names, weights):
        raw_val = features.get(feat_name)
        if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
            missing += 1
            continue
        normed = _normalize_feature(feat_name, float(raw_val))
        if direction == -1:
            normed = 1.0 - normed
        available_values.append((normed, w))

    if not available_values:
        return {"score": 3, "confidence": 0.2, "evidence": "Insufficient features", "method": "feature"}

    # Weighted average of normalized values
    total_w = sum(w for _, w in available_values)
    weighted_sum = sum(v * w for v, w in available_values)
    raw_score_0_1 = weighted_sum / total_w if total_w else 0.5

    # Map to 1–5 scale
    raw_score_1_5 = 1.0 + raw_score_0_1 * 4.0
    final_score = clamp_score(raw_score_1_5)

    # Confidence components
    data_completeness = 1.0 - (missing / len(feat_names))
    # Inter-feature consistency: low std → high consistency
    if len(available_values) > 1:
        vals = [v for v, _ in available_values]
        consistency = 1.0 - min(float(np.std(vals)), 0.5) / 0.5
    else:
        consistency = 0.7
    feature_reliability = 0.65  # base reliability for feature-based approach
    confidence = clamp_confidence(feature_reliability * data_completeness * consistency)

    # Evidence string
    evidence_parts = [f"{fn}={features.get(fn, 'N/A'):.3f}" for fn in feat_names if features.get(fn) is not None]
    evidence = f"Feature-based [{', '.join(evidence_parts)}]"

    return {
        "score":      final_score,
        "confidence": confidence,
        "evidence":   evidence,
        "method":     "feature",
    }


# ---------------------------------------------------------------------------
# Batch scoring for a conversation
# ---------------------------------------------------------------------------

def score_all_facets(
    facets: list[dict[str, Any]],
    features: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Score all facets for a conversation's aggregated features."""
    results: dict[str, dict[str, Any]] = {}
    for facet in facets:
        results[facet["name"]] = score_facet_feature_based(facet, features)
    return results
