"""
data_loader.py — Load and clean facets from CSV; load conversations from JSON.

Usage (CLI):
    python src/data_loader.py --facets data/raw/facets_assignment.csv \
                              --output data/processed/facets_cleaned.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, load_config, save_json

logger = get_logger("data_loader")

# ---------------------------------------------------------------------------
# Facet category assignment rules
# ---------------------------------------------------------------------------
# Each tuple: (keyword_patterns_lower, category_key)
CATEGORY_RULES: list[tuple[list[str], str]] = [
    # Linguistic Quality
    (
        [
            "grammar", "vocabulary", "sentence", "readability", "clarity",
            "language", "spelling", "structure", "brevity", "verbal",
            "comprehension", "auditory", "memory for sound", "alphabetical",
            "numeric filing", "syntax", "word", "writing",
        ],
        "linguistic_quality",
    ),
    # Pragmatics
    (
        [
            "pragmatic", "relevance", "informative", "directness", "politeness",
            "appropriateness", "context", "discourse", "communication",
            "listening", "feedback", "collaboration", "social interaction",
            "non-verbal", "talkativeness", "outspoken", "brevity",
            "cooperation", "participation",
        ],
        "pragmatics",
    ),
    # Safety
    (
        [
            "safety", "harmful", "harm", "disrespect", "aggression",
            "hostility", "dishonest", "ethical", "bias", "privacy",
            "toxic", "misinformation", "violence", "passive-aggressive",
            "cantankerous", "hateful", "coarse", "impudent", "brazen",
            "hatefulness", "psychoticism", "drug", "psychotic",
        ],
        "safety",
    ),
    # Emotion & Empathy
    (
        [
            "emotion", "empathy", "compassion", "sentiment", "warmth",
            "affection", "sensitivity", "joyful", "happiness", "sadness",
            "grief", "morose", "merry", "bliss", "anger", "fear",
            "emotional", "mood", "desperation", "discontentment",
            "compassion fatigue", "comfort", "vulnerability", "spiritual pain",
            "affinity", "warmhearted",
        ],
        "emotion_empathy",
    ),
    # Behavioral Traits
    (
        [
            "honest", "honesty", "genuine", "authentic", "assertive",
            "humility", "humble", "dignity", "self-esteem", "confidence",
            "determinedness", "perseverance", "resilience", "bravery",
            "courageous", "chivalr", "classy", "decency", "civility",
            "conscientiousness", "self-control", "self-improvement",
            "self-direction", "adaptability", "flexibility", "submission",
            "servility", "independence", "impulsivity", "compulsive",
            "rebellious", "martyrdom", "orderliness", "conformity",
        ],
        "behavioral_traits",
    ),
    # Intelligence & Reasoning
    (
        [
            "reasoning", "logic", "intelligence", "common-sense", "creativity",
            "analytical", "problem-solv", "critical", "numerical", "spatial",
            "statistical", "cognitive", "synthesis", "originality", "innovation",
            "data analysis", "iq", "intellect", "working memory",
            "mathematical", "sequence", "analogies", "rapid cognitive",
            "economic reasoning", "estimating",
        ],
        "intelligence_reasoning",
    ),
]

CATEGORY_DISPLAY: dict[str, str] = {
    "linguistic_quality":   "Linguistic Quality",
    "pragmatics":           "Pragmatics",
    "safety":               "Safety",
    "emotion_empathy":      "Emotion & Empathy",
    "behavioral_traits":    "Behavioral Traits",
    "intelligence_reasoning": "Intelligence & Reasoning",
}


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
_HEADER_RE = re.compile(r"^\d+\.\s*")
_TRAILING_RE = re.compile(r"[:\s]+$")


def _clean_facet_name(raw: str) -> str:
    """Remove numeric prefixes, trailing colons/spaces, normalize whitespace."""
    name = _HEADER_RE.sub("", str(raw).strip())
    name = _TRAILING_RE.sub("", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _is_header_or_meta(name: str) -> bool:
    """Return True for category-header rows and useless meta lines."""
    skip_patterns = [
        r"^facets?$",
        r"subcomponents?$",
        r"styles?$",
        r"types?$",
        r"themes?$",
        r"^(moral and ethical|behavioral tendencies|leadership|adaptability|"
        r"listening|innovation|achievement|affiliation|numerical reasoning|"
        r"hexaco personality|big five facet|psychological construct|"
        r"additional common|time orientation|motivational drivers|"
        r"computer skills|moral and ethical).*",
    ]
    lower = name.lower()
    for pat in skip_patterns:
        if re.search(pat, lower):
            return True
    return False


def _assign_category(name: str) -> str:
    """Assign a facet to one of the 6 categories via keyword matching."""
    lower = name.lower()
    scores: dict[str, int] = {k: 0 for k in CATEGORY_DISPLAY}
    for keywords, cat_key in CATEGORY_RULES:
        for kw in keywords:
            if kw in lower:
                scores[cat_key] += 1
    # Pick highest; default to behavioral_traits if no match
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "behavioral_traits"


# ---------------------------------------------------------------------------
# Importance scoring (heuristic)
# ---------------------------------------------------------------------------
HIGH_IMPORTANCE = {
    "Grammar Correctness", "Clarity", "Politeness", "Safety", "Harmfulness",
    "Empathy", "Emotional Intelligence", "Honesty", "Common-sense",
    "Critical Reasoning", "Assertiveness", "Ethical Standards",
    "Compassion", "Trust in Others", "Disagreeableness",
}


def _importance(name: str) -> float:
    """Return importance weight in [0, 1] for a facet."""
    if name in HIGH_IMPORTANCE:
        return 1.0
    lower = name.lower()
    if any(k in lower for k in ["safety", "harm", "toxic", "ethic"]):
        return 0.95
    if any(k in lower for k in ["empathy", "compassion", "honest"]):
        return 0.85
    if any(k in lower for k in ["grammar", "clarity", "logic", "reason"]):
        return 0.80
    return round(0.5 + 0.3 * (abs(hash(name)) % 100) / 100.0, 2)  # deterministic jitter


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_facets_from_csv(
    csv_path: str | Path,
    target_count: int = 300,
) -> list[dict[str, Any]]:
    """
    Read the raw facets CSV, clean names, assign categories, and return
    the top ``target_count`` facets as a list of dicts.

    Returns
    -------
    list[dict] — each dict has keys:
        id, name, category, category_display, weight, importance, description
    """
    df = pd.read_csv(csv_path, header=0)
    raw_names: list[str] = df.iloc[:, 0].dropna().tolist()

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in raw_names:
        name = _clean_facet_name(raw)
        if not name or _is_header_or_meta(name):
            continue
        lower = name.lower()
        if lower not in seen:
            seen.add(lower)
            cleaned.append(name)

    logger.info("After cleaning: %d unique facets (from %d raw)", len(cleaned), len(raw_names))

    # Assign categories
    by_category: dict[str, list[str]] = {k: [] for k in CATEGORY_DISPLAY}
    for name in cleaned:
        cat = _assign_category(name)
        by_category[cat].append(name)

    # Balance: take per-category quota, total = target_count
    per_cat = target_count // len(CATEGORY_DISPLAY)
    remainder = target_count % len(CATEGORY_DISPLAY)
    selected: list[dict[str, Any]] = []
    facet_id = 1

    # Build relationship map (similar facets in same category)
    for idx, (cat_key, names) in enumerate(by_category.items()):
        quota = per_cat + (1 if idx < remainder else 0)
        selected_names = names[:quota]
        # Pad with synthetic entries if not enough real ones
        i = len(selected_names) + 1
        while len(selected_names) < quota:
            selected_names.append(f"{CATEGORY_DISPLAY[cat_key]} Facet {i}")
            i += 1
        for name in selected_names:
            selected.append(
                {
                    "id": facet_id,
                    "name": name,
                    "category": cat_key,
                    "category_display": CATEGORY_DISPLAY[cat_key],
                    "weight": round(0.7 + 0.3 * (facet_id % 10) / 10, 2),
                    "importance": _importance(name),
                    "description": f"Evaluates the degree to which '{name}' is present in the conversation.",
                }
            )
            facet_id += 1

    # Build facet_relationships: top-5 nearest in same category
    cat_ids: dict[str, list[int]] = {k: [] for k in CATEGORY_DISPLAY}
    for f in selected:
        cat_ids[f["category"]].append(f["id"])

    relationships: dict[str, list[int]] = {}
    for f in selected:
        peers = [i for i in cat_ids[f["category"]] if i != f["id"]]
        relationships[str(f["id"])] = peers[:5]

    logger.info("Loaded %d facets across %d categories", len(selected), len(CATEGORY_DISPLAY))
    return selected, relationships


def build_facets_config(
    csv_path: str | Path,
    output_path: str | Path,
    target_count: int = 300,
) -> dict[str, Any]:
    """Full pipeline: CSV → cleaned JSON config."""
    facets, relationships = load_facets_from_csv(csv_path, target_count)
    config = {
        "version": "1.0.0",
        "total": len(facets),
        "categories": list(CATEGORY_DISPLAY.values()),
        "facets": facets,
        "facet_relationships": relationships,
    }
    save_json(config, output_path)
    return config


# ---------------------------------------------------------------------------
# Conversation loader
# ---------------------------------------------------------------------------

def load_conversations(conversations_path: str | Path) -> list[dict[str, Any]]:
    """Load generated conversations from JSON file."""
    import json

    with open(conversations_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded %d conversations from %s", len(data), conversations_path)
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load and clean facets from CSV")
    p.add_argument(
        "--facets",
        default="data/raw/facets_assignment.csv",
        help="Path to raw facets CSV",
    )
    p.add_argument(
        "--output",
        default="data/processed/facets_cleaned.json",
        help="Output path for cleaned facets JSON",
    )
    p.add_argument("--count", type=int, default=300, help="Target facet count")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config()
    build_facets_config(args.facets, args.output, args.count)
    print(f"✅ Facets saved to {args.output}")
