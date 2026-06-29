"""
feature_extractor.py — Extract rich linguistic features for each conversation turn.

Features computed per turn (20+ features):
  lexical_diversity, avg_word_length, sentence_count, avg_sentence_length,
  question_ratio, readability_score, sentiment_polarity, subjectivity_score,
  politeness_markers, formality_score, uncertainty_markers, negation_ratio,
  entity_density, pronoun_ratios, coherence_score, grammar_score,
  safety_flags, empathy_signals, reasoning_markers, turn_length_variance,
  imperative_ratio, passive_voice_ratio, vocabulary_level

Usage:
    python src/feature_extractor.py \
        --input  data/processed/conversations_preprocessed.parquet \
        --output data/processed/conversation_features.parquet
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, load_config, timeit

logger = get_logger("feature_extractor")

# ---------------------------------------------------------------------------
# Optional heavy dependencies with graceful fallbacks
# ---------------------------------------------------------------------------
try:
    from textblob import TextBlob  # type: ignore

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available — using fallback sentiment")

try:
    import textstat  # type: ignore

    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logger.warning("textstat not available — readability will be estimated")

try:
    import spacy  # type: ignore

    try:
        _nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        logger.warning("spaCy model 'en_core_web_sm' not found — run: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available — entity/grammar features will be basic")

# Sentence-BERT for coherence
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    SBERT_AVAILABLE = True
except (ImportError, Exception):
    SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not available — coherence will be heuristic")

# ---------------------------------------------------------------------------
# Lexical helpers
# ---------------------------------------------------------------------------

POLITENESS_MARKERS = frozenset([
    "please", "thank you", "thanks", "appreciate", "sorry", "excuse me",
    "pardon", "kindly", "would you mind", "could you", "would you",
    "if you don't mind", "with respect", "politely",
])

UNCERTAINTY_MARKERS = frozenset([
    "maybe", "perhaps", "possibly", "might", "could", "may",
    "not sure", "i think", "i believe", "probably", "seem",
    "appears", "apparently", "roughly", "approximately",
])

EMPATHY_SIGNALS = frozenset([
    "understand", "feel", "sorry", "compassion", "care", "here for you",
    "support", "empathize", "concern", "listen", "hear you",
    "must be hard", "tough", "difficult", "challenge",
])

REASONING_MARKERS = frozenset([
    "because", "therefore", "thus", "hence", "since", "as a result",
    "consequently", "given that", "due to", "evidence", "suggests",
    "implies", "demonstrate", "analysis", "conclude", "reason",
    "logic", "argument", "data", "proof",
])

NEGATIONS = frozenset([
    "no", "not", "never", "neither", "nor", "nothing",
    "nobody", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "shouldn't", "wouldn't", "couldn't", "isn't", "aren't",
])

SAFETY_BAD_WORDS = frozenset([
    "hate", "kill", "die", "attack", "destroy", "harm", "threat",
    "abuse", "violent", "explode", "weapon", "bomb", "illegal",
    "drugs", "suicide", "self-harm",
])

# Pronouns
FIRST_PERSON  = frozenset(["i", "me", "my", "mine", "myself", "we", "us", "our", "ours"])
SECOND_PERSON = frozenset(["you", "your", "yours", "yourself", "yourselves"])
THIRD_PERSON  = frozenset(["he", "she", "it", "they", "him", "her", "them", "his", "hers", "their"])

IMPERATIVE_STARTERS = frozenset([
    "do", "don't", "please", "ensure", "make", "get", "go",
    "try", "run", "check", "look", "follow", "avoid", "stop",
])

# Formality signals
FORMAL_MARKERS   = frozenset(["therefore", "furthermore", "however", "consequently", "nevertheless", "moreover"])
INFORMAL_MARKERS = frozenset(["gonna", "wanna", "kinda", "sorta", "ya", "nope", "yep", "ok", "okay", "yeah"])


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def sentiment_polarity(text: str) -> float:
    if TEXTBLOB_AVAILABLE:
        return round(TextBlob(text).sentiment.polarity, 4)
    # Fallback: simple positive/negative word count
    lower = text.lower()
    pos_words = {"good", "great", "excellent", "happy", "wonderful", "thank", "helpful"}
    neg_words = {"bad", "terrible", "awful", "angry", "wrong", "broken", "fail", "sorry"}
    pos = sum(1 for w in pos_words if w in lower)
    neg = sum(1 for w in neg_words if w in lower)
    total = pos + neg
    return round((pos - neg) / total, 4) if total else 0.0


def subjectivity_score(text: str) -> float:
    if TEXTBLOB_AVAILABLE:
        return round(TextBlob(text).sentiment.subjectivity, 4)
    # Simple heuristic: presence of opinion words
    opinion_words = {"think", "believe", "feel", "seems", "opinion", "wonder", "imagine"}
    lower_words = set(text.lower().split())
    return round(min(len(lower_words & opinion_words) / 3, 1.0), 4)


def readability_score(text: str) -> float:
    """Flesch-Kincaid grade level; higher = harder."""
    if TEXTSTAT_AVAILABLE:
        return round(textstat.flesch_kincaid_grade(text), 2)
    # Manual approximation
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    words = text.split()
    n_words = len(words)
    n_sent = max(len(sentences), 1)
    syllables = sum(_count_syllables(w) for w in words)
    if n_words == 0:
        return 0.0
    fk = 0.39 * (n_words / n_sent) + 11.8 * (syllables / n_words) - 15.59
    return round(max(0.0, fk), 2)


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:")
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def formality_score(text: str) -> float:
    """0 = informal, 1 = formal."""
    lower = text.lower()
    formal = sum(1 for m in FORMAL_MARKERS if m in lower)
    informal = sum(1 for m in INFORMAL_MARKERS if m in lower)
    total = formal + informal
    if total == 0:
        return 0.5
    return round(formal / total, 4)


def politeness_markers_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for m in POLITENESS_MARKERS if m in lower)


def uncertainty_markers_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for m in UNCERTAINTY_MARKERS if m in lower)


def empathy_signals_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for s in EMPATHY_SIGNALS if s in lower)


def reasoning_markers_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for m in REASONING_MARKERS if m in lower)


def negation_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    n = sum(1 for t in tokens if t.lower() in NEGATIONS)
    return round(n / len(tokens), 4)


def safety_flags_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for w in SAFETY_BAD_WORDS if w in lower)


def pronoun_ratios(tokens: list[str]) -> dict[str, float]:
    n = max(len(tokens), 1)
    lower_toks = [t.lower() for t in tokens]
    return {
        "first_person_ratio":  round(sum(1 for t in lower_toks if t in FIRST_PERSON)  / n, 4),
        "second_person_ratio": round(sum(1 for t in lower_toks if t in SECOND_PERSON) / n, 4),
        "third_person_ratio":  round(sum(1 for t in lower_toks if t in THIRD_PERSON)  / n, 4),
    }


def entity_density(text: str, n_words: int) -> float:
    """Named entities per 100 words."""
    if SPACY_AVAILABLE:
        doc = _nlp(text)
        entity_count = len(doc.ents)
    else:
        # Heuristic: count capitalized phrases not at sentence start
        entity_count = len(re.findall(r"(?<!\. )[A-Z][a-z]+", text))
    if n_words == 0:
        return 0.0
    return round(entity_count / n_words * 100, 2)


def grammar_score(text: str) -> float:
    """
    Grammar quality proxy (0–1).
    Uses spaCy dependency parsing: counts 'dep_' labels that indicate well-formed syntax.
    Falls back to heuristic (sentence completion proxy).
    """
    if SPACY_AVAILABLE:
        doc = _nlp(text)
        if not doc:
            return 0.5
        good_deps = {"nsubj", "obj", "ROOT", "dobj", "iobj", "nsubjpass"}
        has_root = any(t.dep_ == "ROOT" for t in doc)
        has_subj = any(t.dep_ in {"nsubj", "nsubjpass"} for t in doc)
        has_obj  = any(t.dep_ in {"obj", "dobj"} for t in doc)
        score = 0.4 * has_root + 0.3 * has_subj + 0.3 * has_obj
        return round(score, 4)
    # Heuristic: presence of verb + noun
    words = text.lower().split()
    has_verb = any(w in {"is", "are", "was", "were", "be", "been", "have", "has", "do", "can", "will"} for w in words)
    has_noun = len(words) > 3
    return round((has_verb + has_noun) / 2, 2)


def coherence_score(current_text: str, previous_text: str | None) -> float:
    """
    Semantic coherence with the previous turn (0–1).
    Uses sentence-BERT if available, else simple word-overlap Jaccard.
    """
    if previous_text is None:
        return 1.0  # first turn is always coherent with itself
    if SBERT_AVAILABLE:
        try:
            embs = _sbert.encode([current_text, previous_text])
            sim = cosine_similarity([embs[0]], [embs[1]])[0][0]
            return round(float(sim), 4)
        except Exception:
            pass
    # Jaccard fallback
    a, b = set(current_text.lower().split()), set(previous_text.lower().split())
    if not a or not b:
        return 0.5
    return round(len(a & b) / len(a | b), 4)


def imperative_ratio(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    imperative = 0
    for s in sentences:
        first_word = s.strip().split()[0].lower() if s.strip() else ""
        if first_word in IMPERATIVE_STARTERS:
            imperative += 1
    return round(imperative / len(sentences), 4)


def vocabulary_level(tokens: list[str]) -> float:
    """
    Proxy for vocabulary sophistication: proportion of words > 6 characters
    (longer words tend to be more advanced vocabulary).
    """
    if not tokens:
        return 0.0
    long_words = sum(1 for t in tokens if len(t) > 6)
    return round(long_words / len(tokens), 4)


# ---------------------------------------------------------------------------
# Conversation-level aggregation
# ---------------------------------------------------------------------------

def _turn_length_variance(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    lengths = [len(t.split()) for t in texts]
    return round(float(np.std(lengths)), 4)


# ---------------------------------------------------------------------------
# Main feature extraction per row
# ---------------------------------------------------------------------------

def extract_features_for_turn(
    row: dict[str, Any],
    prev_text: str | None,
    conv_texts: list[str],
) -> dict[str, Any]:
    """Compute all features for a single turn row."""
    text = row.get("original_text", "")
    tokens = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_tokens = max(len(tokens), 1)

    pronoun_r = pronoun_ratios(tokens)

    features: dict[str, Any] = {
        # Lexical
        "lexical_diversity":          lexical_diversity(tokens),
        "vocabulary_level":           vocabulary_level(tokens),
        "avg_word_length":            round(sum(len(w) for w in tokens) / n_tokens, 2),
        # Syntax
        "sentence_count":             len(sentences),
        "avg_sentence_length":        round(n_tokens / max(len(sentences), 1), 2),
        "readability_score":          readability_score(text),
        "grammar_score":              grammar_score(text),
        "imperative_ratio":           imperative_ratio(sentences),
        # Pragmatic
        "question_ratio":             round(text.count("?") / max(len(sentences), 1), 4),
        "politeness_markers":         politeness_markers_count(text),
        "formality_score":            formality_score(text),
        "uncertainty_markers":        uncertainty_markers_count(text),
        "negation_ratio":             negation_ratio(tokens),
        # Semantic
        "sentiment_polarity":         sentiment_polarity(text),
        "subjectivity_score":         subjectivity_score(text),
        "coherence_score":            coherence_score(text, prev_text),
        # Safety / emotion / reasoning
        "safety_flags":               safety_flags_count(text),
        "empathy_signals":            empathy_signals_count(text),
        "reasoning_markers":          reasoning_markers_count(text),
        # Pronouns
        "first_person_ratio":         pronoun_r["first_person_ratio"],
        "second_person_ratio":        pronoun_r["second_person_ratio"],
        "third_person_ratio":         pronoun_r["third_person_ratio"],
        # Entity / density
        "entity_density":             entity_density(text, n_tokens),
        # Conversation-level
        "turn_length_variance":       _turn_length_variance(conv_texts),
    }
    return features


# ---------------------------------------------------------------------------
# DataFrame-level extraction
# ---------------------------------------------------------------------------

@timeit
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns to an existing preprocessed DataFrame."""
    feature_rows: list[dict[str, Any]] = []

    for cid, group in df.groupby("conversation_id"):
        group_sorted = group.sort_values("turn_index")
        conv_texts = group_sorted["original_text"].tolist()
        prev_text: str | None = None
        for _, row in group_sorted.iterrows():
            feats = extract_features_for_turn(row.to_dict(), prev_text, conv_texts)
            feature_rows.append({"_row_idx": row.name, **feats})
            prev_text = row["original_text"]

    feat_df = pd.DataFrame(feature_rows).set_index("_row_idx")
    # Merge back
    result = df.copy()
    for col in feat_df.columns:
        result[col] = feat_df[col]
    logger.info("Extracted %d features for %d rows", len(feat_df.columns), len(df))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/processed/conversations_preprocessed.parquet")
    p.add_argument("--output", default="data/processed/conversation_features.parquet")
    args = p.parse_args()

    df_in = pd.read_parquet(args.input)
    df_out = extract_features(df_in)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out, index=False)
    print(f"✅ Features saved to {args.output}  shape={df_out.shape}")
