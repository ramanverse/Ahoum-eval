"""
preprocessing.py — Clean and normalize conversations, produce Parquet output.

Input:  data/raw/generated_conversations.json
Output: data/processed/conversations_preprocessed.parquet

Usage:
    python src/preprocessing.py \
        --input  data/raw/generated_conversations.json \
        --output data/processed/conversations_preprocessed.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, load_config, set_seed, timeit

logger = get_logger("preprocessing")

# ---------------------------------------------------------------------------
# NLTK bootstrap (download if absent)
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize

    def _ensure_nltk():
        for resource in ["punkt", "punkt_tab", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

    _ensure_nltk()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not installed — falling back to simple tokenization")


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s\.,!?;:\-\'\"\(\)]")


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_special: bool = True,
) -> str:
    """Unicode-normalize, optionally lowercase, collapse whitespace."""
    # Unicode normalization (NFKC: canonical decomposition + compat composition)
    text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    if remove_special:
        text = _SPECIAL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, method: str = "nltk") -> list[str]:
    """Tokenize text into word tokens."""
    if method == "nltk" and NLTK_AVAILABLE:
        return word_tokenize(text)
    # Simple whitespace fallback
    return text.split()


def sentence_split(text: str) -> list[str]:
    """Split text into sentences."""
    if NLTK_AVAILABLE:
        return sent_tokenize(text)
    return re.split(r"[.!?]+", text)


# ---------------------------------------------------------------------------
# Conversation-level metadata
# ---------------------------------------------------------------------------

def _speaker_balance(turns: list[dict]) -> dict[str, float]:
    total = len(turns)
    if total == 0:
        return {"user_pct": 0.0, "assistant_pct": 0.0}
    user_turns = sum(1 for t in turns if t.get("speaker") == "user")
    return {
        "user_pct": round(user_turns / total, 3),
        "assistant_pct": round(1 - user_turns / total, 3),
    }


def _sentiment_trend(turns: list[dict]) -> str:
    """Naively classify overall sentiment based on presence of keywords."""
    all_text = " ".join(t.get("text", "") for t in turns).lower()
    positive_kw = {"thank", "great", "help", "excellent", "appreciate", "perfect", "good", "happy"}
    negative_kw = {"sorry", "unfortunately", "cannot", "problem", "issue", "wrong", "angry", "fail", "not"}
    pos = sum(1 for w in positive_kw if w in all_text)
    neg = sum(1 for w in negative_kw if w in all_text)
    if pos > neg + 2:
        return "positive"
    elif neg > pos + 2:
        return "negative"
    return "neutral"


def _topic_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract simple top-frequency keywords (no stopwords) from text."""
    stopwords = {
        "the", "a", "an", "is", "it", "in", "on", "for", "of", "and", "or",
        "to", "i", "you", "we", "they", "this", "that", "my", "your", "be",
        "with", "have", "do", "can", "will", "was", "are", "me", "so", "but",
    }
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=lambda k: freq[k], reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Turn-level features (lightweight, no heavy ML)
# ---------------------------------------------------------------------------

def extract_turn_features(text: str, processed_text: str) -> dict[str, Any]:
    tokens = tokenize(processed_text)
    sentences = sentence_split(text)
    n_tokens = max(len(tokens), 1)
    n_sentences = max(len(sentences), 1)

    unique_tokens = set(tokens)
    question_count = text.count("?")
    exclamation_count = text.count("!")

    return {
        "char_count": len(text),
        "word_count": n_tokens,
        "sentence_count": n_sentences,
        "unique_word_count": len(unique_tokens),
        "avg_word_length": round(
            sum(len(w) for w in tokens) / n_tokens, 2
        ),
        "avg_sentence_length_words": round(n_tokens / n_sentences, 2),
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "has_questions": question_count > 0,
    }


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

@timeit
def preprocess_conversations(
    conversations: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Convert a list of conversation dicts into a flat Parquet-ready DataFrame.

    Columns
    -------
    conversation_id, turn_index, speaker, original_text, processed_text,
    char_count, word_count, sentence_count, unique_word_count,
    avg_word_length, avg_sentence_length_words, question_count,
    exclamation_count, has_questions,
    conv_num_turns, conv_total_words, conv_sentiment, conv_user_pct,
    conv_assistant_pct, conv_topic_keywords, domain, quality_level,
    created_at
    """
    if config is None:
        config = {}
    pp_cfg = config.get("preprocessing", {})
    lowercase = pp_cfg.get("lowercase", True)
    remove_special = pp_cfg.get("handle_special_chars", True)
    tokenizer_method = pp_cfg.get("tokenizer", "nltk")

    rows: list[dict[str, Any]] = []

    for conv in conversations:
        cid = conv.get("conversation_id", "unknown")
        turns = conv.get("turns", [])
        domain = conv.get("domain", "")
        quality = conv.get("quality_level", "")
        created_at = conv.get("created_at", "")

        # Conversation-level metadata
        all_text = " ".join(t.get("text", "") for t in turns)
        balance = _speaker_balance(turns)
        sentiment = _sentiment_trend(turns)
        topic_kws = _topic_keywords(all_text)
        conv_total_words = sum(len(t.get("text", "").split()) for t in turns)

        for idx, turn in enumerate(turns):
            raw_text = turn.get("text", "")
            processed = normalize_text(raw_text, lowercase=lowercase, remove_special=remove_special)
            feats = extract_turn_features(raw_text, processed)

            row: dict[str, Any] = {
                "conversation_id":              cid,
                "turn_index":                   idx,
                "speaker":                      turn.get("speaker", "unknown"),
                "original_text":                raw_text,
                "processed_text":               processed,
                # Turn-level features
                **feats,
                # Conversation-level metadata
                "conv_num_turns":               len(turns),
                "conv_total_words":             conv_total_words,
                "conv_sentiment":               sentiment,
                "conv_user_pct":                balance["user_pct"],
                "conv_assistant_pct":           balance["assistant_pct"],
                "conv_topic_keywords":          "|".join(topic_kws),
                # Labels
                "domain":                       domain,
                "quality_level":                quality,
                "created_at":                   created_at,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Preprocessed %d rows from %d conversations",
        len(df),
        len(conversations),
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess conversation JSON → Parquet")
    p.add_argument("--input",  default="data/raw/generated_conversations.json")
    p.add_argument("--output", default="data/processed/conversations_preprocessed.parquet")
    return p.parse_args()


if __name__ == "__main__":
    import json

    from src.utils import ensure_dir

    args = _parse_args()
    cfg = load_config()
    set_seed(cfg.get("project", {}).get("seed", 42))

    with open(args.input, "r", encoding="utf-8") as fh:
        convs = json.load(fh)

    df = preprocess_conversations(convs, cfg)
    out = Path(args.output)
    ensure_dir(out.parent)
    df.to_parquet(out, index=False)
    logger.info("Saved preprocessed data to %s  shape=%s", out, df.shape)
    print(f"✅ Preprocessed {len(df)} rows saved to {args.output}")
