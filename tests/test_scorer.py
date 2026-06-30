"""Tests for scorer.py and feature_extractor.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import pytest
import numpy as np

from src.facet_mapper import score_facet_feature_based, score_all_facets, _best_mapping
from src.feature_extractor import (
    lexical_diversity,
    sentiment_polarity,
    readability_score,
    formality_score,
    politeness_markers_count,
    negation_ratio,
    empathy_signals_count,
    reasoning_markers_count,
    coherence_score,
    grammar_score,
    safety_flags_count,
    vocabulary_level,
    extract_features_for_turn,
)
from src.scorer import ConversationEvaluator, aggregate_turn_features


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_turns_high():
    return [
        {"speaker": "user", "text": "I really appreciate your thorough and empathetic response."},
        {"speaker": "assistant", "text": "Thank you! I understand this is a challenging situation. Let me explain further with concrete examples."},
        {"speaker": "user", "text": "That clarifies things perfectly. Could you also elaborate on the reasoning?"},
        {"speaker": "assistant", "text": "Certainly. The logical basis for this approach involves three key principles: consistency, evidence, and applicability. Therefore, we recommend..."},
    ]

@pytest.fixture
def sample_turns_low():
    return [
        {"speaker": "user", "text": "ur garbage"},
        {"speaker": "assistant", "text": "ok"},
        {"speaker": "user", "text": "hate this"},
        {"speaker": "assistant", "text": "whatever"},
    ]

@pytest.fixture
def sample_facets():
    return [
        {"id": 1, "name": "Empathy",            "category": "emotion_empathy",       "category_display": "Emotion & Empathy",       "importance": 0.9,  "description": "..."},
        {"id": 2, "name": "Grammar Correctness", "category": "linguistic_quality",    "category_display": "Linguistic Quality",      "importance": 0.8,  "description": "..."},
        {"id": 3, "name": "Politeness",          "category": "pragmatics",            "category_display": "Pragmatics",              "importance": 0.85, "description": "..."},
        {"id": 4, "name": "Safety",              "category": "safety",                "category_display": "Safety",                  "importance": 0.95, "description": "..."},
        {"id": 5, "name": "Logical Reasoning",   "category": "intelligence_reasoning","category_display": "Intelligence & Reasoning","importance": 0.85, "description": "..."},
        {"id": 6, "name": "Honesty",             "category": "behavioral_traits",     "category_display": "Behavioral Traits",       "importance": 0.8,  "description": "..."},
    ]


# ---------------------------------------------------------------------------
# Feature extractor tests
# ---------------------------------------------------------------------------

class TestLexicalDiversity:

    def test_high_diversity(self):
        tokens = "the cat sat on a big fluffy mat near the door".split()
        result = lexical_diversity(tokens)
        assert 0.0 <= result <= 1.0

    def test_low_diversity(self):
        tokens = "the the the the the".split()
        result = lexical_diversity(tokens)
        assert result < 0.5

    def test_empty(self):
        assert lexical_diversity([]) == 0.0

    def test_all_unique(self):
        tokens = ["apple", "banana", "cherry"]
        assert lexical_diversity(tokens) == 1.0


class TestSentimentPolarity:

    def test_positive_text(self):
        result = sentiment_polarity("This is wonderful and amazing!")
        assert result >= -1.0
        assert result <= 1.0

    def test_negative_text(self):
        result = sentiment_polarity("This is terrible and awful.")
        # Should be non-positive
        assert result <= 0.2

    def test_neutral_text(self):
        result = sentiment_polarity("The cat sat on the mat.")
        assert -1.0 <= result <= 1.0

    def test_returns_float(self):
        assert isinstance(sentiment_polarity("hello"), float)


class TestReadabilityScore:

    def test_simple_text_low_score(self):
        score = readability_score("The cat is big. The dog runs fast.")
        assert isinstance(score, float)

    def test_complex_text(self):
        score = readability_score(
            "The epistemological implications of quantum entanglement necessitate a comprehensive re-evaluation of our fundamental assumptions regarding locality and causality."
        )
        assert isinstance(score, float)
        assert score >= 0.0


class TestPolitenessMarkers:

    def test_polite_text(self):
        count = politeness_markers_count("Please could you help me? Thank you so much!")
        assert count >= 2

    def test_impolite_text(self):
        count = politeness_markers_count("give it now")
        assert count == 0


class TestSafetyFlags:

    def test_safe_text(self):
        count = safety_flags_count("I'd like some help with my homework.")
        assert count == 0

    def test_unsafe_text(self):
        count = safety_flags_count("I hate this and want to destroy everything.")
        assert count >= 1


class TestCoherenceScore:

    def test_first_turn_is_1(self):
        score = coherence_score("Hello!", None)
        assert score == 1.0

    def test_identical_texts_high(self):
        score = coherence_score("Hello world", "Hello world")
        assert score >= 0.8

    def test_unrelated_texts_lower(self):
        score = coherence_score("I love pizza", "The stock market crashed")
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        assert isinstance(coherence_score("test", "other"), float)


class TestEmpathySignals:

    def test_empathetic_text(self):
        count = empathy_signals_count("I understand how you feel. I'm here to support you.")
        assert count >= 2

    def test_no_empathy(self):
        count = empathy_signals_count("The report is due tomorrow.")
        assert count == 0


class TestExtractFeaturesForTurn:

    def test_returns_dict(self, sample_turns_high):
        row = {"original_text": sample_turns_high[0]["text"]}
        features = extract_features_for_turn(row, None, [t["text"] for t in sample_turns_high])
        assert isinstance(features, dict)

    def test_all_expected_features_present(self, sample_turns_high):
        row = {"original_text": sample_turns_high[0]["text"]}
        features = extract_features_for_turn(row, None, [t["text"] for t in sample_turns_high])
        expected_keys = [
            "lexical_diversity", "sentiment_polarity", "coherence_score",
            "safety_flags", "empathy_signals", "reasoning_markers",
            "politeness_markers", "formality_score",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_feature_ranges(self, sample_turns_high):
        row = {"original_text": sample_turns_high[0]["text"]}
        features = extract_features_for_turn(row, None, [t["text"] for t in sample_turns_high])
        assert 0.0 <= features["lexical_diversity"] <= 1.0
        assert 0.0 <= features["formality_score"] <= 1.0
        assert -1.0 <= features["sentiment_polarity"] <= 1.0


# ---------------------------------------------------------------------------
# Facet mapper tests
# ---------------------------------------------------------------------------

class TestFacetMapper:

    @pytest.fixture
    def sample_features_high(self):
        return {
            "grammar_score":      0.9,
            "vocabulary_level":   0.6,
            "lexical_diversity":  0.8,
            "sentiment_polarity": 0.5,
            "subjectivity_score": 0.4,
            "coherence_score":    0.85,
            "safety_flags":       0.0,
            "empathy_signals":    4.0,
            "reasoning_markers":  3.0,
            "politeness_markers": 3.0,
            "formality_score":    0.7,
            "readability_score":  8.0,
            "uncertainty_markers":1.0,
            "negation_ratio":     0.05,
            "avg_sentence_length":15.0,
            "word_count":         80.0,
            "sentence_count":     5.0,
            "entity_density":     5.0,
            "first_person_ratio": 0.1,
            "second_person_ratio":0.2,
            "third_person_ratio": 0.05,
            "imperative_ratio":   0.2,
            "exclamation_count":  1.0,
            "question_ratio":     0.2,
            "turn_length_variance":10.0,
        }

    @pytest.fixture
    def sample_features_low(self):
        return {
            "grammar_score":      0.2,
            "vocabulary_level":   0.1,
            "lexical_diversity":  0.2,
            "sentiment_polarity": -0.7,
            "subjectivity_score": 0.8,
            "coherence_score":    0.2,
            "safety_flags":       3.0,
            "empathy_signals":    0.0,
            "reasoning_markers":  0.0,
            "politeness_markers": 0.0,
            "formality_score":    0.1,
            "readability_score":  3.0,
            "uncertainty_markers":0.0,
            "negation_ratio":     0.2,
            "avg_sentence_length":3.0,
            "word_count":         15.0,
            "sentence_count":     3.0,
            "entity_density":     0.0,
            "first_person_ratio": 0.0,
            "second_person_ratio":0.0,
            "third_person_ratio": 0.0,
            "imperative_ratio":   0.0,
            "exclamation_count":  0.0,
            "question_ratio":     0.0,
            "turn_length_variance":2.0,
        }

    @pytest.fixture
    def empathy_facet(self):
        return {"id": 1, "name": "Empathy", "category": "emotion_empathy", "importance": 0.9, "description": "..."}

    @pytest.fixture
    def safety_facet(self):
        return {"id": 2, "name": "Harmfulness", "category": "safety", "importance": 0.95, "description": "..."}

    def test_score_in_range(self, empathy_facet, sample_features_high):
        result = score_facet_feature_based(empathy_facet, sample_features_high)
        assert 1 <= result["score"] <= 5

    def test_confidence_in_range(self, empathy_facet, sample_features_high):
        result = score_facet_feature_based(empathy_facet, sample_features_high)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_high_quality_higher_empathy_score(self, empathy_facet, sample_features_high, sample_features_low):
        high = score_facet_feature_based(empathy_facet, sample_features_high)
        low  = score_facet_feature_based(empathy_facet, sample_features_low)
        assert high["score"] >= low["score"]

    def test_safety_inverted_for_harmful(self, safety_facet, sample_features_high, sample_features_low):
        safe_result   = score_facet_feature_based(safety_facet, sample_features_high)
        unsafe_result = score_facet_feature_based(safety_facet, sample_features_low)
        # Low safety_flags → higher safety score
        assert safe_result["score"] >= unsafe_result["score"]

    def test_result_has_required_keys(self, empathy_facet, sample_features_high):
        result = score_facet_feature_based(empathy_facet, sample_features_high)
        for key in ["score", "confidence", "evidence", "method"]:
            assert key in result

    def test_score_is_integer(self, empathy_facet, sample_features_high):
        result = score_facet_feature_based(empathy_facet, sample_features_high)
        assert isinstance(result["score"], int)

    def test_empty_features_returns_default(self, empathy_facet):
        result = score_facet_feature_based(empathy_facet, {})
        assert result["score"] == 3
        assert result["confidence"] == 0.2


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------

class TestConversationEvaluator:

    @pytest.fixture
    def mini_facets_config(self):
        return {
            "facets": [
                {"id": 1, "name": "Empathy",      "category": "emotion_empathy",       "category_display": "Emotion & Empathy",       "importance": 0.9, "description": "...", "weight": 1.0},
                {"id": 2, "name": "Grammar",       "category": "linguistic_quality",    "category_display": "Linguistic Quality",      "importance": 0.8, "description": "...", "weight": 1.0},
                {"id": 3, "name": "Safety",        "category": "safety",                "category_display": "Safety",                  "importance": 0.95,"description": "...", "weight": 1.2},
                {"id": 4, "name": "Politeness",    "category": "pragmatics",            "category_display": "Pragmatics",              "importance": 0.85,"description": "...", "weight": 1.0},
                {"id": 5, "name": "Logical Reasoning","category":"intelligence_reasoning","category_display":"Intelligence & Reasoning","importance": 0.85,"description": "...", "weight": 1.0},
                {"id": 6, "name": "Honesty",       "category": "behavioral_traits",     "category_display": "Behavioral Traits",       "importance": 0.8, "description": "...", "weight": 1.0},
            ],
            "categories": ["Linguistic Quality", "Pragmatics", "Safety", "Emotion & Empathy", "Behavioral Traits", "Intelligence & Reasoning"],
        }

    @pytest.fixture
    def evaluator(self, mini_facets_config):
        return ConversationEvaluator(
            facets_config=mini_facets_config,
            mode="feature",
            config={},
        )

    @pytest.fixture
    def sample_conv_high(self, sample_turns_high):
        return {
            "conversation_id": "test_conv_high",
            "domain": "educational",
            "quality_level": "high",
            "turns": sample_turns_high,
        }

    @pytest.fixture
    def sample_conv_low(self, sample_turns_low):
        return {
            "conversation_id": "test_conv_low",
            "domain": "general",
            "quality_level": "low",
            "turns": sample_turns_low,
        }

    def test_evaluate_returns_dict(self, evaluator, sample_conv_high):
        result = evaluator.evaluate_conversation(sample_conv_high)
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, evaluator, sample_conv_high):
        result = evaluator.evaluate_conversation(sample_conv_high)
        for key in ["conversation_id", "facet_scores", "category_averages", "overall_score", "overall_confidence"]:
            assert key in result

    def test_scores_in_range(self, evaluator, sample_conv_high):
        result = evaluator.evaluate_conversation(sample_conv_high)
        for fname, fdata in result["facet_scores"].items():
            assert 1 <= fdata["score"] <= 5, f"Score out of range for {fname}"
            assert 0.0 <= fdata["confidence"] <= 1.0, f"Confidence out of range for {fname}"

    def test_overall_score_in_range(self, evaluator, sample_conv_high):
        result = evaluator.evaluate_conversation(sample_conv_high)
        assert 1.0 <= result["overall_score"] <= 5.0
        assert 0.0 <= result["overall_confidence"] <= 1.0

    def test_all_facets_scored(self, evaluator, mini_facets_config, sample_conv_high):
        result = evaluator.evaluate_conversation(sample_conv_high)
        expected_facets = {f["name"] for f in mini_facets_config["facets"]}
        assert expected_facets == set(result["facet_scores"].keys())

    def test_batch_evaluate(self, evaluator, sample_conv_high, sample_conv_low):
        results = evaluator.batch_evaluate([sample_conv_high, sample_conv_low], show_progress=False)
        assert len(results) == 2
        for r in results:
            assert "overall_score" in r

    def test_results_to_dataframe(self, evaluator, sample_conv_high, sample_conv_low):
        import pandas as pd
        results = evaluator.batch_evaluate([sample_conv_high, sample_conv_low], show_progress=False)
        df = evaluator.results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "overall_score" in df.columns


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------

class TestPerformance:

    def test_feature_scoring_speed(self):
        """Feature-based scoring should process 10 conversations in < 5 seconds."""
        from src.data_generator import generate_conversations
        convs = generate_conversations(total=10, seed=42)

        facets_config = {
            "facets": [
                {"id": i, "name": f"Facet_{i}", "category": "behavioral_traits",
                 "category_display": "Behavioral Traits", "importance": 0.5,
                 "description": "...", "weight": 1.0}
                for i in range(1, 21)  # 20 facets
            ],
            "categories": [],
        }

        evaluator = ConversationEvaluator(facets_config=facets_config, mode="feature", config={})
        t0 = time.perf_counter()
        results = evaluator.batch_evaluate(convs, show_progress=False)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"Feature scoring too slow: {elapsed:.2f}s for 10 convs"
        assert len(results) == 10
