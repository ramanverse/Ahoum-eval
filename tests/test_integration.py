"""
Integration tests — runs the full pipeline end-to-end on synthetic data.
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd

from src.data_generator import generate_conversations
from src.data_loader import load_facets_from_csv, build_facets_config
from src.preprocessing import preprocess_conversations
from src.feature_extractor import extract_features
from src.scorer import ConversationEvaluator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_conversations():
    return generate_conversations(total=5, seed=123)

@pytest.fixture(scope="module")
def mini_facets():
    return {
        "facets": [
            {"id": i, "name": n, "category": c, "category_display": cd,
             "importance": 0.8, "description": "Test facet.", "weight": 1.0}
            for i, (n, c, cd) in enumerate([
                ("Empathy",          "emotion_empathy",       "Emotion & Empathy"),
                ("Grammar",          "linguistic_quality",    "Linguistic Quality"),
                ("Safety",           "safety",                "Safety"),
                ("Politeness",       "pragmatics",            "Pragmatics"),
                ("Logical Reasoning","intelligence_reasoning","Intelligence & Reasoning"),
                ("Honesty",          "behavioral_traits",     "Behavioral Traits"),
            ], start=1)
        ],
        "categories": ["Emotion & Empathy", "Linguistic Quality", "Safety", "Pragmatics", "Intelligence & Reasoning", "Behavioral Traits"],
    }

@pytest.fixture(scope="module")
def evaluator(mini_facets):
    return ConversationEvaluator(facets_config=mini_facets, mode="feature", config={})


# ---------------------------------------------------------------------------
# Full Pipeline Test
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_step1_generation(self, tiny_conversations):
        assert len(tiny_conversations) == 5
        for conv in tiny_conversations:
            assert "turns" in conv
            assert len(conv["turns"]) >= 2

    def test_step2_preprocessing(self, tiny_conversations):
        df = preprocess_conversations(tiny_conversations)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "conversation_id" in df.columns
        assert "processed_text" in df.columns

    def test_step3_feature_extraction(self, tiny_conversations):
        df = preprocess_conversations(tiny_conversations)
        df_feats = extract_features(df)
        assert "lexical_diversity" in df_feats.columns
        assert "sentiment_polarity" in df_feats.columns
        assert "coherence_score" in df_feats.columns
        assert len(df_feats) == len(df)

    def test_step4_scoring(self, tiny_conversations, evaluator):
        results = evaluator.batch_evaluate(tiny_conversations, show_progress=False)
        assert len(results) == 5
        for r in results:
            assert "facet_scores" in r
            assert "overall_score" in r
            assert 1.0 <= r["overall_score"] <= 5.0

    def test_step5_dataframe_export(self, tiny_conversations, evaluator):
        results = evaluator.batch_evaluate(tiny_conversations, show_progress=False)
        df = evaluator.results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert "overall_score" in df.columns
        assert len(df) == len(tiny_conversations)

    def test_step6_json_serialization(self, tiny_conversations, evaluator):
        """Ensure results can be JSON-serialized (no numpy types)."""
        results = evaluator.batch_evaluate(tiny_conversations, show_progress=False)
        try:
            json_str = json.dumps(results, default=str)
            reloaded = json.loads(json_str)
            assert len(reloaded) == len(results)
        except Exception as exc:
            pytest.fail(f"JSON serialization failed: {exc}")


# ---------------------------------------------------------------------------
# Output Validation
# ---------------------------------------------------------------------------

class TestOutputValidation:

    def test_facet_scores_schema(self, tiny_conversations, evaluator):
        results = evaluator.batch_evaluate(tiny_conversations[:1], show_progress=False)
        result = results[0]
        for fname, fdata in result["facet_scores"].items():
            assert isinstance(fdata["score"], int), f"{fname}: score not int"
            assert isinstance(fdata["confidence"], float), f"{fname}: confidence not float"
            assert isinstance(fdata["evidence"], str), f"{fname}: evidence not str"
            assert 1 <= fdata["score"] <= 5
            assert 0.0 <= fdata["confidence"] <= 1.0

    def test_category_averages_all_present(self, tiny_conversations, evaluator, mini_facets):
        results = evaluator.batch_evaluate(tiny_conversations[:1], show_progress=False)
        result = results[0]
        expected_cats = set(mini_facets["categories"])
        actual_cats   = set(result["category_averages"].keys())
        assert expected_cats == actual_cats

    def test_num_facets_evaluated_correct(self, tiny_conversations, evaluator, mini_facets):
        results = evaluator.batch_evaluate(tiny_conversations[:1], show_progress=False)
        result = results[0]
        assert result["num_facets_evaluated"] == len(mini_facets["facets"])

    def test_evaluation_mode_recorded(self, tiny_conversations, evaluator):
        results = evaluator.batch_evaluate(tiny_conversations[:1], show_progress=False)
        assert results[0]["evaluation_mode"] == "feature"


# ---------------------------------------------------------------------------
# Reproducibility Test
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_input_same_output(self, mini_facets):
        conv = {
            "conversation_id": "repro_test",
            "domain": "test",
            "turns": [
                {"speaker": "user", "text": "Hello, I need help please."},
                {"speaker": "assistant", "text": "Of course! I'd be happy to assist you with empathy and care."},
            ],
        }
        ev1 = ConversationEvaluator(facets_config=mini_facets, mode="feature", config={})
        ev2 = ConversationEvaluator(facets_config=mini_facets, mode="feature", config={})
        r1 = ev1.evaluate_conversation(conv)
        r2 = ev2.evaluate_conversation(conv)
        assert r1["overall_score"] == r2["overall_score"]
        for fname in r1["facet_scores"]:
            assert r1["facet_scores"][fname]["score"] == r2["facet_scores"][fname]["score"]


# ---------------------------------------------------------------------------
# Data Loader Integration
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration:

    def test_real_csv_to_facets(self):
        csv_path = Path("data/raw/facets_assignment.csv")
        if not csv_path.exists():
            pytest.skip("Real CSV not available")

        facets, relationships = load_facets_from_csv(csv_path, target_count=300)
        assert len(facets) == 300

        # All categories present
        cats = {f["category"] for f in facets}
        assert len(cats) == 6

        # Each category has exactly 50 facets
        from collections import Counter
        cat_counts = Counter(f["category"] for f in facets)
        for cat, count in cat_counts.items():
            assert count == 50, f"Category {cat} has {count} facets (expected 50)"

    def test_build_facets_config_to_file(self, tmp_path):
        csv_path = Path("data/raw/facets_assignment.csv")
        if not csv_path.exists():
            pytest.skip("Real CSV not available")
        out_path = tmp_path / "facets_cleaned.json"
        config = build_facets_config(csv_path, out_path, target_count=50)
        assert out_path.exists()
        assert len(config["facets"]) == 50
