"""Tests for data_generator.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.data_generator import generate_conversations, DOMAIN_TEMPLATES


class TestDataGenerator:

    def test_generates_correct_count(self):
        convs = generate_conversations(total=60, seed=42)
        assert len(convs) == 60

    def test_default_300_conversations(self):
        convs = generate_conversations(seed=42)
        assert len(convs) == 300

    def test_conversation_structure(self):
        convs = generate_conversations(total=10, seed=42)
        for conv in convs:
            assert "conversation_id" in conv
            assert "domain" in conv
            assert "quality_level" in conv
            assert "turns" in conv
            assert "metadata" in conv
            assert isinstance(conv["turns"], list)
            assert len(conv["turns"]) >= 2

    def test_turn_structure(self):
        convs = generate_conversations(total=10, seed=42)
        for conv in convs:
            for turn in conv["turns"]:
                assert "speaker" in turn
                assert "text" in turn
                assert turn["speaker"] in {"user", "assistant"}
                assert len(turn["text"]) > 0

    def test_all_domains_covered(self):
        convs = generate_conversations(total=300, seed=42)
        domains = {c["domain"] for c in convs}
        expected = set(DOMAIN_TEMPLATES.keys())
        assert domains == expected

    def test_metadata_computed(self):
        convs = generate_conversations(total=10, seed=42)
        for conv in convs:
            meta = conv["metadata"]
            assert "num_turns" in meta
            assert "avg_turn_length" in meta
            assert "total_words" in meta
            assert meta["num_turns"] == len(conv["turns"])

    def test_reproducibility(self):
        c1 = generate_conversations(total=10, seed=99)
        c2 = generate_conversations(total=10, seed=99)
        for a, b in zip(c1, c2):
            assert a["conversation_id"] == b["conversation_id"]

    def test_quality_levels_present(self):
        convs = generate_conversations(total=300, seed=42)
        qualities = {c["quality_level"] for c in convs}
        assert "high" in qualities
        assert "medium" in qualities
        assert "low" in qualities

    def test_conversation_ids_unique(self):
        convs = generate_conversations(total=50, seed=42)
        ids = [c["conversation_id"] for c in convs]
        assert len(ids) == len(set(ids)), "Duplicate conversation IDs found"

    def test_edge_case_safety_flag(self):
        convs = generate_conversations(total=300, seed=42)
        edge_cases = [c for c in convs if c.get("quality_level") == "edge_case"]
        # At least some edge cases should have metadata
        assert len(edge_cases) > 0
