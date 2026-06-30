"""Tests for data_loader.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import tempfile
import os

from src.data_loader import (
    _clean_facet_name,
    _is_header_or_meta,
    _assign_category,
    _importance,
    load_facets_from_csv,
    CATEGORY_DISPLAY,
)


class TestCleanFacetName:

    def test_removes_trailing_colon(self):
        assert _clean_facet_name("Democratic Leadership:") == "Democratic Leadership"

    def test_removes_numeric_prefix(self):
        assert _clean_facet_name("800. Sufi practice: Sufi retreat") == "Sufi practice: Sufi retreat"

    def test_normalizes_whitespace(self):
        assert _clean_facet_name("  Big   heartedness  ") == "Big heartedness"

    def test_empty_string(self):
        assert _clean_facet_name("") == ""


class TestIsHeaderOrMeta:

    def test_detects_facets_header(self):
        assert _is_header_or_meta("Facets") is True

    def test_detects_subcomponents(self):
        assert _is_header_or_meta("Listening Skills Subcomponents:") is True

    def test_normal_facet_passes(self):
        assert _is_header_or_meta("Empathy") is False
        assert _is_header_or_meta("Honesty") is False

    def test_behavioral_tendencies_header(self):
        assert _is_header_or_meta("Behavioral Tendencies and Subcomponents:") is True


class TestAssignCategory:

    def test_grammar_is_linguistic(self):
        assert _assign_category("Grammar Correctness") == "linguistic_quality"

    def test_politeness_is_pragmatics(self):
        assert _assign_category("Politeness Markers") == "pragmatics"

    def test_harmful_is_safety(self):
        assert _assign_category("Harmfulness Score") == "safety"

    def test_empathy_is_emotion(self):
        assert _assign_category("Empathy and Compassion") == "emotion_empathy"

    def test_honesty_is_behavioral(self):
        assert _assign_category("Honesty and Integrity") == "behavioral_traits"

    def test_reasoning_is_intelligence(self):
        assert _assign_category("Critical Reasoning Ability") == "intelligence_reasoning"

    def test_unknown_defaults(self):
        result = _assign_category("Xylophone Playing Skill")
        assert result in CATEGORY_DISPLAY.keys()


class TestImportance:

    def test_high_importance_facets(self):
        assert _importance("Safety") == 1.0
        assert _importance("Empathy") == 1.0
        assert _importance("empathy_level") == 0.85
        assert _importance("Grammar") == 0.80  # from grammar keyword

    def test_importance_in_range(self):
        for name in ["Risktaking", "Merriness", "Openness", "Cunningness"]:
            imp = _importance(name)
            assert 0.0 <= imp <= 1.0, f"Importance out of range for {name}: {imp}"


class TestLoadFacetsFromCSV:

    @pytest.fixture
    def sample_csv(self, tmp_path):
        csv_content = """Facets
Empathy
Grammar Correctness
Politeness:
Honesty
Harmfulness
Logical Reasoning
Creativity
Warmheartedness
Safety Compliance
Common-sense
Listening Skills Subcomponents:
"""
        csv_file = tmp_path / "test_facets.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    def test_loads_correct_count(self, sample_csv):
        facets, _ = load_facets_from_csv(sample_csv, target_count=6)
        assert len(facets) == 6

    def test_facet_structure(self, sample_csv):
        facets, _ = load_facets_from_csv(sample_csv, target_count=6)
        for f in facets:
            assert "id" in f
            assert "name" in f
            assert "category" in f
            assert "category_display" in f
            assert "weight" in f
            assert "importance" in f

    def test_ids_are_sequential(self, sample_csv):
        facets, _ = load_facets_from_csv(sample_csv, target_count=6)
        ids = [f["id"] for f in facets]
        assert ids == list(range(1, len(facets) + 1))

    def test_relationships_built(self, sample_csv):
        _, relationships = load_facets_from_csv(sample_csv, target_count=6)
        assert isinstance(relationships, dict)

    def test_real_csv_if_exists(self):
        """Test against actual provided facets CSV if available."""
        real_csv = Path("data/raw/facets_assignment.csv")
        if not real_csv.exists():
            pytest.skip("Real facets CSV not available")
        facets, relationships = load_facets_from_csv(real_csv, target_count=300)
        assert len(facets) == 300
        cats = {f["category"] for f in facets}
        assert len(cats) == 6
