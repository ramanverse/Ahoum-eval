"""
Ahoum Conversation Evaluation Dashboard — Streamlit Multi-Page App
===================================================================
Entry point: ui/app.py

This page serves as the Home / Landing page, which will automatically show
the pages under ui/pages/ in the sidebar navigation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Global style loading utility
# ---------------------------------------------------------------------------
def load_style():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

      html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
      }
      .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e8e8f0;
      }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(12px);
      }

      /* Cards */
      .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }
      .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
      }
      .metric-number {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        margin-top: 0.2rem;
      }

      /* Hero */
      .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
      }
      .hero-subtitle {
        font-size: 1.15rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
      }

      /* Badges */
      .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
      }
      .badge-high      { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
      .badge-medium    { background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid #fbbf24; }
      .badge-low       { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid #f87171; }
      .badge-edge_case { background: rgba(167,139,250,0.2); color: #a78bfa; border: 1px solid #a78bfa; }

      /* Score bars */
      .score-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        gap: 0.8rem;
      }
      .score-label { width: 180px; font-size: 0.82rem; color: rgba(255,255,255,0.7); }
      .score-bar-bg {
        flex: 1;
        height: 8px;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        overflow: hidden;
      }
      .score-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        transition: width 0.6s ease;
      }
      .score-val { width: 30px; font-size: 0.82rem; font-weight: 600; color: #60a5fa; }

      /* Buttons */
      .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: opacity 0.2s;
      }
      .stButton > button:hover { opacity: 0.85; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders (Shared/Cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_facets_config() -> dict:
    facets_path = _ROOT / "data" / "processed" / "facets_cleaned.json"
    if facets_path.exists():
        with open(facets_path) as fh:
            return json.load(fh)
    return {"facets": [], "categories": []}


@st.cache_data(ttl=3600)
def load_sample_evals() -> list:
    p = _ROOT / "data" / "examples" / "sample_evaluations_50.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return []


@st.cache_data(ttl=3600)
def load_conversations() -> list:
    p = _ROOT / "data" / "raw" / "generated_conversations.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return []


def _score_bar(label: str, score: float, max_score: float = 5.0) -> str:
    pct = (score / max_score) * 100
    return f"""
    <div class="score-row">
      <div class="score-label">{label}</div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct:.1f}%"></div>
      </div>
      <div class="score-val">{score:.1f}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Main Entry Point / Home UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Ahoum Conversation Evaluation",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_style()

    # Title & Subtitle
    st.markdown("""
    <div class="hero-title">Ahoum Conversation<br>Evaluation System</div>
    <div class="hero-subtitle">
      Production-ready multi-facet scoring benchmark for AI conversations.<br>
      300 behavioral & linguistic facets · Open-weight LLMs · Confidence scores.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Load specs
    facets_cfg = load_facets_config()
    sample_evals = load_sample_evals()
    convs = load_conversations()

    n_facets = len(facets_cfg.get("facets", []))
    n_evals = len(sample_evals)
    n_convs = len(convs)
    n_cats = len(facets_cfg.get("categories", []))

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{n_facets}</div><div class="metric-label">Total Facets</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{n_cats}</div><div class="metric-label">Categories</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{n_convs}</div><div class="metric-label">Conversations</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{n_evals}</div><div class="metric-label">Sample Evals</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_arch, col_features = st.columns([3, 2])
    with col_arch:
        st.subheader("🏗️ System Architecture")
        st.markdown("""
        ```
        CSV Facets ──┐
                     ├─→ Data Loader ──→ Facet Config (300 facets)
        Conversations┘
              │
              ▼
        Preprocessing  ──→  Parquet (cleaned, tokenized)
              │
              ▼
        Feature Extractor ──→ 25+ linguistic features per turn
              │
              ├──[Mode: Feature]──→ Facet Mapper ─────────────┐
              │                    (weighted scoring)         │
              ├──[Mode: LLM]──────→ Model Manager ────────────┤
              │                    (Mistral-7B)               │
              └──[Mode: Hybrid]───→ Combine scores ───────────┤
                                                              ▼
                                                   300-Facet Score Matrix
                                                   + Confidence Intervals
                                                              │
                                              ┌───────────────┴────────────┐
                                              ▼                            ▼
                                       Streamlit UI                  FastAPI /REST
                                       (This Dashboard)              (Batch Scoring)
        ```
        """)

    with col_features:
        st.subheader("✨ Key Features")
        features_list = [
            ("🔬", "300 Behavioral Facets", "Spanning 6 categories from linguistic quality to reasoning"),
            ("🤖", "Open-Weight LLMs", "Mistral-7B, Llama-2-7B, Qwen2-7B — no closed APIs"),
            ("📊", "Confidence Scores", "Every prediction includes uncertainty quantification"),
            ("⚡", "3 Scoring Modes", "Feature-based (fast), LLM, or Hybrid"),
            ("🐳", "Docker Ready", "One-command deployment"),
            ("🧪", "Full Test Suite", "Unit + integration + performance tests"),
        ]
        for icon, title, desc in features_list:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.6rem;">
              <div style="display:flex; gap:0.7rem; align-items:start;">
                <div style="font-size:1.3rem;">{icon}</div>
                <div>
                  <div style="font-weight:600; font-size:0.9rem;">{title}</div>
                  <div style="font-size:0.78rem; color:rgba(255,255,255,0.55); margin-top:0.1rem;">{desc}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Quick start
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🚀 Quick Start")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Install**")
        st.code("pip install -r requirements.txt\npython -m spacy download en_core_web_sm", language="bash")
    with c2:
        st.markdown("**2. Generate Data**")
        st.code("python src/data_generator.py\npython src/data_loader.py\npython src/generate_sample_evals.py", language="bash")
    with c3:
        st.markdown("**3. Score**")
        st.code("python src/scorer.py --mode feature\nstreamlit run ui/app.py", language="bash")

    # Facet categories preview
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📂 Facet Categories")
    cats = facets_cfg.get("categories", [])
    cat_cols = st.columns(min(len(cats), 3))
    cat_colors = ["#a78bfa", "#60a5fa", "#f87171", "#fbbf24", "#34d399", "#fb7185"]
    for i, cat in enumerate(cats):
        with cat_cols[i % 3]:
            color = cat_colors[i % len(cat_colors)]
            st.markdown(f"""
            <div class="metric-card" style="border-left: 3px solid {color}; margin-bottom:0.8rem;">
              <div style="font-weight:600; color:{color};">{cat}</div>
              <div style="font-size:0.78rem; color:rgba(255,255,255,0.5); margin-top:0.2rem;">50 facets</div>
            </div>
            """, unsafe_allow_html=True)

    st.sidebar.success("Select a page above to explore evaluation details.")


if __name__ == "__main__":
    # If run directly as a script, execute main.
    # Note: Streamlit set_page_config is called inside main.
    main()
