"""
Ahoum Conversation Evaluation Dashboard — Streamlit Multi-Page App
===================================================================
Entry point: ui/app.py

Pages:
  1. Home         — Project overview, stats
  2. Upload & Evaluate — Run scoring on new conversations
  3. Results Explorer  — Browse sample evaluations
  4. Analytics         — Correlations, trends, anomaly detection

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or ui/ directory
# ---------------------------------------------------------------------------
_THIS_DIR  = Path(__file__).resolve().parent
_ROOT      = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ahoum Conversation Evaluation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — dark theme + premium UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* ---------- Global ---------- */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e8e8f0;
  }

  /* ---------- Sidebar ---------- */
  section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
  }

  /* ---------- Cards ---------- */
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

  /* ---------- Hero ---------- */
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

  /* ---------- Badges ---------- */
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

  /* ---------- Score bars ---------- */
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

  /* ---------- Buttons ---------- */
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

  /* ---------- Tabs ---------- */
  .stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6);
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
  }

  /* ---------- DataFrames ---------- */
  .stDataFrame { border-radius: 12px; overflow: hidden; }

  /* ---------- Progress bars ---------- */
  .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #2563eb); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers / cached loaders
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


def _badge(quality: str) -> str:
    return f'<span class="badge badge-{quality}">{quality.replace("_", " ").title()}</span>'


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
# Sidebar Navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
      <div style="font-size:2.5rem;">🧠</div>
      <div style="font-size:1.1rem; font-weight:700; color:#a78bfa;">Ahoum</div>
      <div style="font-size:0.75rem; color:rgba(255,255,255,0.5);">Conversation Evaluator</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📤 Upload & Evaluate", "🔍 Results Explorer", "📊 Analytics"],
        label_visibility="collapsed",
    )

    st.divider()

    # Quick stats
    facets_cfg   = load_facets_config()
    sample_evals = load_sample_evals()
    convs        = load_conversations()

    n_facets    = len(facets_cfg.get("facets", []))
    n_evals     = len(sample_evals)
    n_convs     = len(convs)
    n_cats      = len(facets_cfg.get("categories", []))

    st.markdown(f"""
    <div style="display:grid; gap:0.7rem;">
      <div class="metric-card">
        <div class="metric-number">{n_facets}</div>
        <div class="metric-label">Facets Loaded</div>
      </div>
      <div class="metric-card">
        <div class="metric-number">{n_evals}</div>
        <div class="metric-label">Sample Evaluations</div>
      </div>
      <div class="metric-card">
        <div class="metric-number">{n_convs}</div>
        <div class="metric-label">Conversations</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("v1.0.0 · MIT License")


# ===========================================================================
# PAGE 1: HOME
# ===========================================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-title">Ahoum Conversation<br>Evaluation System</div>
    <div class="hero-subtitle">
      Production-ready multi-facet scoring benchmark for AI conversations.<br>
      300 behavioral & linguistic facets · Open-weight LLMs · Confidence scores.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top metrics row
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

    # Architecture overview
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


# ===========================================================================
# PAGE 2: UPLOAD & EVALUATE
# ===========================================================================
elif page == "📤 Upload & Evaluate":
    st.markdown('<div class="hero-title" style="font-size:2rem;">Upload & Evaluate</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV or JSON conversation file and run multi-facet evaluation.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_upload, col_config = st.columns([3, 2])

    with col_config:
        st.subheader("⚙️ Configuration")
        eval_mode = st.selectbox(
            "Scoring Mode",
            ["feature", "hybrid"],
            help="'feature' is fast (no GPU needed). 'hybrid' adds LLM refinement.",
        )
        model_name = st.selectbox(
            "LLM Model (for hybrid/llm mode)",
            [
                "mistralai/Mistral-7B-Instruct-v0.1",
                "meta-llama/Llama-2-7b-chat-hf",
                "Qwen/Qwen2-7B-Instruct",
            ],
        )
        n_facets_slider = st.slider(
            "Number of Facets to Evaluate",
            min_value=10,
            max_value=min(n_facets, 300) if n_facets > 0 else 300,
            value=min(50, n_facets) if n_facets > 0 else 50,
            step=10,
        )
        show_low_conf = st.checkbox("Highlight low-confidence predictions", value=True)
        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.05)

    with col_upload:
        st.subheader("📁 Upload Conversation File")
        uploaded = st.file_uploader(
            "Upload CSV or JSON",
            type=["csv", "json"],
            help="JSON: list of conversation dicts with 'turns' key. CSV: columns conversation_id, speaker, text.",
        )

        # Show example format
        with st.expander("📋 Expected JSON Format"):
            st.json({
                "conversation_id": "conv_001",
                "domain": "customer_service",
                "quality_level": "high",
                "turns": [
                    {"speaker": "user", "text": "Hello, I need help..."},
                    {"speaker": "assistant", "text": "Of course! ..."},
                ],
            })

    st.divider()

    # --- Evaluation ---
    if uploaded is not None:
        st.success(f"✅ File uploaded: **{uploaded.name}** ({uploaded.size // 1024} KB)")

        # Parse
        import io
        import pandas as pd

        conversations_to_eval = []
        if uploaded.name.endswith(".json"):
            try:
                data = json.load(io.BytesIO(uploaded.read()))
                if isinstance(data, list):
                    conversations_to_eval = data
                elif isinstance(data, dict):
                    conversations_to_eval = [data]
                st.info(f"Loaded {len(conversations_to_eval)} conversation(s)")
            except Exception as e:
                st.error(f"JSON parse error: {e}")
        else:
            try:
                df_upload = pd.read_csv(io.BytesIO(uploaded.read()))
                st.dataframe(df_upload.head(10), width="stretch")
                st.info("CSV preview shown above. For full evaluation, convert to JSON format.")
            except Exception as e:
                st.error(f"CSV parse error: {e}")

        if conversations_to_eval and st.button("🚀 Run Evaluation", type="primary"):
            facets_data = load_facets_config()
            if not facets_data.get("facets"):
                st.error("❌ Facets not loaded. Run `python src/data_loader.py` first.")
            else:
                # Subset facets
                facets_subset = facets_data.copy()
                facets_subset["facets"] = facets_data["facets"][:n_facets_slider]

                progress_bar = st.progress(0, text="Initializing evaluation...")
                status_text  = st.empty()

                try:
                    from src.scorer import ConversationEvaluator
                    from src.utils import load_config

                    cfg = load_config()
                    evaluator = ConversationEvaluator(
                        facets_config=facets_subset,
                        model_name=model_name,
                        mode=eval_mode,
                        config=cfg,
                    )

                    results = []
                    for i, conv in enumerate(conversations_to_eval):
                        progress = (i + 1) / len(conversations_to_eval)
                        progress_bar.progress(progress, text=f"Evaluating {i+1}/{len(conversations_to_eval)}...")
                        result = evaluator.evaluate_conversation(conv)
                        results.append(result)
                        status_text.text(f"✅ {conv.get('conversation_id', f'conv_{i}')} — overall score: {result['overall_score']:.2f}")

                    progress_bar.progress(1.0, text="✅ Evaluation complete!")

                    # Results display
                    st.subheader("📊 Results")
                    for res in results:
                        with st.expander(f"🗣️ {res['conversation_id']} — Score: {res['overall_score']:.2f} | Conf: {res['overall_confidence']:.2f}"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**Category Averages**")
                                bars_html = "".join(
                                    _score_bar(cat, avg)
                                    for cat, avg in res.get("category_averages", {}).items()
                                )
                                st.markdown(bars_html, unsafe_allow_html=True)
                            with col_b:
                                st.markdown("**Top Facets**")
                                top_facets = sorted(
                                    res.get("facet_scores", {}).items(),
                                    key=lambda x: x[1]["score"],
                                    reverse=True,
                                )[:10]
                                for fname, fdata in top_facets:
                                    c_color = "#f87171" if fdata["confidence"] < conf_threshold and show_low_conf else "#34d399"
                                    st.markdown(
                                        f"**{fname}**: {fdata['score']}/5 "
                                        f"<span style='color:{c_color};'>conf={fdata['confidence']:.2f}</span>",
                                        unsafe_allow_html=True,
                                    )

                    # Export
                    results_json = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        "⬇️ Download Results (JSON)",
                        data=results_json,
                        file_name="evaluation_results.json",
                        mime="application/json",
                    )

                except Exception as exc:
                    st.error(f"Evaluation error: {exc}")
    else:
        # Demo with sample data
        st.info("👆 Upload a file above, or explore pre-computed results in **Results Explorer**.")

        if st.button("📦 Load Sample Conversation for Demo"):
            convs = load_conversations()
            if convs:
                sample = convs[0]
                st.subheader("Sample Conversation Preview")
                for turn in sample.get("turns", []):
                    role = turn["speaker"].upper()
                    emoji = "👤" if turn["speaker"] == "user" else "🤖"
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:0.5rem;">
                      <div style="font-size:0.75rem; color:rgba(255,255,255,0.5);">{emoji} {role}</div>
                      <div style="margin-top:0.3rem;">{turn['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No conversations found. Run `python src/data_generator.py` first.")


# ===========================================================================
# PAGE 3: RESULTS EXPLORER
# ===========================================================================
elif page == "🔍 Results Explorer":
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown('<div class="hero-title" style="font-size:2rem;">Results Explorer</div>', unsafe_allow_html=True)
    st.caption("Browse and filter pre-computed evaluation results.")
    st.markdown("<br>", unsafe_allow_html=True)

    sample_evals = load_sample_evals()
    if not sample_evals:
        st.error("❌ No sample evaluations found. Run `python src/generate_sample_evals.py` first.")
        st.stop()

    # Build summary DataFrame
    rows = []
    for ev in sample_evals:
        row = {
            "ID":           ev["conversation_id"],
            "Domain":       ev.get("domain", ""),
            "Quality":      ev.get("quality_level", ""),
            "Overall Score": ev.get("overall_score", 0),
            "Confidence":   ev.get("overall_confidence", 0),
            "Turns":        len(ev.get("turns", [])),
        }
        for cat, avg in ev.get("category_averages", {}).items():
            row[cat] = avg
        rows.append(row)
    df_summary = pd.DataFrame(rows)

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        qual_filter = st.multiselect("Quality Level", df_summary["Quality"].unique().tolist(), default=df_summary["Quality"].unique().tolist())
    with col_f2:
        domain_filter = st.multiselect("Domain", df_summary["Domain"].unique().tolist(), default=df_summary["Domain"].unique().tolist())
    with col_f3:
        score_range = st.slider("Overall Score Range", 1.0, 5.0, (1.0, 5.0), 0.1)

    mask = (
        df_summary["Quality"].isin(qual_filter)
        & df_summary["Domain"].isin(domain_filter)
        & df_summary["Overall Score"].between(*score_range)
    )
    df_filtered = df_summary[mask]
    st.caption(f"Showing {len(df_filtered)} / {len(df_summary)} evaluations")

    # Summary table
    st.subheader("📋 Evaluation Summary")
    st.dataframe(
        df_filtered.style.background_gradient(
            subset=["Overall Score"], cmap="RdYlGn", vmin=1, vmax=5
        ),
        width="stretch",
        height=300,
    )

    # Charts row
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Score Distributions")
    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            df_filtered,
            x="Overall Score",
            color="Quality",
            nbins=20,
            title="Score Distribution by Quality",
            color_discrete_map={
                "high": "#34d399", "medium": "#fbbf24",
                "low": "#f87171", "edge_case": "#a78bfa",
            },
            template="plotly_dark",
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_scatter = px.scatter(
            df_filtered,
            x="Overall Score",
            y="Confidence",
            color="Domain",
            size_max=10,
            title="Quality vs Confidence",
            template="plotly_dark",
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Heatmap
    cat_cols_available = [c for c in df_filtered.columns if c not in ["ID", "Domain", "Quality", "Overall Score", "Confidence", "Turns"]]
    if cat_cols_available:
        st.subheader("🔥 Category Score Heatmap")
        heatmap_data = df_filtered[["ID"] + cat_cols_available].set_index("ID")
        fig_heat = px.imshow(
            heatmap_data.T,
            title="Facet Category Scores per Conversation",
            color_continuous_scale="Viridis",
            aspect="auto",
            template="plotly_dark",
            zmin=1, zmax=5,
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0",
            height=300,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Detail view
    st.subheader("🔎 Conversation Detail")
    selected_id = st.selectbox("Select Conversation", df_filtered["ID"].tolist())
    if selected_id:
        selected_eval = next((e for e in sample_evals if e["conversation_id"] == selected_id), None)
        if selected_eval:
            col_detail1, col_detail2 = st.columns([2, 3])
            with col_detail1:
                st.markdown("**Conversation**")
                for turn in selected_eval.get("turns", []):
                    emoji = "👤" if turn["speaker"] == "user" else "🤖"
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:0.4rem; padding:0.8rem;">
                      <div style="font-size:0.7rem; color:rgba(255,255,255,0.45);">{emoji} {turn['speaker'].upper()}</div>
                      <div style="font-size:0.88rem; margin-top:0.2rem;">{turn['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            with col_detail2:
                st.markdown("**Facet Scores (Top 20)**")
                facet_scores = selected_eval.get("facet_scores", {})
                top_20 = sorted(facet_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:20]
                bars = "".join(_score_bar(name, data["score"]) for name, data in top_20)
                st.markdown(bars, unsafe_allow_html=True)

                # Download
                st.download_button(
                    "⬇️ Export as JSON",
                    data=json.dumps(selected_eval, indent=2, default=str),
                    file_name=f"{selected_id}_eval.json",
                    mime="application/json",
                )


# ===========================================================================
# PAGE 4: ANALYTICS
# ===========================================================================
elif page == "📊 Analytics":
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy import stats

    st.markdown('<div class="hero-title" style="font-size:2rem;">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.caption("Facet correlations · Domain performance · Anomaly detection · Confidence analysis")
    st.markdown("<br>", unsafe_allow_html=True)

    sample_evals = load_sample_evals()
    if not sample_evals:
        st.error("❌ No sample evaluations found.")
        st.stop()

    # Build category DataFrame
    rows = []
    for ev in sample_evals:
        row = {
            "ID": ev["conversation_id"],
            "Domain": ev.get("domain", ""),
            "Quality": ev.get("quality_level", ""),
            "Overall Score": ev.get("overall_score", 0),
            "Confidence": ev.get("overall_confidence", 0),
        }
        for cat, avg in ev.get("category_averages", {}).items():
            row[cat] = avg
        rows.append(row)
    df = pd.DataFrame(rows)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 Category Performance",
        "🔗 Correlations",
        "⚠️ Anomaly Detection",
        "📐 Confidence Analysis",
    ])

    cat_cols = [c for c in df.columns if c not in ["ID", "Domain", "Quality", "Overall Score", "Confidence"]]

    with tab1:
        st.subheader("Category Performance by Domain")
        if cat_cols:
            df_melt = df.melt(id_vars=["Domain", "Quality"], value_vars=cat_cols, var_name="Category", value_name="Score")
            fig = px.box(
                df_melt,
                x="Category",
                y="Score",
                color="Domain",
                template="plotly_dark",
                title="Score Distribution per Category by Domain",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0",
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Bar: avg per category
            cat_avgs = df[cat_cols].mean().reset_index()
            cat_avgs.columns = ["Category", "Mean Score"]
            fig2 = px.bar(
                cat_avgs, x="Category", y="Mean Score",
                title="Mean Score per Category (all conversations)",
                color="Mean Score",
                color_continuous_scale="Viridis",
                template="plotly_dark",
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Facet Category Correlation Matrix")
        if len(cat_cols) >= 2:
            corr = df[cat_cols + ["Overall Score"]].corr()
            fig_corr = px.imshow(
                corr,
                title="Category Score Correlations",
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                template="plotly_dark",
            )
            fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough category columns for correlation matrix.")

    with tab3:
        st.subheader("Anomaly Detection — Unusual Conversation Patterns")
        # Z-score based
        if cat_cols:
            df_z = df[cat_cols].apply(stats.zscore, nan_policy="omit")
            df["max_z"] = df_z.abs().max(axis=1)
            anomalies = df[df["max_z"] > 1.5][["ID", "Domain", "Quality", "Overall Score", "max_z"]].copy()
            anomalies.columns = ["ID", "Domain", "Quality", "Overall Score", "Anomaly Score"]
            anomalies = anomalies.sort_values("Anomaly Score", ascending=False)

            st.markdown(f"**{len(anomalies)} conversations** with unusual score patterns (|z| > 1.5):")
            st.dataframe(anomalies, width="stretch")

            fig_anom = px.scatter(
                df, x="Overall Score", y="max_z",
                color="Quality", hover_data=["ID", "Domain"],
                title="Overall Score vs Anomaly Score",
                template="plotly_dark",
                color_discrete_map={
                    "high": "#34d399", "medium": "#fbbf24",
                    "low": "#f87171", "edge_case": "#a78bfa",
                },
            )
            fig_anom.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Anomaly threshold")
            fig_anom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0")
            st.plotly_chart(fig_anom, use_container_width=True)

    with tab4:
        st.subheader("Confidence Distribution Analysis")
        fig_conf = px.violin(
            df, y="Confidence", x="Quality",
            color="Quality",
            box=True, points="all",
            template="plotly_dark",
            title="Confidence Distribution by Quality Level",
            color_discrete_map={
                "high": "#34d399", "medium": "#fbbf24",
                "low": "#f87171", "edge_case": "#a78bfa",
            },
        )
        fig_conf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f0")
        st.plotly_chart(fig_conf, use_container_width=True)

        # Stats table
        conf_stats = df.groupby("Quality")["Confidence"].agg(["mean", "std", "min", "max"]).round(3).reset_index()
        conf_stats.columns = ["Quality", "Mean Confidence", "Std Dev", "Min", "Max"]
        st.dataframe(conf_stats, width="stretch")

        # Export
        st.download_button(
            "⬇️ Download Full Analytics CSV",
            data=df.to_csv(index=False),
            file_name="ahoum_analytics.csv",
            mime="text/csv",
        )
