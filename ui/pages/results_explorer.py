import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT))

from ui.app import load_facets_config, load_sample_evals, _score_bar, load_style

load_style()

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

            # Export
            st.download_button(
                "⬇️ Export as JSON",
                data=json.dumps(selected_eval, indent=2, default=str),
                file_name=f"{selected_id}_eval.json",
                mime="application/json",
            )
