import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT))

from ui.app import load_sample_evals, load_style

load_style()

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

    conf_stats = df.groupby("Quality")["Confidence"].agg(["mean", "std", "min", "max"]).round(3).reset_index()
    conf_stats.columns = ["Quality", "Mean Confidence", "Std Dev", "Min", "Max"]
    st.dataframe(conf_stats, width="stretch")

    st.download_button(
        "⬇️ Download Full Analytics CSV",
        data=df.to_csv(index=False),
        file_name="ahoum_analytics.csv",
        mime="text/csv",
    )
