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

from ui.app import load_sample_evals, load_style, get_confidence_color, confidence_badge

load_style()

st.markdown('<div class="hero-title" style="font-size:2rem;">Analytics Dashboard</div>', unsafe_allow_html=True)
st.caption("Perform advanced statistical analyses, facet correlations, and anomaly diagnostics on datasets.")
st.markdown("<br>", unsafe_allow_html=True)

sample_evals = load_sample_evals()
if not sample_evals:
    st.error("❌ No evaluations found. Execute the preprocessing and scoring chains first.")
    st.stop()

# Build DataFrame
rows = []
for ev in sample_evals:
    row = {
        "ID":           ev["conversation_id"],
        "Domain":       ev.get("domain", "N/A"),
        "Quality":      ev.get("quality_level", "N/A"),
        "Overall Score": ev.get("overall_score", 0.0),
        "Confidence":   ev.get("overall_confidence", 0.0),
    }
    for cat, avg in ev.get("category_averages", {}).items():
        row[cat] = avg
    rows.append(row)
df = pd.DataFrame(rows)

tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Category Performance",
    "🔗 Correlation Matrices",
    "⚠️ Anomaly Diagnostics",
    "📐 Confidence Profiling",
])

cat_cols = [c for c in df.columns if c not in ["ID", "Domain", "Quality", "Overall Score", "Confidence"]]

# ---------------------------------------------------------------------------
# Tab 1: Category Performance
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Category Performance Distribution")
    if cat_cols:
        df_melt = df.melt(id_vars=["Domain", "Quality"], value_vars=cat_cols, var_name="Category", value_name="Score")
        fig = px.box(
            df_melt,
            x="Category",
            y="Score",
            color="Domain",
            template="plotly_dark",
            title="Category Score Distributions by Domain",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        cat_avgs = df[cat_cols].mean().reset_index()
        cat_avgs.columns = ["Category", "Mean Score"]
        fig2 = px.bar(
            cat_avgs, x="Category", y="Mean Score",
            title="Mean Category Ratings Across Entire Dataset",
            color="Mean Score",
            color_continuous_scale="Viridis",
            template="plotly_dark",
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No category scores detected in evaluations.")

# ---------------------------------------------------------------------------
# Tab 2: Correlation Matrices
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Facet Category Correlation Analysis")
    if len(cat_cols) >= 2:
        corr = df[cat_cols + ["Overall Score"]].corr()
        fig_corr = px.imshow(
            corr,
            title="Linear Correlation of Facet Categories",
            color_continuous_scale="RdBu",
            zmin=-1.0, zmax=1.0,
            template="plotly_dark",
        )
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Insufficient category dimensions to calculate correlation.")

# ---------------------------------------------------------------------------
# Tab 3: Anomaly Diagnostics
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Anomaly Detection — Deviation Ratings")
    if cat_cols:
        # Calculate z-score of category averages
        df_z = df[cat_cols].apply(stats.zscore, nan_policy="omit")
        df["max_z"] = df_z.abs().max(axis=1)
        anomalies = df[df["max_z"] > 1.5][["ID", "Domain", "Quality", "Overall Score", "max_z"]].copy()
        anomalies.columns = ["ID", "Domain", "Quality", "Overall Rating", "Anomaly Z-Score"]
        anomalies = anomalies.sort_values("Anomaly Z-Score", ascending=False)

        st.markdown(f"Detected **{len(anomalies)} anomalies** exhibiting unusual category behaviors (|z| > 1.5):")
        st.dataframe(anomalies, use_container_width=True)

        fig_anom = px.scatter(
            df, x="Overall Score", y="max_z",
            color="Quality", hover_data=["ID", "Domain"],
            title="Rating Intensity vs. Behavior Anomaly Index",
            template="plotly_dark",
            color_discrete_map={
                "high": "#10b981", "medium": "#fbbf24",
                "low": "#f87171", "edge_case": "#a78bfa",
            },
        )
        fig_anom.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Anomaly Boundary Limit")
        fig_anom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_anom, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Confidence Profiling
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Confidence Score Distribution & Analysis")
    fig_conf = px.violin(
        df, y="Confidence", x="Quality",
        color="Quality",
        box=True, points="all",
        template="plotly_dark",
        title="Confidence Dispersion Across Quality Metrics",
        color_discrete_map={
            "high": "#10b981", "medium": "#fbbf24",
            "low": "#f87171", "edge_case": "#a78bfa",
        },
    )
    fig_conf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    st.plotly_chart(fig_conf, use_container_width=True)

    # Statistical breakdown with color coded interpretation
    conf_stats = df.groupby("Quality")["Confidence"].agg(["mean", "std", "min", "max"]).reset_index()
    conf_stats.columns = ["Quality Segment", "Mean Confidence", "Std Dev", "Min Confidence", "Max Confidence"]
    
    # Styled Grid
    st.dataframe(conf_stats.style.format({
        "Mean Confidence": "{:.3f}",
        "Std Dev": "{:.3f}",
        "Min Confidence": "{:.3f}",
        "Max Confidence": "{:.3f}",
    }), use_container_width=True)
    
    # Textual guidelines
    st.markdown("""
    💡 **Confidence Rating Guide**:
    * <span style="color:#10b981; font-weight:700;">● Green (High, ≥ 0.90)</span>: Reliable feature alignment or fully validated by local LLMs.
    * <span style="color:#fbbf24; font-weight:700;">● Yellow (Medium, 0.70 - 0.89)</span>: Sufficient turn length and semantic feature density.
    * <span style="color:#f97316; font-weight:700;">● Orange (Low, 0.50 - 0.69)</span>: Shorter turns or semantic ambiguity, possibly requiring manual QA.
    * <span style="color:#f87171; font-weight:700;">● Red (Critical, < 0.50)</span>: Extreme turn sparsity or anomalous formatting. Review required.
    """, unsafe_allow_html=True)
    
    st.download_button(
        "⬇️ Export Analytics Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="ahoum_statistical_analytics.csv",
        mime="text/csv",
    )
