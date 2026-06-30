import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.ui.app import load_facets_config, load_conversations, _score_bar, load_style, confidence_badge, get_confidence_color

load_style()

st.markdown('<div class="hero-title" style="font-size:2rem;">Upload & Evaluate</div>', unsafe_allow_html=True)
st.caption("Upload conversation datasets and trigger real-time, high-throughput multi-facet evaluation.")
st.markdown("<br>", unsafe_allow_html=True)

# Layout Setup
col_upload, col_config = st.columns([3, 2])

with col_config:
    st.subheader("⚙️ Configuration")
    eval_mode = st.selectbox(
        "Scoring Mode",
        ["feature", "hybrid"],
        help="'feature' runs 100% locally using rules and NLP features. 'hybrid' invokes open-weights LLMs for low-confidence scores.",
    )
    model_name = st.selectbox(
        "LLM Refinement Model",
        [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "meta-llama/Llama-2-7b-chat-hf",
            "Qwen/Qwen2-7B-Instruct",
        ],
    )
    facets_data = load_facets_config()
    n_facets = len(facets_data.get("facets", []))

    n_facets_slider = st.slider(
        "Facets Limit (Sub-sampling for speed)",
        min_value=10,
        max_value=min(n_facets, 300) if n_facets > 0 else 300,
        value=min(300, n_facets) if n_facets > 0 else 300,
        step=10,
    )
    show_low_conf = st.checkbox("Highlight low confidence predictions", value=True)
    conf_threshold = st.slider("Confidence Cutoff threshold", 0.0, 1.0, 0.6, 0.05)

with col_upload:
    st.subheader("📁 Ingest Conversation Log")
    uploaded = st.file_uploader(
        "Drag & Drop CSV or JSON",
        type=["csv", "json"],
        help="JSON: list of conversation dicts with a 'turns' key. CSV: columns conversation_id, speaker, text.",
    )

    with st.expander("📝 Expected JSON Input Schema"):
        st.json({
            "conversation_id": "conv_demo_123",
            "domain": "customer_service",
            "turns": [
                {"speaker": "user", "text": "I need help with my account."},
                {"speaker": "assistant", "text": "Sure, I can help you with that. Can I get your ID?"}
            ]
        })

st.divider()

if uploaded is not None:
    st.success(f"Ingested file: **{uploaded.name}** ({uploaded.size // 1024} KB)")

    conversations_to_eval = []
    if uploaded.name.endswith(".json"):
        try:
            data = json.load(io.BytesIO(uploaded.read()))
            if isinstance(data, list):
                conversations_to_eval = data
            elif isinstance(data, dict):
                conversations_to_eval = [data]
            st.info(f"Successfully loaded {len(conversations_to_eval)} conversations.")
        except Exception as e:
            st.error(f"JSON Parse Error: {e}")
    else:
        try:
            df_upload = pd.read_csv(io.BytesIO(uploaded.read()))
            st.dataframe(df_upload.head(5), use_container_width=True)
            st.info("CSV parsed. For multi-turn scoring, convert rows to standard nested JSON.")
        except Exception as e:
            st.error(f"CSV Parse Error: {e}")

    if conversations_to_eval:
        st.markdown("### ⚡ Run Evaluation Pipeline")
        if st.button("🚀 Start Bulk Evaluation", type="primary"):
            if not facets_data.get("facets"):
                st.error("❌ Facets configuration file not found. Clean raw CSV first.")
            else:
                facets_subset = facets_data.copy()
                facets_subset["facets"] = facets_data["facets"][:n_facets_slider]

                # Create performance placeholder metrics
                st.markdown("#### 📈 Real-Time Performance Monitor")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                stat_throughput = m_col1.empty()
                stat_latency = m_col2.empty()
                stat_eta = m_col3.empty()
                stat_memory = m_col4.empty()

                progress_bar = st.progress(0, text="Initializing scorer...")
                status_text = st.empty()

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
                    total_convs = len(conversations_to_eval)
                    start_time = time.perf_counter()

                    for idx, conv in enumerate(conversations_to_eval):
                        t_conv_start = time.perf_counter()
                        
                        # Pipeline call
                        result = evaluator.evaluate_conversation(conv)
                        results.append(result)
                        
                        t_conv_end = time.perf_counter()
                        elapsed = t_conv_end - start_time
                        avg_latency = elapsed / (idx + 1)
                        convs_per_sec = (idx + 1) / elapsed
                        eta = (total_convs - (idx + 1)) * avg_latency

                        # Update metrics UI
                        stat_throughput.metric("Throughput", f"{convs_per_sec:.2f} conv/sec")
                        stat_latency.metric("Avg Latency", f"{avg_latency:.3f} s")
                        stat_eta.metric("Time Remaining", f"{eta:.1f} s")
                        stat_memory.metric("Model Status", "Local RAM Warm" if eval_mode == "feature" else "GPU Active")

                        progress = (idx + 1) / total_convs
                        progress_bar.progress(progress, text=f"Processing {idx+1}/{total_convs}...")
                        status_text.text(f"Scored {conv.get('conversation_id')} — rating: {result['overall_score']:.2f}")

                    progress_bar.progress(1.0, text="✅ Evaluation complete!")
                    st.balloons()

                    # Save results in state to support downstream exports
                    st.session_state["eval_results"] = results
                    st.success(f"Successfully evaluated {len(results)} conversations across {n_facets_slider} facets!")

                except Exception as exc:
                    st.error(f"Evaluation pipeline crashed: {exc}")

if "eval_results" in st.session_state:
    results = st.session_state["eval_results"]
    st.divider()
    st.markdown("### 📤 Export & Download Center")
    
    # Render detailed results preview
    with st.expander("🔬 View Evaluation Scores Preview"):
        for res in results[:5]:
            st.markdown(f"**Conversation ID**: `{res['conversation_id']}` | Overall Score: `{res['overall_score']:.2f}` | Confidence: {confidence_badge(res['overall_confidence'])}", unsafe_allow_html=True)
    
    # Excel and CSV building
    # Standard CSV
    csv_rows = []
    for r in results:
        for fname, fdata in r["facet_scores"].items():
            csv_rows.append({
                "conversation_id": r["conversation_id"],
                "domain": r.get("domain", "N/A"),
                "facet_name": fname,
                "score": fdata["score"],
                "confidence": fdata["confidence"],
                "method": fdata["method"],
            })
    df_export = pd.DataFrame(csv_rows)

    c_exp1, c_exp2, c_exp3 = st.columns(3)
    
    # Download JSON
    with c_exp1:
        st.download_button(
            "⬇️ Download Raw JSON",
            data=json.dumps(results, indent=2, default=str),
            file_name="ahoum_eval_results.json",
            mime="application/json",
            key="download_json",
        )
    # Download CSV
    with c_exp2:
        st.download_button(
            "⬇️ Download CSV Spreadsheet",
            data=df_export.to_csv(index=False),
            file_name="ahoum_eval_results.csv",
            mime="text/csv",
            key="download_csv",
        )
    # Download Excel
    with c_exp3:
        # Build multi-sheet excel
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
            # Sheet 1: Summary Ratings
            summary_rows = []
            for r in results:
                summary_rows.append({
                    "conversation_id": r["conversation_id"],
                    "domain": r.get("domain", "N/A"),
                    "overall_score": r["overall_score"],
                    "overall_confidence": r["overall_confidence"],
                    "num_facets": r["num_facets_evaluated"],
                })
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Session Summary", index=False)
            # Sheet 2: Flat Facets
            df_export.to_excel(writer, sheet_name="Facet Details", index=False)
            
        towrite.seek(0)
        st.download_button(
            "⬇️ Download Excel Workbook",
            data=towrite.read(),
            file_name="ahoum_eval_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel",
        )
