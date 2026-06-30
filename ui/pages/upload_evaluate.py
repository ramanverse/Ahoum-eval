import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT))

from ui.app import load_facets_config, load_conversations, _score_bar, load_style

load_style()

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
    facets_data = load_facets_config()
    n_facets = len(facets_data.get("facets", []))

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

if uploaded is not None:
    st.success(f"✅ File uploaded: **{uploaded.name}** ({uploaded.size // 1024} KB)")

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
        if not facets_data.get("facets"):
            st.error("❌ Facets not loaded. Run `python src/data_loader.py` first.")
        else:
            facets_subset = facets_data.copy()
            facets_subset["facets"] = facets_data["facets"][:n_facets_slider]

            progress_bar = st.progress(0, text="Initializing evaluation...")
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
                for i, conv in enumerate(conversations_to_eval):
                    progress = (i + 1) / len(conversations_to_eval)
                    progress_bar.progress(progress, text=f"Evaluating {i+1}/{len(conversations_to_eval)}...")
                    result = evaluator.evaluate_conversation(conv)
                    results.append(result)
                    status_text.text(f"✅ {conv.get('conversation_id', f'conv_{i}')} — overall score: {result['overall_score']:.2f}")

                progress_bar.progress(1.0, text="✅ Evaluation complete!")

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
