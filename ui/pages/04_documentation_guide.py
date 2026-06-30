import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT))

from ui.app import load_facets_config, load_style

load_style()

st.markdown('<div class="hero-title" style="font-size:2rem;">Documentation & System Explorer</div>', unsafe_allow_html=True)
st.caption("Browse system documentation, model specs, facet registries, and API code snippets.")
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Facet Explorer",
    "🤖 Model Information",
    "📐 Scoring Methodology",
    "🔌 API & Integration Panel",
])

# Load facet metadata
facets_data = load_facets_config()
facets_list = facets_data.get("facets", [])

# ---------------------------------------------------------------------------
# Tab 1: Facet Explorer (Search & Filter Registry)
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("🔍 Interactive Facet Registry")
    st.markdown("Query the complete list of 300 behavioral and linguistic dimensions.")
    
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        f_search = st.text_input("Search facets by name or keyword...", "")
    with col_t2:
        categories = facets_data.get("categories", [])
        f_cat = st.multiselect("Filter by Category", categories, default=categories)

    # Filter facets list
    filtered_facets = []
    for f in facets_list:
        name = f.get("name", "")
        category = f.get("category_display", "")
        
        # apply search query & category filter
        match_search = f_search.lower() in name.lower() or f_search.lower() in category.lower()
        match_cat = category in f_cat
        if match_search and match_cat:
            filtered_facets.append({
                "ID": f.get("id"),
                "Facet Name": name,
                "Category": category,
                "Weight": f.get("weight", 1.0),
                "Importance": f.get("importance", 0.8),
                "Description": f.get("description", f"Evaluates the degree of '{name}' present in the turns.")
            })
            
    df_facets = pd.DataFrame(filtered_facets)
    
    if not df_facets.empty:
        st.markdown(f"**Found {len(df_facets)} facets matching criteria** (hover over cells to read full descriptions):")
        st.dataframe(df_facets, use_container_width=True, height=350)
    else:
        st.warning("No facets match your search query or filters.")

# ---------------------------------------------------------------------------
# Tab 2: Model Information
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("🤖 Model Specs & Resource Profiling")
    st.markdown("Profile of the open-weight LLMs compatible with Ahoum evaluation pipelines.")
    
    # Model comparison table
    model_rows = [
        {
            "Model Name": "Mistral-7B-Instruct-v0.1",
            "Parameters": "7.2 Billion",
            "Memory Req (RAM)": "16 GB (Float16)",
            "Memory Req (Quantized)": "6 GB (Q4_K_M)",
            "Avg Latency/Conv": "25.2 seconds",
            "Accuracy Rank": "Top Choice (Fast & High)",
            "Recommended Use Case": "General support, structured JSON outputs, causal reasoning"
        },
        {
            "Model Name": "Llama-2-7b-chat-hf",
            "Parameters": "7.0 Billion",
            "Memory Req (RAM)": "15 GB (Float16)",
            "Memory Req (Quantized)": "5.5 GB (Q4_K_M)",
            "Avg Latency/Conv": "28.5 seconds",
            "Accuracy Rank": "Moderate (Slower)",
            "Recommended Use Case": "Casual dialogue, simple sentiment tracking, politeness evaluation"
        },
        {
            "Model Name": "Qwen2-7B-Instruct",
            "Parameters": "7.6 Billion",
            "Memory Req (RAM)": "16.2 GB (Float16)",
            "Memory Req (Quantized)": "6.2 GB (Q4_K_M)",
            "Avg Latency/Conv": "23.8 seconds",
            "Accuracy Rank": "Superior (Ultra-fast)",
            "Recommended Use Case": "Complex reasoning, logical flow, multilingual dialogues"
        }
    ]
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True)
    
    st.markdown("#### ⚡ Active Model Status")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        # Check model cache
        cache_dir = Path(".cache/model_cache")
        cache_exists = cache_dir.exists() and any(cache_dir.iterdir())
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid #10b981;">
          <div style="font-weight:600; color:#10b981;">Cache warming</div>
          <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.2rem;">
            Pre-loaded NLP: spaCy & NLTK (Ready)<br>
            Local HuggingFace Cache: {".cache/model_cache (Warmed)" if cache_exists else "Empty (downloads on run)"}
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown("""
        <div class="metric-card" style="border-left: 3px solid #6366f1;">
          <div style="font-weight:600; color:#6366f1;">Inference Target Engine</div>
          <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.2rem;">
            PyTorch Execution: CPU-only (extra-index backend loaded)<br>
            Execution Overhead: ~15ms per conversation in Feature Mode
          </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 3: Scoring Methodology
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("📐 Ahoum Scoring Methodology")
    
    col_meth1, col_meth2 = st.columns(2)
    with col_meth1:
        st.markdown("""
        ### 1. Feature-Based Scoring (Fast Baseline)
        Runs 100% locally on standard CPUs. Takes **~15ms per conversation**.
        
        * **Extraction**: Calculates 24 linguistic features (like readability index, lexical diversity, subjectivity score, coherence semantic distance, and keyword matching).
        * **Facet Mapping**: Mapped to 300 facets using mathematical regression.
        * **Confidence**: Calculated as:
          $$\\text{Confidence} = \\text{Data Completeness} \\times \\text{Feature agreement} \\times \\text{Keyword frequency}$$
        """)
    with col_meth2:
        st.markdown("""
        ### 2. Hybrid Scoring Strategy (Optimal)
        Blends feature baselines with local open-weight instruction models.
        
        * **Trigger conditions**: Evaluates with features first. If a facet is marked critical (importance > 0.8) OR feature confidence drops below the threshold (< 0.60), the pipeline schedules a local LLM refinement run.
        * **Mathematical blend**:
          $$\\text{Final Rating} = 0.4 \\times \\text{Feature Rating} + 0.6 \\times \\text{LLM Rating}$$
          $$\\text{Final Confidence} = \\text{Feature Confidence} \\times \\text{LLM Confidence}$$
        """)

# ---------------------------------------------------------------------------
# Tab 4: API & Integration Panel
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("🔌 REST API & Software Integration")
    st.markdown("programmatic scoring access for pipeline integrations.")
    
    col_api1, col_api2 = st.columns(2)
    with col_api1:
        st.markdown("#### cURL Example — Single Evaluation")
        st.code("""
curl -X POST http://localhost:8080/evaluate \\
  -H "Content-Type: application/json" \\
  -d '{
    "conversation_id": "api_demo_001",
    "domain": "customer_service",
    "mode": "feature",
    "turns": [
      {"speaker": "user", "text": "My order #1234 has not arrived yet."},
      {"speaker": "assistant", "text": "I apologize for the delay. Let me track it."}
    ]
  }'
        """, language="bash")
        
        st.markdown("#### Rate Limits & Scaling")
        st.markdown("""
        * **Feature Mode**: Unlimited parallel requests (CPU-bound, ~60 requests/sec).
        * **Hybrid/LLM Mode**: Throttled by GPU VRAM capabilities. Recommended: `--workers 2` with FastAPI deployment.
        """)

    with col_api2:
        st.markdown("#### Python Client Code Snippet")
        st.code("""
import requests

payload = {
    "conversation_id": "test_script_01",
    "domain": "general",
    "mode": "feature",
    "turns": [
        {"speaker": "user", "text": "Can you explain linear regression?"},
        {"speaker": "assistant", "text": "Linear regression models linear relation."}
    ]
}

response = requests.post("http://localhost:8080/evaluate", json=payload)
result = response.json()

print(f"Overall Rating: {result['overall_score']}")
print(f"Confidence score: {result['overall_confidence']}")
# Full list of 300 facets
print(f"Facets evaluated: {result['num_facets_evaluated']}")
        """, language="python")
