# Ahoum Conversation Evaluation System

[![CI — Test & Build](https://github.com/ramanverse/Ahoum-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/ramanverse/Ahoum-eval/actions/workflows/ci.yml)
[![Docker Pulls](https://img.shields.io/badge/docker-ghcr.io-blue.svg?logo=docker)](https://github.com/ramanverse/Ahoum-eval/pkgs/container/Ahoum-eval-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](requirements.txt)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=Streamlit)](https://ahoumfacets.streamlit.app/)
[![FastAPI Docs](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg?logo=FastAPI)](http://localhost:8080/docs)

A production-ready, scalable, and highly configurable multi-facet conversation evaluation benchmark system that scores conversation turns and whole sessions on **300 behavioral and linguistic facets** using open-weight Large Language Models (LLMs) and advanced feature engineering.

Developed for AI/ML engineering contexts to establish reliable, reproducible, and robust metric scoring systems.

---

## 🏗️ Architecture Overview

The system is designed to scale dynamically to **5000+ facets** without architectural redesign. It utilizes a **hybrid scoring approach** that balances speed and accuracy:

```
                      +-----------------------------+
                      |    Facets_Assignment.csv    |
                      +--------------+--------------+
                                     |
                                     v
                        +------------+------------+
                        |  data_loader.py (Phase 3)|
                        +------------+------------+
                                     |
                                     v
                        +-------------+-------------+
                        |    facets_cleaned.json    |
                        +-------------+-------------+
                                     |
                                     v
+------------------------+     +-----+-----+     +--------------------------+
| generated_convs.json   | --> | Preprocess| --> | conversations_           |
| (data_generator.py)    |     | (Phase 4) |     | preprocessed.parquet     |
+------------------------+     +-----------+     +------------+-------------+
                                                              |
                                                              v
+-------------------------+                       +-----------+-----------+
| conversation_features.  | <---------------------+ feature_extractor.py  |
| csv (Phase 5)           |                       | (25+ features - Phase 5)|
+-----------+-------------+                       +-----------------------+
            |
            v
+-----------+-------------+
| scorer.py (Phase 6)     | <--- Mode 1: Feature-Based Scoring (Fast, No GPU)
|                         | <--- Mode 2: LLM-Augmented Scoring (Accurate, GPU)
|                         | <--- Mode 3: Hybrid Blend (Automatic Fallback)
+-----------+-------------+
            |
            +------------> [Parquet/JSON Score Outputs]
            |
            +------------> [FastAPI REST endpoints / Streamlit Dashboard]
```

### Key Components

1. **`data_generator.py`**: Generates 300 diverse multi-turn synthetic conversations across 6 domains (Customer Service, AI Assistant, Tech Support, Emotional Support, E-commerce, Educational) and 4 quality levels.
2. **`data_loader.py`**: Reads raw `facets_assignment.csv` (containing ~399 raw facets), cleans names, maps them into 6 categories (50 facets each), and builds relationships.
3. **`preprocessing.py`**: Performs unicode normalization, regex cleaning, tokenization (NLTK/spaCy), and speaker/metadata extraction.
4. **`feature_extractor.py`**: Extracts 25+ linguistic features per turn (readability, subjectivity, formal/informal cues, grammar proxy, coherence, etc.).
5. **`facet_mapper.py`**: Performs rule-based mapping from extracted features to all 300 facets for rapid baseline evaluation.
6. **`model_manager.py`**: Wraps open-weight instruction-tuned LLMs (Mistral-7B, Llama-2-7B, Qwen2-7B) with local response caching and structured JSON validation.
7. **`scorer.py`**: Executes the evaluation modes (Feature, LLM, Hybrid) and produces unified scores (1–5 scale) + confidence scores (0–1 scale).

---

## 🚀 Quick Start (Local Development)

### 1. Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/ramanverse/Ahoum-eval.git
cd Ahoum-eval

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install CPU-optimized requirements (avoids massive CUDA download)
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch>=2.2.0
pip install -r requirements.txt

# Download NLP models
python3 -m spacy download en_core_web_sm
```

### 2. Run the Data & Scoring Pipeline

```bash
# Clean raw facets assignment csv
python3 src/data_loader.py --facets data/raw/facets_assignment.csv --output data/processed/facets_cleaned.json

# Generate synthetic conversations
python3 src/data_generator.py --output data/raw/generated_conversations.json

# Preprocess conversations
python3 src/preprocessing.py --input data/raw/generated_conversations.json --output data/processed/conversations_preprocessed.parquet

# Extract features
python3 src/feature_extractor.py --input data/processed/conversations_preprocessed.parquet --output data/processed/conversation_features.parquet

# Generate 50 sample evaluations for the Dashboard
python3 src/generate_sample_evals.py
```

### 3. Run the Test Suite

```bash
# Run unit and integration tests (80 tests total)
pytest tests/ -v
```

---

## 🏃 Usage Examples

### Scoring Engine CLI

Run the scoring engine on raw conversations using **feature-based baseline mode** (takes ~2 seconds total):
```bash
python3 src/scorer.py \
  --conversations data/raw/generated_conversations.json \
  --facets data/processed/facets_cleaned.json \
  --output data/processed/scores.parquet \
  --mode feature
```

To run with **hybrid mode** (uses LLM to resolve low-confidence scores):
```bash
export HF_TOKEN="your_huggingface_token"
python3 src/scorer.py \
  --conversations data/raw/generated_conversations.json \
  --facets data/processed/facets_cleaned.json \
  --output data/processed/scores.parquet \
  --mode hybrid \
  --model mistralai/Mistral-7B-Instruct-v0.2
```

### Streamlit Dashboard UI

Launch the interactive dark glassmorphism dashboard UI:
```bash
streamlit run src/ui/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** to browse local pages, or view the live production deployment at **[https://ahoumfacets.streamlit.app/](https://ahoumfacets.streamlit.app/)**:
- **🏠 Home**: System overview and metrics
- **📤 Upload & Evaluate**: Process raw conversation logs
- **🔍 Results Explorer**: View pre-computed evaluations and individual turn analysis
- **📊 Analytics**: View category breakdowns, correlation heatmaps, and scatter plots

---

## 🐳 Docker Deployment

The application is containerized with a highly optimized multi-stage build.

### Build and Run with Script Helper

Use the helper script to manage your containers:
```bash
# Start all containers in the background (FastAPI on 8080, Streamlit on 8501)
bash docker/build.sh up

# Run container health and API smoke tests
bash docker/build.sh test

# Follow stdout logs
bash docker/build.sh logs

# Stop containers
bash docker/build.sh down
```

### Pull Directly from GitHub Packages (GHCR)

If you don't want to build locally, you can pull the images pushed by the CI/CD pipeline:
```bash
# Authenticate to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Pull and run the API
docker run -d -p 8080:8080 --name ahoum-api ghcr.io/ramanverse/ahoum-eval-api:latest

# Pull and run the UI
docker run -d -p 8501:8501 --name ahoum-ui ghcr.io/ramanverse/ahoum-eval-ui:latest
```

---

## ☁️ Streamlit Community Cloud Deployment

You can deploy the Streamlit Dashboard UI for free on [Streamlit Community Cloud](https://share.streamlit.io/).

### Steps to Deploy:
1. **Fork or Push** this repository to your own GitHub account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and click **"New app"**.
3. Select your repository, branch (`main`), and set the main file path to:
   ```
   ui/app.py
   ```
4. Under **"Advanced settings..."**, add any environment variables (like `HF_TOKEN`) if you plan to use LLM/hybrid modes in production.
5. Click **"Deploy!"** — Streamlit will automatically install dependencies from `requirements.txt`, read theme configurations from `.streamlit/config.toml`, and make your dashboard live at a custom URL.

---


## 📊 API Reference

The FastAPI service runs on port `8080` (or `80` if using the Nginx proxy profile).

### GET `/health`
Returns system status.
```json
{
  "status": "ok",
  "facets_loaded": true,
  "version": "1.0.0"
}
```

### POST `/evaluate`
Evaluate a single conversation turns-list across all 300 facets.

**Request Body (`application/json`):**
```json
{
  "conversation_id": "demo_conv_123",
  "domain": "customer_service",
  "mode": "feature",
  "turns": [
    {"speaker": "user", "text": "Hello, my order is missing!"},
    {"speaker": "assistant", "text": "I can help with that. Let me look up your account."}
  ]
}
```

**Response Body (`application/json`):**
```json
{
  "conversation_id": "demo_conv_123",
  "domain": "customer_service",
  "overall_score": 84.5,
  "overall_confidence": 0.88,
  "num_facets_evaluated": 300,
  "category_averages": {
    "linguistic_quality": 87.2,
    "pragmatics": 85.0,
    "safety": 99.1,
    "emotion_empathy": 76.5,
    "behavioral_traits": 80.2,
    "intelligence_reasoning": 79.0
  },
  "facet_scores": {
    "Grammar": {
      "score": 5,
      "confidence": 0.95,
      "evidence": "Polished text, 0 grammatical syntax issues observed.",
      "method": "feature"
    }
  },
  "evaluation_mode": "feature",
  "latency_seconds": 0.124
}
```

---

## 🧪 Demo Notebook

Explore [notebooks/demo_evaluation_pipeline.ipynb](notebooks/demo_evaluation_pipeline.ipynb) to walk through the pipeline step-by-step with visualizations, radar plots, and category breakdowns.

---

## 🛡️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
