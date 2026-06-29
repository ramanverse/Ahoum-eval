# Ahoum Conversation Evaluation System

A production-ready, scalable, and highly configurable multi-facet conversation evaluation benchmark system that scores conversation turns and whole sessions on 300 behavioral and linguistic facets using open-weight Large Language Models (LLMs) and advanced feature engineering.

Developed for AI/ML engineering contexts to establish reliable, reproducible, and robust metric scoring systems.

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
| scorer.py (Phase 6)     | <--- Mode 1: Feature-Based Scoring (Fast)
|                         | <--- Mode 2: LLM-Augmented Scoring (Accurate)
|                         | <--- Mode 3: Hybrid Blend
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

## 🚀 Installation & Setup

### Option 1: Using pip

1. **Clone the project & navigate to directory:**
   ```bash
   cd ahoum-conversation-eval
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Initialize dataset generation & processing pipeline:**
   ```bash
   # 1. Clean raw facets assignment csv
   python src/data_loader.py --facets data/raw/facets_assignment.csv --output data/processed/facets_cleaned.json
   
   # 2. Generate synthetic conversations
   python src/data_generator.py --output data/raw/generated_conversations.json
   
   # 3. Preprocess conversations
   python src/preprocessing.py --input data/raw/generated_conversations.json --output data/processed/conversations_preprocessed.parquet
   
   # 4. Extract features
   python src/feature_extractor.py --input data/processed/conversations_preprocessed.parquet --output data/processed/conversation_features.parquet
   
   # 5. Generate 50 sample evaluations
   python src/generate_sample_evals.py
   ```

5. **Run tests:**
   ```bash
   pytest tests/
   ```

---

## 🏃 Usage Examples

### Running the Scorer

To run the scoring engine using the feature-based baseline mode:
```bash
python src/scorer.py --conversations data/raw/generated_conversations.json --facets data/processed/facets_cleaned.json --output data/processed/scores.parquet --mode feature
```

To run with **hybrid mode** (refining low-confidence predictions using LLM):
```bash
python src/scorer.py --conversations data/raw/generated_conversations.json --facets data/processed/facets_cleaned.json --output data/processed/scores.parquet --mode hybrid --model mistralai/Mistral-7B-Instruct-v0.1
```

### Launching the UI Dashboard
Start the interactive Streamlit dashboard:
```bash
streamlit run ui/app.py
```

### Launching the REST API
Start the FastAPI scoring endpoint:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## 🐳 Docker Deployment

The system is fully containerized. You can run both the Streamlit UI and the FastAPI server simultaneously using Docker Compose:

1. **Build and run the containers:**
   ```bash
   docker-compose up --build
   ```

2. **Access services:**
   - **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI Documentation (Swagger)**: [http://localhost:8080/docs](http://localhost:8080/docs)
   - **API Health Check**: [http://localhost:8080/health](http://localhost:8080/health)

---

## 📊 API Documentation

### POST `/evaluate`
Score a single conversation.

**Request Body:**
```json
{
  "conversation_id": "conv_001",
  "domain": "customer_service",
  "mode": "feature",
  "turns": [
    {"speaker": "user", "text": "My package has not arrived yet."},
    {"speaker": "assistant", "text": "I am sorry for the delay. Let me check your tracking information."}
  ]
}
```

---

## 🛡️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
