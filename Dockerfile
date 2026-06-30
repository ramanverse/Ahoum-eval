# =============================================================================
# Ahoum Conversation Evaluation — Multi-stage Dockerfile
# =============================================================================
# Stage 1: builder — installs all Python deps into a virtualenv
# Stage 2: runtime — slim image that copies only the venv + app code
# =============================================================================

# --------------------------------------------------------------------------- #
# Stage 1: Builder
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps using lean CPU-only requirements (avoids ~650MB CUDA libs)
COPY requirements-docker.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Download NLP models into the venv layer
RUN python -m spacy download en_core_web_sm || true
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)" || true


# --------------------------------------------------------------------------- #
# Stage 2: Runtime base
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy application code
COPY src/       ./src/
COPY api/       ./api/
COPY ui/        ./ui/
COPY config.yaml .

# Copy pre-generated data (avoids rebuilding inside container)
COPY data/processed/facets_cleaned.json    ./data/processed/facets_cleaned.json
COPY data/raw/generated_conversations.json ./data/raw/generated_conversations.json
COPY data/examples/sample_evaluations_50.json ./data/examples/sample_evaluations_50.json

# Create writable runtime dirs
RUN mkdir -p data/processed data/raw data/examples .cache/model_cache logs

# Non-root user for security
RUN addgroup --system ahoum && adduser --system --ingroup ahoum ahoum
RUN chown -R ahoum:ahoum /app
USER ahoum


# --------------------------------------------------------------------------- #
# Stage 3a: Streamlit UI image
# --------------------------------------------------------------------------- #
FROM runtime AS ui

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]


# --------------------------------------------------------------------------- #
# Stage 3b: FastAPI API image
# --------------------------------------------------------------------------- #
FROM runtime AS api

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8080/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--log-level", "info"]
