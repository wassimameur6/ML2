FROM python:3.12-slim

# Metadata
LABEL maintainer="serfy-bank@example.com"
LABEL description="Churn Prediction Backend API with AI Agent"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Copy agent module
COPY agent/ ./agent/

# Copy processors directory (contains model copied by Jenkins)
COPY processors/ ./processors/

# Verify model exists (CRITICAL)
RUN if [ ! -f /app/processors/models/best_model_final.pkl ]; then \
        echo ""; \
        echo "ERROR: MODEL NOT FOUND!"; \
        echo ""; \
        echo "The file best_model_final.pkl does not exist!"; \
        echo ""; \
        echo "SOLUTION:"; \
        echo "   Jenkins must run this script BEFORE build:"; \
        echo "   → python Jenkins/register_best_model.py"; \
        echo ""; \
        echo "Current content of /app/processors/:"; \
        ls -lah /app/processors/ 2>/dev/null || echo "   (empty)"; \
        echo ""; \
        exit 1; \
    fi && \
    echo "Model found: /app/processors/models/best_model_final.pkl" && \
    ls -lh /app/processors/models/best_model_final.pkl

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
