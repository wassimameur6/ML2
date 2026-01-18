FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .

# Copy agent module
COPY ../agent /app/agent

# Copy artifacts (will be mounted as volume in production)
COPY ../artifacts /app/artifacts

ENV ARTIFACTS_PATH=/app/artifacts
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
