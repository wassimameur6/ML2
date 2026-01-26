# Customer Churn Prevention System

A production-ready ML system for predicting customer churn and automatically sending personalized retention offers using RAG (Retrieval-Augmented Generation).

## Overview

This system combines classical machine learning (Random Forest classifier) with modern AI (OpenAI GPT + ChromaDB vector search) to:

1. **Predict** customer churn probability
2. **Retrieve** the best matching retention offer from a knowledge base
3. **Personalize** the offer using GPT
4. **Send** the personalized email to at-risk customers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                          │
│                   (webapp:8501)                              │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│                   (serving-api:8080)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │   Model     │  │   Agent     │  │    Vector Store      │ │
│  │ (sklearn)   │  │  (OpenAI)   │  │    (ChromaDB)        │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Production-Ready Components

- **Retry Logic**: Exponential backoff for API calls
- **Circuit Breaker**: Prevents cascading failures
- **Rate Limiting**: Protects against abuse
- **Request Tracking**: Unique request IDs and response times
- **Metrics Endpoint**: Prometheus-compatible metrics
- **Health Checks**: Detailed health status for monitoring
- **Thread Safety**: Safe for concurrent requests
- **CORS Support**: Cross-origin request handling
- **Input Validation**: Pydantic-based request validation

### Business Logic

| Risk Level | Probability | Action |
|------------|-------------|--------|
| HIGH | ≥75% | AI agent sends personalized retention offer |
| MEDIUM | 50-74% | Warning, customer flagged for monitoring |
| LOW | <50% | No immediate action needed |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key

### 1. Clone and Configure

```bash
cd ML2
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and OPENROUTER_API_KEY
```

### OpenRouter Setup (Client Chat)

Add the following to your `.env` file to enable the client chat agent:

```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### 2. Run with Docker

```bash
docker-compose up --build
```

### 3. Access

- **Web UI**: http://localhost:8501
- **Client Chat UI**: http://localhost:8502
- **API Docs**: http://localhost:8080/docs
- **Metrics**: http://localhost:8080/metrics

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch predictions |
| `/agent/test` | POST | Test AI agent directly |

### Example Request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CLIENTNUM": 123456789,
    "Customer_Age": 45,
    "Gender": "M",
    "Dependent_count": 2,
    "Education_Level": "Graduate",
    "Marital_Status": "Married",
    "Income_Category": "$60K - $80K",
    "Card_Category": "Blue",
    "Months_on_book": 36,
    "Total_Relationship_Count": 4,
    "Months_Inactive_12_mon": 2,
    "Contacts_Count_12_mon": 3,
    "Credit_Limit": 10000.0,
    "Total_Revolving_Bal": 1500.0,
    "Avg_Open_To_Buy": 8500.0,
    "Total_Amt_Chng_Q4_Q1": 1.5,
    "Total_Trans_Amt": 5000.0,
    "Total_Trans_Ct": 50,
    "Total_Ct_Chng_Q4_Q1": 1.2,
    "Avg_Utilization_Ratio": 0.15
  }'
```

## Project Structure

```
ML2/
├── agent/                    # RAG AI Agent
│   ├── churn_agent.py       # Main agent logic
│   └── offer_vectorstore.py # ChromaDB vector store
├── serving/                  # FastAPI backend
│   ├── api.py               # REST API endpoints
│   ├── Dockerfile.full      # Production Dockerfile
│   └── requirements.txt
├── webapp/                   # Streamlit frontend
│   ├── app.py               # Web UI
│   ├── client_app.py        # Client chat UI
│   ├── client_api.py        # Client API helpers
│   ├── client_ui.py         # Client UI components
│   ├── client_styles.py     # Client UI styles
│   ├── Dockerfile
│   └── requirements.txt
├── artifacts/                # ML model files
│   ├── model.pickle
│   ├── scaler.pickle
│   ├── label_encoders.pickle
│   └── feature_cols.pickle
├── data/                     # Data files
│   ├── churn2.csv           # Customer database
│   └── retention_offers.json # Offer templates
├── scripts/                  # Utility scripts
│   ├── train_model.py
│   └── test_rag.py
├── tests/                    # Test suite
├── docker-compose.yml
├── .env                      # Environment config
└── README.md
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `OPENAI_MODEL` | Model for personalization | `gpt-3.5-turbo` |
| `OPENROUTER_API_KEY` | OpenRouter API key (client chat) | (required for chat) |
| `OPENROUTER_BASE_URL` | OpenRouter base URL | `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | OpenRouter model | `openai/gpt-4o-mini` |
| `OPENROUTER_APP_URL` | OpenRouter app URL for telemetry | `http://localhost` |
| `OPENROUTER_APP_TITLE` | OpenRouter app title | `Serfy Bank Client Service` |
| `SMTP_HOST` | Email server | `smtp.gmail.com` |
| `SMTP_PORT` | Email port | `587` |
| `SMTP_USER` | Email username | (optional) |
| `SMTP_PASSWORD` | Email password | (optional) |
| `COMPANY_NAME` | Company name for emails | `Premium Bank` |
| `RATE_LIMIT_REQUESTS` | Max requests per window | `100` |
| `RATE_LIMIT_WINDOW` | Rate limit window (seconds) | `60` |

## Development

### Run Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Run Locally (without Docker)

```bash
# Terminal 1: API
cd serving
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8080

# Terminal 2: Web UI
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

### Train Model

```bash
python scripts/train_model.py
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "checks": {
    "model_loaded": true,
    "scaler_loaded": true,
    "customer_db_loaded": true,
    "customer_count": 10127
  },
  "version": "2.0.0"
}
```

### Metrics

```bash
curl http://localhost:8080/metrics
```

Response:
```json
{
  "api_metrics": {
    "requests_total": 150,
    "predictions_total": 45,
    "agents_triggered": 12,
    "errors_total": 2,
    "avg_response_time_ms": 234.5
  },
  "agent_metrics": {
    "total_tokens_used": 15000,
    "total_requests": 12,
    "circuit_breaker_state": "closed"
  }
}
```

## Technology Stack

- **ML**: scikit-learn (Random Forest)
- **AI**: OpenAI GPT-3.5-turbo, text-embedding-3-small
- **Vector DB**: ChromaDB
- **Backend**: FastAPI, Pydantic
- **Frontend**: Streamlit
- **Containerization**: Docker, Docker Compose

## License

MLOps Workshop 2025-2026
