"""
FastAPI Backend - REST API for Churn Prediction and Retention System
With AI Agent: RAG + LLM for intelligent offer selection and email generation

Combined API supporting:
- Single and batch predictions
- CSV upload for bulk predictions
- AI Agent pipeline (RAG + LLM)
- Human-in-the-Loop feedback
"""
import os
import sys
import io
import json
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.churn_agent import ChurnAgent, CustomerProfile, PredictionResult, OfferRecommendation
from agent.offer_vectorstore import OfferVectorStore
from agent.customer_repository import CustomerRepository
from agent.openrouter_client import OpenRouterClient
from agent.client_chat_agent import ClientChatAgent


app = FastAPI(
    title="Churn Prediction & Retention AI Agent API",
    description="""
    AI-powered customer churn prediction and personalized retention system.

    **Features:**
    - Single and batch predictions
    - CSV file upload for bulk predictions
    - AI Agent Pipeline (RAG + LLM)
    - Human-in-the-Loop feedback collection

    **AI Agent Pipeline:**
    1. ML Model predicts churn probability
    2. RAG (semantic search) finds relevant offers
    3. LLM ranks and selects the best offer
    4. LLM generates personalized email content
    5. Human-in-the-Loop feedback collection
    """,
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

# Paths configuration
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# In Docker, BACKEND_DIR is /app, so data is at /app/data (mounted volume)
# Locally, go up to project root
if os.path.exists(os.path.join(BACKEND_DIR, 'data')):
    DATA_PATH = os.path.join(BACKEND_DIR, 'data')
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BACKEND_DIR))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
PROCESSORS_DIR = os.path.join(BACKEND_DIR, "processors")
MODEL_PATH = os.path.join(PROCESSORS_DIR, "models", "best_model_final.pkl")
PREPROCESSOR_PATH = os.path.join(PROCESSORS_DIR, "preprocessor.pkl")
FEATURE_NAMES_PATH = os.path.join(PROCESSORS_DIR, "feature_names.pkl")
METADATA_PATH = os.path.join(PROCESSORS_DIR, "models", "best_model_final_metadata.pkl")

# Global variables for batch prediction
batch_model = None
preprocessor = None
feature_names = None
model_metadata = {}

# Initialize AI Agent
agent = ChurnAgent()
vectorstore = OfferVectorStore(data_path=DATA_PATH)
customer_repo = CustomerRepository(DATA_PATH)
openrouter_client = OpenRouterClient()
client_chat_agent = ClientChatAgent(customer_repo, openrouter_client, agent.email_service)


# ============================================================================
# Pydantic Models
# ============================================================================

class CustomerData(BaseModel):
    """Customer data for prediction"""
    CLIENTNUM: int
    Customer_Age: int
    Gender: str
    Dependent_count: int
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: float
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: float
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float


class PredictionResponse(BaseModel):
    client_num: int
    churn_probability: float
    churn_risk: str
    is_churning: bool


class OfferResponse(BaseModel):
    offer_id: str
    title: str
    description: str
    offer_type: str
    value: str
    relevance_score: float


class EmailRequest(BaseModel):
    client_num: int
    offer_id: str
    use_ai: bool = True  # Use RAG + LLM for personalization


class EmailResponse(BaseModel):
    success: bool
    message: str
    to_email: Optional[str] = None
    error: Optional[str] = None
    feedback_token: Optional[str] = None
    ai_generated: bool = False


class CampaignRequest(BaseModel):
    customer_ids: Optional[List[int]] = None
    risk_threshold: float = Field(default=0.5, ge=0, le=1)
    send_emails: bool = False
    max_customers: int = Field(default=100, ge=1, le=1000)


class ClientChatRequest(BaseModel):
    client_num: int = Field(..., description="Customer CLIENTNUM")
    message: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[Dict[str, Any]]] = None


class ClientChatResponse(BaseModel):
    message: str


class CampaignResponse(BaseModel):
    total_customers: int
    emails_sent: int
    emails_failed: int
    campaign_details: List[Dict[str, Any]]
    agent_config: Optional[Dict[str, Any]] = None


class CustomerSearchRequest(BaseModel):
    income_category: str
    card_category: str
    tenure_months: int
    churn_risk: str
    top_k: int = 3


class CustomerInput(BaseModel):
    """Schema for batch prediction input"""
    customer_age: int = Field(45, ge=18, le=100, description="Customer age")
    gender: str = Field("M", description="Gender (M/F)")
    dependent_count: int = Field(3, ge=0, le=10, description="Number of dependents")
    education_level: str = Field("Graduate", description="Education level")
    marital_status: str = Field("Married", description="Marital status")
    income_category: str = Field("$60K - $80K", description="Income category")
    card_category: str = Field("Blue", description="Card type")
    months_on_book: int = Field(39, ge=0, description="Tenure in months")
    total_relationship_count: int = Field(5, ge=1, le=6, description="Number of products")
    months_inactive_12_mon: int = Field(1, ge=0, le=12, description="Inactive months (last 12)")
    contacts_count_12_mon: int = Field(3, ge=0, description="Contacts (last 12 months)")
    credit_limit: float = Field(12691.0, ge=0, description="Credit limit")
    total_revolving_bal: int = Field(777, ge=0, description="Total revolving balance")
    avg_open_to_buy: float = Field(11914.0, ge=0, description="Average open to buy")
    total_amt_chng_q4_q1: float = Field(1.335, description="Amount change Q4/Q1")
    total_trans_amt: int = Field(1144, ge=0, description="Total transaction amount")
    total_trans_ct: int = Field(42, ge=0, description="Total transaction count")
    total_ct_chng_q4_q1: float = Field(1.625, description="Count change Q4/Q1")
    avg_utilization_ratio: float = Field(0.061, ge=0, le=1, description="Average utilization ratio")


# ============================================================================
# PREPROCESSING FUNCTIONS (for batch predictions)
# ============================================================================

def preprocess_raw_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply churn preprocessing (feature engineering)
    Replicates the logic from preprocessing.ipynb
    """
    df = df.copy()

    # Lowercase columns
    df.columns = df.columns.str.lower()

    # Convert to categorical
    categorical_cols = ['gender', 'education_level', 'marital_status', 'income_category', 'card_category']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Fill missing values
    if 'marital_status' in df.columns:
        df['marital_status'] = df['marital_status'].replace('Unknown', 'Married')

    if 'income_category' in df.columns:
        df['income_category'] = df['income_category'].replace('Unknown', 'Less than $40K')

    # Feature Engineering
    if 'months_on_book' in df.columns and 'customer_age' in df.columns:
        df['tenure_per_age'] = df['months_on_book'] / (df['customer_age'] * 12)

    if 'avg_utilization_ratio' in df.columns and 'customer_age' in df.columns:
        df['utilisation_per_age'] = df['avg_utilization_ratio'] / df['customer_age']

    if 'credit_limit' in df.columns and 'customer_age' in df.columns:
        df['credit_lim_per_age'] = df['credit_limit'] / df['customer_age']

    if 'total_trans_amt' in df.columns and 'credit_limit' in df.columns:
        df['total_trans_amt_per_credit_lim'] = df['total_trans_amt'] / df['credit_limit']

    if 'total_trans_ct' in df.columns and 'credit_limit' in df.columns:
        df['total_trans_ct_per_credit_lim'] = df['total_trans_ct'] / df['credit_limit']

    return df


def apply_preprocessor(df: pd.DataFrame, preprocessor_obj, feature_names_list) -> np.ndarray:
    """Apply sklearn ColumnTransformer"""
    X_transformed = preprocessor_obj.transform(df)

    if feature_names_list is not None:
        expected_features = len(feature_names_list)
        actual_features = X_transformed.shape[1]

        if expected_features != actual_features:
            print(f"Warning: Expected {expected_features} features, got {actual_features}")

    return X_transformed


# ============================================================================
# STARTUP EVENT - Load batch prediction model
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global batch_model, preprocessor, feature_names, model_metadata

    print("="*80)
    print("STARTING CHURN PREDICTION API")
    print("="*80)

    # 1. Load Preprocessor
    try:
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded: {PREPROCESSOR_PATH}")
    except Exception as e:
        print(f"Preprocessor not found: {e}")
        preprocessor = None

    # 2. Load Feature Names
    try:
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"Feature names loaded: {len(feature_names)} features")
    except Exception as e:
        print(f"Feature names not found: {e}")
        feature_names = None

    # 3. Load Model
    try:
        with open(MODEL_PATH, 'rb') as f:
            batch_model = pickle.load(f)
        print(f"Batch model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"Batch model not found: {e}")
        batch_model = None

    # 4. Load Metadata
    try:
        with open(METADATA_PATH, 'rb') as f:
            model_metadata = pickle.load(f)
        print(f"Model metadata loaded")
    except Exception as e:
        print(f"Metadata not available: {e}")
        model_metadata = {}

    print("="*80)
    print("AI Agent initialized")
    print(f"  - RAG: {'enabled' if agent.vectorstore else 'disabled'}")
    print(f"  - LLM: {'enabled' if agent.openai_client else 'disabled'}")
    print(f"  - Batch prediction: {'enabled' if batch_model and preprocessor else 'disabled'}")
    print("="*80)


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page"""
    index_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "webapp", "index.html"
    )
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": agent.model is not None,
        "batch_model_loaded": batch_model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "vectorstore_ready": agent.vectorstore is not None,
        "email_configured": agent.email_service.is_configured(),
        "openai_configured": agent.openai_client is not None,
        "ai_agent_status": {
            "rag": "enabled" if agent.vectorstore else "disabled",
            "llm_ranking": "enabled" if agent.openai_client else "disabled",
            "llm_email_gen": "enabled" if agent.openai_client else "disabled"
        }
    }


@app.get("/model-info")
def get_model_info():
    """Get loaded model information"""
    if not model_metadata:
        return {"error": "Model metadata not available"}

    return {
        "model_name": model_metadata.get('model_name'),
        "model_type": model_metadata.get('model_type'),
        "metrics": model_metadata.get('metrics'),
        "training_time_sec": model_metadata.get('training_time_sec'),
        "timestamp": model_metadata.get('timestamp'),
        "global_score": model_metadata.get('global_score')
    }


@app.get("/features")
def get_features():
    """List of expected features"""
    if feature_names is None:
        return {"error": "Feature names not available"}

    return {
        "total_features": len(feature_names),
        "feature_names": feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names)
    }


# ============================================================================
# Customer Endpoints
# ============================================================================

@app.get("/customers", response_model=List[Dict])
async def get_customers(
    limit: int = Query(default=100, ge=1, le=10000),
    offset: int = Query(default=0, ge=0)
):
    """Get list of all customers"""
    customers = agent.get_all_customers()
    return customers[offset:offset + limit]


@app.get("/customers/{client_num}")
async def get_customer(client_num: int):
    """Get customer by client number"""
    customer = agent.get_customer(client_num)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    return {
        "client_num": customer.client_num,
        "name": customer.full_name,
        "email": customer.email,
        "phone": customer.phone,
        "age": customer.age,
        "gender": customer.gender,
        "income_category": customer.income_category,
        "card_category": customer.card_category,
        "months_on_book": customer.months_on_book,
        "credit_limit": customer.credit_limit,
        "total_trans_amt": customer.total_trans_amt,
        "months_inactive": customer.months_inactive_12_mon
    }


@app.get("/customers/{client_num}/predict", response_model=PredictionResponse)
async def predict_customer_churn(client_num: int):
    """Predict churn probability for a specific customer"""
    customer = agent.get_customer(client_num)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer_data = agent.customers_df[
        agent.customers_df['CLIENTNUM'] == client_num
    ].iloc[0].to_dict()

    prediction = agent.predict(customer_data)

    return PredictionResponse(
        client_num=prediction.client_num,
        churn_probability=prediction.churn_probability,
        churn_risk=prediction.churn_risk,
        is_churning=prediction.is_churning
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict churn for provided customer data"""
    prediction = agent.predict(customer.model_dump())

    return PredictionResponse(
        client_num=prediction.client_num,
        churn_probability=prediction.churn_probability,
        churn_risk=prediction.churn_risk,
        is_churning=prediction.is_churning
    )


# ============================================================================
# Batch Prediction Endpoints
# ============================================================================

@app.post("/predict-batch")
async def predict_batch(customers: List[CustomerInput]):
    """Batch prediction for multiple customers"""
    if not batch_model or not preprocessor:
        raise HTTPException(status_code=503, detail="Batch prediction service not available")

    try:
        # Convert to DataFrame
        df_input = pd.DataFrame([c.dict() for c in customers])

        # 1. Feature Engineering
        df_processed = preprocess_raw_churn(df_input)

        # 2. Apply preprocessor
        X = apply_preprocessor(df_processed, preprocessor, feature_names)

        # 3. Predict
        predictions = batch_model.predict(X)

        # 4. Predict proba
        probas = None
        if hasattr(batch_model, 'predict_proba'):
            probas = batch_model.predict_proba(X)

        # Format results
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "index": i,
                "prediction": int(pred),
                "prediction_label": "Churn" if pred == 1 else "Non-Churn"
            }

            if probas is not None:
                result["probabilities"] = {
                    "non_churn": float(probas[i][0]),
                    "churn": float(probas[i][1])
                }

            results.append(result)

        return {
            "count": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    """Upload CSV, get predictions, download result"""
    if not batch_model or not preprocessor:
        raise HTTPException(status_code=503, detail="Batch prediction service not available")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Read CSV
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))

        print(f"CSV received: {len(df_input)} rows, {len(df_input.columns)} columns")

        # 1. Feature Engineering
        df_processed = preprocess_raw_churn(df_input)

        # 2. Apply preprocessor
        X = apply_preprocessor(df_processed, preprocessor, feature_names)

        # 3. Predict
        predictions = batch_model.predict(X)

        # 4. Probabilities
        if hasattr(batch_model, 'predict_proba'):
            probas = batch_model.predict_proba(X)
            df_result = df_input.copy()
            df_result['churn_prediction'] = predictions
            df_result['proba_non_churn'] = probas[:, 0]
            df_result['proba_churn'] = probas[:, 1]
        else:
            df_result = df_input.copy()
            df_result['churn_prediction'] = predictions

        # Save to buffer
        output = io.StringIO()
        df_result.to_csv(output, index=False)
        output.seek(0)

        # Return file
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")


# ============================================================================
# AI Agent Endpoints - RAG + LLM
# ============================================================================

@app.get("/customers/{client_num}/recommendations", response_model=List[OfferResponse])
async def get_recommendations(
    client_num: int,
    top_k: int = Query(default=3, ge=1, le=10),
    use_rag: bool = Query(default=True, description="Use RAG semantic search"),
    use_llm: bool = Query(default=True, description="Use LLM for ranking")
):
    """
    Get AI-powered offer recommendations for a customer.

    Pipeline:
    1. RAG: Semantic search in vector store for relevant offers
    2. LLM: Intelligent ranking based on customer profile
    """
    customer = agent.get_customer(client_num)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer_data = agent.customers_df[
        agent.customers_df['CLIENTNUM'] == client_num
    ].iloc[0].to_dict()

    prediction = agent.predict(customer_data)

    # Use AI Agent pipeline
    offers = agent.recommend_offers(
        customer, prediction,
        top_k=top_k,
        use_rag=use_rag,
        use_llm_ranking=use_llm
    )

    return [
        OfferResponse(
            offer_id=o.offer_id,
            title=o.title,
            description=o.description,
            offer_type=o.offer_type,
            value=o.value,
            relevance_score=o.relevance_score
        )
        for o in offers
    ]


@app.get("/customers/{client_num}/ai-recommendation")
async def get_ai_recommendation(client_num: int):
    """
    Get full AI analysis and recommendation for a customer.
    Uses RAG + LLM to select offers and explain the decision.
    """
    customer = agent.get_customer(client_num)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer_data = agent.customers_df[
        agent.customers_df['CLIENTNUM'] == client_num
    ].iloc[0].to_dict()

    prediction = agent.predict(customer_data)
    offers = agent.recommend_offers(customer, prediction, use_rag=True, use_llm_ranking=True)

    # Get AI explanation
    ai_recommendation = agent.generate_ai_recommendation(customer, prediction, offers)

    return {
        "client_num": client_num,
        "customer_name": customer.full_name,
        "churn_risk": prediction.churn_risk,
        "churn_probability": prediction.churn_probability,
        "ai_recommendation": ai_recommendation,
        "recommended_offers": [{"id": o.offer_id, "title": o.title, "score": o.relevance_score} for o in offers],
        "agent_pipeline": {
            "rag_used": agent.vectorstore is not None,
            "llm_used": agent.openai_client is not None
        }
    }


@app.get("/high-risk-customers")
async def get_high_risk_customers(
    threshold: float = Query(default=0.5, ge=0, le=1),
    limit: int = Query(default=50, ge=1, le=500)
):
    """Get customers with high churn risk"""
    high_risk = agent.get_high_risk_customers(threshold)[:limit]
    return {
        "total_high_risk": len(agent.get_high_risk_customers(threshold)),
        "returned": len(high_risk),
        "threshold": threshold,
        "customers": high_risk
    }


# ============================================================================
# Client Chat Endpoint (OpenRouter)
# ============================================================================

@app.post("/client/chat", response_model=ClientChatResponse)
async def client_chat(request: ClientChatRequest):
    """Chat endpoint for client-facing UI using OpenRouter."""
    try:
        reply = await client_chat_agent.reply(
            client_num=request.client_num,
            message=request.message,
            history=request.history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {exc}") from exc

    return ClientChatResponse(message=reply)


# ============================================================================
# Offer Endpoints
# ============================================================================

@app.get("/offers")
async def get_all_offers():
    """Get all available retention offers"""
    return vectorstore.get_all_offers()


@app.get("/offers/{offer_id}")
async def get_offer(offer_id: str):
    """Get specific offer by ID"""
    offer = vectorstore.get_offer_by_id(offer_id)
    if not offer:
        raise HTTPException(status_code=404, detail="Offer not found")
    return offer


@app.post("/offers/search")
async def search_offers(request: CustomerSearchRequest):
    """Search offers using RAG for a customer profile"""
    offers = vectorstore.search_for_customer(
        income_category=request.income_category,
        card_category=request.card_category,
        tenure_months=request.tenure_months,
        churn_risk=request.churn_risk,
        n_results=request.top_k
    )
    return {"offers": offers}


# ============================================================================
# Email & Campaign Endpoints
# ============================================================================

@app.post("/send-email", response_model=EmailResponse)
async def send_retention_email(request: EmailRequest):
    """
    Send AI-generated retention email to a customer.

    When use_ai=True:
    - LLM generates personalized email content
    - Includes feedback link for Human-in-the-Loop
    """
    customer = agent.get_customer(request.client_num)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    offer = vectorstore.get_offer_by_id(request.offer_id)
    if not offer:
        raise HTTPException(status_code=404, detail="Offer not found")

    # Get prediction for AI personalization
    customer_data = agent.customers_df[
        agent.customers_df['CLIENTNUM'] == request.client_num
    ].iloc[0].to_dict()
    prediction = agent.predict(customer_data)

    offer_rec = OfferRecommendation(
        offer_id=offer['offer_id'],
        title=offer['title'],
        description=offer['description'],
        offer_type=offer['offer_type'],
        value=offer['value'],
        relevance_score=1.0,
        email_subject=offer['email_subject'],
        email_body=offer['email_body']
    )

    # Send with AI features
    result = agent.send_retention_email(
        customer, offer_rec, prediction,
        use_llm_email=request.use_ai,
        include_feedback_link=True
    )

    return EmailResponse(
        success=result['success'],
        message=result.get('message', 'Email processed'),
        to_email=result.get('to'),
        error=result.get('error'),
        feedback_token=result.get('feedback_token'),
        ai_generated=request.use_ai and agent.openai_client is not None
    )


@app.post("/campaign", response_model=CampaignResponse)
async def run_retention_campaign(request: CampaignRequest):
    """
    Run AI-powered retention campaign.

    Pipeline for each customer:
    1. Predict churn risk
    2. RAG: Find relevant offers via semantic search
    3. LLM: Rank and select best offer
    4. LLM: Generate personalized email
    5. Send email with feedback link
    """
    result = agent.run_retention_campaign(
        customer_ids=request.customer_ids,
        risk_threshold=request.risk_threshold,
        send_emails=request.send_emails,
        max_customers=request.max_customers
    )

    return CampaignResponse(**result)


# ============================================================================
# Human-in-the-Loop Feedback Endpoints
# ============================================================================

def _load_pending_feedback() -> Dict:
    """Load pending feedback data"""
    feedback_file = os.path.join(DATA_PATH, 'pending_feedback.json')
    try:
        with open(feedback_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_pending_feedback(data: Dict):
    """Save pending feedback data"""
    feedback_file = os.path.join(DATA_PATH, 'pending_feedback.json')
    with open(feedback_file, 'w') as f:
        json.dump(data, f, indent=2)


def _append_to_prod_data(feedback_entry: Dict):
    """Append feedback to prod_data.csv for model retraining"""
    import pandas as pd
    prod_file = os.path.join(DATA_PATH, 'prod_data.csv')

    new_row = {
        'client_num': feedback_entry['client_num'],
        'offer_id': feedback_entry['offer_id'],
        'churn_probability': feedback_entry['churn_probability'],
        'feedback': feedback_entry['feedback'],
        'timestamp': feedback_entry['feedback_timestamp']
    }

    try:
        df = pd.read_csv(prod_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([new_row])

    df.to_csv(prod_file, index=False)


def _apply_offer_to_customer(feedback_entry: Dict):
    """
    Apply the accepted offer to the customer's record in churn2.csv.
    Updates customer data based on the offer type.
    """
    import pandas as pd

    if feedback_entry.get('feedback') != 'accept':
        return  # Only apply if accepted

    client_num = feedback_entry['client_num']
    offer_id = feedback_entry['offer_id']

    # Load offers to get offer details
    offers_file = os.path.join(DATA_PATH, 'retention_offers.json')
    try:
        with open(offers_file, 'r') as f:
            offers_data = json.load(f)
            # Handle both {"offers": [...]} and [...] formats
            offers = offers_data.get('offers', offers_data) if isinstance(offers_data, dict) else offers_data
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Could not load offers file")
        return

    # Find the offer
    offer = next((o for o in offers if isinstance(o, dict) and o.get('offer_id') == offer_id), None)
    if not offer:
        print(f"Offer {offer_id} not found")
        return

    # Load customer data
    churn_file = os.path.join(DATA_PATH, 'churn2.csv')
    try:
        df = pd.read_csv(churn_file)
    except FileNotFoundError:
        print(f"Could not load customer data file")
        return

    # Find customer
    customer_mask = df['CLIENTNUM'] == client_num
    if not customer_mask.any():
        print(f"Customer {client_num} not found")
        return

    # Apply offer based on type
    offer_type = offer.get('offer_type', '')

    if offer_type == 'upgrade':
        # Card upgrade: Blue -> Silver -> Gold -> Platinum
        card_upgrade = {'Blue': 'Silver', 'Silver': 'Gold', 'Gold': 'Platinum', 'Platinum': 'Platinum'}
        current_card = df.loc[customer_mask, 'Card_Category'].values[0]
        df.loc[customer_mask, 'Card_Category'] = card_upgrade.get(current_card, current_card)

    elif offer_type == 'credit_increase':
        # Increase credit limit by 20%
        df.loc[customer_mask, 'Credit_Limit'] = df.loc[customer_mask, 'Credit_Limit'] * 1.20

    elif offer_type == 'rate_reduction':
        # Lower utilization ratio (simulating lower rates = more spending power)
        df.loc[customer_mask, 'Avg_Utilization_Ratio'] = df.loc[customer_mask, 'Avg_Utilization_Ratio'] * 0.85

    elif offer_type == 'cashback':
        # Increase transaction amount (customer uses card more for cashback)
        df.loc[customer_mask, 'Total_Trans_Amt'] = df.loc[customer_mask, 'Total_Trans_Amt'] * 1.15
        df.loc[customer_mask, 'Total_Trans_Ct'] = df.loc[customer_mask, 'Total_Trans_Ct'] * 1.10

    elif offer_type == 'rewards':
        # Increase transaction count (more engagement)
        df.loc[customer_mask, 'Total_Trans_Ct'] = df.loc[customer_mask, 'Total_Trans_Ct'] * 1.15

    elif offer_type == 'fee_waiver':
        # Reduce months inactive (customer becomes more active)
        df.loc[customer_mask, 'Months_Inactive_12_mon'] = max(0, df.loc[customer_mask, 'Months_Inactive_12_mon'].values[0] - 1)

    elif offer_type == 'retention':
        # General retention: reduce inactive months, increase transactions
        df.loc[customer_mask, 'Months_Inactive_12_mon'] = max(0, df.loc[customer_mask, 'Months_Inactive_12_mon'].values[0] - 2)
        df.loc[customer_mask, 'Total_Trans_Ct'] = df.loc[customer_mask, 'Total_Trans_Ct'] * 1.10

    # Mark customer as "Existing Customer" if they were at risk (accepted retention offer)
    df.loc[customer_mask, 'Attrition_Flag'] = 'Existing Customer'

    # Save updated data
    df.to_csv(churn_file, index=False)
    print(f"Applied offer {offer_id} ({offer_type}) to customer {client_num}")


@app.get("/feedback/{token}/{action}", response_class=HTMLResponse)
async def process_feedback(token: str, action: str):
    """
    Process customer feedback from email link.
    This is the Human-in-the-Loop endpoint.

    - action: 'accept' or 'decline'
    """
    if action not in ['accept', 'decline']:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'accept' or 'decline'")

    pending = _load_pending_feedback()

    if token not in pending:
        return HTMLResponse(content="""
        <html>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>⚠️ Invalid or Expired Link</h1>
                <p>This feedback link is no longer valid.</p>
            </body>
        </html>
        """, status_code=404)

    feedback_entry = pending[token]

    if feedback_entry.get('status') != 'pending':
        return HTMLResponse(content=f"""
        <html>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>ℹ️ Already Recorded</h1>
                <p>Your feedback was already recorded as: <strong>{feedback_entry.get('feedback', 'unknown')}</strong></p>
            </body>
        </html>
        """)

    # Record feedback
    feedback_entry['status'] = 'completed'
    feedback_entry['feedback'] = action
    feedback_entry['feedback_timestamp'] = datetime.now().isoformat()

    pending[token] = feedback_entry
    _save_pending_feedback(pending)

    # Append to prod_data.csv for future model retraining
    _append_to_prod_data(feedback_entry)

    # Apply offer to customer data if accepted
    if action == 'accept':
        _apply_offer_to_customer(feedback_entry)

    # Return thank you page with Serfy Bank branding
    if action == 'accept':
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Thank You - Serfy Bank</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f5f5; min-height: 100vh; display: flex; align-items: center; justify-content: center;">
            <div style="background: #ffffff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); max-width: 500px; width: 90%; text-align: center; overflow: hidden;">
                <div style="background-color: #ffffff; padding: 30px; border-bottom: 3px solid #E5A229;">
                    <h1 style="margin: 0; color: #1a1a1a; font-size: 24px;">Serfy Bank</h1>
                </div>
                <div style="background: linear-gradient(90deg, #E5A229 0%, #C78B1F 100%); height: 4px;"></div>
                <div style="padding: 40px;">
                    <div style="width: 80px; height: 80px; background-color: #d4edda; border-radius: 50%; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 40px;">✓</span>
                    </div>
                    <h2 style="color: #28a745; margin: 0 0 15px 0;">Thank You!</h2>
                    <p style="color: #1a1a1a; font-size: 16px; margin: 0 0 10px 0;">We're glad you liked our offer, <strong>{feedback_entry['customer_name']}</strong>!</p>
                    <p style="color: #555; font-size: 14px; margin: 0;">Your offer has been activated. Thank you for being a valued customer.</p>
                </div>
                <div style="background-color: #1a1a1a; padding: 20px;">
                    <p style="margin: 0; color: #888; font-size: 12px;"><span style="color: #E5A229;">© 2025 Serfy Bank</span>. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """)
    else:
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Feedback Recorded - Serfy Bank</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f5f5; min-height: 100vh; display: flex; align-items: center; justify-content: center;">
            <div style="background: #ffffff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); max-width: 500px; width: 90%; text-align: center; overflow: hidden;">
                <div style="background-color: #ffffff; padding: 30px; border-bottom: 3px solid #E5A229;">
                    <h1 style="margin: 0; color: #1a1a1a; font-size: 24px;">Serfy Bank</h1>
                </div>
                <div style="background: linear-gradient(90deg, #E5A229 0%, #C78B1F 100%); height: 4px;"></div>
                <div style="padding: 40px;">
                    <div style="width: 80px; height: 80px; background-color: #FDF8EF; border-radius: 50%; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 40px;">📝</span>
                    </div>
                    <h2 style="color: #E5A229; margin: 0 0 15px 0;">Feedback Recorded</h2>
                    <p style="color: #1a1a1a; font-size: 16px; margin: 0 0 10px 0;">Thank you for your feedback, <strong>{feedback_entry['customer_name']}</strong>.</p>
                    <p style="color: #555; font-size: 14px; margin: 0;">We'll work on finding better offers that match your needs.</p>
                </div>
                <div style="background-color: #1a1a1a; padding: 20px;">
                    <p style="margin: 0; color: #888; font-size: 12px;"><span style="color: #E5A229;">© 2025 Serfy Bank</span>. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """)


@app.get("/feedback/stats")
@app.get("/feedback-stats")
async def get_feedback_stats():
    """Get feedback statistics for Human-in-the-Loop monitoring"""
    pending = _load_pending_feedback()

    total = len(pending)
    completed = sum(1 for f in pending.values() if f.get('status') == 'completed')
    accepted = sum(1 for f in pending.values() if f.get('feedback') == 'accept')
    declined = sum(1 for f in pending.values() if f.get('feedback') == 'decline')
    pending_count = sum(1 for f in pending.values() if f.get('status') == 'pending')

    return {
        "total_emails": total,
        "total_sent": total,
        "completed": completed,
        "accepted": accepted,
        "declined": declined,
        "pending_response": pending_count,
        "acceptance_rate": accepted / (accepted + declined) if (accepted + declined) > 0 else 0
    }


# ============================================================================
# Serve Static Frontend Files
# ============================================================================

WEBAPP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webapp")

@app.get("/styles.css")
async def serve_css():
    """Serve CSS file"""
    return FileResponse(os.path.join(WEBAPP_DIR, "styles.css"), media_type="text/css")

@app.get("/app.js")
async def serve_js():
    """Serve JavaScript file"""
    return FileResponse(os.path.join(WEBAPP_DIR, "app.js"), media_type="application/javascript")

@app.get("/srf.jpeg")
async def serve_logo():
    """Serve logo image"""
    return FileResponse(os.path.join(WEBAPP_DIR, "srf.jpeg"), media_type="image/jpeg")


# ============================================================================
# Statistics Endpoint
# ============================================================================

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    customers_df = agent.customers_df

    total_customers = len(customers_df)
    attrited = len(customers_df[customers_df['Attrition_Flag'] == 'Attrited Customer'])
    existing = total_customers - attrited

    # Get feedback stats
    pending = _load_pending_feedback()

    return {
        "total_customers": total_customers,
        "existing_customers": existing,
        "attrited_customers": attrited,
        "attrition_rate": attrited / total_customers,
        "total_offers": len(vectorstore.get_all_offers()),
        "email_service_configured": agent.email_service.is_configured(),
        "ai_agent": {
            "rag_enabled": agent.vectorstore is not None,
            "llm_enabled": agent.openai_client is not None
        },
        "feedback_stats": {
            "total_emails_sent": len(pending),
            "responses_received": sum(1 for f in pending.values() if f.get('status') == 'completed'),
            "pending": sum(1 for f in pending.values() if f.get('status') == 'pending')
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
