"""
Test the API locally without Docker.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set artifacts path for local testing
os.environ["ARTIFACTS_PATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

from fastapi.testclient import TestClient
import serving.api as api_module
from serving.api import app

# Manually load model artifacts for testing
import pickle
artifacts_path = os.environ["ARTIFACTS_PATH"]
with open(f"{artifacts_path}/model.pickle", 'rb') as f:
    api_module.model = pickle.load(f)
with open(f"{artifacts_path}/scaler.pickle", 'rb') as f:
    api_module.scaler = pickle.load(f)
with open(f"{artifacts_path}/label_encoders.pickle", 'rb') as f:
    api_module.label_encoders = pickle.load(f)
with open(f"{artifacts_path}/feature_cols.pickle", 'rb') as f:
    api_module.feature_cols = pickle.load(f)

client = TestClient(app)

def test_health():
    response = client.get("/health")
    print("Health check:", response.json())
    assert response.status_code == 200

def test_prediction():
    # Sample customer data (likely to churn based on features)
    customer = {
        "CLIENTNUM": 999999999,
        "Customer_Age": 45,
        "Gender": "M",
        "Dependent_count": 2,
        "Education_Level": "Graduate",
        "Marital_Status": "Married",
        "Income_Category": "$60K - $80K",
        "Card_Category": "Blue",
        "Months_on_book": 36,
        "Total_Relationship_Count": 2,
        "Months_Inactive_12_mon": 4,  # High inactivity
        "Contacts_Count_12_mon": 4,   # High contacts (complaints?)
        "Credit_Limit": 5000,
        "Total_Revolving_Bal": 0,
        "Avg_Open_To_Buy": 5000,
        "Total_Amt_Chng_Q4_Q1": 0.5,   # Low change
        "Total_Trans_Amt": 1000,       # Low transactions
        "Total_Trans_Ct": 15,          # Low count
        "Total_Ct_Chng_Q4_Q1": 0.5,
        "Avg_Utilization_Ratio": 0.0
    }

    # Test without triggering agent
    response = client.post("/predict?trigger_agent=false", json=customer)
    print("\nPrediction result:")
    print(response.json())
    assert response.status_code == 200

if __name__ == "__main__":
    print("Testing API locally...\n")
    test_health()
    test_prediction()
    print("\n✅ All tests passed!")
