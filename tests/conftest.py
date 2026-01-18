"""
Pytest configuration and fixtures for the test suite.
"""
import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def api_key_available():
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing."""
    return {
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
    }


@pytest.fixture
def high_risk_customer_data(sample_customer_data):
    """Sample high-risk customer data."""
    data = sample_customer_data.copy()
    data["Months_Inactive_12_mon"] = 6
    data["Total_Trans_Ct"] = 10
    data["Total_Trans_Amt"] = 500
    return data


@pytest.fixture
def sample_customer_profile():
    """Sample customer profile for vector store queries."""
    return {
        "income_category": "$60K - $80K",
        "card_category": "Blue",
        "months_on_book": 36,
        "churn_probability": 0.75,
        "age": 45,
        "total_trans_amt": 5000
    }


@pytest.fixture
def sample_offer():
    """Sample retention offer for testing."""
    return {
        "offer_id": "test_001",
        "title": "Test Cashback Offer",
        "offer_type": "cashback",
        "value": "5% cashback",
        "email_subject": "Special offer from {company_name}",
        "email_body": "Dear customer, thank you for {tenure} months with us.",
        "target_profile": {
            "income_category": ["$60K - $80K", "$80K - $120K"],
            "card_category": ["Blue", "Silver"],
            "min_tenure_months": 12,
            "churn_risk": ["high", "medium"]
        }
    }
