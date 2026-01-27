"""
Unit tests for the FastAPI prediction API.
"""
import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        # Import here to avoid issues with model loading
        from serving.api import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "service" in data

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "vectorstore_ready" in data

    def test_predict_endpoint_validation(self, client):
        """Test prediction endpoint validates input."""
        # Missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_success(self, client):
        """Test successful prediction."""
        customer_data = {
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
        response = client.post("/predict", json=customer_data)
        assert response.status_code == 200
        data = response.json()
        assert "client_num" in data
        assert "churn_probability" in data
        assert "churn_risk" in data
        assert "is_churning" in data

    def test_predict_low_risk(self, client):
        """Test prediction returns low risk for good customer."""
        customer_data = {
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
            "Months_Inactive_12_mon": 1,  # Low inactivity
            "Contacts_Count_12_mon": 2,
            "Credit_Limit": 10000.0,
            "Total_Revolving_Bal": 1500.0,
            "Avg_Open_To_Buy": 8500.0,
            "Total_Amt_Chng_Q4_Q1": 1.5,
            "Total_Trans_Amt": 8000.0,  # High transaction amount
            "Total_Trans_Ct": 80,  # High transaction count
            "Total_Ct_Chng_Q4_Q1": 1.2,
            "Avg_Utilization_Ratio": 0.15
        }
        response = client.post("/predict", json=customer_data)
        assert response.status_code == 200
        data = response.json()
        assert data["churn_risk"] in ["low", "medium"]

    def test_predict_high_risk(self, client):
        """Test prediction returns high risk for at-risk customer."""
        customer_data = {
            "CLIENTNUM": 999888777,
            "Customer_Age": 55,
            "Gender": "F",
            "Dependent_count": 0,
            "Education_Level": "College",
            "Marital_Status": "Single",
            "Income_Category": "Less than $40K",
            "Card_Category": "Blue",
            "Months_on_book": 48,
            "Total_Relationship_Count": 1,  # Low relationship count
            "Months_Inactive_12_mon": 6,  # High inactivity
            "Contacts_Count_12_mon": 5,  # High contacts (complaints?)
            "Credit_Limit": 2000.0,
            "Total_Revolving_Bal": 0.0,  # No balance
            "Avg_Open_To_Buy": 2000.0,
            "Total_Amt_Chng_Q4_Q1": 0.3,  # Low change
            "Total_Trans_Amt": 500.0,  # Very low transactions
            "Total_Trans_Ct": 5,  # Very low count
            "Total_Ct_Chng_Q4_Q1": 0.2,
            "Avg_Utilization_Ratio": 0.0
        }
        response = client.post("/predict", json=customer_data)
        assert response.status_code == 200
        data = response.json()
        assert data["churn_risk"] == "high"
        assert data["is_churning"] is True


class TestRateLimiting:
    """Tests for rate limiting."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from serving.api import app
        return TestClient(app)

    def test_rate_limit_not_applied_to_health(self, client):
        """Health endpoint should not be rate limited."""
        # Make many requests
        for _ in range(50):
            response = client.get("/health")
            assert response.status_code == 200


class TestPredictionResponse:
    """Tests for prediction response format."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from serving.api import app
        return TestClient(app)

    def test_response_has_correct_types(self, client):
        """Test response has correct data types."""
        customer_data = {
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
        response = client.post("/predict", json=customer_data)
        data = response.json()

        assert isinstance(data["client_num"], int)
        assert isinstance(data["churn_probability"], float)
        assert isinstance(data["churn_risk"], str)
        assert isinstance(data["is_churning"], bool)
        assert data["churn_risk"] in ["low", "medium", "high"]
        assert 0 <= data["churn_probability"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
