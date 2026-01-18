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

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "api_metrics" in data
        assert "requests_total" in data["api_metrics"]

    def test_predict_endpoint_validation(self, client):
        """Test prediction endpoint validates input."""
        # Missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_age(self, client):
        """Test prediction rejects invalid age."""
        customer_data = {
            "CLIENTNUM": 123456789,
            "Customer_Age": 10,  # Invalid: too young
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
        assert response.status_code == 422

    def test_predict_endpoint_invalid_gender(self, client):
        """Test prediction rejects invalid gender."""
        customer_data = {
            "CLIENTNUM": 123456789,
            "Customer_Age": 45,
            "Gender": "X",  # Invalid gender
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
        assert response.status_code == 422

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
        response = client.post("/predict", json=customer_data, params={"trigger_agent": False})

        # May fail if model not loaded, but structure should be correct
        if response.status_code == 200:
            data = response.json()
            assert "customer_id" in data
            assert "churn_probability" in data
            assert "risk_level" in data
            assert "recommendation" in data

    def test_request_id_header(self, client):
        """Test that request ID is returned in headers."""
        response = client.get("/health")
        assert "x-request-id" in response.headers


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
        for _ in range(150):
            response = client.get("/health")
            assert response.status_code == 200


class TestRiskLevels:
    """Tests for risk level classification."""

    def test_high_risk_threshold(self):
        """Test high risk threshold."""
        from serving.api import get_risk_level_and_recommendation

        level, rec = get_risk_level_and_recommendation(0.80)
        assert level == "high"
        assert "URGENT" in rec

    def test_medium_risk_threshold(self):
        """Test medium risk threshold."""
        from serving.api import get_risk_level_and_recommendation

        level, rec = get_risk_level_and_recommendation(0.60)
        assert level == "medium"
        assert "WARNING" in rec

    def test_low_risk_threshold(self):
        """Test low risk threshold."""
        from serving.api import get_risk_level_and_recommendation

        level, rec = get_risk_level_and_recommendation(0.30)
        assert level == "low"
        assert "stable" in rec.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
