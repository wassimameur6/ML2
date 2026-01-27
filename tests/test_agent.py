"""
Unit tests for the Churn Prevention Agent.
"""
import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.churn_agent import (
    ChurnAgent,
    CustomerProfile,
    PredictionResult,
    OfferRecommendation,
    EmailService
)


class TestCustomerProfile:
    """Tests for the CustomerProfile class."""

    def test_from_dict(self):
        """Test creating CustomerProfile from dictionary."""
        data = {
            "CLIENTNUM": 123456789,
            "First_Name": "John",
            "Last_Name": "Doe",
            "Email": "john@example.com",
            "Phone_Number": "555-1234",
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

        profile = CustomerProfile.from_dict(data)

        assert profile.client_num == 123456789
        assert profile.first_name == "John"
        assert profile.last_name == "Doe"
        assert profile.age == 45
        assert profile.income_category == "$60K - $80K"

    def test_full_name(self):
        """Test full_name property."""
        data = {"First_Name": "Jane", "Last_Name": "Smith"}
        profile = CustomerProfile.from_dict(data)
        assert profile.full_name == "Jane Smith"

    def test_to_description(self):
        """Test natural language description generation."""
        data = {
            "Income_Category": "$60K - $80K",
            "Card_Category": "Blue",
            "Months_on_book": 36,
            "Months_Inactive_12_mon": 1,
            "Total_Trans_Amt": 6000,
            "Avg_Utilization_Ratio": 0.3,
            "Credit_Limit": 10000,
            "Customer_Age": 45,
            "Marital_Status": "Married"
        }
        profile = CustomerProfile.from_dict(data)
        desc = profile.to_description()

        assert "$60K - $80K" in desc
        assert "Blue" in desc
        assert "36 months" in desc


class TestPredictionResult:
    """Tests for the PredictionResult class."""

    def test_low_risk_classification(self):
        """Test low risk classification."""
        result = PredictionResult.from_probability(123, 0.2)
        assert result.churn_risk == "low"
        assert result.is_churning is False

    def test_medium_risk_classification(self):
        """Test medium risk classification."""
        result = PredictionResult.from_probability(123, 0.45)
        assert result.churn_risk == "medium"
        assert result.is_churning is False

    def test_high_risk_classification(self):
        """Test high risk classification."""
        result = PredictionResult.from_probability(123, 0.8)
        assert result.churn_risk == "high"
        assert result.is_churning is True

    def test_boundary_churning_threshold(self):
        """Test churning boundary at 0.5."""
        result_below = PredictionResult.from_probability(123, 0.49)
        result_at = PredictionResult.from_probability(123, 0.5)

        assert result_below.is_churning is False
        assert result_at.is_churning is True


class TestChurnAgent:
    """Tests for the ChurnAgent class."""

    def test_agent_initialization(self):
        """Test agent initializes without error when API key is present."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        agent = ChurnAgent()
        assert agent is not None
        assert agent.model is not None

    def test_agent_has_email_service(self):
        """Test agent has email service configured."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        agent = ChurnAgent()
        assert agent.email_service is not None


class TestEmailService:
    """Tests for the EmailService class."""

    def test_email_service_configuration(self):
        """Test email service reads environment variables."""
        service = EmailService()
        assert service.smtp_host == os.getenv('SMTP_HOST', 'smtp.gmail.com')
        assert service.smtp_port == int(os.getenv('SMTP_PORT', 587))

    def test_is_configured(self):
        """Test is_configured returns correct status."""
        service = EmailService()
        # Should be configured if SMTP_USER is set
        expected = bool(os.getenv('SMTP_USER', ''))
        assert service.is_configured() == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
