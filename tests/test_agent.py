"""
Unit tests for the Churn Prevention Agent.
"""
import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.churn_agent import (
    ChurnPreventionAgent,
    get_agent,
    reset_agent,
    retry_with_backoff,
    CircuitBreaker
)


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in closed state."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)
        assert cb.state == "closed"
        assert cb.can_execute() is True

    def test_opens_after_threshold_failures(self):
        """Circuit breaker should open after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"
        assert cb.can_execute() is False

    def test_success_resets_failures(self):
        """Recording success should reset failure count."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb.failures == 0
        assert cb.state == "closed"


class TestRetryDecorator:
    """Tests for the retry_with_backoff decorator."""

    def test_successful_function_returns_immediately(self):
        """Function that succeeds should return without retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_failure(self):
        """Function should retry on failure."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        """Should raise exception after max retries exhausted."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent error")

        with pytest.raises(ValueError):
            always_fails()

        assert call_count == 3


class TestChurnPreventionAgent:
    """Tests for the ChurnPreventionAgent class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset agent before each test."""
        reset_agent()
        yield
        reset_agent()

    def test_singleton_returns_same_instance(self):
        """get_agent should return the same instance."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        agent1 = get_agent()
        agent2 = get_agent()
        assert agent1 is agent2

    def test_agent_requires_api_key(self):
        """Agent should raise error if no API key."""
        # Temporarily remove API key
        original_key = os.environ.pop("OPENAI_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                ChurnPreventionAgent(initialize_db=False)
        finally:
            # Restore API key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_retrieve_best_offers(self):
        """Test offer retrieval from vector store."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        agent = get_agent()

        customer_profile = {
            "income_category": "$60K - $80K",
            "card_category": "Blue",
            "months_on_book": 36,
            "churn_probability": 0.8,
            "age": 45,
            "total_trans_amt": 5000
        }

        offers, retrieval_time = agent.retrieve_best_offers(customer_profile)

        assert isinstance(offers, list)
        assert len(offers) > 0
        assert retrieval_time >= 0

    def test_usage_stats_tracking(self):
        """Test that usage stats are tracked."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        agent = get_agent()
        stats = agent.get_usage_stats()

        assert "total_tokens_used" in stats
        assert "total_requests" in stats
        assert "model" in stats
        assert "circuit_breaker_state" in stats


class TestFallbackPersonalization:
    """Tests for fallback personalization when OpenAI is unavailable."""

    def test_fallback_returns_valid_response(self):
        """Fallback should return a valid response structure."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        agent = get_agent()

        offer = {
            "offer_id": "test_001",
            "title": "Test Offer",
            "email_subject": "Special offer from {company_name}",
            "email_body": "Dear customer, thank you for {tenure} months with us."
        }

        customer_data = {
            "months_on_book": 24
        }

        result = agent._fallback_personalization(
            offer, customer_data,
            offer["email_subject"], offer["email_body"]
        )

        assert result["success"] is True
        assert result["fallback"] is True
        assert result["offer_id"] == "test_001"
        assert "Premium Bank" in result["subject"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
