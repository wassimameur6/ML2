"""
Unit tests for the Offer Vector Store.
"""
import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.offer_vectorstore import (
    OfferVectorStore,
    get_vectorstore,
    initialize_vectorstore,
    reset_vectorstore
)


class TestOfferVectorStore:
    """Tests for the OfferVectorStore class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset vector store before each test."""
        reset_vectorstore()
        yield
        reset_vectorstore()

    def test_singleton_returns_same_instance(self):
        """get_vectorstore should return the same instance."""
        vs1 = get_vectorstore()
        vs2 = get_vectorstore()
        assert vs1 is vs2

    def test_churn_risk_level_low(self):
        """Test low churn risk classification."""
        vs = get_vectorstore()
        assert vs._get_churn_risk_level(0.3) == "low"

    def test_churn_risk_level_medium(self):
        """Test medium churn risk classification."""
        vs = get_vectorstore()
        assert vs._get_churn_risk_level(0.5) == "medium"

    def test_churn_risk_level_high(self):
        """Test high churn risk classification."""
        vs = get_vectorstore()
        assert vs._get_churn_risk_level(0.8) == "high"

    def test_adjacent_risk_matching(self):
        """Test adjacent risk level matching."""
        vs = get_vectorstore()

        # Medium can match low, medium, and high
        assert vs._check_adjacent_risk("medium", "low,medium") is True
        assert vs._check_adjacent_risk("medium", "high") is True

        # High can only match medium and high
        assert vs._check_adjacent_risk("high", "low") is False
        assert vs._check_adjacent_risk("high", "medium,high") is True

    def test_health_check(self):
        """Test health check returns status."""
        vs = get_vectorstore()
        health = vs.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]

    def test_find_best_offers_returns_list(self):
        """Test that find_best_offers returns a list."""
        # Skip if no API key (needed for embeddings)
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        vs = initialize_vectorstore()

        customer_profile = {
            "income_category": "$60K - $80K",
            "card_category": "Blue",
            "months_on_book": 36,
            "churn_probability": 0.8
        }

        offers = vs.find_best_offers(customer_profile, n_results=3)

        assert isinstance(offers, list)
        assert len(offers) <= 3

    def test_offers_have_required_fields(self):
        """Test that returned offers have required fields."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        vs = initialize_vectorstore()

        customer_profile = {
            "income_category": "$60K - $80K",
            "card_category": "Blue",
            "months_on_book": 36,
            "churn_probability": 0.8
        }

        offers = vs.find_best_offers(customer_profile)

        if offers:
            offer = offers[0]
            required_fields = ["offer_id", "title", "relevance_score", "matches"]
            for field in required_fields:
                assert field in offer, f"Missing field: {field}"

    def test_offers_sorted_by_relevance(self):
        """Test that offers are sorted by relevance score."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        vs = initialize_vectorstore()

        customer_profile = {
            "income_category": "$60K - $80K",
            "card_category": "Blue",
            "months_on_book": 36,
            "churn_probability": 0.8
        }

        offers = vs.find_best_offers(customer_profile, n_results=5)

        if len(offers) > 1:
            scores = [o["relevance_score"] for o in offers]
            assert scores == sorted(scores, reverse=True)


class TestQueryTextGeneration:
    """Tests for query text generation."""

    def test_query_text_includes_profile_info(self):
        """Test that query text includes customer profile info."""
        vs = get_vectorstore()

        profile = {
            "income_category": "$80K - $120K",
            "card_category": "Gold",
            "months_on_book": 48,
            "total_trans_amt": 10000,
            "churn_probability": 0.75
        }

        query = vs._create_query_text(profile, "high")

        assert "$80K - $120K" in query
        assert "Gold" in query
        assert "48" in query
        assert "high" in query


class TestCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_deterministic(self):
        """Test that same profile generates same cache key."""
        vs = get_vectorstore()

        profile = {
            "income_category": "$60K - $80K",
            "card_category": "Blue",
            "churn_probability": 0.75
        }

        key1 = vs._create_cache_key(profile)
        key2 = vs._create_cache_key(profile)

        assert key1 == key2

    def test_different_profiles_different_keys(self):
        """Test that different profiles generate different cache keys."""
        vs = get_vectorstore()

        profile1 = {
            "income_category": "$60K - $80K",
            "card_category": "Blue",
            "churn_probability": 0.75
        }

        profile2 = {
            "income_category": "$80K - $120K",
            "card_category": "Gold",
            "churn_probability": 0.80
        }

        key1 = vs._create_cache_key(profile1)
        key2 = vs._create_cache_key(profile2)

        assert key1 != key2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
