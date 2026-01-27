"""
Unit tests for the Offer Vector Store.
"""
import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.offer_vectorstore import OfferVectorStore


class TestOfferVectorStore:
    """Tests for the OfferVectorStore class."""

    @pytest.fixture
    def vectorstore(self):
        """Create a vectorstore instance for testing."""
        return OfferVectorStore()

    def test_vectorstore_initialization(self, vectorstore):
        """Test vectorstore initializes correctly."""
        assert vectorstore is not None
        assert vectorstore.offers is not None
        assert len(vectorstore.offers) > 0

    def test_offers_loaded(self, vectorstore):
        """Test that offers are loaded from JSON."""
        assert isinstance(vectorstore.offers, list)
        assert len(vectorstore.offers) > 0
        # Check first offer has expected fields
        offer = vectorstore.offers[0]
        assert "offer_id" in offer
        assert "title" in offer

    def test_collection_exists(self, vectorstore):
        """Test that collection is created."""
        assert vectorstore.collection is not None
        assert vectorstore.collection.name == "retention_offers"

    def test_index_offers(self, vectorstore):
        """Test indexing offers into vector store."""
        # Index should work without errors
        vectorstore.index_offers()
        # Collection should have offers
        assert vectorstore.collection.count() > 0

    def test_search_offers_returns_list(self, vectorstore):
        """Test that search_offers returns a list."""
        vectorstore.index_offers()
        results = vectorstore.search_offers("high income customer", n_results=3)
        assert isinstance(results, list)

    def test_create_offer_document(self, vectorstore):
        """Test document creation from offer."""
        offer = vectorstore.offers[0]
        doc = vectorstore._create_offer_document(offer)
        assert isinstance(doc, str)
        assert offer["title"] in doc


class TestOfferMetadata:
    """Tests for offer metadata handling."""

    @pytest.fixture
    def vectorstore(self):
        """Create a vectorstore instance for testing."""
        return OfferVectorStore()

    def test_offers_have_target_profile(self, vectorstore):
        """Test that offers have target_profile field."""
        for offer in vectorstore.offers:
            assert "target_profile" in offer
            target = offer["target_profile"]
            assert "income_category" in target
            assert "card_category" in target
            assert "churn_risk" in target

    def test_offers_have_email_templates(self, vectorstore):
        """Test that offers have email templates."""
        for offer in vectorstore.offers:
            assert "email_subject" in offer
            assert "email_body" in offer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
