"""
Offer Vector Store - RAG system for semantic offer matching
"""
import os
import json
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OfferVectorStore:
    """Vector store for semantic search over retention offers"""

    def __init__(self, data_path: str = None, persist_directory: str = None):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = data_path or os.path.join(base_path, 'data')
        self.persist_directory = persist_directory or os.path.join(self.data_path, 'chroma_db')

        self.offers = self._load_offers()
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._get_or_create_collection()

        api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=api_key) if api_key else None

    def _load_offers(self) -> List[Dict]:
        """Load offers from JSON file"""
        offers_path = os.path.join(self.data_path, 'retention_offers.json')
        with open(offers_path, 'r') as f:
            return json.load(f)['offers']

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one"""
        return self.client.get_or_create_collection(
            name="retention_offers",
            metadata={"description": "Retention offers for churn prevention"}
        )

    def _create_offer_document(self, offer: Dict) -> str:
        """Create searchable document from offer"""
        target = offer['target_profile']
        return f"""
        Offer: {offer['title']}
        Type: {offer['offer_type']}
        Description: {offer['description']}
        Value: {offer['value']}
        Target Income: {', '.join(target['income_category'])}
        Target Card: {', '.join(target['card_category'])}
        Min Tenure: {target['min_tenure_months']} months
        Risk Levels: {', '.join(target['churn_risk'])}
        """

    def index_offers(self, force_reindex: bool = False):
        """Index all offers into vector store"""
        existing = self.collection.count()

        if existing > 0 and not force_reindex:
            print(f"Collection already has {existing} offers indexed")
            return

        if force_reindex:
            self.client.delete_collection("retention_offers")
            self.collection = self._get_or_create_collection()

        documents = []
        metadatas = []
        ids = []

        for offer in self.offers:
            documents.append(self._create_offer_document(offer))
            metadatas.append({
                'offer_id': offer['offer_id'],
                'title': offer['title'],
                'offer_type': offer['offer_type'],
                'value': offer['value'],
                'min_tenure': offer['target_profile']['min_tenure_months'],
                'income_categories': ','.join(offer['target_profile']['income_category']),
                'card_categories': ','.join(offer['target_profile']['card_category']),
                'risk_levels': ','.join(offer['target_profile']['churn_risk'])
            })
            ids.append(offer['offer_id'])

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Indexed {len(documents)} offers")

    def search_offers(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """Search offers by semantic similarity"""
        where_clause = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                if isinstance(value, list):
                    conditions.append({key: {"$in": value}})
                else:
                    conditions.append({key: value})
            if len(conditions) == 1:
                where_clause = conditions[0]
            elif len(conditions) > 1:
                where_clause = {"$and": conditions}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )

        offers = []
        if results['ids'] and results['ids'][0]:
            for i, offer_id in enumerate(results['ids'][0]):
                offer_data = next((o for o in self.offers if o['offer_id'] == offer_id), None)
                if offer_data:
                    distance = results['distances'][0][i] if results['distances'] else 0
                    offers.append({
                        **offer_data,
                        'relevance_score': 1 / (1 + distance)
                    })

        return offers

    def search_for_customer(
        self,
        income_category: str,
        card_category: str,
        tenure_months: int,
        churn_risk: str,
        n_results: int = 3
    ) -> List[Dict]:
        """Find best offers for a specific customer profile"""
        query = f"""
        Customer needs retention offer.
        Income level: {income_category}
        Card type: {card_category}
        Customer for {tenure_months} months
        Churn risk: {churn_risk}
        Looking for personalized retention offer to prevent customer attrition.
        """

        all_results = self.search_offers(query, n_results=10)

        filtered = []
        for offer in all_results:
            target = offer['target_profile']

            if income_category not in target['income_category']:
                continue
            if card_category not in target['card_category']:
                continue
            if tenure_months < target['min_tenure_months']:
                continue
            if churn_risk not in target['churn_risk']:
                continue

            filtered.append(offer)

        if len(filtered) < n_results:
            for offer in all_results:
                if offer not in filtered:
                    filtered.append(offer)
                if len(filtered) >= n_results:
                    break

        return filtered[:n_results]

    def rank_offers_with_llm(
        self,
        customer_profile: Dict,
        candidate_offers: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:
        """Use LLM to rank offers for a specific customer"""
        if not self.openai_client or not candidate_offers:
            return candidate_offers[:top_k]

        offers_text = "\n".join([
            f"- {o['offer_id']}: {o['title']} - {o['description']}"
            for o in candidate_offers
        ])

        prompt = f"""Given this customer profile:
- Income: {customer_profile.get('income_category', 'Unknown')}
- Card Type: {customer_profile.get('card_category', 'Unknown')}
- Tenure: {customer_profile.get('months_on_book', 0)} months
- Monthly Transactions: {customer_profile.get('total_trans_ct', 0)}
- Credit Utilization: {customer_profile.get('avg_utilization_ratio', 0):.1%}
- Months Inactive: {customer_profile.get('months_inactive_12_mon', 0)}

Rank these retention offers from most to least suitable (return only offer IDs separated by commas):
{offers_text}

Return ONLY the offer IDs in order, comma-separated. Example: OFF001,OFF005,OFF003"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )

            ranked_ids = response.choices[0].message.content.strip().split(',')
            ranked_ids = [id.strip() for id in ranked_ids]

            ranked_offers = []
            for offer_id in ranked_ids:
                offer = next((o for o in candidate_offers if o['offer_id'] == offer_id), None)
                if offer:
                    ranked_offers.append(offer)

            for offer in candidate_offers:
                if offer not in ranked_offers:
                    ranked_offers.append(offer)

            return ranked_offers[:top_k]

        except Exception as e:
            print(f"LLM ranking error: {e}")
            return candidate_offers[:top_k]

    def get_offer_by_id(self, offer_id: str) -> Optional[Dict]:
        """Get a specific offer by ID"""
        return next((o for o in self.offers if o['offer_id'] == offer_id), None)

    def get_all_offers(self) -> List[Dict]:
        """Get all available offers"""
        return self.offers


if __name__ == "__main__":
    store = OfferVectorStore()

    store.index_offers()

    print("\n--- Semantic Search Test ---")
    results = store.search_offers("customer spending less, needs incentive to stay")
    for offer in results[:3]:
        print(f"- {offer['title']}: {offer['relevance_score']:.2f}")

    print("\n--- Customer Match Test ---")
    matches = store.search_for_customer(
        income_category="$60K - $80K",
        card_category="Silver",
        tenure_months=18,
        churn_risk="high"
    )
    for offer in matches:
        print(f"- {offer['title']}")
