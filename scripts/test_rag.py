"""
Test RAG accuracy and speed for the offer retrieval system.
Evaluates how well the system matches offers to customer profiles.
"""
import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.offer_vectorstore import OfferVectorStore, initialize_vectorstore


def test_vectorstore_initialization():
    """Test that vector store loads offers correctly."""
    print("=" * 60)
    print("TEST 1: Vector Store Initialization")
    print("=" * 60)

    start = time.time()
    vectorstore = initialize_vectorstore()
    init_time = time.time() - start

    offers = vectorstore.get_all_offers()
    print(f"✓ Loaded {len(offers)} offers in {init_time:.3f}s")

    return vectorstore, init_time


def test_retrieval_accuracy(vectorstore: OfferVectorStore):
    """
    Test retrieval accuracy with various customer profiles.
    Each test case has expected offer types that should match.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Retrieval Accuracy")
    print("=" * 60)

    test_cases = [
        {
            "name": "High-income Gold card, high churn risk",
            "profile": {
                "income_category": "$80K - $120K",
                "card_category": "Gold",
                "months_on_book": 36,
                "churn_probability": 0.85,
                "age": 45,
                "total_trans_amt": 15000
            },
            "expected_types": ["premium_service", "advisory", "bonus_points"]
        },
        {
            "name": "Low-income Blue card, high churn risk",
            "profile": {
                "income_category": "Less than $40K",
                "card_category": "Blue",
                "months_on_book": 8,
                "churn_probability": 0.90,
                "age": 28,
                "total_trans_amt": 500
            },
            "expected_types": ["fee_waiver", "statement_credit", "payment_flexibility"]
        },
        {
            "name": "Mid-income Silver card, medium churn risk",
            "profile": {
                "income_category": "$60K - $80K",
                "card_category": "Silver",
                "months_on_book": 24,
                "churn_probability": 0.55,
                "age": 35,
                "total_trans_amt": 8000
            },
            "expected_types": ["cashback", "bonus_points", "upgrade", "credit_increase"]
        },
        {
            "name": "High-income Platinum card, low churn risk",
            "profile": {
                "income_category": "$120K +",
                "card_category": "Platinum",
                "months_on_book": 48,
                "churn_probability": 0.25,
                "age": 55,
                "total_trans_amt": 50000
            },
            "expected_types": ["premium_service", "advisory", "insurance"]
        },
        {
            "name": "New customer, high churn risk",
            "profile": {
                "income_category": "$40K - $60K",
                "card_category": "Blue",
                "months_on_book": 4,
                "churn_probability": 0.80,
                "age": 30,
                "total_trans_amt": 200
            },
            "expected_types": ["statement_credit", "fee_waiver", "partner_discount"]
        }
    ]

    results = []
    total_retrieval_time = 0

    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}: {case['name']}")
        print(f"  Profile: income={case['profile']['income_category']}, "
              f"card={case['profile']['card_category']}, "
              f"churn={case['profile']['churn_probability']*100:.0f}%")

        start = time.time()
        offers = vectorstore.find_best_offers(case['profile'], n_results=3)
        retrieval_time = time.time() - start
        total_retrieval_time += retrieval_time

        if offers:
            top_offer = offers[0]
            offer_type = top_offer['offer_type']
            is_match = offer_type in case['expected_types']

            print(f"  Top offer: {top_offer['title']} (type: {offer_type})")
            print(f"  Relevance score: {top_offer['relevance_score']:.4f}")
            print(f"  Matches expected: {'✓ YES' if is_match else '✗ NO'}")
            print(f"  Expected types: {case['expected_types']}")
            print(f"  Retrieval time: {retrieval_time*1000:.2f}ms")

            results.append({
                "case": case['name'],
                "matched": is_match,
                "offer_type": offer_type,
                "expected": case['expected_types'],
                "score": top_offer['relevance_score'],
                "time_ms": retrieval_time * 1000
            })
        else:
            print(f"  ✗ No offers found!")
            results.append({
                "case": case['name'],
                "matched": False,
                "offer_type": None,
                "expected": case['expected_types'],
                "score": 0,
                "time_ms": retrieval_time * 1000
            })

    # Summary
    accuracy = sum(1 for r in results if r['matched']) / len(results) * 100
    avg_time = total_retrieval_time / len(results) * 1000

    print("\n" + "-" * 60)
    print("ACCURACY SUMMARY")
    print("-" * 60)
    print(f"Total test cases: {len(results)}")
    print(f"Correct matches: {sum(1 for r in results if r['matched'])}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average retrieval time: {avg_time:.2f}ms")

    return results, accuracy, avg_time


def test_retrieval_speed(vectorstore: OfferVectorStore, n_iterations: int = 50):
    """
    Benchmark retrieval speed with multiple iterations.
    """
    print("\n" + "=" * 60)
    print(f"TEST 3: Speed Benchmark ({n_iterations} iterations)")
    print("=" * 60)

    # Random-ish profiles for benchmarking
    profiles = [
        {"income_category": "$60K - $80K", "card_category": "Blue", "months_on_book": 24, "churn_probability": 0.7},
        {"income_category": "$80K - $120K", "card_category": "Silver", "months_on_book": 36, "churn_probability": 0.5},
        {"income_category": "Less than $40K", "card_category": "Blue", "months_on_book": 12, "churn_probability": 0.9},
        {"income_category": "$120K +", "card_category": "Gold", "months_on_book": 48, "churn_probability": 0.3},
        {"income_category": "$40K - $60K", "card_category": "Silver", "months_on_book": 18, "churn_probability": 0.6},
    ]

    times = []
    start_total = time.time()

    for i in range(n_iterations):
        profile = profiles[i % len(profiles)]
        start = time.time()
        vectorstore.find_best_offers(profile, n_results=3)
        times.append(time.time() - start)

    total_time = time.time() - start_total

    avg_time = sum(times) / len(times) * 1000
    min_time = min(times) * 1000
    max_time = max(times) * 1000
    p95_time = sorted(times)[int(len(times) * 0.95)] * 1000

    print(f"Total time for {n_iterations} queries: {total_time:.3f}s")
    print(f"Average latency: {avg_time:.2f}ms")
    print(f"Min latency: {min_time:.2f}ms")
    print(f"Max latency: {max_time:.2f}ms")
    print(f"P95 latency: {p95_time:.2f}ms")
    print(f"Throughput: {n_iterations/total_time:.1f} queries/second")

    return {
        "iterations": n_iterations,
        "total_time_s": total_time,
        "avg_ms": avg_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "p95_ms": p95_time,
        "throughput": n_iterations / total_time
    }


def main():
    print("\n" + "=" * 60)
    print("RAG SYSTEM EVALUATION")
    print("=" * 60)
    print("Testing ChromaDB + OpenAI embeddings for offer retrieval\n")

    # Test 1: Initialization
    vectorstore, init_time = test_vectorstore_initialization()

    # Test 2: Accuracy
    accuracy_results, accuracy, avg_accuracy_time = test_retrieval_accuracy(vectorstore)

    # Test 3: Speed
    speed_results = test_retrieval_speed(vectorstore)

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Vector Store: ChromaDB with OpenAI embeddings (text-embedding-3-small)")
    print(f"Number of offers in knowledge base: {len(vectorstore.get_all_offers())}")
    print(f"\nInitialization time: {init_time*1000:.2f}ms")
    print(f"Retrieval accuracy: {accuracy:.1f}%")
    print(f"Average retrieval latency: {speed_results['avg_ms']:.2f}ms")
    print(f"P95 retrieval latency: {speed_results['p95_ms']:.2f}ms")
    print(f"Throughput: {speed_results['throughput']:.1f} queries/second")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if accuracy >= 80 and speed_results['avg_ms'] < 100:
        print("✓ RAG system is PRODUCTION READY")
        print("  - High accuracy for offer matching")
        print("  - Fast retrieval times")
    elif accuracy >= 60:
        print("△ RAG system needs improvement")
        print("  - Consider adding more offers to knowledge base")
        print("  - Or improve offer descriptions for better matching")
    else:
        print("✗ RAG system needs significant work")
        print("  - Review offer targeting criteria")
        print("  - Consider different embedding model")


if __name__ == "__main__":
    main()
