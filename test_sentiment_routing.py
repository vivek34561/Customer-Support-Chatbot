"""
Test Sentiment-Based Routing
==============================

This script tests the hybrid sentiment analysis approach
that escalates angry customers even if their intent suggests
lower-cost buckets.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import CustomerSupportGraph


def test_sentiment_escalation():
    """Test sentiment-based escalation with different queries"""
    
    print("=" * 80)
    print("SENTIMENT-BASED ROUTING TEST")
    print("=" * 80)
    print("\nInitializing chatbot...\n")
    
    chatbot = CustomerSupportGraph()
    
    test_cases = [
        {
            "query": "What is your refund policy?",
            "description": "Neutral question (should NOT escalate)",
            "expected_bucket": "BUCKET_A"
        },
        {
            "query": "How can I track my order?",
            "description": "Simple informational query (should NOT escalate)",
            "expected_bucket": "BUCKET_A"
        },
        {
            "query": "I need help canceling my order",
            "description": "Neutral request (should stay in BUCKET_B)",
            "expected_bucket": "BUCKET_B"
        },
        {
            "query": "This is terrible! I want my money back immediately!",
            "description": "Angry refund request (should ESCALATE to BUCKET_C)",
            "expected_bucket": "BUCKET_C"
        },
        {
            "query": "Your customer service is absolutely useless! Nobody is helping me!",
            "description": "Very frustrated customer (should ESCALATE to BUCKET_C)",
            "expected_bucket": "BUCKET_C"
        },
        {
            "query": "I'm extremely disappointed with this purchase. This is unacceptable!",
            "description": "Disappointed and angry (should ESCALATE to BUCKET_C)",
            "expected_bucket": "BUCKET_C"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"{'='*80}")
        print(f"Query: \"{test_case['query']}\"")
        print()
        
        # Process query
        result = chatbot.process(test_case['query'])
        
        # Check results
        actual_bucket = result['bucket']
        expected_bucket = test_case['expected_bucket']
        sentiment = result.get('sentiment_label', 'UNKNOWN')
        sentiment_score = result.get('sentiment_score', 0.0)
        has_anger = result.get('has_anger_keywords', False)
        escalated = result.get('action') == 'escalate_sentiment'
        
        passed = actual_bucket == expected_bucket
        
        print(f"\n📊 RESULTS:")
        print(f"   Intent: {result['predicted_intent']} ({result['confidence']:.1%})")
        print(f"   Sentiment: {sentiment} ({sentiment_score:.1%})")
        print(f"   Anger Keywords: {'✅ YES' if has_anger else '❌ NO'}")
        print(f"   Bucket: {actual_bucket} (expected: {expected_bucket})")
        print(f"   Cost Tier: {result['cost_tier']}")
        
        if escalated:
            print(f"   ⚠️  ESCALATED BY SENTIMENT!")
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n   Status: {status}")
        
        results.append({
            'case': test_case['description'],
            'passed': passed,
            'escalated': escalated
        })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    escalated = sum(1 for r in results if r['escalated'])
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}/{total} ({passed/total:.1%})")
    print(f"Failed: {total - passed}/{total}")
    print(f"Sentiment Escalations: {escalated}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Sentiment-based routing works correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the results above.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    test_sentiment_escalation()
