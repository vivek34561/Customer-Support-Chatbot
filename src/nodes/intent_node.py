"""
Intent Classification Node
===========================

Routes user query through intent classifier.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from intent_router import IntentRouter
from src.state import ChatbotState
from transformers import pipeline


class IntentNode:
    """Node for intent classification and routing with sentiment analysis"""
    
    # Anger/frustration keywords for escalation
    ANGER_KEYWORDS = [
        'terrible', 'horrible', 'worst', 'useless', 'garbage', 'pathetic',
        'frustrated', 'angry', 'furious', 'disappointed', 'unacceptable',
        'ridiculous', 'disgusted', 'outraged', 'demand', 'immediately',
        'never', 'always', '!!', 'wtf', 'damn', 'awful', 'disgusting',
        'incompetent', 'idiots', 'stupid', 'hate', 'fed up'
    ]
    
    def __init__(self):
        """Initialize intent router and sentiment analyzer"""
        self.router = IntentRouter()
        
        # Load sentiment analyzer (lazy loading - only loads on first use)
        self.sentiment_analyzer = None
        
        print("  ✓ Intent classification node initialized")
    
    def _get_sentiment_analyzer(self):
        """Lazy load sentiment analyzer"""
        if self.sentiment_analyzer is None:
            print("  → Loading sentiment analyzer...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU
            )
            print("  ✓ Sentiment analyzer loaded")
        return self.sentiment_analyzer
    
    def _analyze_sentiment(self, message: str) -> dict:
        """Analyze sentiment with keyword filtering"""
        analyzer = self._get_sentiment_analyzer()
        sentiment = analyzer(message)[0]
        
        # Check for anger keywords
        message_lower = message.lower()
        has_anger = any(keyword in message_lower for keyword in self.ANGER_KEYWORDS)
        
        return {
            'label': sentiment['label'],
            'score': sentiment['score'],
            'has_anger': has_anger
        }
    
    def __call__(self, state: ChatbotState) -> ChatbotState:
        """
        Classify intent, analyze sentiment, and determine routing bucket
        
        Args:
            state: Current chatbot state
            
        Returns:
            Updated state with routing information and sentiment
        """
        query = state['user_query']
        
        print(f"  → Classifying intent...")
        
        # Route message
        routing_result = self.router.route_message(query)
        
        # Analyze sentiment
        print(f"  → Analyzing sentiment...")
        sentiment_result = self._analyze_sentiment(query)
        
        # Determine if escalation is needed (Hybrid approach)
        is_negative = sentiment_result['label'] == 'NEGATIVE'
        high_confidence = sentiment_result['score'] > 0.75
        has_anger = sentiment_result['has_anger']
        
        # Escalate if: NEGATIVE sentiment + high confidence + anger keywords
        should_escalate = is_negative and high_confidence and has_anger
        
        original_bucket = routing_result['bucket']
        
        # Override bucket if sentiment requires escalation
        if should_escalate and original_bucket != 'BUCKET_C':
            print(f"  ⚠️  Sentiment escalation triggered!")
            print(f"     Sentiment: {sentiment_result['label']} ({sentiment_result['score']:.1%})")
            print(f"     Anger keywords detected: YES")
            print(f"     Overriding {original_bucket} → BUCKET_C")
            routing_result['bucket'] = 'BUCKET_C'
            routing_result['cost_tier'] = 'high'
            routing_result['action'] = 'escalate_sentiment'
        
        # Update state with routing info
        state['predicted_intent'] = routing_result['predicted_intent']
        state['confidence'] = routing_result['confidence']
        state['bucket'] = routing_result['bucket']
        state['cost_tier'] = routing_result['cost_tier']
        state['action'] = routing_result['action']
        
        # Update state with sentiment info
        state['sentiment_label'] = sentiment_result['label']
        state['sentiment_score'] = sentiment_result['score']
        state['has_anger_keywords'] = sentiment_result['has_anger']
        
        print(f"  ✓ Intent: {routing_result['predicted_intent']} "
              f"({routing_result['confidence']:.1%} confidence)")
        print(f"  ✓ Sentiment: {sentiment_result['label']} "
              f"({sentiment_result['score']:.1%} confidence)")
        print(f"  ✓ Bucket: {routing_result['bucket']} "
              f"({routing_result['cost_tier']} cost)")
        
        return state
