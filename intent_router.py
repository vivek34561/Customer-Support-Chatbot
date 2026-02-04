"""
Intent Router Module
====================
Reusable module for customer support intent classification and routing.

Flow:
    User Message → Clean Text → TF-IDF → Logistic Regression → 
    Predicted Intent + Confidence → Route to Bucket

Usage:
    from intent_router import IntentRouter
    
    router = IntentRouter()
    result = router.route_message("I want to cancel my order")
    print(result['bucket'])  # BUCKET_B
    print(result['predicted_intent'])  # cancel_order
"""

import re
import pickle
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class IntentRouter:
    """
    Intent classification and routing for customer support messages.
    
    Attributes:
        vectorizer: TF-IDF vectorizer for text transformation
        model: Logistic Regression model for intent prediction
        routing_config: Dictionary containing routing rules
        confidence_threshold: Minimum confidence to trust predictions
    """
    
    def __init__(
        self, 
        models_dir: str = "models",
        confidence_threshold: Optional[float] = None
    ):
        """
        Initialize the Intent Router.
        
        Args:
            models_dir: Directory containing model files (default: "models")
            confidence_threshold: Override default confidence threshold
        """
        self.models_dir = Path(models_dir)
        self.vectorizer = None
        self.model = None
        self.routing_config = None
        self.intent_to_bucket = {}
        self.confidence_threshold = confidence_threshold
        
        # Load all components
        self._load_vectorizer()
        self._load_model()
        self._load_routing_config()
        self._build_intent_mapping()
    
    def _load_vectorizer(self):
        """Load the TF-IDF vectorizer from pickle file."""
        vectorizer_path = self.models_dir / "tfidf_vectorizer.pkl"
        
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {vectorizer_path}"
            )
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"✓ Loaded TF-IDF vectorizer from {vectorizer_path}")
    
    def _load_model(self):
        """Load the Logistic Regression model from pickle file."""
        model_path = self.models_dir / "logistic_regression_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Fix sklearn version compatibility issue
        # In sklearn 1.8.0+, multi_class attribute was removed
        if not hasattr(self.model, 'multi_class'):
            self.model.multi_class = 'auto'
        
        print(f"✓ Loaded Logistic Regression model from {model_path}")
    
    def _load_routing_config(self):
        """Load routing configuration from JSON file."""
        config_path = self.models_dir / "routing_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Routing config not found at {config_path}"
            )
        
        with open(config_path, 'r') as f:
            self.routing_config = json.load(f)
        
        # Set confidence threshold if not provided
        if self.confidence_threshold is None:
            self.confidence_threshold = self.routing_config.get(
                'confidence_threshold', 0.5
            )
        
        print(f"✓ Loaded routing config from {config_path}")
    
    def _build_intent_mapping(self):
        """Build reverse mapping from intent to bucket."""
        intent_routing = self.routing_config.get('intent_routing', {})
        
        for bucket_name, bucket_info in intent_routing.items():
            for intent in bucket_info.get('intents', []):
                self.intent_to_bucket[intent] = {
                    'bucket': bucket_name,
                    'description': bucket_info.get('description', ''),
                    'cost': bucket_info.get('cost', 'Unknown')
                }
        
        print(f"✓ Built intent mapping for {len(self.intent_to_bucket)} intents")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text using the same preprocessing as training.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        # Remove placeholders like {{Order Number}}, {{Invoice Number}}, etc.
        text_clean = re.sub(r'\{\{.*?\}\}', '', text)
        
        # Convert to lowercase
        text_clean = text_clean.lower()
        
        # Remove extra whitespaces
        text_clean = text_clean.strip()
        text_clean = re.sub(r'\s+', ' ', text_clean)
        
        return text_clean
    
    def predict_intent(self, text: str) -> Dict[str, Any]:
        """
        Predict intent and confidence for a given text.
        
        Args:
            text: Input text (already cleaned)
            
        Returns:
            Dictionary containing:
                - predicted_intent: The predicted intent label
                - confidence: Confidence score (0-1)
                - probabilities: Dictionary of all intent probabilities
        """
        # Transform text using TF-IDF
        text_tfidf = self.vectorizer.transform([text])
        
        # Predict intent
        predicted_intent = self.model.predict(text_tfidf)[0]
        
        # Get probabilities (handle sklearn compatibility)
        try:
            probabilities = self.model.predict_proba(text_tfidf)[0]
        except AttributeError:
            # Fallback for compatibility issues
            # Use decision function for binary classification or return uniform
            if hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(text_tfidf)[0]
                # Convert to pseudo-probabilities
                if isinstance(decision, float):
                    probabilities = [1.0, 0.0]
                else:
                    probabilities = decision
            else:
                # Return uniform probabilities as fallback
                n_classes = len(self.model.classes_)
                probabilities = [1.0/n_classes] * n_classes
        
        confidence = float(max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            intent: float(prob) 
            for intent, prob in zip(self.model.classes_, probabilities)
        }
        
        # Sort by probability (top 3)
        top_3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'predicted_intent': predicted_intent,
            'confidence': confidence,
            'probabilities': prob_dict,
            'top_3_predictions': top_3
        }
    
    def get_routing_decision(
        self, 
        intent: str, 
        confidence: float
    ) -> Dict[str, Any]:
        """
        Determine routing bucket based on intent and confidence.
        
        Args:
            intent: Predicted intent
            confidence: Confidence score
            
        Returns:
            Dictionary containing routing decision
        """
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return {
                'bucket': 'BUCKET_C',
                'action': 'LOW_CONFIDENCE_ESCALATE',
                'reason': f'Low confidence ({confidence:.2%}) - Escalate to human',
                'cost_tier': 'High'
            }
        
        # Get routing info for intent
        routing_info = self.intent_to_bucket.get(intent)
        
        if routing_info is None:
            # Unknown intent - escalate
            return {
                'bucket': 'BUCKET_C',
                'action': 'UNKNOWN_INTENT_ESCALATE',
                'reason': f'Unknown intent "{intent}" - Escalate to human',
                'cost_tier': 'High'
            }
        
        # Determine action based on bucket
        bucket = routing_info['bucket']
        
        if bucket == 'BUCKET_A':
            reason = "Direct database/FAQ lookup - No LLM needed"
        elif bucket == 'BUCKET_B':
            reason = "RAG + Small LLM for procedural response"
        else:  # BUCKET_C
            reason = "Escalate to Big LLM or Human agent"
        
        return {
            'bucket': bucket,
            'action': bucket,
            'reason': reason,
            'cost_tier': routing_info['cost']
        }
    
    def route_message(self, user_message: str) -> Dict[str, Any]:
        """
        Complete pipeline: User message → Intent → Routing decision.
        
        This is the main entry point for the router.
        
        Args:
            user_message: Raw user input
            
        Returns:
            Dictionary containing:
                - user_message: Original message
                - cleaned_text: Preprocessed text
                - predicted_intent: Predicted intent
                - confidence: Confidence score
                - bucket: Routing bucket (BUCKET_A/B/C)
                - action: Recommended action
                - reason: Explanation for routing decision
                - cost_tier: Cost tier (Zero/Low/High)
                - top_3_predictions: Top 3 intent predictions with scores
        """
        # Step 1: Clean text
        cleaned_text = self.clean_text(user_message)
        
        # Step 2: Predict intent
        prediction = self.predict_intent(cleaned_text)
        
        # Step 3: Get routing decision
        routing = self.get_routing_decision(
            prediction['predicted_intent'],
            prediction['confidence']
        )
        
        # Compile complete result
        result = {
            'user_message': user_message,
            'cleaned_text': cleaned_text,
            'predicted_intent': prediction['predicted_intent'],
            'confidence': prediction['confidence'],
            'bucket': routing['bucket'],
            'action': routing['action'],
            'reason': routing['reason'],
            'cost_tier': routing['cost_tier'],
            'top_3_predictions': prediction['top_3_predictions']
        }
        
        return result
    
    def batch_route(self, messages: list) -> list:
        """
        Route multiple messages in batch.
        
        Args:
            messages: List of user messages
            
        Returns:
            List of routing results
        """
        return [self.route_message(msg) for msg in messages]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the routing configuration.
        
        Returns:
            Dictionary containing routing statistics
        """
        intent_routing = self.routing_config.get('intent_routing', {})
        
        total_intents = len(self.intent_to_bucket)
        
        bucket_stats = {}
        for bucket_name, bucket_info in intent_routing.items():
            count = len(bucket_info.get('intents', []))
            bucket_stats[bucket_name] = {
                'count': count,
                'percentage': (count / total_intents * 100) if total_intents > 0 else 0,
                'cost': bucket_info.get('cost', 'Unknown'),
                'description': bucket_info.get('description', '')
            }
        
        return {
            'total_intents': total_intents,
            'confidence_threshold': self.confidence_threshold,
            'bucket_distribution': bucket_stats
        }


def main():
    """Example usage of IntentRouter."""
    print("=" * 80)
    print("Intent Router - Example Usage")
    print("=" * 80)
    
    # Initialize router
    router = IntentRouter()
    
    print("\n" + "=" * 80)
    print("Router Statistics:")
    print("=" * 80)
    stats = router.get_stats()
    print(f"Total Intents: {stats['total_intents']}")
    print(f"Confidence Threshold: {stats['confidence_threshold']}")
    print("\nBucket Distribution:")
    for bucket, info in stats['bucket_distribution'].items():
        print(f"  {bucket}: {info['count']} intents ({info['percentage']:.1f}%) - {info['cost']} cost")
    
    # Test messages
    test_messages = [
        "I want to cancel my order",
        "What payment methods do you accept?",
        "I need to speak with a human agent",
    ]
    
    print("\n" + "=" * 80)
    print("Testing with sample messages:")
    print("=" * 80)
    
    for msg in test_messages:
        result = router.route_message(msg)
        print(f"\n📨 Message: {msg}")
        print(f"🎯 Intent: {result['predicted_intent']} ({result['confidence']:.2%})")
        print(f"🗂️  Bucket: {result['bucket']} ({result['cost_tier']} cost)")
        print(f"⚡ Action: {result['action']}")


if __name__ == "__main__":
    main()
