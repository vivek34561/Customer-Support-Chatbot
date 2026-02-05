"""
State Definitions
=================

State types for the RAG chatbot graph.
"""

from typing import TypedDict, Annotated, Sequence, List, Dict
from langchain_core.messages import BaseMessage


class ChatbotState(TypedDict):
    """State for RAG chatbot graph"""
    # User input
    user_query: str
    
    # Intent routing
    predicted_intent: str
    confidence: float
    bucket: str
    
    # RAG retrieval
    retrieved_documents: List[Dict]
    retrieved_context: str
    
    # LLM generation
    final_response: str
    
    # Conversation history
    messages: Annotated[Sequence[BaseMessage], "conversation history"]
    
    # Sentiment analysis
    sentiment_label: str
    sentiment_score: float
    has_anger_keywords: bool
    
    # Metadata
    cost_tier: str
    action: str
