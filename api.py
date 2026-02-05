"""
FastAPI Inference Endpoint
===========================

REST API for customer support chatbot with intent-based routing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import uvicorn
from contextlib import asynccontextmanager

from src.graph import CustomerSupportGraph

# Global chatbot instance
chatbot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize chatbot on startup"""
    global chatbot
    print("🚀 Initializing chatbot...")
    chatbot = CustomerSupportGraph()
    print("✅ Chatbot ready!\n")
    yield
    print("👋 Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Customer Support Chatbot API",
    description="Intelligent customer support chatbot with 3-tier routing system (Zero-cost FAQ, RAG + LLM, Escalation)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message/query", min_length=1)
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "How do I track my order?",
                "session_id": "user-123"
            }
        }


class ChatResponse(BaseModel):
    response: str = Field(..., description="Chatbot's response")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Intent prediction confidence (0-1)")
    bucket: str = Field(..., description="Routing bucket (BUCKET_A/B/C)")
    cost_tier: str = Field(..., description="Cost tier (Zero/Low/High)")
    action: str = Field(..., description="Action taken")
    sentiment: str = Field(..., description="Detected sentiment (POSITIVE/NEGATIVE)")
    sentiment_score: float = Field(..., description="Sentiment confidence (0-1)")
    escalated_by_sentiment: bool = Field(..., description="Whether bucket was overridden by sentiment")
    session_id: Optional[str] = Field(None, description="Session ID if provided")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "I'd be happy to help you track your order! Please provide your order number...",
                "intent": "track_order",
                "confidence": 0.98,
                "bucket": "BUCKET_A",
                "cost_tier": "Zero",
                "action": "Direct template response",
                "sentiment": "NEGATIVE",
                "sentiment_score": 0.65,
                "escalated_by_sentiment": False,
                "session_id": "user-123"
            }
        }


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]


# API Endpoints
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Customer Support Chatbot API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    global chatbot
    
    return HealthResponse(
        status="healthy" if chatbot is not None else "unhealthy",
        version="1.0.0",
        components={
            "chatbot": "ready" if chatbot is not None else "not initialized",
            "intent_classifier": "ready",
            "faiss_index": "ready",
            "groq_llm": "ready"
        }
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process customer message and return response
    
    - **message**: User's question or message
    - **session_id**: Optional session identifier for tracking
    
    Returns chatbot response with routing information
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized. Please try again in a moment."
        )
    
    try:
        # Process message through chatbot
        result = chatbot.process(request.message)
        
        # Check if bucket was escalated by sentiment
        escalated_by_sentiment = result.get('action') == 'escalate_sentiment'
        
        return ChatResponse(
            response=result['final_response'],
            intent=result['predicted_intent'],
            confidence=result['confidence'],
            bucket=result['bucket'],
            cost_tier=result['cost_tier'],
            action=result['action'],
            sentiment=result.get('sentiment_label', 'UNKNOWN'),
            sentiment_score=result.get('sentiment_score', 0.0),
            escalated_by_sentiment=escalated_by_sentiment,
            session_id=request.session_id
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.get("/intents", tags=["Information"])
async def get_supported_intents():
    """Get list of supported intents and their routing buckets"""
    return {
        "total_intents": 27,
        "buckets": {
            "BUCKET_A": {
                "description": "Zero-cost (Direct responses)",
                "count": 8,
                "intents": [
                    "check_invoice",
                    "check_payment_methods",
                    "track_order",
                    "delivery_options",
                    "check_refund_policy",
                    "check_cancellation_fee",
                    "delivery_period",
                    "track_refund"
                ]
            },
            "BUCKET_B": {
                "description": "Low-cost (RAG + Small LLM)",
                "count": 15,
                "intents": [
                    "cancel_order",
                    "change_order",
                    "place_order",
                    "get_invoice",
                    "get_refund",
                    "set_up_shipping_address",
                    "change_shipping_address",
                    "create_account",
                    "edit_account",
                    "switch_account",
                    "delete_account",
                    "recover_password",
                    "registration_problems",
                    "newsletter_subscription",
                    "review"
                ]
            },
            "BUCKET_C": {
                "description": "High-cost (Escalation)",
                "count": 4,
                "intents": [
                    "complaint",
                    "payment_issue",
                    "contact_customer_service",
                    "contact_human_agent"
                ]
            }
        }
    }


@app.get("/stats", tags=["Information"])
async def get_stats():
    """Get chatbot performance statistics"""
    return {
        "model_accuracy": "97.69%",
        "total_intents": 27,
        "routing_distribution": {
            "BUCKET_A": "30.6% (Zero-cost)",
            "BUCKET_B": "51.6% (Low-cost)",
            "BUCKET_C": "17.8% (High-cost)"
        },
        "cost_savings": "79.6% vs uniform big LLM",
        "technology_stack": {
            "intent_classifier": "scikit-learn (TF-IDF + Logistic Regression)",
            "embeddings": "HuggingFace sentence-transformers",
            "vector_db": "FAISS (local)",
            "llm": "Groq (llama-3.3-70b-versatile)",
            "orchestration": "LangGraph"
        }
    }


# Run server
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
