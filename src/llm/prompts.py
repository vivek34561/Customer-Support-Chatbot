"""
Prompt Templates
================

System and user prompts for different scenarios.
"""


# RAG System Prompt (BUCKET_B)
RAG_SYSTEM_PROMPT = """You are a helpful customer support assistant.

Use the provided context from the knowledge base to answer the customer's question.
If the context contains relevant information, use it to provide a helpful answer.
If the context doesn't contain relevant information, provide a general helpful response and suggest contacting support.

IMPORTANT INSTRUCTIONS:
- Be professional, concise, and customer-focused
- Always maintain a friendly and empathetic tone
- DO NOT include any internal reasoning, thinking process, or XML tags like <think>, <reasoning>, etc.
- Output ONLY the customer-facing response
- Keep responses clear and well-structured"""


def get_rag_prompt(context: str, query: str) -> str:
    """
    Create RAG prompt with context
    
    Args:
        context: Retrieved context from knowledge base
        query: User's question
        
    Returns:
        Formatted prompt string
    """
    return f"""Context from knowledge base:
{context}

Customer Question: {query}

Please provide a helpful response based on the context above."""


# Escalation Prompt (BUCKET_C)
ESCALATION_SYSTEM_PROMPT = """You are a senior customer support specialist handling escalated issues.

The customer has been routed to you because their issue requires:
- Extra attention and care
- Complex problem-solving
- Empathetic handling of complaints or sensitive situations

Approach:
1. Acknowledge their concern with empathy
2. Ask clarifying questions if needed
3. Provide detailed, personalized solutions
4. Offer additional assistance or follow-up

Be professional, empathetic, and solution-focused."""


def get_escalation_prompt(query: str, intent: str) -> str:
    """
    Create escalation prompt
    
    Args:
        query: User's question/complaint
        intent: Detected intent
        
    Returns:
        Formatted prompt string
    """
    return f"""Customer Issue (Intent: {intent}):
{query}

Please provide a thoughtful, empathetic response that addresses their concern."""


# Direct Response Templates (BUCKET_A)
DIRECT_RESPONSE_TEMPLATES = {
    'check_invoice': "I can help you check your invoice. Please provide your order number or email address, and I'll look it up for you right away.",
    
    'check_payment_methods': """We accept the following payment methods:
• Credit/Debit cards (Visa, Mastercard, American Express, Discover)
• PayPal
• Apple Pay
• Google Pay
• Bank transfer (for orders over $500)

All payments are secure and encrypted.""",
    
    'track_order': "I'd be happy to help you track your order! Please provide your order number (found in your confirmation email), and I'll get you the latest shipping information.",
    
    'delivery_options': """We offer several delivery options:

Standard Delivery: 5-7 business days (Free on orders $50+)
Express Delivery: 2-3 business days ($9.99)
Next Day Delivery: Next business day ($19.99)

Shipping costs may vary by location and will be calculated at checkout.""",
    
    'check_refund_policy': """Our refund policy:

• 30-day return window from purchase date
• Items must be unused and in original packaging
• Full refund to original payment method
• Free return shipping on defective items
• 10% restocking fee may apply on some items

To start a return, visit your account's order history or contact us with your order number.""",
    
    'check_cancellation_fee': """Cancellation policy:

• Free cancellation within 24 hours of order placement
• After 24 hours: 10% processing fee may apply
• Orders already shipped cannot be cancelled (but can be returned)
• Digital products/services: cancellation only before download/activation

To cancel, go to your order details or contact us immediately.""",
    
    'delivery_period': """Delivery timeframes:

Standard Delivery: 5-7 business days
Express Delivery: 2-3 business days  
Next Day Delivery: 1 business day (order by 2 PM)

Note: Business days exclude weekends and holidays. You'll receive tracking information once your order ships.""",
    
    'track_refund': """To track your refund:

1. Provide your order number
2. I'll check the refund status in our system
3. Refund timeline:
   • Processing: 3-5 business days
   • Bank/card credit: 5-10 business days after processing
   
You'll receive an email confirmation once the refund is processed."""
}


def get_direct_response(intent: str, fallback: str = None) -> str:
    """
    Get direct response template for intent
    
    Args:
        intent: Detected intent
        fallback: Fallback message if intent not found
        
    Returns:
        Response template
    """
    return DIRECT_RESPONSE_TEMPLATES.get(
        intent,
        fallback or "I can help you with that. Could you please provide more details?"
    )


def has_direct_response(intent: str) -> bool:
    """
    Check if a direct response template exists for the intent
    
    Args:
        intent: Intent to check
        
    Returns:
        True if template exists, False otherwise
    """
    return intent in DIRECT_RESPONSE_TEMPLATES
