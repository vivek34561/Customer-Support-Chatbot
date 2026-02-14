"""
LLM Generation Node
===================

Generates response using LLM with context.
"""

import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler

try:
    from langchain.callbacks.tracers import LangChainTracer
except Exception:
    try:
        from langchain_core.tracers import LangChainTracer
    except Exception:
        LangChainTracer = None

try:
    from langsmith import Client as LangSmithClient
except Exception:
    LangSmithClient = None

from src.state import ChatbotState
from src.llm import (
    LLMFactory,
    RAG_SYSTEM_PROMPT,
    ESCALATION_SYSTEM_PROMPT,
    get_rag_prompt,
    get_escalation_prompt,
    get_direct_response
)
from src.config import (
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT,
    LLM_INPUT_COST_PER_1M,
    LLM_OUTPUT_COST_PER_1M
)


def _extract_usage_from_response(response) -> dict:
    """Extract token usage from an LLM response."""
    usage = {}

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata

    if not usage and hasattr(response, "response_metadata"):
        metadata = response.response_metadata or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}

    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0

    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens) + int(output_tokens)
    }


class CostTrackingCallback(BaseCallbackHandler):
    """Attach cost/token metadata to LangSmith runs."""

    def __init__(self, client: LangSmithClient):
        self.client = client

    def on_llm_end(self, response, *, run_id, **kwargs):
        if not self.client:
            return

        usage = _extract_usage_from_response(response)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost_usd = (
            (input_tokens * LLM_INPUT_COST_PER_1M) + (output_tokens * LLM_OUTPUT_COST_PER_1M)
        ) / 1_000_000
        input_cost = (input_tokens * LLM_INPUT_COST_PER_1M) / 1_000_000
        output_cost = (output_tokens * LLM_OUTPUT_COST_PER_1M) / 1_000_000

        try:
            self.client.update_run(
                run_id,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=usage.get("total_tokens", 0),
                prompt_cost=round(input_cost, 6),
                completion_cost=round(output_cost, 6),
                total_cost=round(cost_usd, 6),
                extra={
                    "metadata": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": round(cost_usd, 6)
                    },
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": usage.get("total_tokens", 0),
                        "input_cost": round(input_cost, 6),
                        "output_cost": round(output_cost, 6),
                        "total_cost": round(cost_usd, 6)
                    }
                }
            )
        except Exception:
            pass


class GenerateNode:
    """Node for LLM response generation"""
    
    def __init__(self):
        """Initialize LLM factory"""
        self.llm_factory = LLMFactory()
        self._callbacks = []
        if LangChainTracer and LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2.lower() == "true":
            self._callbacks = [LangChainTracer(project_name=LANGCHAIN_PROJECT)]
            if LangSmithClient:
                self._callbacks.append(CostTrackingCallback(LangSmithClient()))
        print("  ✓ Generation node initialized")

    def _extract_usage(self, response) -> dict:
        """Extract token usage from LLM response metadata"""
        return _extract_usage_from_response(response)
    
    def _clean_response(self, response: str) -> str:
        """
        Clean LLM response by removing thinking tags and formatting issues
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned response
        """
        # Remove <think>...</think> tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove other common internal tags
        response = re.sub(r'<reasoning>.*?</reasoning>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<internal>.*?</internal>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up excessive whitespace
        response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)  # Multiple newlines to max 2
        response = response.strip()
        
        return response
    
    def _generate_bucket_a_response(self, state: ChatbotState) -> str:
        """
        Generate response for BUCKET_A (no LLM needed)
        
        Note: If the template is missing, the graph's _should_retrieve() method
        will have already routed this to BUCKET_B (RAG), so this method
        will only be called if the template exists.
        
        Args:
            state: Current state
            
        Returns:
            Direct response
        """
        state['llm_usage'] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        intent = state['predicted_intent']
        return get_direct_response(intent)
    
    def _generate_bucket_b_response(self, state: ChatbotState) -> str:
        """
        Generate response for BUCKET_B (RAG + Small LLM)
        
        Args:
            state: Current state
            
        Returns:
            Generated response
        """
        llm = self.llm_factory.get_llm_for_bucket('BUCKET_B')
        
        # Create messages
        system_msg = SystemMessage(content=RAG_SYSTEM_PROMPT)
        user_msg = HumanMessage(
            content=get_rag_prompt(
                state['retrieved_context'],
                state['user_query']
            )
        )
        
        # Generate response
        if self._callbacks:
            response = llm.invoke([system_msg, user_msg], config={"callbacks": self._callbacks})
        else:
            response = llm.invoke([system_msg, user_msg])

        # Store token usage for cost calculation
        state['llm_usage'] = self._extract_usage(response)
        
        # Clean response (remove thinking tags, formatting issues)
        cleaned_response = self._clean_response(response.content)
        
        return cleaned_response
    
    def _generate_bucket_c_response(self, state: ChatbotState) -> str:
        """
        Generate response for BUCKET_C (Big LLM or escalation)
        
        Args:
            state: Current state
            
        Returns:
            Generated response
        """
        # For now, return escalation message
        # Can be extended to use GPT-4 or route to human
        state['llm_usage'] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        intent = state['predicted_intent']
        
        escalation_messages = {
            'complaint': "I understand you're experiencing an issue and I sincerely apologize for any inconvenience. I'm connecting you with a senior support specialist who can better assist you and ensure your concern is fully addressed.",
            
            'payment_issue': "Payment issues require immediate attention. I'm escalating this to our payment support team right away. They'll contact you within the next 30 minutes to resolve this. In the meantime, please don't attempt any additional payments.",
            
            'contact_human_agent': "I'm connecting you with a human agent now. They'll be with you shortly. Thank you for your patience.",
            
            'contact_customer_service': "Let me transfer you to our customer service team. They have access to more tools and resources to help with your request. Please hold for just a moment.",
        }
        
        return escalation_messages.get(
            intent,
            "I'm escalating your request to a specialist who can provide better assistance. You'll be connected shortly. Thank you for your patience."
        )
    
    def __call__(self, state: ChatbotState) -> ChatbotState:
        """
        Generate response based on bucket
        
        Args:
            state: Current chatbot state
            
        Returns:
            Updated state with final response
        """
        bucket = state['bucket']
        
        print(f"  → Generating response for {bucket}...")
        
        # Generate based on bucket
        if bucket == 'BUCKET_A':
            response = self._generate_bucket_a_response(state)
        
        elif bucket == 'BUCKET_B':
            response = self._generate_bucket_b_response(state)
        
        elif bucket == 'BUCKET_C':
            response = self._generate_bucket_c_response(state)
        
        else:
            response = "I apologize, but I'm unable to process your request at the moment. Please contact our support team directly."
        
        # Update state
        state['final_response'] = response
        
        # Update conversation history
        if 'messages' not in state:
            state['messages'] = []
        
        state['messages'].append(HumanMessage(content=state['user_query']))
        state['messages'].append(AIMessage(content=response))
        
        print(f"  ✓ Response generated")
        
        return state
