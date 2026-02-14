"""
LLM Generation Node
===================

Generates response using LLM with context.
"""

import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.state import ChatbotState
from src.llm import (
    LLMFactory,
    RAG_SYSTEM_PROMPT,
    ESCALATION_SYSTEM_PROMPT,
    get_rag_prompt,
    get_escalation_prompt,
    get_direct_response
)


class GenerateNode:
    """Node for LLM response generation"""
    
    def __init__(self):
        """Initialize LLM factory"""
        self.llm_factory = LLMFactory()
        print("  ✓ Generation node initialized")
    
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
        response = llm.invoke([system_msg, user_msg])
        
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
