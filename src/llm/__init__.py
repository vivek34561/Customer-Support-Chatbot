"""
LLM Module
"""

from src.llm.models import get_small_llm, get_big_llm, LLMFactory
from src.llm.prompts import (
    RAG_SYSTEM_PROMPT,
    ESCALATION_SYSTEM_PROMPT,
    get_rag_prompt,
    get_escalation_prompt,
    get_direct_response,
    has_direct_response
)

__all__ = [
    'get_small_llm',
    'get_big_llm',
    'LLMFactory',
    'RAG_SYSTEM_PROMPT',
    'ESCALATION_SYSTEM_PROMPT',
    'get_rag_prompt',
    'get_escalation_prompt',
    'get_direct_response',
    'has_direct_response'
]
