"""
Customer Support RAG Graph
===========================

LangGraph state machine for customer support chatbot.
"""

from langgraph.graph import StateGraph, END

from src.state import ChatbotState
from src.nodes import IntentNode, RetrieveNode, GenerateNode
from src.llm.prompts import has_direct_response


class CustomerSupportGraph:
    """LangGraph state machine for customer support"""
    
    def __init__(self):
        """Initialize graph with nodes"""
        print("Initializing Customer Support Graph...")
        
        # Initialize nodes
        self.intent_node = IntentNode() # Intent classification node (BUCKET_A, BUCKET_B, BUCKET_C)
        self.retrieve_node = RetrieveNode() # Retrieval node (only invoked for BUCKET_B)
        self.generate_node = GenerateNode() # Generation node (invoked for all buckets, but with different context)
        
        # Build graph
        self.graph = self._build_graph()
        
        print("✓ Graph initialized\n")
    
    def _should_retrieve(self, state: ChatbotState) -> str:
        """
        Conditional edge: Determine if retrieval is needed
        
        Routing logic:
        - BUCKET_A with missing template → Route to BUCKET_B (retrieve)
        - BUCKET_B → Always retrieve
        - BUCKET_C → No retrieval (escalation)
        
        Args:
            state: Current state
            
        Returns:
            Next node name ('retrieve' or 'generate')
        """
        bucket = state['bucket']
        
        # BUCKET_A: Check if template exists
        if bucket == 'BUCKET_A':
            intent = state['predicted_intent']
            if not has_direct_response(intent):
                # Template missing - fallback to RAG
                print(f"  ⚠️  Template missing for '{intent}' → Routing to RAG (BUCKET_B)")
                state['bucket'] = 'BUCKET_B'
                state['cost_tier'] = 'low'
                return "retrieve"
            else:
                # Template exists - no retrieval needed
                return "generate"
        
        # BUCKET_B: Always retrieve
        if bucket == 'BUCKET_B':
            return "retrieve"
        
        # BUCKET_C: Escalation - no retrieval
        else:
            return "generate"
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine
        
        Graph flow:
            START → intent → [conditional] → retrieve → generate → END
                                    ↓
                                  generate → END (if BUCKET_A or BUCKET_C)
        """
        # Create graph
        workflow = StateGraph(ChatbotState)
        
        # Add nodes
        workflow.add_node("intent", self.intent_node) # Intent classification node (BUCKET_A, BUCKET_B, BUCKET_C)
        workflow.add_node("retrieve", self.retrieve_node) # Retrieval node (only invoked for BUCKET_B)
        workflow.add_node("generate", self.generate_node) # Generation node (invoked for all buckets, but with different context)
        
        # Set entry point
        workflow.set_entry_point("intent") # Start with intent classification
        
        # Add conditional edge after intent
        workflow.add_conditional_edges(
            "intent", # From intent node
            self._should_retrieve,
            {
                "retrieve": "retrieve",  # If BUCKET_B, go to retrieve
                "generate": "generate"  # If BUCKET_A or BUCKET_C, skip retrieval and go to generate
            }
        )
        
        # Add edges
        workflow.add_edge("retrieve", "generate") # After retrieval, always go to generate
        workflow.add_edge("generate", END)
        print(workflow.compile())
        # Compile
        return workflow.compile()
    
    def process(self, user_query: str) -> dict:
        """
        Process user query through the graph
        
        Args:
            user_query: User's question
            
        Returns:
            Final state with response
        """
        print(f"\n{'='*80}")
        print(f"Processing: {user_query}")
        print('='*80)
        
        # Initialize state
        initial_state = {
            'user_query': user_query,
            'predicted_intent': '',
            'confidence': 0.0,
            'bucket': '',
            'retrieved_documents': [],
            'retrieved_context': '',
            'final_response': '',
            'llm_usage': {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            'messages': [],
            'cost_tier': '',
            'action': ''
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        print('='*80 + '\n')
        
        return final_state
    
    def get_response(self, user_query: str) -> str:
        """
        Convenience method to get just the response
        
        Args:
            user_query: User's question
            
        Returns:
            Generated response
        """
        state = self.process(user_query)
        return state['final_response']
