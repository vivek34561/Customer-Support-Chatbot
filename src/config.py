"""
Configuration for RAG Chatbot
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# FAISS configuration (local vector database)
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
FAISS_METADATA_PATH = DATA_DIR / "faiss_metadata.json"

# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Embedding configuration (HuggingFace)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 dimension

# RAG configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 3

# LLM configuration (Groq)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")  # Groq model
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500
BIG_LLM_MODEL = os.getenv("BIG_LLM_MODEL", "llama-3.3-70b-versatile")  # For escalations

# Dataset
DATASET_URL = "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"