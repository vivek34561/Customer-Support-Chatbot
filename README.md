# 🤖 Intelligent Customer Support Chatbot – Cost-Optimized RAG System

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/LangGraph-latest-green.svg" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/FastAPI-latest-009688.svg" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20DB-orange.svg" alt="FAISS"/>
  <img src="https://img.shields.io/badge/Groq-LLM-purple.svg" alt="Groq"/>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"/>
</p>

<p align="center">
  <strong>A production-grade customer support chatbot with intelligent 3-tier routing achieving 79.6% cost reduction through adaptive RAG and sentiment-based escalation.</strong>
</p>

<p align="center">
  <a href="#-features">✨ Features</a> •
  <a href="#-architecture">🏗️ Architecture</a> •
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#-performance-metrics">📊 Performance</a> •
  <a href="#-project-structure">📂 Project Structure</a> •
  <a href="#-deployment">🚀 Deployment</a>
</p>

---

## 🎯 Overview

This is a **production-ready customer support chatbot** built with **LangChain, LangGraph, and FastAPI** that intelligently routes queries through a cost-optimized 3-tier system.

Instead of processing all queries equally, the system:

* Routes **FAQ queries** to template responses (zero cost)
* Uses **RAG + Local LLM** for procedural questions (low cost)
* Escalates **complex/sensitive issues** to human agents (high cost)
* Applies **sentiment analysis** to detect frustrated customers and prioritize them
* Achieves **97.69% intent classification accuracy** on 27 intents

This results in **79.6% cost reduction** while maintaining high-quality responses.

---

## 💡 Why This Project?

**The Problem:**
Traditional chatbots either:
* Use expensive LLMs for every query (high cost)
* Use templates for everything (poor user experience)
* Don't detect customer emotion (frustrated customers get automated responses)

**The Solution:**
Build a **self-routing, emotion-aware chatbot** that:
* Decides *how* to answer based on query complexity
* Uses *sentiment analysis* to catch angry customers
* Optimizes *cost* without sacrificing quality
* Provides a *production REST API* for integration

**Key Innovation:**
Combines intent classification, sentiment analysis, and adaptive RAG in a unified LangGraph workflow.

---

## ✨ Key Features

### 🎯 Intelligent Routing
* **3-Bucket System**: Zero-cost templates → Low-cost RAG → High-cost escalation
* **97.69% Accuracy**: TF-IDF + Logistic Regression intent classifier (27 intents)
* **Confidence Fallback**: Low-confidence queries safely routed to RAG

### 😊 Sentiment-Based Escalation
* **Hybrid Analysis**: Combines DistilBERT sentiment + keyword detection
* **Smart Override**: Routes angry customers to human support even for simple queries
* **False Positive Prevention**: Keyword filter prevents neutral questions from escalating

### 🧠 Local-First Architecture
* **HuggingFace Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (runs offline)
* **FAISS Vector Database**: No cloud dependencies, version-controllable index
* **Groq LLM**: Ultra-fast inference (750+ tokens/sec), free tier available

### 🚀 Production Ready
* **FastAPI REST API**: 5 endpoints with auto-generated docs
* **Web Chat UI**: Beautiful responsive interface
* **Lazy Loading**: Models load on first request (deployment-friendly)
* **Response Cleaning**: Automatically removes LLM thinking tags and formatting issues

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Cost Reduction** | 79.6% vs uniform LLM |
| **Intent Accuracy** | 97.69% (27 intents) |
| **Zero-Cost Routing** | 30.6% of queries |
| **Low-Cost Routing** | 51.6% of queries |
| **High-Cost Routing** | 17.8% of queries |
| **Dataset Size** | 26,872 examples |
| **Response Time** | <2s (after model load) |

---

## 🚀 Quick Start

### Prerequisites

* Python 3.10+
* Groq API key (free at [console.groq.com](https://console.groq.com))

### Environment Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
ENABLE_SENTIMENT_ANALYSIS=true
```

### Install Dependencies

```bash
**Note:** Embedding model (~90MB) downloads automatically on first run and is cached locally.

### Run the Chatbot

#### Option 1: Streamlit UI (Recommended) 🎨

**Quick Launch:**
```bash
# Windows
launch_streamlit.bat

# Linux/Mac
chmod +x launch_streamlit.sh
./launch_streamlit.sh
```

**Manual Launch:**
```bash
# Terminal 1: Start FastAPI backend
python api.py

# Terminal 2: Start Streamlit UI
streamlit run streamlit_app.py
```

The Streamlit UI will open automatically at `http://localhost:8501` with:
- 💬 Real-time chat interface
- 📊 Live analytics dashboard
- 🎭 Sentiment tracking
- 🎯 Intent visualization
- 📦 Bucket distribution stats

#### Option 2: Interactive CLI

```bash
python src/main.py interactive
```

#### Option 3: REST API Only

```bash
# Start FastAPI server
python api.py

# API runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

#### Option 4: HTML Chat UI (Legacy)

Start the API then open `chat_ui.html` in your browser.

### Test the API

```bash
python test_api.py
```

---

## 📂 Project Structure

```
Customer Support Chatbot/
├── src/
│   ├── config.py                      # Configuration & environment
│   ├── faiss_index_builder.py         # FAISS index builder
│   ├── retriever.py                   # RAG retriever
│   │
│   ├── state/
│   │   └── state.py                   # ChatbotState TypedDict
│   │
│   ├── llm/
│   │   ├── models.py                  # Groq LLM setup
│   │   └── prompts.py                 # System prompts & templates
│   │
│   ├── nodes/
│   │   ├── intent_node.py             # Intent + sentiment analysis
│   │   ├── retrieve_node.py           # FAISS retrieval
│   │   └── generate_node.py           # LLM generation + cleaning
│   │
│   ├── graph/
│   │   └── chatbot_graph.py           # LangGraph workflow
│   │
│   └── main.py                        # Main interface
│
├── models/
│   ├── tfidf_vectorizer.pkl           # TF-IDF model
│   ├── intent_classifier.pkl          # Logistic Regression
│   └── routing_config.json            # Bucket mappings
│
├── data/
│   └── faiss_index/                   # Vector DB (created at build)
│
├── Notebook/
│   └── sementic_analysis.ipynb        # Sentiment testing
│
├── api.py                             # FastAPI server
├── streamlit_app.py                   # Streamlit UI (recommended)
├── chat_ui.html                       # HTML UI (legacy)
├── launch_streamlit.bat               # Windows launcher
├── launch_streamlit.sh                # Linux/Mac launcher
├── intent_router.py                   # Routing module
├── build_rag_index.py                 # Index builder script
├── test_api.py                        # API tests
├── requirements.txt                   # Dependencies
├── .env                               # API keys (create this)
└── README.md                          # This file
```

---

## 🧪 Testing & Evaluation

### Test Intent Classification

```bash
python intent_router.py
```

### Test Sentiment Analysis

```bash
python test_sentiment_routing.py
```

### Test Latency & Performance

```bash
python test_latency.py
```

**This will:**
* Test 11+ queries across all buckets (A, B, C)
* Measure client-side and server-side latency
* Report statistics (mean, median, min, max, stddev)
* Show bucket-wise performance breakdown
* Provide performance insights and recommendations

**Sample Output:**
```
📊 LATENCY STATISTICS
Overall Performance:
  Total Requests: 11
  Successful: 11

Server Processing Time:
  Mean:   145.32 ms
  Median: 89.45 ms
  Min:    45.12 ms
  Max:    542.78 ms

Bucket Performance Breakdown:
BUCKET_A (4 requests):
  Mean:   52.34 ms   (fastest - template responses)
  
BUCKET_B (5 requests):
  Mean:   198.67 ms  (RAG + LLM generation)
  
BUCKET_C (2 requests):
  Mean:   78.91 ms   (fast escalation)

💡 Performance Insights:
  • BUCKET_A is 3.8x faster than BUCKET_B
  • Average network overhead: 12.3 ms
```

### Dry-Run Evaluation (500 samples)

```bash
python dry_run_evaluation.py
```

### Test RAG Retrieval

```bash
python src/retriever.py
```

---

## 🔬 How It Works

### 1. Intent Classification

* **Model**: TF-IDF + Logistic Regression
* **Accuracy**: 97.69% on 27 intents
* **Processing**: <10ms per query
* **Output**: Intent label + confidence score

**Example:**
```
Input: "How do I track my order?"
Output: track_order (98% confidence) → BUCKET_A
```

### 2. Sentiment Analysis (Hybrid Approach)

* **Model**: DistilBERT SST-2 (sentiment) + keyword filter (anger detection)
* **Logic**: 
  - Classify sentiment (POSITIVE/NEGATIVE)
  - Check for anger keywords (terrible, frustrated, useless, etc.)
  - Override bucket to BUCKET_C if: NEGATIVE + high confidence + anger detected

**Example:**
```
Input: "This is terrible! I want my money back!"
Sentiment: NEGATIVE (92%) + anger keywords detected
Action: Override BUCKET_A → BUCKET_C (escalate)
```

### 3. FAISS Retrieval (BUCKET_B only)

* **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
* **Index**: IndexFlatIP (cosine similarity)
* **Top-K**: 3 most relevant documents
* **Storage**: Local files (no cloud)

### 4. Response Generation

* **BUCKET_A**: Return template (no LLM)
* **BUCKET_B**: RAG prompt + Groq LLM + response cleaning
* **BUCKET_C**: Escalation message

**Response Cleaning:**
- Removes `<think>...</think>` tags
- Removes internal reasoning
- Cleans whitespace

---

## 🎯 Three-Bucket Routing System

### BUCKET_A: Zero-Cost (30.6% of queries)

**Intents:** 
- check_invoice
- check_payment_methods
- check_refund_policy
- check_cancellation_fee
- delivery_period
- delivery_options
- track_order
- track_refund

**Handling:** Template responses, no LLM
**Cost:** $0

### BUCKET_B: Low-Cost (51.6% of queries)

**Intents:**
- cancel_order, change_order
- create_account, edit_account, delete_account
- get_invoice, get_refund
- change_shipping_address, set_up_shipping_address
- place_order, recover_password
- registration_problems, newsletter_subscription
- review, switch_account

**Handling:** FAISS retrieval → Groq generation
**Cost:** ~$0.0001 per query

### BUCKET_C: High-Cost (17.8% of queries)

**Intents:**
- complaint
- payment_issue
- contact_customer_service
- contact_human_agent

**Handling:** Escalation message or human handoff
**Cost:** Variable
**Trigger:** Intent-based OR sentiment override

---

## 🚀 Deployment

### Local Development

```bash
python api.py
# Access at http://localhost:8000
```

### Production Deployment

**Supported Platforms:**
- Railway.app (Recommended - $5 free credit/month)
- Fly.io (Free tier: 256MB x 3 machines)
- Render.com (Free tier: 512MB RAM - requires optimization)

**Deployment Guide:** See [DEPLOYMENT.md](DEPLOYMENT.md)

**Memory Optimization:**
- Lazy loading enabled by default
- Sentiment analysis optional (`ENABLE_SENTIMENT_ANALYSIS=false` saves ~200MB)
- Recommended: 2GB RAM for full features

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/chat` | POST | Process message |
| `/intents` | GET | List all intents |
| `/stats` | GET | Performance metrics |

**API Documentation:** `http://localhost:8000/docs` (auto-generated)

#### Chat Endpoint Example

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I track my order?"}'
```

**Response includes latency measurement:**
```json
{
  "response": "To track your order, please visit...",
  "intent": "track_order",
  "confidence": 0.98,
  "bucket": "BUCKET_A",
  "cost_tier": "Zero",
  "action": "Direct template response",
  "sentiment": "POSITIVE",
  "sentiment_score": 0.89,
  "escalated_by_sentiment": false,
  "latency_ms": 52.34,
  "session_id": null
}
```

**Latency Monitoring:**
* `latency_ms`: Server-side processing time (in milliseconds)
* Response headers include `X-Process-Time` for total request time
* Console logs show timing for every request: `⏱️ POST /chat - 0.052s`

---

## 📊 What Makes This Project Strong

✅ **Production-Grade Architecture**
- Not a demo - ready for real deployment
- FastAPI with proper error handling
- Lazy loading for deployment optimization

✅ **Cost Optimization**
- 79.6% cost reduction proven on 500-query evaluation
- Smart routing means most queries use zero/low-cost paths

✅ **Emotion Intelligence**
- Sentiment analysis catches frustrated customers
- Hybrid approach prevents false positives

✅ **Real ML Engineering**
- 97.69% classification accuracy
- Proper train/test split
- Confidence-based fallbacks

✅ **Modern Stack**
- LangGraph state machines
- Local-first (works offline after setup)
- Clean separation of concerns

This is the kind of chatbot used in:
* E-commerce customer support
* SaaS helpdesks
* Internal IT support
* Banking/fintech support

---

## 🔮 Future Improvements

- [ ] Multi-language support
- [ ] Conversation memory (chat history)
- [ ] Streaming responses
- [ ] Custom knowledge base upload
- [ ] Fine-tuned intent classifier
- [ ] A/B testing framework
- [ ] Analytics dashboard
- [ ] Docker containerization

---

## 👨‍💻 Author

**Vivek Kumar Gupta**
AI Engineering Student | GenAI & Agentic Systems Builder

* **GitHub**: [https://github.com/vivek34561](https://github.com/vivek34561)
* **LinkedIn**: [https://linkedin.com/in/vivek-gupta-0400452b6](https://linkedin.com/in/vivek-gupta-0400452b6)
* **Portfolio**: [https://resume-sepia-seven.vercel.app/](https://resume-sepia-seven.vercel.app/)

---

## 🙏 Acknowledgments

- **Bitext** for the customer support dataset
- **HuggingFace** for sentence-transformers and DistilBERT
- **Meta AI** for FAISS vector database
- **Groq** for lightning-fast LLM inference
- **LangChain team** for LangGraph framework

---

## 📄 License

MIT License © 2025 Vivek Kumar Gupta

---

## 🚀 Ready to Run?

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up .env with GROQ_API_KEY

# 3. Build FAISS index
python build_rag_index.py --limit 100

# 4. Run interactively
python src/main.py interactive

# OR run API
python api.py
### FAISS Vector Database
- ✅ No cloud service needed
- ✅ No API keys for vector DB
- ✅ Fast for small-medium datasets
- ✅ Can version control the index

### Groq LLM
- ✅ Ultra-fast inference (750+ tokens/sec)
- ✅ Free tier available
- ✅ Low latency
- ✅ Cost-effective

## 📦 Dependencies

```
# Core
pandas, numpy, scikit-learn

# Embeddings
sentence-transformers, torch

# Vector DB
faiss-cpu

# LLM
groq

# Orchestration
langchain, langchain-groq, langgraph
```

## 🔑 API Keys Needed

**Required:**
- Groq API Key (free at console.groq.com)

**Optional:**
- None! Embeddings and vector DB are local

## 📈 Cost Comparison

**Traditional Approach (uniform GPT-4):**
- 100% of queries → GPT-4 → High cost

**Our Approach:**
- 30.6% → Zero cost (templates)
- 51.6% → Low cost (Groq)
- 17.8% → High cost (escalation)
- **Result: 79.6% cost reduction**

## 🎓 Dataset

Bitext Customer Support Dataset v11
- **Size:** 26,872 instruction-response pairs
- **Intents:** 27 categories
- **Language:** English
- **Domain:** E-commerce customer support

## 🤝 Contributing

Feel free to open issues or submit PRs!

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Bitext for the customer support dataset
- HuggingFace for sentence-transformers
- Meta for FAISS
- Groq for fast LLM inference
- LangChain team for LangGraph

---

**Ready to run?**
```bash
pip install -r requirements-rag.txt
python build_rag_index.py --limit 100
python src/main.py interactive
```
