# 🏗️ Setup Guide - Architecture & Deployment

> Complete technical reference for the Investment Agent System

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Deep Dive](#component-deep-dive)
3. [Configuration Reference](#configuration-reference)
4. [Deployment Options](#deployment-options)
5. [Production Considerations](#production-considerations)

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INVESTMENT AGENT SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    USER QUERY                             │   │
│  │  "Should I invest in Reliance Industries?"                │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  RAG RETRIEVAL                            │   │
│  │  ┌─────────┐    ┌──────────────┐    ┌────────────────┐   │   │
│  │  │  Query  │───▶│  Embeddings  │───▶│  FAISS Vector  │   │   │
│  │  │ Parser  │    │ (HuggingFace)│    │     Store      │   │   │
│  │  └─────────┘    └──────────────┘    └────────────────┘   │   │
│  │                                              │            │   │
│  │                         ┌────────────────────┘            │   │
│  │                         ▼                                 │   │
│  │              ┌────────────────────┐                       │   │
│  │              │  Relevant Chunks   │                       │   │
│  │              │  (Top K = 5)       │                       │   │
│  │              └─────────┬──────────┘                       │   │
│  └────────────────────────┼──────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               MULTI-AGENT DEBATE LAYER                    │   │
│  │                                                           │   │
│  │   ┌─────────────┐              ┌─────────────────┐       │   │
│  │   │  PRO AGENT  │              │  AGAINST AGENT  │       │   │
│  │   │   🟢        │              │      🔴         │       │   │
│  │   │  Bullish    │              │    Bearish      │       │   │
│  │   │   Case      │              │     Case        │       │   │
│  │   └──────┬──────┘              └────────┬────────┘       │   │
│  │          │                              │                │   │
│  │          └──────────────┬───────────────┘                │   │
│  │                         ▼                                │   │
│  └─────────────────────────┼────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   JURY SPECIALISTS                        │   │
│  │                                                           │   │
│  │  ┌─────────────┐  ┌───────────┐  ┌─────────┐  ┌────────┐ │   │
│  │  │ FUNDAMENTALS│  │   RISK    │  │   ESG   │  │SENTIMENT│ │   │
│  │  │     📊      │  │    ⚠️     │  │   🌱    │  │   💭   │ │   │
│  │  │ Revenue,    │  │ Market,   │  │ Environ │  │ Mgmt   │ │   │
│  │  │ Margins,    │  │ Regulatory│  │ Social, │  │ Tone,  │ │   │
│  │  │ ROE, Debt   │  │ Ops Risk  │  │ Govern. │  │ Narrative│ │   │
│  │  └──────┬──────┘  └─────┬─────┘  └────┬────┘  └────┬───┘ │   │
│  │         └───────────────┼─────────────┴────────────┘     │   │
│  │                         ▼                                │   │
│  └─────────────────────────┼────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    JUDGE AGENT                            │   │
│  │                        ⚖️                                 │   │
│  │                                                           │   │
│  │   Weighs all evidence • Considers all perspectives        │   │
│  │   Identifies key factors • Acknowledges dissent           │   │
│  │   Renders final decision with confidence score            │   │
│  │                                                           │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   FINAL OUTPUT                            │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Decision: BUY | SELL | HOLD                       │  │   │
│  │  │  Confidence: 0.78 (78%)                            │  │   │
│  │  │  Reasoning: Full chain of thought                  │  │   │
│  │  │  Key Considerations: Top 5 factors                 │  │   │
│  │  │  Dissenting Views: Counter-arguments               │  │   │
│  │  │  Risk Warnings: What could invalidate              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. RAG System (`RAGSystem` class)

**Purpose:** Retrieves relevant information from your PDF documents

**Components:**
- **Document Loader:** PyPDFLoader for PDF parsing
- **Text Splitter:** RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 (local, free)
- **Vector Store:** FAISS (local, no cloud dependency)

**Configuration:**
```python
RAG Configuration
├── chunk_size: 1000          # Characters per chunk
├── chunk_overlap: 200        # Overlap between chunks
├── embedding_model: all-MiniLM-L6-v2
└── top_k_retrieval: 5        # Documents to retrieve
```

### 2. Agent Framework (`InvestmentAgentSystem` class)

**LangGraph Workflow:**
```python
workflow = StateGraph(GraphState)

# Nodes (in execution order)
1. retrieve_documents  →  RAG lookup
2. pro_agent          →  Bullish analysis (parallel)
3. against_agent      →  Bearish analysis (parallel)
4. jury_fundamentals  →  Financial metrics
5. jury_risk          →  Risk assessment
6. jury_esg           →  ESG scoring
7. jury_sentiment     →  Sentiment analysis
8. judge_agent        →  Final decision
```

### 3. LLM Integration (OpenRouter)

**Why OpenRouter?**
- Access to 50+ models via single API
- Pay-per-use (no monthly minimums)
- Easy model switching
- Fallback options

**Supported Models:**
| Model | ID | Best For |
|-------|----|---------| 
| Claude 3.5 Sonnet | `anthropic/claude-3.5-sonnet` | Best overall |
| GPT-4 Turbo | `openai/gpt-4-turbo` | Reasoning tasks |
| Llama 3 70B | `meta-llama/llama-3-70b-instruct` | Cost efficiency |
| Mixtral 8x7B | `mistralai/mixtral-8x7b-instruct` | Speed |

---

## Configuration Reference

### Environment Variables (`.env`)

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-...    # Your API key

# Optional - Model
DEFAULT_MODEL=anthropic/claude-3.5-sonnet
TEMPERATURE=0.3                     # 0.0-1.0 (lower = more consistent)
MAX_TOKENS=4096                     # Response length limit

# Optional - RAG
DOCUMENTS_DIR=./                    # PDF location
TOP_K_RETRIEVAL=5                   # Documents per query
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional - Analysis
CONFIDENCE_THRESHOLD=0.6            # Minimum confidence for decisions
```

### Programmatic Configuration

```python
from investment_agent_system import Config, InvestmentAgentSystem

# Custom configuration
config = Config(
    openrouter_api_key="sk-or-v1-...",
    default_model="anthropic/claude-3.5-sonnet",
    temperature=0.2,
    max_tokens=8192,
    chunk_size=1500,
    top_k_retrieval=10,
    documents_dir="./data/annual_reports/"
)

system = InvestmentAgentSystem(config)
```

---

## Deployment Options

### Option 1: Local Development (Current Setup)

```
Your Machine
├── Python 3.10+
├── investment_agent_system.py
├── .env (API key)
└── PDF documents
```

**Pros:** Simple, private, no cloud costs
**Cons:** Only works on your machine

---

### Option 2: Docker Container

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY investment_agent_system.py .
COPY .env .

# Create documents directory
RUN mkdir -p /app/documents
VOLUME /app/documents

CMD ["python", "investment_agent_system.py"]
```

**Build & Run:**
```bash
docker build -t investment-agent .
docker run -v ./pdfs:/app/documents investment-agent
```

---

### Option 3: Cloud Deployment (AWS)

**Architecture:**
```
API Gateway → Lambda → ECS Container
                          ↓
                    S3 (PDFs)
                          ↓
                    OpenRouter API
```

**Terraform snippet:**
```hcl
resource "aws_ecs_task_definition" "investment_agent" {
  family = "investment-agent"
  container_definitions = jsonencode([{
    name  = "investment-agent"
    image = "your-ecr-repo/investment-agent:latest"
    environment = [
      {
        name  = "OPENROUTER_API_KEY"
        value = var.openrouter_key
      }
    ]
  }])
}
```

---

### Option 4: Streamlit Web App

**Create `app.py`:**
```python
import streamlit as st
from investment_agent_system import InvestmentAgentSystem, Config

st.title("🚀 Investment Agent System")

# Initialize
@st.cache_resource
def get_system():
    config = Config()
    system = InvestmentAgentSystem(config)
    system.load_documents()
    return system

system = get_system()

# UI
ticker = st.text_input("Ticker Symbol", "RIL")
company = st.text_input("Company Name", "Reliance Industries")
query = st.text_area("Analysis Query", "Should I invest?")

if st.button("Analyze"):
    with st.spinner("Running 7-agent analysis..."):
        result = system.analyze(query, ticker, company)
    
    st.header("📊 Decision")
    decision = result['decision']
    st.metric("Recommendation", decision.get('DECISION'))
    st.metric("Confidence", f"{decision.get('CONFIDENCE', 0)*100:.0f}%")
    
    st.header("📝 Reasoning")
    st.write(decision.get('REASONING'))
```

**Run:**
```bash
pip install streamlit
streamlit run app.py
```

---

## Production Considerations

### 1. Error Handling

The system includes built-in error handling:
```python
try:
    response = self.llm.invoke([HumanMessage(content=prompt)])
except Exception as e:
    logger.error(f"Agent error: {e}")
    state["errors"].append(str(e))
```

### 2. Rate Limiting

OpenRouter has rate limits. For high-volume use:
```python
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=60))
def call_llm_with_retry(prompt):
    return llm.invoke([HumanMessage(content=prompt)])
```

### 3. Caching

Cache embeddings and frequent queries:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieve(query: str):
    return rag.retrieve(query)
```

### 4. Monitoring

Add logging for production:
```python
import structlog

logger = structlog.get_logger()
logger.info("analysis_started", ticker=ticker, company=company)
logger.info("analysis_complete", decision=decision, confidence=confidence)
```

### 5. Security

- Never commit `.env` to git
- Rotate API keys periodically
- Use secrets manager in production (AWS Secrets Manager, HashiCorp Vault)

---

## Performance Optimization

| Optimization | Impact | Implementation |
|-------------|--------|----------------|
| Reduce TOP_K | Faster, less thorough | `TOP_K_RETRIEVAL=3` |
| Smaller chunks | Faster indexing | `chunk_size=500` |
| GPU embeddings | 10x faster | Install `faiss-gpu` |
| Async agents | Parallel execution | Use async/await |
| Model caching | Faster startup | Pre-download models |

---

## Monitoring Dashboard Metrics

For production deployments, track:

| Metric | Description | Target |
|--------|-------------|--------|
| Analysis time | End-to-end duration | < 5 min |
| Token usage | Tokens per analysis | < 50K |
| Error rate | Failed analyses | < 1% |
| RAG recall | Relevant docs found | > 80% |
| Decision distribution | BUY/SELL/HOLD ratio | Balanced |

---

*This guide covers setup through production deployment. For advanced patterns, see `ADVANCED_PATTERNS.md`.*
