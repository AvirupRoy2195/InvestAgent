# 🚀 Investment Agent System

> **Multi-Agent Investment Analysis Framework** with Streamlit UI, Free LLMs & RAG

A production-ready investment analysis system using 7 specialized AI agents with a beautiful web interface. Upload your PDF annual reports and get BUY/SELL/HOLD recommendations with full reasoning.

---

## ✨ Features

- 🤖 **7 AI Agents** - Pro, Against, Judge + 4 Jury Specialists
- 📄 **PDF Upload** - Drag & drop annual reports for RAG analysis
- 🆓 **100% Free LLMs** - No API costs (via OpenRouter free tier)
- 🌐 **Beautiful Streamlit UI** - Modern web interface
- 📊 **Transparent Reasoning** - Full chain of thought
- 💾 **Export Reports** - Download as TXT or JSON

---

## 🎯 Quick Start (3 Steps)

### 1️⃣ Install Dependencies

```powershell
cd c:\Users\aviru\Downloads\Invest_agent
pip install -r requirements.txt
```

### 2️⃣ Get Free API Key

1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up (free)
3. Copy your API key

### 3️⃣ Run the App

```powershell
streamlit run app.py
```

The app opens at `http://localhost:8501` 🎉

---

## 📸 Screenshot

```
┌─────────────────────────────────────────────────────────────────┐
│  🚀 Investment Agent System                                      │
│  Multi-Agent Investment Analysis powered by AI                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐   ┌───────────────────────────────────────┐│
│  │ ⚙️ CONFIG       │   │  📊 Analysis                          ││
│  │                 │   │                                        ││
│  │ API Key: ****   │   │  Company: Reliance Industries          ││
│  │                 │   │  Ticker: RIL                           ││
│  │ Model:          │   │                                        ││
│  │ [OLMo 3.1 32B]  │   │  Query: Should I invest in this        ││
│  │                 │   │  company for long-term growth?         ││
│  │ 📄 Upload PDFs  │   │                                        ││
│  │ [Browse...]     │   │  [🚀 Run Multi-Agent Analysis]         ││
│  │                 │   │                                        ││
│  │ ✅ 4 files      │   │  ┌─────────────────────────────────┐   ││
│  │ loaded          │   │  │  🟢 BUY                         │   ││
│  │                 │   │  │  Confidence: 78%                │   ││
│  └─────────────────┘   │  └─────────────────────────────────┘   ││
│                        └───────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Free LLM Models

All models are **100% FREE** via OpenRouter:

| Model | Best For |
|-------|----------|
| **OLMo 3.1 32B Think** ⭐ | Reasoning (Recommended) |
| **DeepSeek V3.1 Nex** | General performance |
| **Nvidia Nemotron 30B** | Fast inference |
| **Xiaomi MiMo V2 Flash** | Quick responses |

---

## 🏗️ System Architecture

```
PDF Upload → RAG Indexing → Query
                              ↓
         ┌────────────────────┼────────────────────┐
         ↓                    ↓                    ↓
    [Pro Agent]         [Against Agent]      [Jury Council]
       🟢                    🔴              ├─ 📊 Fundamentals
    Bullish Case         Bearish Case       ├─ ⚠️ Risk
                                            ├─ 🌱 ESG
                                            └─ 💭 Sentiment
         └────────────────────┼────────────────────┘
                              ↓
                       [Judge Agent ⚖️]
                              ↓
                    ┌─────────────────┐
                    │ BUY/SELL/HOLD   │
                    │ + Confidence    │
                    │ + Full Reasoning│
                    └─────────────────┘
```

---

## 📦 Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |
| `investment_agent_system.py` | Core 7-agent framework |
| `requirements.txt` | Python dependencies |
| `.env` | API configuration |
| `README.md` | Documentation |

---

## 💻 Command Line Mode

You can also run without the UI:

```python
from investment_agent_system import InvestmentAgentSystem, Config

config = Config()
system = InvestmentAgentSystem(config)
system.load_documents(["path/to/annual_report.pdf"])

result = system.analyze(
    query="Should I invest in this company?",
    ticker="RIL",
    company_name="Reliance Industries"
)

print(result["decision"])
```

---

## 📄 Supported Documents

Upload any financial PDF:
- ✅ Annual Reports (10-K)
- ✅ Quarterly Reports (10-Q)
- ✅ Investor Presentations
- ✅ Broker Research Reports
- ✅ Earnings Transcripts

---

## ⚡ Tips for Best Results

1. **Upload recent reports** - Last 1-2 years
2. **Include multiple documents** - More context = better analysis
3. **Use specific queries** - "What are the growth catalysts?" vs "Tell me about the company"
4. **Try different models** - Some work better for certain companies

---

## 🔧 Configuration

Edit `.env` to customize:

```bash
# Your OpenRouter API key
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Model (see free options above)
DEFAULT_MODEL=allenai/olmo-3.1-32b-think:free

# Analysis settings
TEMPERATURE=0.3
MAX_TOKENS=4096
TOP_K_RETRIEVAL=5
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "API Key not found"
Enter your key in the Streamlit sidebar or set in `.env`

### "No documents loaded"
Upload PDF files using the sidebar uploader

### Slow first run
First run downloads embedding model (~90MB). Subsequent runs are faster.

---

## 📞 Resources

- **OpenRouter**: [openrouter.ai](https://openrouter.ai)
- **LangGraph**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **Streamlit**: [streamlit.io](https://streamlit.io)

---

## ⚠️ Disclaimer

This is an AI-powered analysis tool for educational purposes. Always combine with your own research and professional financial advice before making investment decisions.

---

**Happy Investing! 🚀📈**
