# ⚡ Quick Start Checklist

> Get your Investment Agent System running in 5 minutes!

---

## ✅ Pre-Flight Checklist

### Step 1: Verify Files (30 seconds)

```powershell
cd c:\Users\aviru\Downloads\Invest_agent

# Check all files are present
dir
```

**Expected files:**
- [ ] `investment_agent_system.py` ✓
- [ ] `requirements.txt` ✓
- [ ] `.env` ✓
- [ ] `.env.example` ✓
- [ ] `README.md` ✓
- [ ] PDF files (RIL, RGICL annual reports) ✓

---

### Step 2: Install Dependencies (2-3 minutes)

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate

# Install all packages
pip install -r requirements.txt
```

**First-time notes:**
- 📦 Downloads ~500MB of packages
- 🧠 Embedding model (~90MB) downloads on first run
- ⏱️ Takes 2-5 minutes depending on internet speed

---

### Step 3: Verify API Key (30 seconds)

```powershell
# Check .env file
type .env
```

**Expected output:**
```
OPENROUTER_API_KEY=sk-or-v1-96f94370...
DEFAULT_MODEL=anthropic/claude-3.5-sonnet
DOCUMENTS_DIR=./
```

---

### Step 4: Run First Analysis (10-15 minutes)

```powershell
python investment_agent_system.py
```

**What happens:**
1. 📄 Loads PDF documents (your annual reports)
2. 🔍 Creates vector embeddings for search
3. 🤖 Runs 7-agent analysis pipeline
4. 📊 Generates investment recommendation

---

## 🎯 Expected Output

```
======================================================================
🚀 INVESTMENT AGENT SYSTEM - Multi-Agent Analysis Framework
======================================================================

📦 Initializing Investment Agent System...

📄 Loading financial documents...
   Loading: RIL-Integrated-Annual-Report-2024-25.pdf
     → 245 chunks
   Loading: 2023-24-rgicl-annual-report.pdf
     → 180 chunks
   ✅ Indexed 425 total chunks

======================================================================
📊 Running Investment Analysis...
======================================================================

Retrieved 5 relevant document chunks
✅ Pro Agent completed analysis
✅ Against Agent completed analysis
✅ Jury FUNDAMENTALS completed analysis
✅ Jury RISK completed analysis
✅ Jury ESG completed analysis
✅ Jury SENTIMENT completed analysis
✅ Judge rendered final decision

======================================================================
INVESTMENT ANALYSIS REPORT
Company: Reliance Industries Limited (RIL)
======================================================================

📊 FINAL DECISION
----------------------------------------
Decision: BUY
Confidence: 0.78

📝 REASONING
[Detailed multi-paragraph analysis...]

🔑 KEY CONSIDERATIONS
  1. Strong revenue diversification
  2. Jio digital platform growth
  3. Retail expansion momentum
  ...

💾 Report saved to: investment_report_RIL_20241230_160000.txt
💾 JSON data saved to: investment_analysis_RIL_20241230_160000.json
```

---

## 🔧 Common Issues & Fixes

### ❌ "ModuleNotFoundError: No module named 'langchain'"

**Fix:**
```powershell
pip install -r requirements.txt --upgrade
```

---

### ❌ "OPENROUTER_API_KEY not found"

**Fix:**
```powershell
# Verify .env file exists and has your key
type .env

# If missing, create it:
echo OPENROUTER_API_KEY=sk-or-v1-your-key-here > .env
```

---

### ❌ "No documents loaded" or "0 chunks"

**Fix:**
```powershell
# Check PDFs exist in directory
dir *.pdf

# If PDFs are elsewhere, update DOCUMENTS_DIR in .env
# Or copy PDFs to current directory
```

---

### ❌ "Connection timeout" or API errors

**Fix:**
1. Check internet connection
2. Verify API key is valid at [openrouter.ai/keys](https://openrouter.ai/keys)
3. Check API credits at [openrouter.ai/usage](https://openrouter.ai/usage)

---

### ❌ Slow first run (downloading models)

**This is normal!** First run downloads:
- Embedding model: ~90MB
- Takes 1-2 minutes

Subsequent runs are much faster.

---

## 📊 Verify Success

After running, you should have **2 new files**:

```powershell
dir investment_*
```

**Expected:**
- `investment_report_RIL_YYYYMMDD_HHMMSS.txt` - Human-readable report
- `investment_analysis_RIL_YYYYMMDD_HHMMSS.json` - Full JSON data

---

## 🚀 Next Steps

### Analyze Different Company

Edit the `main()` function in `investment_agent_system.py`:

```python
# Change these values:
result = system.analyze(
    query="Should I invest? Analyze financials and risks.",
    ticker="RGICL",  # Changed
    company_name="Reliance General Insurance"  # Changed
)
```

### Ask Specific Questions

```python
result = system.analyze(
    query="What are the key risks in the insurance business?",
    ticker="RGICL",
    company_name="Reliance General Insurance"
)
```

### Add More Documents

Simply drop PDFs into the folder - they'll be auto-detected!

---

## 💡 Pro Tips

| Tip | Why |
|-----|-----|
| Use Claude 3.5 Sonnet | Best quality/cost balance |
| Add earnings transcripts | Richer context for analysis |
| Reduce `TOP_K_RETRIEVAL` to 3 | Faster (but less thorough) |
| Keep PDFs organized | One folder per company works best |

---

## ✅ Checklist Complete!

You should now have:
- [ ] Dependencies installed
- [ ] API key configured
- [ ] First analysis completed
- [ ] Report files generated

**Congratulations! 🎉 Your Investment Agent System is ready!**

---

## 📞 Need Help?

1. Check the **README.md** for detailed documentation
2. Check the **Troubleshooting** section above
3. Verify API credits at [openrouter.ai](https://openrouter.ai)

---

*Happy Investing! 🚀📈*
