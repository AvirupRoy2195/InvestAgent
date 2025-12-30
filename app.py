"""
Investment Agent System - Streamlit Web Interface
==================================================
A modern web UI for the multi-agent investment analysis system.
Features:
- PDF document upload for RAG
- Multiple free LLM model selection
- Interactive analysis with progress tracking
- Beautiful report visualization

Run with: streamlit run app.py
"""

import streamlit as st
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
import time

# Import our investment system
from investment_agent_system import (
    InvestmentAgentSystem, 
    Config, 
    RAGSystem,
    FREE_MODELS,
    AGENT_MODEL_MAPPING,
    generate_report
)

# Page configuration
st.set_page_config(
    page_title="🚀 Investment Agent System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .decision-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .decision-sell {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .decision-hold {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'system' not in st.session_state:
    st.session_state.system = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()


def save_uploaded_files(uploaded_files):
    """Save uploaded files to temp directory and return paths"""
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = Path(st.session_state.temp_dir) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(file_path))
    return saved_paths


def initialize_system(api_key: str, model: str):
    """Initialize the investment agent system"""
    config = Config(
        openrouter_api_key=api_key,
        default_model=model,
        documents_dir=st.session_state.temp_dir
    )
    return InvestmentAgentSystem(config)


def render_decision_badge(decision: str):
    """Render a colored badge for the decision"""
    if decision == "BUY":
        st.markdown('<div class="decision-buy">🟢 BUY</div>', unsafe_allow_html=True)
    elif decision == "SELL":
        st.markdown('<div class="decision-sell">🔴 SELL</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="decision-hold">🟡 HOLD</div>', unsafe_allow_html=True)


def render_jury_scores(jury_verdicts: dict):
    """Render jury specialist scores as metrics"""
    cols = st.columns(4)
    
    specialists = [
        ("Fundamentals", "fundamentals", "📊"),
        ("Risk", "risk", "⚠️"),
        ("ESG", "esg", "🌱"),
        ("Sentiment", "sentiment", "💭")
    ]
    
    for col, (name, key, emoji) in zip(cols, specialists):
        with col:
            verdict = jury_verdicts.get(key, {})
            # Try different score key formats
            score = verdict.get(f"{key.upper()}_SCORE") or verdict.get("SCORE") or verdict.get("score") or "N/A"
            if isinstance(score, (int, float)):
                score_display = f"{score:.0%}" if score <= 1 else f"{score}"
            else:
                score_display = str(score)
            
            st.metric(
                label=f"{emoji} {name}",
                value=score_display
            )


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Key
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Get your free API key at openrouter.ai"
    )
    
    # Model Selection
    st.markdown("### 🤖 Select Model (Free)")
    selected_model_name = st.selectbox(
        "Choose LLM",
        options=list(FREE_MODELS.keys()),
        index=0,
        help="All models are FREE to use via OpenRouter"
    )
    selected_model = FREE_MODELS[selected_model_name]
    
    st.caption(f"Model ID: `{selected_model}`")
    
    st.divider()
    
    # PDF Upload Section
    st.markdown("### 📄 Upload Documents")
    st.caption("Upload PDF annual reports, financial statements, or research reports")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload company annual reports, 10-K filings, or research reports"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        for f in uploaded_files:
            st.caption(f"📁 {f.name} ({f.size / 1024:.1f} KB)")
    
    # Load Documents Button
    if st.button("📥 Load Documents for Analysis", type="primary", disabled=not uploaded_files):
        if not api_key:
            st.error("Please enter your OpenRouter API key")
        else:
            with st.spinner("Processing documents..."):
                # Save uploaded files
                saved_paths = save_uploaded_files(uploaded_files)
                st.session_state.uploaded_files = saved_paths
                
                # Initialize system
                st.session_state.system = initialize_system(api_key, selected_model)
                
                # Load documents into RAG
                chunk_count = st.session_state.system.load_documents(saved_paths)
                
                st.session_state.documents_loaded = True
                st.success(f"✅ Loaded {chunk_count} document chunks!")
    
    st.divider()
    
    # System Status
    st.markdown("### 📊 System Status")
    if st.session_state.documents_loaded:
        st.success("✅ Documents Loaded")
        st.caption(f"📂 {len(st.session_state.uploaded_files)} files indexed")
    else:
        st.warning("⚠️ No documents loaded")
    
    # Model info
    st.info(f"🤖 Model: {selected_model_name}")


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<h1 class="main-header">🚀 Investment Agent System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Agent Investment Analysis powered by AI • 7 Specialized Agents • Free LLMs</p>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Results", "ℹ️ About"])

with tab1:
    st.markdown("## 🔍 Run Investment Analysis")
    
    if not st.session_state.documents_loaded:
        st.info("👈 Please upload PDF documents in the sidebar first, then click 'Load Documents for Analysis'")
        
        # Show sample workflow
        st.markdown("### 📋 How It Works")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 1️⃣ Upload")
            st.write("Upload PDF annual reports or financial documents")
        
        with col2:
            st.markdown("#### 2️⃣ Configure")
            st.write("Choose a free LLM model and enter company details")
        
        with col3:
            st.markdown("#### 3️⃣ Analyze")
            st.write("7 AI agents debate and analyze the investment")
        
        with col4:
            st.markdown("#### 4️⃣ Decide")
            st.write("Get BUY/SELL/HOLD recommendation with reasoning")
    
    else:
        # Analysis Form
        st.markdown("### 🏢 Company Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input(
                "Stock Ticker",
                value="RIL",
                placeholder="e.g., RIL, TCS, INFY"
            )
        
        with col2:
            company_name = st.text_input(
                "Company Name",
                value="Reliance Industries Limited",
                placeholder="Full company name"
            )
        
        query = st.text_area(
            "Analysis Query",
            value="Should I invest in this company for long-term wealth creation? Analyze growth prospects, financial health, risks, and provide a recommendation.",
            height=100,
            placeholder="What would you like to analyze?"
        )
        
        # Analysis Options
        st.markdown("### ⚙️ Analysis Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            focus_fundamentals = st.checkbox("📊 Deep Fundamentals", value=True)
        with col2:
            focus_risk = st.checkbox("⚠️ Risk Analysis", value=True)
        with col3:
            focus_esg = st.checkbox("🌱 ESG Factors", value=True)
        
        # Run Analysis Button
        st.markdown("---")
        
        if st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True):
            if not ticker or not company_name:
                st.error("Please enter both ticker and company name")
            else:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Agent status display
                agent_cols = st.columns(7)
                agent_status = {
                    "RAG": agent_cols[0].empty(),
                    "Pro": agent_cols[1].empty(),
                    "Against": agent_cols[2].empty(),
                    "Fundamentals": agent_cols[3].empty(),
                    "Risk": agent_cols[4].empty(),
                    "ESG": agent_cols[5].empty(),
                    "Judge": agent_cols[6].empty(),
                }
                
                # Initialize all as pending
                for name, placeholder in agent_status.items():
                    placeholder.markdown(f"⏳ {name}")
                
                try:
                    # Simulate progress (actual progress would need callbacks)
                    status_text.text("🔍 Retrieving relevant documents...")
                    agent_status["RAG"].markdown("🔄 RAG")
                    progress_bar.progress(10)
                    
                    # Run actual analysis
                    status_text.text("🤖 Running multi-agent analysis...")
                    
                    result = st.session_state.system.analyze(
                        query=query,
                        ticker=ticker,
                        company_name=company_name
                    )
                    
                    # Update progress
                    for name in agent_status:
                        agent_status[name].markdown(f"✅ {name}")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Store result
                    st.session_state.analysis_result = result
                    
                    # Show success message
                    st.success("🎉 Analysis complete! Check the Results tab for detailed findings.")
                    
                    # Quick preview
                    decision = result.get("decision", {})
                    if decision and not decision.get("error"):
                        st.markdown("### 📊 Quick Summary")
                        render_decision_badge(decision.get("DECISION", "HOLD"))
                        st.metric("Confidence", f"{decision.get('CONFIDENCE', 0):.0%}")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)

with tab2:
    st.markdown("## 📈 Analysis Results")
    
    if st.session_state.analysis_result is None:
        st.info("No analysis results yet. Run an analysis in the Analysis tab first.")
    else:
        result = st.session_state.analysis_result
        decision = result.get("decision", {})
        
        # Header with decision
        st.markdown(f"### {result.get('company_name', 'Company')} ({result.get('ticker', 'N/A')})")
        st.caption(f"Analysis completed: {result.get('timestamp', 'N/A')}")
        
        # Main decision display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 🎯 Decision")
            render_decision_badge(decision.get("DECISION", "HOLD"))
            
            confidence = decision.get("CONFIDENCE", 0)
            if isinstance(confidence, (int, float)):
                st.metric("Confidence Score", f"{confidence:.0%}")
            else:
                st.metric("Confidence Score", str(confidence))
        
        with col2:
            st.markdown("#### 📝 Summary")
            reasoning = decision.get("REASONING", "No reasoning provided")
            if isinstance(reasoning, str):
                st.write(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)
            else:
                st.write(str(reasoning))
        
        st.divider()
        
        # Jury Scores
        st.markdown("### ⚖️ Jury Specialist Scores")
        jury_verdicts = result.get("jury_verdicts", {})
        if jury_verdicts:
            render_jury_scores(jury_verdicts)
        else:
            st.warning("No jury verdicts available")
        
        st.divider()
        
        # Pro vs Against
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Bullish Case (Pro Agent)")
            pro_case = result.get("pro_case", {})
            if pro_case:
                bullish_points = pro_case.get("KEY_BULLISH_POINTS") or pro_case.get("raw_response", "No data")
                if isinstance(bullish_points, list):
                    for point in bullish_points:
                        st.markdown(f"✓ {point}")
                else:
                    st.write(str(bullish_points)[:500])
            else:
                st.info("No bullish case data")
        
        with col2:
            st.markdown("### 🔴 Bearish Case (Against Agent)")
            against_case = result.get("against_case", {})
            if against_case:
                bearish_points = against_case.get("KEY_BEARISH_POINTS") or against_case.get("raw_response", "No data")
                if isinstance(bearish_points, list):
                    for point in bearish_points:
                        st.markdown(f"✗ {point}")
                else:
                    st.write(str(bearish_points)[:500])
            else:
                st.info("No bearish case data")
        
        st.divider()
        
        # Key Considerations & Dissenting Views
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔑 Key Considerations")
            considerations = decision.get("KEY_CONSIDERATIONS", [])
            if considerations:
                for i, point in enumerate(considerations, 1):
                    st.markdown(f"{i}. {point}")
            else:
                st.info("No key considerations listed")
        
        with col2:
            st.markdown("### ⚠️ Dissenting Views")
            dissenting = decision.get("DISSENTING_VIEWS", [])
            if dissenting:
                for point in dissenting:
                    st.markdown(f"• {point}")
            else:
                st.info("No dissenting views recorded")
        
        st.divider()
        
        # Download Options
        st.markdown("### 💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Text report
            report = generate_report(result)
            st.download_button(
                label="📄 Download Text Report",
                data=report,
                file_name=f"investment_report_{result.get('ticker', 'analysis')}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # JSON data
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📊 Download JSON Data",
                data=json_data,
                file_name=f"investment_analysis_{result.get('ticker', 'analysis')}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        # Raw JSON (expandable)
        with st.expander("🔧 View Raw JSON Response"):
            st.json(result)

with tab3:
    st.markdown("## ℹ️ About Investment Agent System")
    
    st.markdown("""
    ### 🏗️ Architecture
    
    This system uses **7 specialized AI agents** working together to analyze investments.
    **Each agent uses a DIFFERENT free LLM model** for diverse, robust analysis:
    
    | Agent | Role | LLM Model |
    |-------|------|-------|
    | **Pro Agent** 🟢 | Bullish Advocate | OLMo 3.1 32B Think ⭐ |
    | **Against Agent** 🔴 | Bearish Advocate | DeepSeek V3.1 Nex |
    | **Judge** ⚖️ | Final Decision | OLMo 3.1 32B Think ⭐ |
    | **Fundamentals** 📊 | Financial Metrics | DeepSeek V3.1 Nex |
    | **Risk** ⚠️ | Risk Assessment | Nvidia Nemotron 30B |
    | **ESG** 🌱 | Sustainability | OLMo 3 32B Think |
    | **Sentiment** 💭 | Market Psychology | Xiaomi MiMo V2 Flash |
    
    ---
    
    ### 🤖 Multi-Model Architecture
    
    **Why multiple models?**
    - 🎯 **Diverse perspectives**: Each model has different training data and reasoning patterns
    - ⚡ **Specialized strengths**: Some models excel at analysis, others at quick inference
    - 🔒 **Reduced bias**: Multiple models cross-check each other's conclusions
    - 🆓 **100% FREE**: All models are free tier via OpenRouter
    
    **Model Assignments:**
    
    ---
    
    ### 🤖 All Free LLM Models Used
    
    All **5 models** are **100% FREE** via OpenRouter:
    
    | Model | Agent Assignment | Strength |
    |-------|-----------------|----------|
    | **OLMo 3.1 32B Think** ⭐ | Pro Agent, Judge | Best reasoning & decision-making |
    | **OLMo 3 32B Think** | ESG Jury | Strong ESG analysis |
    | **DeepSeek V3.1 Nex** | Against, Fundamentals | Deep financial analysis |
    | **Nvidia Nemotron 30B** | Risk Jury | Fast risk assessment |
    | **Xiaomi MiMo V2 Flash** | Sentiment Jury | Quick sentiment analysis |
    
    ---
    
    ### 📄 Supported Documents
    
    Upload any PDF financial documents:
    - Annual Reports (10-K, Annual Report)
    - Quarterly Reports (10-Q)
    - Investor Presentations
    - Research/Broker Reports
    - Earnings Transcripts
    
    ---
    
    ### ⚡ Quick Start
    
    1. **Get API Key**: Sign up at [openrouter.ai](https://openrouter.ai) (free)
    2. **Upload Documents**: Add PDF annual reports in sidebar
    3. **Enter Company**: Ticker and name
    4. **Run Analysis**: Click the button and wait ~2-5 minutes
    5. **Review Results**: See decision with full reasoning
    
    ---
    
    ### 💡 Tips for Best Results
    
    - Upload **recent annual reports** (last 1-2 years)
    - Include **earnings transcripts** for sentiment analysis
    - Add **broker research** for external perspectives
    - Use **specific queries** for focused analysis
    
    ---
    
    ### 📞 Support
    
    - **OpenRouter Docs**: [openrouter.ai/docs](https://openrouter.ai/docs)
    - **LangGraph Docs**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
    """)
    
    st.divider()
    
    st.markdown("""
    ---
    *Built with ❤️ using LangGraph, OpenRouter, and Streamlit*
    
    **Remember**: This is an AI-powered analysis tool. Always combine with your own research and professional financial advice.
    """)


# Footer
st.divider()
st.caption("Investment Agent System v1.1 • Multi-Model Multi-Agent Framework • 5 Free LLMs via OpenRouter")
