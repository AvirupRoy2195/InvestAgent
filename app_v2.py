"""
Agentic Investment System - Streamlit Web Interface V2
=======================================================
Enhanced UI for 12-agent courtroom-style investment analysis.
Features:
- PDF upload with semantic chunking
- Query understanding chatbox
- Full courtroom debate visualization
- Critique (Media) accountability display

Run with: streamlit run app_v2.py
"""

# Delayed imports for performance
import streamlit as st
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="⚖️ Agentic Investment Courtroom",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
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
    .phase-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
    }
    .pro-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
    }
    .against-card {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
    }
    .jury-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
    }
    .judge-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
    }
    .critique-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 4px solid #9c27b0;
    }
    .decision-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; padding: 1rem 2rem; border-radius: 10px;
        font-size: 1.5rem; font-weight: bold; text-align: center;
    }
    .decision-sell {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        color: white; padding: 1rem 2rem; border-radius: 10px;
        font-size: 1.5rem; font-weight: bold; text-align: center;
    }
    .decision-hold {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white; padding: 1rem 2rem; border-radius: 10px;
        font-size: 1.5rem; font-weight: bold; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Session state
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
    """Save uploaded files to temp directory"""
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = Path(st.session_state.temp_dir) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(file_path))
    return saved_paths


@st.cache_resource
def get_system_engine(api_key: str):
    from agentic_rag_system import AgenticRAGSystem, Config
    os.environ["OPENROUTER_API_KEY"] = api_key
    config = Config(openrouter_api_key=api_key)
    return AgenticRAGSystem(config)


def render_decision_badge(decision: str):
    """Render colored badge for decision"""
    if decision in ["BUY", "INVEST"]:
        st.markdown(f'<div class="decision-buy">🟢 {decision}</div>', unsafe_allow_html=True)
    elif decision in ["SELL", "NOT_TO_INVEST"]:
        st.markdown(f'<div class="decision-sell">🔴 {decision}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="decision-hold">🟡 {decision}</div>', unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="Enter your API key...",
        help="Get your free API key at openrouter.ai"
    )
    
    if api_key:
        try:
            st.session_state.system = get_system_engine(api_key)
        except Exception as e:
            st.error(f"Initialization Error: {e}")
    
    st.caption("🤖 Using 7 FREE LLMs auto-assigned to agents")
    
    st.divider()
    
    st.markdown("### 📄 Step 1: Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
    
    if st.button("📥 Load Documents", type="primary", disabled=not uploaded_files):
        if not api_key:
            st.error("Please enter your OpenRouter API key")
        else:
            with st.spinner("Processing documents with semantic chunking..."):
                saved_paths = save_uploaded_files(uploaded_files)
                st.session_state.uploaded_files = saved_paths
                # System is already initialized via get_system_engine if API key is present
                chunk_count = st.session_state.system.load_documents(saved_paths)
                st.session_state.documents_loaded = True
                
                # Auto-populate inputs from metadata
                meta = st.session_state.system.get_extracted_metadata()
                if meta:
                    if meta.get("ticker"):
                        st.session_state.input_ticker = meta["ticker"]
                    if meta.get("company_name"):
                        st.session_state.input_company = meta["company_name"]
                    st.toast(f"✅ Detected: {meta.get('company_name')} ({meta.get('ticker')})")
                
                st.success(f"✅ Loaded {chunk_count} semantic chunks!")
    
    st.divider()
    st.markdown("### 📊 System")
    if st.session_state.documents_loaded:
        st.success("✅ Ready for analysis")
    else:
        st.warning("⚠️ Upload documents first")


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="main-header">⚖️ Agentic Investment Courtroom</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#666;font-size:1.1rem;">12-Agent System • Semantic RAG • Courtroom Debate • Media Critique</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "⚖️ Courtroom", "📰 Critique", "ℹ️ About"])

with tab1:
    st.markdown("## 🔍 Step 2: Run Analysis")
    
    # Always show input section
    # Initialize session state for inputs if not needed
    if "input_ticker" not in st.session_state:
        st.session_state.input_ticker = ""
    if "input_company" not in st.session_state:
        st.session_state.input_company = ""

    # Always show input section
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Stock Ticker", key="input_ticker", placeholder="e.g. AAPL")
    with col2:
        company_name = st.text_input("Company Name", key="input_company", placeholder="e.g. Apple Inc.")
    
    query = st.text_area(
        "Investment Query",
        height=100,
        placeholder="Enter your investment question here (e.g. Should I invest in this company? Analyze risks)..."
    )
    
    if not st.session_state.documents_loaded:
        st.info("👆 Enter your query above, but you must **Upload PDF Documents** in the sidebar to run the analysis.")
        
        st.markdown("### 🏛️ Process Flow")
        # Visual Flowchart using custom CSS
        st.markdown("""
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; text-align: center; margin-bottom: 20px;">
            <div style="flex: 1; min-width: 80px;">🧠<br>Query</div>
            <div style="align-self: center;">→</div>
            <div style="flex: 1; min-width: 80px;">🟢🔴<br>Debate</div>
            <div style="align-self: center;">→</div>
            <div style="flex: 1; min-width: 80px;">👥<br>Jury</div>
            <div style="align-self: center;">→</div>
            <div style="flex: 1; min-width: 80px;">👨‍⚖️<br>Judge</div>
            <div style="align-self: center;">→</div>
            <div style="flex: 1; min-width: 80px;">🌐<br>Media</div>
            <div style="align-self: center;">→</div>
            <div style="flex: 1; min-width: 80px; font-weight: bold; color: #4fd1c5;">👑<br>King</div>
        </div>
        """, unsafe_allow_html=True)
            
    # Run button
    run_disabled = not st.session_state.documents_loaded
    if st.button("⚖️ Run Courtroom Analysis", type="primary", use_container_width=True, disabled=run_disabled):
            if not ticker or not company_name:
                st.error("Please enter ticker and company name")
            else:
                try:
                    progress = st.progress(0)
                    with st.status("🚀 Orchestrating 12 Agents...", expanded=True) as status:
                        progress.progress(10)
                        
                        status.write("🧠 Query Understanding & Planning...")
                        # We need to access the system without re-initializing
                        
                        status.write("⚖️ Pro vs Against Debate (Searching Web)...")
                        
                        result = st.session_state.system.analyze(
                            query=query,
                            ticker=ticker,
                            company_name=company_name
                        )
                        
                        status.write("👥 Jury Deliberation...")
                        progress.progress(60)
                        status.write("👨‍⚖️ Judge Rendering Verdict...")
                        progress.progress(80)
                        status.write("🌐 Media Critique & King Agent Validation...")
                        progress.progress(100)
                        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                    
                    st.session_state.analysis_result = result
                    st.success("🎉 Courtroom analysis complete! See Courtroom & Critique tabs.")
                    
                    # Quick preview
                    final = result.get("final_verdict", {})
                    if final and not final.get("error"):
                        st.markdown("### 👑 Royal Verdict")
                        render_decision_badge(final.get("OFFICIAL_VERDICT", "HOLD"))
                        st.caption(f"Status: {final.get('VALIDATION_STATUS', 'PENDING')}")
                        
                        conf = final.get("FINAL_CONFIDENCE", 0)
                        if isinstance(conf, (int, float)):
                            st.metric("Confidence", f"{conf:.0%}")
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.exception(e)

with tab2:
    st.markdown("## ⚖️ Courtroom Transcript")
    
    if st.session_state.analysis_result is None:
        st.info("Run an analysis first to see the courtroom debate.")
    else:
        result = st.session_state.analysis_result
        
        # Market Snapshot
        if result.get("financial_metrics") and not result["financial_metrics"].get("error"):
            st.markdown("### 📈 Market Snapshot")
            m = result["financial_metrics"]
            cols = st.columns(4)
            cols[0].metric("Price", f"{m.get('Currency','')} {m.get('Current Price')}")
            cols[1].metric("Market Cap", m.get("Market Cap", "N/A"))
            cols[2].metric("P/E Ratio", m.get("Trailing PE", "N/A"))
            cols[3].metric("Beta", m.get("Beta", "N/A"))
            st.divider()
            
        # PRO CASE
        st.markdown("### 🟢 Pro Agent (Bullish Advocate)")
        pro_case = result.get("pro_case", {})
        
        with st.expander("📖 Opening Statement", expanded=True):
            opening = pro_case.get("opening", {})
            if opening:
                st.markdown(f'<div class="phase-card pro-card">', unsafe_allow_html=True)
                st.write(opening.get("opening_statement", opening.get("raw_response", "N/A")))
                if opening.get("key_bullish_points"):
                    st.markdown("**Key Points:**")
                    for p in opening["key_bullish_points"]:
                        st.markdown(f"✓ {p}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("⚔️ Rebuttal"):
            rebuttal = pro_case.get("rebuttal", {})
            if rebuttal and not rebuttal.get("error"):
                if rebuttal.get("rebuttal_points"):
                    for p in rebuttal["rebuttal_points"]:
                        st.markdown(f"• {p}")
        
        with st.expander("🎯 Closing Statement"):
            closing = pro_case.get("closing", {})
            if closing:
                st.write(closing.get("closing_statement", closing.get("raw_response", "N/A")))
        
        st.divider()
        
        # AGAINST CASE
        st.markdown("### 🔴 Against Agent (Bearish Advocate)")
        against_case = result.get("against_case", {})
        
        with st.expander("📖 Opening Statement", expanded=True):
            opening = against_case.get("opening", {})
            if opening:
                st.markdown(f'<div class="phase-card against-card">', unsafe_allow_html=True)
                st.write(opening.get("opening_statement", opening.get("raw_response", "N/A")))
                if opening.get("key_bearish_points"):
                    st.markdown("**Key Risks:**")
                    for p in opening["key_bearish_points"]:
                        st.markdown(f"✗ {p}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("⚔️ Rebuttal"):
            rebuttal = against_case.get("rebuttal", {})
            if rebuttal and not rebuttal.get("error"):
                if rebuttal.get("rebuttal_points"):
                    for p in rebuttal["rebuttal_points"]:
                        st.markdown(f"• {p}")
        
        with st.expander("🎯 Closing Statement"):
            closing = against_case.get("closing", {})
            if closing:
                st.write(closing.get("closing_statement", closing.get("raw_response", "N/A")))
        
        st.divider()
        
        # JURY
        st.markdown("### 👥 Jury Deliberations")
        jury = result.get("jury_verdicts", {})
        if jury:
            cols = st.columns(4)
            specialists = [("Fundamentals", "📊"), ("Risk", "⚠️"), ("ESG", "🌱"), ("Sentiment", "💭")]
            for col, (spec, emoji) in zip(cols, specialists):
                with col:
                    v = jury.get(spec.lower(), {})
                    score = v.get(f"{spec.lower()}_score", v.get("score", "N/A"))
                    verdict = v.get("verdict", "N/A")
                    st.metric(f"{emoji} {spec}", f"{verdict}", f"Score: {score}")
        
        st.divider()
        
        # JUDGE
        st.markdown("### 👨‍⚖️ Judge Verdict (Initial)")
        decision = result.get("decision", {})
        if decision and not decision.get("error"):
            st.markdown(f'<div class="phase-card judge-card">', unsafe_allow_html=True)
            render_decision_badge(decision.get("DECISION", "HOLD"))
            st.markdown(f"**Confidence:** {decision.get('CONFIDENCE', 'N/A')}")
            st.markdown("**Reasoning:**")
            st.write(decision.get("REASONING", "N/A"))
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.divider()
        
        # KING AGENT
        st.markdown("### 👑 King Agent (Royal Verdict)")
        final = result.get("final_verdict", {})
        if final and not final.get("error"):
            st.markdown(f'<div class="phase-card" style="border-left: 4px solid gold; background: #fffbe6;">', unsafe_allow_html=True)
            render_decision_badge(final.get("OFFICIAL_VERDICT", "HOLD"))
            st.markdown(f"**Validation Status:** {final.get('VALIDATION_STATUS', 'N/A')}")
            st.markdown(f"**Confidence:** {final.get('FINAL_CONFIDENCE', 'N/A')}")
            
            st.markdown("#### Executive Summary")
            st.write(final.get("EXECUTIVE_SUMMARY", "N/A"))
            
            if final.get("KEY_DRIVERS"):
                st.markdown("#### Key Drivers")
                for d in final["KEY_DRIVERS"]:
                    st.markdown(f"• {d}")
            
            if final.get("ACTIONABLE_ADVICE"):
                st.markdown("#### Actionable Advice")
                st.info(final["ACTIONABLE_ADVICE"])
                
            st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("## 📰 Media Critique (External Accountability)")
    
    if st.session_state.analysis_result is None:
        st.info("Run an analysis first to see the media critique.")
    else:
        result = st.session_state.analysis_result
        critique = result.get("critique_report", {})
        
        if critique and not critique.get("error"):
            st.markdown(f'<div class="phase-card critique-card">', unsafe_allow_html=True)
            
            st.markdown(f"### 📰 {critique.get('headline', 'Investment Analysis Report')}")
            
            cols = st.columns(3)
            cols[0].metric("Verdict Fairness", f"{critique.get('verdict_fairness', 0):.0%}")
            cols[1].metric("Confidence", f"{critique.get('confidence_in_verdict', 0):.0%}")
            cols[2].metric("Recommendation", critique.get("recommendation", "N/A"))
            
            st.markdown("### Critique Summary")
            st.write(critique.get("critique_summary", "N/A"))
            
            if critique.get("potential_biases_detected"):
                st.markdown("### ⚠️ Potential Biases Detected")
                for b in critique["potential_biases_detected"]:
                    st.markdown(f"• {b}")
            
            if critique.get("overlooked_factors"):
                st.markdown("### 🔍 Overlooked Factors")
                for f in critique["overlooked_factors"]:
                    st.markdown(f"• {f}")
            
            if critique.get("public_accountability_notes"):
                st.markdown("### 📋 Investor Notes")
                for n in critique["public_accountability_notes"]:
                    st.markdown(f"• {n}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Pass/Fail indicator
            if result.get("critique_passed"):
                st.success("✅ Verdict PASSED media scrutiny")
            else:
                st.warning("⚠️ Verdict flagged for additional review")
        else:
            st.warning("No critique report available")
        
        # Export
        st.divider()
        st.markdown("### 💾 Export Results")
        col1, col2 = st.columns(2)
        with col1:
            from agentic_rag_system import generate_report
            report = generate_report(result)
            st.download_button("📄 Download Report", report, 
                             f"courtroom_report_{result.get('ticker', 'analysis')}.txt", "text/plain")
        with col2:
            st.download_button("📊 Download JSON", json.dumps(result, indent=2, default=str),
                             f"analysis_{result.get('ticker', 'data')}.json", "application/json")

with tab4:
    st.markdown("## ℹ️ About the System")
    st.markdown("""
    ### 🏛️ 12-Agent Architecture
    
    | Layer | Agents | Role |
    |-------|--------|------|
    | **Orchestration** | Query Understanding, Planner | Parse input, create execution plan |
    | **Debate** | Pro Agent, Against Agent | Bullish/Bearish advocates |
    | **Jury** | Fundamentals, Risk, ESG, Sentiment | 4 specialist evaluators |
    | **Verdict** | Judge | Final BUY/SELL/HOLD decision |
    | **Accountability** | Critique (Media) | External review and validation |
    
    ### 📚 Semantic Agentic RAG
    
    - **NLTK sentence tokenization** (not fixed-size splits)
    - **Tiktoken** for token counting and budget management
    - **Multi-query retrieval** with relevance reranking
    - **Context-aware chunks** with document metadata
    
    ### ⚖️ Courtroom Flow
    
    1. **Opening Statements** - Pro and Against present initial cases
    2. **Cross-Examination** - Agents rebut each other's arguments
    3. **Closing Statements** - Final summaries
    4. **Jury Deliberation** - 4 specialists score the debate
    5. **Judge Verdict** - Final decision with reasoning
    6. **Media Critique** - External accountability check
    
    ---
    *Built with LangGraph, OpenRouter, and Streamlit*
    """)

# Footer
st.divider()
st.caption("Agentic Investment Courtroom v2.0 • 12 Agents • Semantic RAG • Courtroom Debate • Media Critique")
