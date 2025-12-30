"""
Investment Agent System - Multi-Agent Investment Analysis Framework
====================================================================
A production-ready investment analysis system using LangGraph with:
- 7 specialized agents (Pro, Against, Judge, 4 Jury specialists)
- RAG integration for PDF document analysis
- OpenRouter LLM integration for flexible model selection
- Designed for Indian market analysis (RIL, RGICL, etc.)

Author: Investment AI Team
Version: 1.0.0
"""

import os
import json
import logging
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# LangChain & LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """System configuration with defaults"""
    # OpenRouter Configuration
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # FREE Model Configuration (OpenRouter free tier models)
    # Available free models:
    # - nvidia/nemotron-3-nano-30b-a3b:free (Fast, good for general tasks)
    # - xiaomi/mimo-v2-flash:free (Fast inference)
    # - allenai/olmo-3-32b-think:free (Good reasoning)
    # - allenai/olmo-3.1-32b-think:free (Latest OLMo with thinking)
    # - nex-agi/deepseek-v3.1-nex-n1:free (DeepSeek based)
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "allenai/olmo-3.1-32b-think:free"))
    temperature: float = 0.3
    max_tokens: int = 4096
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_retrieval: int = 5
    
    # Document paths
    documents_dir: str = field(default_factory=lambda: os.getenv("DOCUMENTS_DIR", "./"))
    
    # Analysis settings
    confidence_threshold: float = 0.6
    require_unanimous_jury: bool = False


# Available free models for selection
FREE_MODELS = {
    "OLMo 3.1 32B Think ⭐ (Recommended)": "allenai/olmo-3.1-32b-think:free",
    "OLMo 3 32B Think": "allenai/olmo-3-32b-think:free",
    "DeepSeek V3.1 Nex": "nex-agi/deepseek-v3.1-nex-n1:free",
    "Nvidia Nemotron 30B": "nvidia/nemotron-3-nano-30b-a3b:free",
    "Xiaomi MiMo V2 Flash": "xiaomi/mimo-v2-flash:free",
}

# Agent-specific model mapping for diverse analysis
# Each agent uses a different model for specialized performance
AGENT_MODEL_MAPPING = {
    # Pro Agent: OLMo 3.1 for best reasoning on bullish arguments
    "pro_agent": "allenai/olmo-3.1-32b-think:free",
    
    # Against Agent: DeepSeek for rigorous risk identification
    "against_agent": "nex-agi/deepseek-v3.1-nex-n1:free",
    
    # Judge Agent: OLMo 3.1 for best overall decision-making
    "judge_agent": "allenai/olmo-3.1-32b-think:free",
    
    # Jury Fundamentals: DeepSeek for deep financial analysis
    "jury_fundamentals": "nex-agi/deepseek-v3.1-nex-n1:free",
    
    # Jury Risk: Nvidia Nemotron for fast risk assessment
    "jury_risk": "nvidia/nemotron-3-nano-30b-a3b:free",
    
    # Jury ESG: OLMo 3 for ESG reasoning
    "jury_esg": "allenai/olmo-3-32b-think:free",
    
    # Jury Sentiment: Xiaomi MiMo Flash for quick sentiment analysis
    "jury_sentiment": "xiaomi/mimo-v2-flash:free",
}


# ============================================================================
# DATA MODELS
# ============================================================================

class InvestmentDecision(Enum):
    """Investment decision types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class AgentOpinion:
    """Structured opinion from an agent"""
    agent_name: str
    stance: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    key_points: List[str]
    evidence: List[str]
    concerns: List[str]
    recommendation: str


@dataclass
class JuryVerdict:
    """Aggregated verdict from jury specialists"""
    fundamentals_score: float
    risk_score: float
    esg_score: float
    sentiment_score: float
    overall_assessment: str
    key_concerns: List[str]


@dataclass
class FinalDecision:
    """Final investment decision with full reasoning"""
    ticker: str
    company_name: str
    decision: InvestmentDecision
    confidence: float
    pro_case: AgentOpinion
    against_case: AgentOpinion
    jury_verdict: JuryVerdict
    judge_reasoning: str
    key_considerations: List[str]
    dissenting_views: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GraphState(TypedDict):
    """State shared across all agents in the graph"""
    # Input
    query: str
    ticker: str
    company_name: str
    
    # RAG context
    retrieved_documents: List[str]
    document_sources: List[str]
    
    # Agent outputs
    pro_opinion: Optional[Dict]
    against_opinion: Optional[Dict]
    jury_verdicts: Dict[str, Dict]
    
    # Final output
    final_decision: Optional[Dict]
    
    # Metadata
    iteration_count: int
    errors: List[str]


# ============================================================================
# LLM SETUP (OpenRouter)
# ============================================================================

def create_llm(config: Config, model_override: str = None) -> ChatOpenAI:
    """Create LLM client using OpenRouter"""
    return ChatOpenAI(
        model=model_override or config.default_model,
        openai_api_key=config.openrouter_api_key,
        openai_api_base=config.openrouter_base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        default_headers={
            "HTTP-Referer": "https://investment-agent.local",
            "X-Title": "Investment Agent System"
        }
    )


# ============================================================================
# RAG SYSTEM
# ============================================================================

class RAGSystem:
    """Retrieval-Augmented Generation for financial documents"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vectorstore = None
        self.documents_loaded = False
    
    def load_documents(self, pdf_paths: List[str] = None) -> int:
        """Load and index PDF documents"""
        if pdf_paths is None:
            # Auto-discover PDFs in documents directory
            doc_dir = Path(self.config.documents_dir)
            pdf_paths = list(doc_dir.glob("*.pdf"))
        
        all_chunks = []
        for pdf_path in pdf_paths:
            try:
                logger.info(f"Loading: {pdf_path}")
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                chunks = self.text_splitter.split_documents(pages)
                
                # Add source metadata
                for chunk in chunks:
                    chunk.metadata["source_file"] = str(pdf_path)
                
                all_chunks.extend(chunks)
                logger.info(f"  → {len(chunks)} chunks from {pdf_path.name if hasattr(pdf_path, 'name') else pdf_path}")
            except Exception as e:
                logger.error(f"Error loading {pdf_path}: {e}")
        
        if all_chunks:
            self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            self.documents_loaded = True
            logger.info(f"✅ Indexed {len(all_chunks)} total chunks")
        
        return len(all_chunks)
    
    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """Retrieve relevant document chunks"""
        if not self.documents_loaded or self.vectorstore is None:
            return []
        
        k = k or self.config.top_k_retrieval
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", 0),
                "relevance_score": float(score)
            }
            for doc, score in results
        ]


# ============================================================================
# AGENT PROMPTS
# ============================================================================

PRO_AGENT_PROMPT = """You are a BULLISH investment analyst advocating for investing in {company_name} ({ticker}).

Your role is to build the STRONGEST POSSIBLE CASE for buying this stock. Focus on:
1. Growth opportunities and catalysts
2. Competitive advantages and moats
3. Strong financials and profitability trends
4. Positive market position and industry tailwinds
5. Management quality and strategic vision

CONTEXT FROM DOCUMENTS:
{context}

USER QUERY: {query}

Provide your bullish analysis with:
1. KEY BULLISH POINTS (3-5 strongest arguments)
2. SUPPORTING EVIDENCE (cite specific data from documents)
3. GROWTH CATALYSTS (upcoming positive events)
4. FAIR VALUE ESTIMATE (if data available)
5. BULL CASE CONFIDENCE (0.0 to 1.0)

Be thorough but focused. Acknowledge risks briefly but emphasize opportunities.
Format as structured JSON."""

AGAINST_AGENT_PROMPT = """You are a BEARISH investment analyst arguing AGAINST investing in {company_name} ({ticker}).

Your role is to identify ALL RISKS and reasons NOT to invest. Focus on:
1. Financial weaknesses and red flags
2. Competitive threats and market risks
3. Valuation concerns (overvalued?)
4. Management issues or governance concerns
5. Industry headwinds and macro risks

CONTEXT FROM DOCUMENTS:
{context}

USER QUERY: {query}

Provide your bearish analysis with:
1. KEY BEARISH POINTS (3-5 strongest arguments against)
2. RISK EVIDENCE (cite specific concerns from documents)
3. DOWNSIDE SCENARIOS (what could go wrong)
4. RISK-ADJUSTED CONCERNS
5. BEAR CASE CONFIDENCE (0.0 to 1.0)

Be thorough in identifying risks. This protects investors from losses.
Format as structured JSON."""

JURY_FUNDAMENTALS_PROMPT = """You are a FUNDAMENTALS ANALYST on the investment jury.

Analyze {company_name} ({ticker}) focusing ONLY on financial fundamentals:
- Revenue growth and trends
- Profitability margins (gross, operating, net)
- Return metrics (ROE, ROA, ROIC)
- Balance sheet strength (debt levels, liquidity)
- Cash flow quality
- Earnings quality and sustainability

CONTEXT FROM DOCUMENTS:
{context}

Provide a fundamentals assessment:
1. FUNDAMENTALS SCORE (0.0 to 1.0)
2. KEY METRICS (with specific numbers)
3. STRENGTHS (2-3 points)
4. WEAKNESSES (2-3 points)
5. TREND DIRECTION (improving/stable/declining)

Format as structured JSON."""

JURY_RISK_PROMPT = """You are a RISK ANALYST on the investment jury.

Analyze {company_name} ({ticker}) focusing ONLY on risks:
- Market and competitive risks
- Regulatory and legal risks
- Operational risks
- Financial risks (leverage, liquidity)
- Concentration risks (customer, supplier, geographic)
- Black swan scenarios

CONTEXT FROM DOCUMENTS:
{context}

Provide a risk assessment:
1. RISK SCORE (0.0 to 1.0, higher = more risky)
2. TOP RISKS (ranked by severity)
3. RISK MITIGANTS (how company manages risks)
4. WORST CASE SCENARIO
5. RISK-ADJUSTED RECOMMENDATION

Format as structured JSON."""

JURY_ESG_PROMPT = """You are an ESG ANALYST on the investment jury.

Analyze {company_name} ({ticker}) focusing on Environmental, Social, Governance:
- Environmental practices and commitments
- Social responsibility (employees, community)
- Corporate governance quality
- Board composition and independence
- Executive compensation alignment
- Sustainability initiatives

CONTEXT FROM DOCUMENTS:
{context}

Provide an ESG assessment:
1. ESG SCORE (0.0 to 1.0)
2. ENVIRONMENTAL SCORE
3. SOCIAL SCORE
4. GOVERNANCE SCORE
5. KEY ESG CONCERNS
6. ESG OPPORTUNITIES

Format as structured JSON."""

JURY_SENTIMENT_PROMPT = """You are a MARKET SENTIMENT ANALYST on the investment jury.

Analyze {company_name} ({ticker}) focusing on market psychology:
- Management tone and confidence
- Strategic narrative and vision
- Communication quality
- Industry positioning narrative
- Investor relations quality
- Forward guidance clarity

CONTEXT FROM DOCUMENTS:
{context}

Provide a sentiment assessment:
1. SENTIMENT SCORE (0.0 to 1.0)
2. MANAGEMENT CONFIDENCE LEVEL
3. NARRATIVE QUALITY
4. KEY MESSAGE THEMES
5. CREDIBILITY ASSESSMENT

Format as structured JSON."""

JUDGE_PROMPT = """You are the CHIEF INVESTMENT JUDGE making the final decision.

You have received arguments from:
1. PRO AGENT (bullish case):
{pro_opinion}

2. AGAINST AGENT (bearish case):
{against_opinion}

3. JURY SPECIALISTS:
{jury_verdicts}

ORIGINAL QUERY: {query}
COMPANY: {company_name} ({ticker})

Your job is to WEIGH ALL EVIDENCE and render a FINAL VERDICT.

Consider:
- Strength of bullish vs bearish arguments
- Quality of evidence presented
- Jury specialist scores
- Risk-reward balance
- Overall conviction level

RENDER YOUR DECISION:
1. DECISION: BUY, SELL, or HOLD
2. CONFIDENCE: 0.0 to 1.0
3. REASONING: Detailed explanation (3-5 paragraphs)
4. KEY CONSIDERATIONS: Top 5 factors that drove decision
5. DISSENTING VIEWS: Important counter-arguments to note
6. RISK WARNINGS: What could invalidate this decision
7. TIME HORIZON: Short/Medium/Long term outlook

Format as structured JSON with all fields."""


# ============================================================================
# AGENT NODES
# ============================================================================

class InvestmentAgentSystem:
    """Multi-agent investment analysis system using LangGraph
    
    Each agent uses a different free LLM model for diverse analysis:
    - Pro Agent: OLMo 3.1 32B Think (best reasoning)
    - Against Agent: DeepSeek V3.1 Nex (rigorous analysis)
    - Judge: OLMo 3.1 32B Think (best decision-making)
    - Jury Fundamentals: DeepSeek V3.1 Nex (financial depth)
    - Jury Risk: Nvidia Nemotron 30B (fast risk eval)
    - Jury ESG: OLMo 3 32B Think (ESG reasoning)
    - Jury Sentiment: Xiaomi MiMo V2 Flash (quick sentiment)
    """
    
    def __init__(self, config: Config = None, use_multi_model: bool = True):
        self.config = config or Config()
        self.use_multi_model = use_multi_model
        
        # Create model instances for each agent
        if use_multi_model:
            self.agent_llms = self._create_agent_llms()
            logger.info("✅ Multi-model mode: Each agent uses a different LLM")
            for agent, model in AGENT_MODEL_MAPPING.items():
                logger.info(f"   {agent}: {model}")
        else:
            # Single model for all agents (fallback)
            self.llm = create_llm(self.config)
            self.agent_llms = None
            logger.info(f"✅ Single-model mode: All agents use {self.config.default_model}")
        
        self.rag = RAGSystem(self.config)
        self.graph = self._build_graph()
    
    def _create_agent_llms(self) -> Dict[str, ChatOpenAI]:
        """Create separate LLM instances for each agent"""
        agent_llms = {}
        for agent_name, model_id in AGENT_MODEL_MAPPING.items():
            agent_llms[agent_name] = create_llm(self.config, model_override=model_id)
            logger.debug(f"Created LLM for {agent_name}: {model_id}")
        return agent_llms
    
    def _get_agent_llm(self, agent_name: str) -> ChatOpenAI:
        """Get the LLM for a specific agent"""
        if self.use_multi_model and self.agent_llms:
            return self.agent_llms.get(agent_name, self.agent_llms.get("judge_agent"))
        return self.llm
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("pro_agent", self._pro_agent)
        workflow.add_node("against_agent", self._against_agent)
        workflow.add_node("jury_fundamentals", self._jury_fundamentals)
        workflow.add_node("jury_risk", self._jury_risk)
        workflow.add_node("jury_esg", self._jury_esg)
        workflow.add_node("jury_sentiment", self._jury_sentiment)
        workflow.add_node("judge_agent", self._judge_agent)
        
        # Define edges
        workflow.set_entry_point("retrieve_documents")
        
        # After retrieval, run Pro and Against in parallel concept 
        # (LangGraph handles this via separate paths that converge)
        workflow.add_edge("retrieve_documents", "pro_agent")
        workflow.add_edge("retrieve_documents", "against_agent")
        
        # Run jury after debates
        workflow.add_edge("pro_agent", "jury_fundamentals")
        workflow.add_edge("against_agent", "jury_fundamentals")
        workflow.add_edge("jury_fundamentals", "jury_risk")
        workflow.add_edge("jury_risk", "jury_esg")
        workflow.add_edge("jury_esg", "jury_sentiment")
        
        # Judge makes final decision
        workflow.add_edge("jury_sentiment", "judge_agent")
        workflow.add_edge("judge_agent", END)
        
        return workflow.compile()
    
    def _format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents for prompts"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = Path(doc["source"]).name if doc.get("source") else "Unknown"
            context_parts.append(
                f"[Document {i} - {source}, Page {doc.get('page', '?')}]\n{doc['content']}\n"
            )
        return "\n".join(context_parts)
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM JSON response with fallback"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            # Return raw response wrapped in dict
            return {"raw_response": response, "parse_error": True}
    
    # --- Node implementations ---
    
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents using RAG"""
        query = f"{state['company_name']} {state['ticker']} {state['query']}"
        
        try:
            results = self.rag.retrieve(query)
            state["retrieved_documents"] = [r["content"] for r in results]
            state["document_sources"] = [r["source"] for r in results]
            logger.info(f"Retrieved {len(results)} relevant document chunks")
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            state["retrieved_documents"] = []
            state["document_sources"] = []
            state["errors"].append(f"RAG error: {str(e)}")
        
        return state
    
    def _pro_agent(self, state: GraphState) -> GraphState:
        """Pro agent builds bullish case using OLMo 3.1 32B Think"""
        context = self._format_context([
            {"content": doc, "source": src}
            for doc, src in zip(state["retrieved_documents"], state["document_sources"])
        ])
        
        prompt = PRO_AGENT_PROMPT.format(
            company_name=state["company_name"],
            ticker=state["ticker"],
            context=context,
            query=state["query"]
        )
        
        try:
            llm = self._get_agent_llm("pro_agent")
            response = llm.invoke([HumanMessage(content=prompt)])
            state["pro_opinion"] = self._parse_llm_response(response.content)
            logger.info(f"✅ Pro Agent completed analysis (Model: {AGENT_MODEL_MAPPING.get('pro_agent', 'default')})")
        except Exception as e:
            logger.error(f"Pro Agent error: {e}")
            state["pro_opinion"] = {"error": str(e)}
            state["errors"].append(f"Pro Agent error: {str(e)}")
        
        return state
    
    def _against_agent(self, state: GraphState) -> GraphState:
        """Against agent builds bearish case using DeepSeek V3.1 Nex"""
        context = self._format_context([
            {"content": doc, "source": src}
            for doc, src in zip(state["retrieved_documents"], state["document_sources"])
        ])
        
        prompt = AGAINST_AGENT_PROMPT.format(
            company_name=state["company_name"],
            ticker=state["ticker"],
            context=context,
            query=state["query"]
        )
        
        try:
            llm = self._get_agent_llm("against_agent")
            response = llm.invoke([HumanMessage(content=prompt)])
            state["against_opinion"] = self._parse_llm_response(response.content)
            logger.info(f"✅ Against Agent completed analysis (Model: {AGENT_MODEL_MAPPING.get('against_agent', 'default')})")
        except Exception as e:
            logger.error(f"Against Agent error: {e}")
            state["against_opinion"] = {"error": str(e)}
            state["errors"].append(f"Against Agent error: {str(e)}")
        
        return state
    
    def _jury_fundamentals(self, state: GraphState) -> GraphState:
        """Jury fundamentals specialist"""
        return self._run_jury_agent(state, "fundamentals", JURY_FUNDAMENTALS_PROMPT)
    
    def _jury_risk(self, state: GraphState) -> GraphState:
        """Jury risk specialist"""
        return self._run_jury_agent(state, "risk", JURY_RISK_PROMPT)
    
    def _jury_esg(self, state: GraphState) -> GraphState:
        """Jury ESG specialist"""
        return self._run_jury_agent(state, "esg", JURY_ESG_PROMPT)
    
    def _jury_sentiment(self, state: GraphState) -> GraphState:
        """Jury sentiment specialist"""
        return self._run_jury_agent(state, "sentiment", JURY_SENTIMENT_PROMPT)
    
    def _run_jury_agent(self, state: GraphState, specialty: str, prompt_template: str) -> GraphState:
        """Generic jury agent runner - uses specialty-specific LLM"""
        context = self._format_context([
            {"content": doc, "source": src}
            for doc, src in zip(state["retrieved_documents"], state["document_sources"])
        ])
        
        prompt = prompt_template.format(
            company_name=state["company_name"],
            ticker=state["ticker"],
            context=context
        )
        
        try:
            # Get the specific LLM for this jury specialist
            agent_key = f"jury_{specialty}"
            llm = self._get_agent_llm(agent_key)
            response = llm.invoke([HumanMessage(content=prompt)])
            if "jury_verdicts" not in state or state["jury_verdicts"] is None:
                state["jury_verdicts"] = {}
            state["jury_verdicts"][specialty] = self._parse_llm_response(response.content)
            logger.info(f"✅ Jury {specialty.upper()} completed analysis (Model: {AGENT_MODEL_MAPPING.get(agent_key, 'default')})")
        except Exception as e:
            logger.error(f"Jury {specialty} error: {e}")
            if "jury_verdicts" not in state or state["jury_verdicts"] is None:
                state["jury_verdicts"] = {}
            state["jury_verdicts"][specialty] = {"error": str(e)}
            state["errors"].append(f"Jury {specialty} error: {str(e)}")
        
        return state
    
    def _judge_agent(self, state: GraphState) -> GraphState:
        """Judge synthesizes all opinions and renders verdict using OLMo 3.1 32B Think"""
        prompt = JUDGE_PROMPT.format(
            company_name=state["company_name"],
            ticker=state["ticker"],
            query=state["query"],
            pro_opinion=json.dumps(state["pro_opinion"], indent=2),
            against_opinion=json.dumps(state["against_opinion"], indent=2),
            jury_verdicts=json.dumps(state["jury_verdicts"], indent=2)
        )
        
        try:
            llm = self._get_agent_llm("judge_agent")
            response = llm.invoke([HumanMessage(content=prompt)])
            state["final_decision"] = self._parse_llm_response(response.content)
            logger.info(f"✅ Judge rendered final decision (Model: {AGENT_MODEL_MAPPING.get('judge_agent', 'default')})")
        except Exception as e:
            logger.error(f"Judge error: {e}")
            state["final_decision"] = {"error": str(e)}
            state["errors"].append(f"Judge error: {str(e)}")
        
        return state
    
    # --- Public API ---
    
    def load_documents(self, pdf_paths: List[str] = None) -> int:
        """Load PDF documents for RAG"""
        return self.rag.load_documents(pdf_paths)
    
    def analyze(self, query: str, ticker: str, company_name: str) -> Dict:
        """Run full investment analysis"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting analysis: {company_name} ({ticker})")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*60}\n")
        
        initial_state: GraphState = {
            "query": query,
            "ticker": ticker,
            "company_name": company_name,
            "retrieved_documents": [],
            "document_sources": [],
            "pro_opinion": None,
            "against_opinion": None,
            "jury_verdicts": {},
            "final_decision": None,
            "iteration_count": 0,
            "errors": []
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "query": query,
            "decision": final_state.get("final_decision"),
            "pro_case": final_state.get("pro_opinion"),
            "against_case": final_state.get("against_opinion"),
            "jury_verdicts": final_state.get("jury_verdicts"),
            "documents_analyzed": len(final_state.get("retrieved_documents", [])),
            "errors": final_state.get("errors", []),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# REPORT GENERATOR
# ============================================================================

def generate_report(result: Dict) -> str:
    """Generate a formatted investment report"""
    report = []
    report.append("=" * 70)
    report.append(f"INVESTMENT ANALYSIS REPORT")
    report.append(f"Company: {result['company_name']} ({result['ticker']})")
    report.append(f"Generated: {result['timestamp']}")
    report.append("=" * 70)
    report.append("")
    
    # Decision Summary
    decision = result.get("decision", {})
    if decision and not decision.get("error"):
        report.append("📊 FINAL DECISION")
        report.append("-" * 40)
        report.append(f"Decision: {decision.get('DECISION', 'N/A')}")
        report.append(f"Confidence: {decision.get('CONFIDENCE', 'N/A')}")
        report.append("")
        
        if decision.get('REASONING'):
            report.append("📝 REASONING")
            report.append(decision['REASONING'])
            report.append("")
        
        if decision.get('KEY_CONSIDERATIONS'):
            report.append("🔑 KEY CONSIDERATIONS")
            for i, point in enumerate(decision['KEY_CONSIDERATIONS'], 1):
                report.append(f"  {i}. {point}")
            report.append("")
        
        if decision.get('DISSENTING_VIEWS'):
            report.append("⚠️ DISSENTING VIEWS")
            for view in decision['DISSENTING_VIEWS']:
                report.append(f"  • {view}")
            report.append("")
    
    # Pro Case Summary
    pro = result.get("pro_case", {})
    if pro and not pro.get("error"):
        report.append("🟢 BULLISH CASE (Pro Agent)")
        report.append("-" * 40)
        if pro.get("KEY_BULLISH_POINTS"):
            for point in pro["KEY_BULLISH_POINTS"]:
                report.append(f"  ✓ {point}")
        report.append("")
    
    # Against Case Summary
    against = result.get("against_case", {})
    if against and not against.get("error"):
        report.append("🔴 BEARISH CASE (Against Agent)")
        report.append("-" * 40)
        if against.get("KEY_BEARISH_POINTS"):
            for point in against["KEY_BEARISH_POINTS"]:
                report.append(f"  ✗ {point}")
        report.append("")
    
    # Jury Scores
    jury = result.get("jury_verdicts", {})
    if jury:
        report.append("⚖️ JURY SPECIALIST SCORES")
        report.append("-" * 40)
        for specialty, verdict in jury.items():
            if not verdict.get("error"):
                score_key = f"{specialty.upper()}_SCORE"
                score = verdict.get(score_key, verdict.get("SCORE", "N/A"))
                report.append(f"  {specialty.capitalize()}: {score}")
        report.append("")
    
    # Errors
    if result.get("errors"):
        report.append("❌ ERRORS ENCOUNTERED")
        report.append("-" * 40)
        for error in result["errors"]:
            report.append(f"  • {error}")
        report.append("")
    
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for the investment analysis system"""
    print("\n" + "=" * 70)
    print("🚀 INVESTMENT AGENT SYSTEM - Multi-Agent Analysis Framework")
    print("=" * 70 + "\n")
    
    # Initialize configuration
    config = Config(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        documents_dir=os.getenv("DOCUMENTS_DIR", "./")
    )
    
    if not config.openrouter_api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in environment!")
        print("   Please create a .env file with: OPENROUTER_API_KEY=your-key-here")
        return
    
    # Initialize system
    print("📦 Initializing Investment Agent System...")
    system = InvestmentAgentSystem(config)
    
    # Load documents
    print("\n📄 Loading financial documents...")
    doc_count = system.load_documents()
    print(f"   Loaded {doc_count} document chunks")
    
    if doc_count == 0:
        print("\n⚠️ WARNING: No documents loaded!")
        print("   Place PDF files in the current directory or set DOCUMENTS_DIR")
    
    # Example analysis
    print("\n" + "=" * 70)
    print("📊 Running Investment Analysis...")
    print("=" * 70 + "\n")
    
    # Analyze Reliance Industries (based on available PDFs)
    result = system.analyze(
        query="Should I invest in this company? Analyze financial performance, growth prospects, and risks.",
        ticker="RIL",
        company_name="Reliance Industries Limited"
    )
    
    # Generate and print report
    report = generate_report(result)
    print(report)
    
    # Save report
    report_path = Path(config.documents_dir) / f"investment_report_{result['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n💾 Report saved to: {report_path}")
    
    # Save raw JSON result
    json_path = Path(config.documents_dir) / f"investment_analysis_{result['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"💾 JSON data saved to: {json_path}")


if __name__ == "__main__":
    main()
