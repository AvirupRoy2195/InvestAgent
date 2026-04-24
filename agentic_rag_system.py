"""
Agentic RAG Investment System - Enhanced Multi-Agent Framework
===============================================================
A production-ready 12-agent investment analysis system with:
- Query Understanding, Planner, Task Scheduler, Orchestrator
- Semantic RAG with NLP pipeline (NLTK, tiktoken)
- Courtroom Debate: Pro/Against opening → cross-exam → closing
- Jury (4 specialists) observes and deliberates
- Judge renders verdict
- Critique Agent (Media) provides external accountability

Author: Investment AI Team
Version: 2.0.0
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# NLP imports
import nltk
import tiktoken

# LangChain & LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# Web search disabled (ddgs compatibility issue)
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import yfinance as yf  # Re-enabled for financials

# Avoid network work at import time. If NLTK tokenizers are unavailable,
# SemanticChunker.sentence_tokenize falls back to a regex splitter.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    pass

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Enhanced system configuration"""
    # OpenRouter
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "google/gemma-4-26b-a4b-it:free"))
    temperature: float = 0.3
    max_tokens: int = 4096
    
    # Semantic RAG Configuration
    semantic_chunk_size: int = 1500
    min_chunk_size: int = 200
    chunk_overlap: int = 200
    use_sentence_splitting: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_retrieval: int = 8
    max_retrieval_tokens: int = 6000
    
    # Documents
    documents_dir: str = field(default_factory=lambda: os.getenv("DOCUMENTS_DIR", "./"))
    
    # Courtroom settings
    enable_cross_examination: bool = True
    debate_rounds: int = 1
    
    # Critique settings
    critique_confidence_threshold: float = 0.7
    max_critique_loops: int = 2


# Model mapping for all agents (now with sub-agents for Judge and King)
# Uses the user-supplied OpenRouter free text-generation models.
AGENT_MODEL_MAPPING = {
    # Orchestration Layer
    "query_understanding": "google/gemma-4-26b-a4b-it:free",
    "planner": "minimax/minimax-m2.5:free",
    
    # Courtroom Debate Agents - using different models for pro/against
    "pro_agent": "z-ai/glm-4.5-air:free",
    "against_agent": "minimax/minimax-m2.5:free",
    
    # Judge Agent - Dual LLM for harder reasoning
    "judge_agent": "google/gemma-4-26b-a4b-it:free",
    "judge_sub_agent": "qwen/qwen3-coder:free",
    
    # Jury Specialists - all different models
    "jury_fundamentals": "google/gemma-4-26b-a4b-it:free",
    "jury_risk": "minimax/minimax-m2.5:free",
    "jury_esg": "z-ai/glm-4.5-air:free",
    "jury_sentiment": "qwen/qwen3-coder:free",
    
    # Media/Critique
    "critique_agent": "z-ai/glm-4.5-air:free",
    
    # King Agent - Dual LLM for royal deliberation
    "king_agent": "minimax/minimax-m2.5:free",
    "king_sub_agent": "google/gemma-4-26b-a4b-it:free",
}

# Restrict the selectable registry to the requested model set. The NVIDIA
# entry is kept available, but not assigned to ChatOpenAI agents because it is
# named as an embedding/VL model.
FREE_MODELS = {
    "Google Gemma 4 26B A4B IT": "google/gemma-4-26b-a4b-it:free",
    "MiniMax M2.5": "minimax/minimax-m2.5:free",
    "GLM 4.5 Air": "z-ai/glm-4.5-air:free",
    "Qwen3 Coder": "qwen/qwen3-coder:free",
    "NVIDIA Llama Nemotron Embed VL 1B V2": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
}


# ============================================================================
# SEMANTIC RAG SYSTEM
# ============================================================================

class SemanticChunker:
    """Context-aware semantic chunking with NLP pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        """Fast approximate token count (4 chars ≈ 1 token for English)"""
        # Using approximation for speed - ~70% faster than exact tiktoken
        return len(text) // 4 + 1
    
    def count_tokens_exact(self, text: str) -> int:
        """Exact token count using tiktoken (slower, use sparingly)"""
        return len(self.tokenizer.encode(text))
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            # Fallback to simple split
            return re.split(r'(?<=[.!?])\s+', text)
    
    def create_semantic_chunks(self, text: str, metadata: Dict = None) -> List[Document]:
        """Create semantically coherent chunks preserving sentence boundaries"""
        sentences = self.sentence_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds max, split it
            if sentence_tokens > self.config.semantic_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                # Split long sentence
                words = sentence.split()
                temp = []
                temp_tokens = 0
                for word in words:
                    word_tokens = self.count_tokens(word)
                    if temp_tokens + word_tokens > self.config.semantic_chunk_size:
                        chunks.append(" ".join(temp))
                        temp = [word]
                        temp_tokens = word_tokens
                    else:
                        temp.append(word)
                        temp_tokens += word_tokens
                if temp:
                    current_chunk = temp
                    current_tokens = temp_tokens
            elif current_tokens + sentence_tokens > self.config.semantic_chunk_size:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Filter out chunks that are too small
        chunks = [c for c in chunks if self.count_tokens(c) >= self.config.min_chunk_size]
        
        # Convert to Documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata["chunk_index"] = i
            doc_metadata["token_count"] = self.count_tokens(chunk)
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents


class AgenticRAGSystem:
    """Enhanced RAG with semantic chunking and intelligent retrieval"""
    
    def __init__(self, config: Config):
        self.config = config
        self.chunker = SemanticChunker(config)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.extracted_metadata = {}
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        
        # PERSISTENCE: Try loading existing index
        try:
            if os.path.exists("faiss_index"):
                self.vectorstore = FAISS.load_local(
                    "faiss_index", 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self.documents_loaded = True
                self._restore_metadata_from_vectorstore()
                logger.info("✅ Loaded existing FAISS index from disk")
            else:
                self.vectorstore = None
                self.documents_loaded = False
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            self.vectorstore = None
            self.documents_loaded = False
        
    def _restore_metadata_from_vectorstore(self):
        """Restore best-effort company metadata from persisted FAISS documents."""
        if self.extracted_metadata:
            return

        try:
            docstore = getattr(self.vectorstore, "docstore", None)
            stored_docs = getattr(docstore, "_dict", {})
            sample_docs = list(stored_docs.values())[:5]
            text_parts = []
            for doc in sample_docs:
                metadata = getattr(doc, "metadata", {}) or {}
                source_name = metadata.get("file_name") or Path(metadata.get("source_file", "")).name
                if source_name:
                    text_parts.append(source_name)
                page_content = getattr(doc, "page_content", "")
                if page_content:
                    text_parts.append(page_content[:2000])

            combined_text = "\n".join(text_parts)
            if combined_text.strip():
                self._extract_initial_metadata(combined_text)
        except Exception as e:
            logger.warning(f"Metadata restore failed: {e}")

    def load_documents(self, pdf_paths: List[str] = None) -> int:
        """Load and index PDF documents with semantic chunking"""
        if pdf_paths is None:
            doc_dir = Path(self.config.documents_dir)
            pdf_paths = list(doc_dir.glob("*.pdf"))
        
        all_chunks = []
        for pdf_path in pdf_paths:
            try:
                logger.info(f"📄 Loading: {pdf_path}")
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                
                for page in pages:
                    metadata = {
                        "source_file": str(pdf_path),
                        "page": page.metadata.get("page", 0),
                        "file_name": Path(pdf_path).name
                    }
                    chunks = self.chunker.create_semantic_chunks(page.page_content, metadata)
                    all_chunks.extend(chunks)
                
                # Extract metadata from first page of first doc
                if not self.extracted_metadata and pages:
                    try:
                        self._extract_initial_metadata(pages[0].page_content)
                    except Exception as e:
                        logger.warning(f"Metadata extraction failed: {e}")
                
                logger.info(f"  → {len([c for c in all_chunks if c.metadata.get('source_file') == str(pdf_path)])} semantic chunks")
            except Exception as e:
                logger.error(f"Error loading {pdf_path}: {e}")
        
        if all_chunks:
            self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            self.vectorstore.save_local("faiss_index")  # PERSIST TO DISK
            self.documents_loaded = True
            logger.info(f"✅ Indexed {len(all_chunks)} semantic chunks and saved to disk")
        
        return len(all_chunks)

    def _extract_initial_metadata(self, text: str):
        """Extract company info from text using fast regex patterns"""
        metadata = {"ticker": None, "company_name": None}
        
        # Fast regex-based extraction (no LLM calls)
        indian_companies = {
            "RELIANCE": ("RIL", "Reliance Industries Limited"),
            "RIL": ("RIL", "Reliance Industries Limited"),
            "RGICL": ("RIL", "Reliance General Insurance Company Limited"),
            "TATA CONSULTANCY": ("TCS", "Tata Consultancy Services Limited"),
            "INFOSYS": ("INFY", "Infosys Limited"),
            "HDFC BANK": ("HDFCBANK", "HDFC Bank Limited"),
            "ICICI BANK": ("ICICIBANK", "ICICI Bank Limited"),
            "WIPRO": ("WIPRO", "Wipro Limited"),
            "TCS": ("TCS", "Tata Consultancy Services Limited"),
            "BHARTI AIRTEL": ("BHARTIARTL", "Bharti Airtel Limited"),
            "ASIAN PAINTS": ("ASIANPAINT", "Asian Paints Limited"),
            "MARUTI": ("MARUTI", "Maruti Suzuki India Limited"),
        }
        
        # Check for known company names
        for key, (ticker, name) in indian_companies.items():
            if key.lower() in text.lower():
                metadata["ticker"] = ticker
                metadata["company_name"] = name
                break
        
        # If not found, try regex patterns
        if not metadata["company_name"]:
            match = re.search(r'([A-Z][A-Za-z\s]+(?:Limited|Ltd|Inc|Corp))', text[:5000])
            if match:
                metadata["company_name"] = match.group(1).strip()
        
        self.extracted_metadata = metadata
        if metadata["ticker"] or metadata["company_name"]:
            logger.info(f"📄 Extracted Metadata: {self.extracted_metadata}")
    
    def get_extracted_metadata(self) -> Dict:
        return getattr(self, 'extracted_metadata', {})
    
    def multi_query_retrieve(self, query: str, company: str, ticker: str) -> List[Dict]:
        """Multi-query retrieval with different perspectives"""
        if not self.documents_loaded or self.vectorstore is None:
            return []
        
        # Generate multiple query variants
        queries = [
            query,
            f"{company} {ticker} financial performance",
            f"{company} revenue profit growth",
            f"{company} risks challenges concerns",
            f"{ticker} investment analysis"
        ]
        
        all_results = []
        seen_content = set()
        
        for q in queries:
            results = self.vectorstore.similarity_search_with_score(q, k=self.config.top_k_retrieval)
            for doc, score in results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source_file", "unknown"),
                        "page": doc.metadata.get("page", 0),
                        "relevance_score": float(score),
                        "token_count": doc.metadata.get("token_count", 0)
                    })
        
        # Sort by relevance and fit within token budget
        all_results.sort(key=lambda x: x["relevance_score"])
        
        final_results = []
        total_tokens = 0
        for r in all_results:
            if total_tokens + r["token_count"] <= self.config.max_retrieval_tokens:
                final_results.append(r)
                total_tokens += r["token_count"]
        
        logger.info(f"📚 Retrieved {len(final_results)} chunks ({total_tokens} tokens)")
        return final_results


# ============================================================================
# GRAPH STATE
# ============================================================================

class GraphState(TypedDict):
    """State shared across all agents"""
    # Input
    query: str
    ticker: str
    company_name: str
    
    # Orchestration
    parsed_query: Optional[Dict]
    execution_plan: Optional[Dict]
    
    # RAG
    retrieved_documents: List[str]
    document_sources: List[str]
    financial_metrics: Optional[Dict]
    
    # Courtroom - Opening
    pro_opening: Optional[Dict]
    against_opening: Optional[Dict]
    
    # Courtroom - Cross-Examination
    pro_rebuttal: Optional[Dict]
    against_rebuttal: Optional[Dict]
    
    # Courtroom - Closing
    pro_closing: Optional[Dict]
    against_closing: Optional[Dict]
    
    # Jury
    jury_observations: Dict[str, Dict]
    jury_deliberations: Dict[str, Dict]
    
    # Judge
    judge_verdict: Optional[Dict]
    
    # Critique (Media)
    critique_report: Optional[Dict]
    critique_passed: bool
    
    # Super Agent (Result)
    final_verdict: Optional[Dict]
    
    # Meta
    current_phase: str
    iteration_count: int
    errors: List[str]


# ============================================================================
# AGENT PROMPTS
# ============================================================================

QUERY_UNDERSTANDING_PROMPT = """You are the QUERY ORCHESTRATOR. Analyze the user's investment query.

QUERY: {query}

Return JSON:
{{
    "intent": "investment_analysis|general_info|comparative",
    "key_topics": ["topic1", "topic2"],
    "required_specialties": ["fundamentals", "risk", "sentiment"],
    "time_horizon": "short|medium|long"
}}"""

# Sub-Agent Strategy Prompts
PRO_STRATEGY_PROMPT = """You are a LEGAL STRATEGIST for the PRO (Bullish) team. 
Your goal: Brainstorm 3 manipulative, logical, and psychologically persuasive arguments to convince a jury to INVEST.

Context:
{web_context}

FINANCIAL METRICS:
{financial_data}

Company: {company_name} ({ticker})

Return JSON:
{{
    "strategy_angles": ["Angle 1: The Visionary Future", "Angle 2: Undervalued Gem", "Angle 3: Market Dominance"],
    "psychological_hooks": ["Fear of Missing Out (FOMO)", "Authority bias"],
    "key_evidence_to_highlight": ["specific revenue growth", "new product launch"]
}}"""

AGAINST_STRATEGY_PROMPT = """You are a LEGAL STRATEGIST for the AGAINST (Bearish) team. 
Your goal: Brainstorm 3 manipulative, logical, and psychologically persuasive arguments to convince a jury NOT TO INVEST.

Context:
{web_context}

FINANCIAL METRICS:
{financial_data}

Company: {company_name} ({ticker})

Return JSON:
{{
    "strategy_angles": ["Angle 1: Hidden Risks", "Angle 2: Overhyped Valuation", "Angle 3: Management Red Flags"],
    "psychological_hooks": ["Loss Aversion", "Skepticism"],
    "weaknesses_to_exploit": ["declining margins", "legal troubles"]
}}"""

PLANNER_PROMPT = """You are the Planner Agent. Create an execution plan for this analysis.

PARSED QUERY: {parsed_query}
COMPANY: {company_name} ({ticker})

Create a plan as JSON:
{{
    "analysis_type": "full_courtroom|quick_assessment|comparison",
    "agents_to_invoke": ["list of agents needed"],
    "focus_areas": ["fundamentals", "risk", "esg", "sentiment"],
    "enable_cross_examination": true/false,
    "expected_duration": "short|medium|long"
}}"""

PRO_OPENING_PROMPT = """You are the PRO AGENT (Bullish Advocate) for {company_name} ({ticker}).

STRATEGY PLAN (from your legal strategist):
{strategy}

WEB RESEARCH:
{web_context}

DOCUMENT EVIDENCE:
{context}

USER QUERY: {query}

Construct a powerful, persuasive OPENING STATEMENT. Use the strategy angles provided.
Be manipulative but backed by logic. Make the jury feel they MUST invest.

Return JSON:
{{
    "opening_statement": "Your 3-paragraph opening speech",
    "key_bullish_points": ["Point 1", "Point 2", "Point 3"],
    "sentiment_score": 0.8-1.0
}}"""

AGAINST_OPENING_PROMPT = """You are the AGAINST AGENT (Bearish Advocate) for {company_name} ({ticker}).

STRATEGY PLAN (from your legal strategist):
{strategy}

WEB RESEARCH:
{web_context}

DOCUMENT EVIDENCE:
{context}

USER QUERY: {query}

Construct a powerful, persuasive OPENING STATEMENT. Use the strategy angles provided.
Be skeptical, cynical, and logical. Expose the flaws.

Return JSON:
{{
    "opening_statement": "Your 3-paragraph opening speech",
    "key_bearish_points": ["Point 1", "Point 2", "Point 3"],
    "sentiment_score": 0.0-0.2
}}"""

CROSS_EXAMINATION_PROMPT = """You are the {agent_role} AGENT in cross-examination.

OPPONENT'S OPENING STATEMENT:
{opponent_opening}

Your task: REBUT the opponent's key arguments. Identify weaknesses in their case.

Return JSON:
{{
    "rebuttal_points": ["Point-by-point rebuttals"],
    "weaknesses_identified": ["holes in opponent's argument"],
    "counter_evidence": ["evidence that contradicts opponent"],
    "maintained_position": "summary of your unchanged stance"
}}"""

CLOSING_STATEMENT_PROMPT = """You are the {agent_role} AGENT delivering your CLOSING STATEMENT.

YOUR OPENING: {own_opening}
CROSS-EXAMINATION RESULTS: {cross_exam}
OPPONENT'S ARGUMENTS: {opponent_arguments}

Deliver a powerful closing argument summarizing your strongest case.

Return JSON:
{{
    "closing_statement": "Your final 2-3 paragraph argument",
    "strongest_argument": "Your single most compelling point",
    "response_to_opponent": "Why opponent's case is weaker",
    "final_recommendation": "BUY/SELL/HOLD from your perspective",
    "final_confidence": 0.0-1.0
}}"""

JURY_OBSERVATION_PROMPT = """You are a JURY SPECIALIST ({specialty}) observing the courtroom debate.

PRO OPENING:
{pro_opening}

AGAINST OPENING:
{against_opening}

CROSS-EXAMINATION:
{cross_exam}

DOCUMENT EVIDENCE:
{context}

FINANCIAL DATA:
{financial_data}

Take observation notes focusing on {specialty}:

Return JSON:
{{
    "observations": ["key observations from debate"],
    "verdict_implication": "positive|negative",
    "score": 1-10 (10 = perfect for this specialty)
}}"""

METADATA_EXTRACTION_PROMPT = """Analyze the following text from a financial report document (cover page/intro) and extract the Company Name and Stock Ticker.

TEXT:
{text}

Return JSON:
{{
    "company_name": "Full Company Name",
    "ticker": "TICKER Symbol",
    "year": "Report Year"
}}"""

JURY_DELIBERATION_PROMPT = """You are a JURY SPECIALIST ({specialty}) in final deliberation.

FULL DEBATE TRANSCRIPT:
- Pro Opening: {pro_opening}
- Against Opening: {against_opening}
- Cross-Examination: {cross_exam}
- Pro Closing: {pro_closing}
- Against Closing: {against_closing}

FINANCIAL DATA:
{financial_data}

YOUR EARLIER OBSERVATIONS: {observations}

Render your specialist verdict:

Return JSON:
{{
    "{specialty}_score": 0.0-1.0,
    "verdict": "BUY|SELL|HOLD",
    "key_factors": ["factors that drove your decision"],
    "concerns_for_judge": ["important points for the judge"],
    "confidence": 0.0-1.0
}}"""

JUDGE_VERDICT_PROMPT = """You are the CHIEF INVESTMENT JUDGE rendering the final verdict.

FULL COURTROOM TRANSCRIPT:

=== PRO AGENT (Bullish Case) ===
Opening: {pro_opening}
Rebuttal: {pro_rebuttal}
Closing: {pro_closing}

=== AGAINST AGENT (Bearish Case) ===
Opening: {against_opening}
Rebuttal: {against_rebuttal}
Closing: {against_closing}

=== JURY DELIBERATIONS ===
{jury_deliberations}

=== FINANCIAL DATA ===
{financial_data}

COMPANY: {company_name} ({ticker})
ORIGINAL QUERY: {query}

Weigh all evidence and render your FINAL VERDICT:

Return JSON:
{{
    "DECISION": "INVEST|NOT_TO_INVEST",
    "CONFIDENCE": 0-100 (integer),
    "REASONING": "3-5 paragraph detailed reasoning",
    "KEY_CONSIDERATIONS": ["Top 5 factors"],
    "DISSENTING_VIEWS": ["Important counter-arguments"],
    "RISK_WARNINGS": ["What could invalidate this decision"],
    "TIME_HORIZON": "short|medium|long term outlook"
}}"""

CRITIQUE_PROMPT = """You are the CRITIQUE AGENT - the "Media" observing this investment trial.

Your role is to provide EXTERNAL ACCOUNTABILITY like a financial journalist covering a court case.

JUDGE'S VERDICT:
{judge_verdict}

FULL TRIAL SUMMARY:
- Pro case strength: {pro_summary}
- Against case strength: {against_summary}
- Jury consensus: {jury_summary}

WEB SEARCH NEWS (LATEST):
{web_context}

Provide your media critique:

Return JSON:
{{
    "headline": "Your headline summarizing the verdict",
    "critique_summary": "2-3 paragraph analysis of the trial",
    "verdict_fairness": 0.0-1.0,
    "confidence_in_verdict": 0.0-1.0,
    "potential_biases_detected": ["any biases in reasoning"],
    "overlooked_factors": ["important factors not considered"],
    "public_accountability_notes": ["what investors should know"],
    "recommendation": "ACCEPT|REVISE|REJECT"
}}"""

KING_AGENT_PROMPT = """You are the KING AGENT (ROYAL VALIDATOR).
Your role is to review the entire case, validate the process, and publish the ROYAL VERDICT.

FULL CASE HISTORY:
1. QUERY: {query}
2. JUDGE'S VERDICT: {judge_verdict}
3. MEDIA CRITIQUE: {critique_report}

Validate the consistency of the verdict with the evidence and critique.
Make the final authoritative decision.

Return JSON:
{{
    "FINAL_DECISION": "INVEST|HOLD|NOT_TO_INVEST",
    "OFFICIAL_VERDICT": "INVEST|HOLD|NOT_TO_INVEST",
    "VALIDATION_STATUS": "VALIDATED|CORRECTED",
    "CONFIDENCE": 0-100,
    "FINAL_CONFIDENCE": 0-100,
    "EXECUTIVE_SUMMARY": "Concise 1-paragraph summary for the user",
    "KEY_DRIVERS": ["Top 3 decisive factors"],
    "ACTIONABLE_ADVICE": "What the user should do next"
}}"""


# ============================================================================
# INVESTMENT AGENT SYSTEM
# ============================================================================

class InvestmentAgentSystem:
    """12-Agent Investment Analysis System with Courtroom Debate"""
    
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.rag = AgenticRAGSystem(self.config)
        # self.search = DuckDuckGoSearchRun()  # Disabled
        self.agent_llms = self._create_agent_llms()
        self.graph = self._build_graph()
        logger.info("✅ Investment Agent System initialized (12 agents)")
    
    def _create_agent_llms(self) -> Dict[str, ChatOpenAI]:
        """Create LLM instances for each agent"""
        return {
            name: ChatOpenAI(
                model=model,
                openai_api_key=self.config.openrouter_api_key,
                openai_api_base=self.config.openrouter_base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                default_headers={"HTTP-Referer": "https://investment-agent.local", "X-Title": "Investment Agent"}
            )
            for name, model in AGENT_MODEL_MAPPING.items()
        }
    
    def _get_llm(self, agent: str) -> ChatOpenAI:
        return self.agent_llms.get(agent, self.agent_llms["judge_agent"])
    
    
    @staticmethod
    def _repair_json(text: str) -> str:
        """Repair common JSON issues from LLM output.

        Handles:
        - Literal newlines / tabs / carriage-returns inside string values
        - Trailing commas before } or ]
        """
        result = []
        in_string = False
        escape_next = False
        for char in text:
            if escape_next:
                result.append(char)
                escape_next = False
                continue
            if char == '\\':
                result.append(char)
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                result.append(char)
                continue
            if in_string:
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    continue          # skip bare \r
                elif char == '\t':
                    result.append('\\t')
                else:
                    result.append(char)
            else:
                result.append(char)
        repaired = ''.join(result)
        # Remove trailing commas before } or ]
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        return repaired

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response with robust fallback"""
        # 1. Extract from Markdown fences if present
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        # 2. Try direct parse
        try:
            return json.loads(json_str.strip())
        except Exception:
            pass

        # 3. Try regex extraction (first { to last })
        match = re.search(r'(\{.*\})', response, re.DOTALL)
        candidate = match.group(1) if match else json_str

        try:
            return json.loads(candidate)
        except Exception:
            pass

        # 4. Repair common LLM JSON issues (newlines in strings, trailing commas)
        try:
            repaired = self._repair_json(candidate)
            return json.loads(repaired)
        except Exception:
            pass

        logger.warning(f"❌ JSON Parse Failed. Response preview: {response[:200]}...")
        return {"raw_response": response, "error": "JSON Parsing Failed", "parse_error": True}
    
    @staticmethod
    def _to_percent(value: Any, default: float = 50.0) -> int:
        """Normalize confidence values to an integer percentage."""
        try:
            if isinstance(value, str):
                value = value.strip().rstrip("%")
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        if number <= 1:
            number *= 100
        return int(max(0, min(100, round(number))))

    @staticmethod
    def _to_ratio(value: Any, default: float = 0.5) -> float:
        """Normalize confidence values to a 0-1 ratio."""
        try:
            if isinstance(value, str):
                value = value.strip().rstrip("%")
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        if number > 1:
            number /= 100
        return max(0.0, min(1.0, number))

    def _format_context(self, docs: List[str], sources: List[str]) -> str:
        """Format retrieved documents"""
        if not docs:
            return "No documents available."
        parts = []
        for i, (doc, src) in enumerate(zip(docs, sources), 1):
            parts.append(f"[Doc {i} - {Path(src).name}]\n{doc}\n")
        return "\n".join(parts)
    
    def _run_search(self, query: str) -> str:
        """Web search disabled - return empty for graceful degradation"""
        # Search wrapper was removed - skip web search
        # This allows the agents to work with document context only
        return ""
            
    def _fetch_financials(self, ticker: str) -> Dict:
        """Fetch quantitative data from yfinance"""
        if not ticker: 
            return {"error": "No ticker provided"}
        
        # Indian stock ticker mapping
        indian_ticker_map = {
            "RIL": "RELIANCE.NS",
            "RELIANCE": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "INFY": "INFY.NS",
            "INFOSYS": "INFY.NS",
            "HDFCBANK": "HDFCBANK.NS",
            "ICICIBANK": "ICICIBANK.NS",
            "SBIN": "SBIN.NS",
            "HINDUNILVR": "HINDUNILVR.NS",
            "TATAMOTORS": "TATAMOTORS.NS",
            "TATAPOWER": "TATAPOWER.NS",
            "WIPRO": "WIPRO.NS",
            "BHARTIARTL": "BHARTIARTL.NS",
            "LT": "LT.NS",
            "ITC": "ITC.NS",
            "MARUTI": "MARUTI.NS",
            "KOTAKBANK": "KOTAKBANK.NS",
            "ASIANPAINT": "ASIANPAINT.NS",
        }
        
        # Normalize ticker
        normalized_ticker = ticker.upper().strip()
        
        try:
            # Check if it's a known Indian stock
            if normalized_ticker in indian_ticker_map:
                yf_ticker = indian_ticker_map[normalized_ticker]
                logger.info(f"Using mapped ticker: {yf_ticker}")
            # If already has exchange suffix, use as-is
            elif "." in normalized_ticker:
                yf_ticker = normalized_ticker
            # Try raw ticker first
            else:
                yf_ticker = normalized_ticker
            
            stock = yf.Ticker(yf_ticker)
            info = stock.info
            
            # If no price found, try with .NS suffix for Indian stocks
            if not info.get("regularMarketPrice") and not info.get("currentPrice"):
                if "." not in yf_ticker:
                    logger.info(f"Retrying with .NS suffix for {normalized_ticker}")
                    stock = yf.Ticker(f"{normalized_ticker}.NS")
                    info = stock.info
            
            # If still no data, try without suffix (for US stocks)
            if not info.get("regularMarketPrice") and not info.get("currentPrice"):
                if ".NS" in yf_ticker:
                    base_ticker = yf_ticker.replace(".NS", "").replace(".BSE", "")
                    logger.info(f"Retrying as US stock: {base_ticker}")
                    stock = yf.Ticker(base_ticker)
                    info = stock.info

            metrics = {
                "Current Price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
                "Market Cap": info.get("marketCap", "N/A"),
                "Trailing PE": info.get("trailingPE", "N/A"),
                "Forward PE": info.get("forwardPE", "N/A"),
                "Revenue Growth": info.get("revenueGrowth", "N/A"),
                "Beta": info.get("beta", "N/A"),
                "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
                "Recommendation": info.get("recommendationKey", "N/A"),
                "Currency": info.get("currency", "USD")
            }
            # Format Market Cap
            mc = metrics["Market Cap"]
            if isinstance(mc, (int, float)):
                if mc > 1e12:
                    metrics["Market Cap"] = f"{mc / 1e12:.2f}T"
                elif mc > 1e9:
                    metrics["Market Cap"] = f"{mc / 1e9:.2f}B"
                elif mc > 1e6:
                    metrics["Market Cap"] = f"{mc / 1e6:.2f}M"
            
            logger.info(f"Fetched financials for {yf_ticker}: Price={metrics['Current Price']}, PE={metrics['Trailing PE']}")
            return metrics
        except Exception as e:
            logger.error(f"yfinance failed for {ticker}: {e}")
            return {"error": f"Data unavailable: {str(e)}"}
    
    # ========== ORCHESTRATION NODES ==========
    
    def _query_understanding(self, state: GraphState) -> GraphState:
        """Parse and understand the user query"""
        prompt = QUERY_UNDERSTANDING_PROMPT.format(query=state["query"])
        try:
            response = self._get_llm("query_understanding").invoke([HumanMessage(content=prompt)])
            state["parsed_query"] = self._parse_response(response.content)
            logger.info("🧠 Query Understanding complete")
        except Exception as e:
            state["parsed_query"] = {"intent": "investment_analysis", "error": str(e)}
            state["errors"].append(f"Query Understanding: {e}")
        return state
    
    def _planner(self, state: GraphState) -> GraphState:
        """Create execution plan"""
        prompt = PLANNER_PROMPT.format(
            parsed_query=json.dumps(state["parsed_query"]),
            company_name=state["company_name"],
            ticker=state["ticker"]
        )
        try:
            response = self._get_llm("planner").invoke([HumanMessage(content=prompt)])
            state["execution_plan"] = self._parse_response(response.content)
            logger.info("📋 Planner complete")
        except Exception as e:
            state["execution_plan"] = {"analysis_type": "full_courtroom", "error": str(e)}
            state["errors"].append(f"Planner: {e}")
        return state
    
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents using semantic RAG"""
        state["current_phase"] = "rag_retrieval"
        
        # DEBUG: Log RAG state
        logger.info(f"🔍 DEBUG: documents_loaded={self.rag.documents_loaded}, vectorstore={self.rag.vectorstore is not None}")
        
        try:
            results = self.rag.multi_query_retrieve(
                state["query"], 
                state["company_name"], 
                state["ticker"]
            )
            state["retrieved_documents"] = [r["content"] for r in results]
            state["document_sources"] = [r["source"] for r in results]
            
            # DEBUG: Log what was retrieved
            logger.info(f"📚 Retrieved {len(results)} documents, {len(state['retrieved_documents'])} in state")
            if results:
                logger.info(f"   First doc preview: {results[0]['content'][:100]}...")
            else:
                logger.warning("⚠️ NO DOCUMENTS RETRIEVED - Check vectorstore!")
            
            # Fetch real-time financial metrics
            state["financial_metrics"] = self._fetch_financials(state["ticker"])
                
        except Exception as e:
            logger.error(f"❌ RAG Retrieval Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            state["retrieved_documents"] = []
            state["document_sources"] = []
            state["errors"].append(f"RAG: {e}")
        return state
    
    # ========== COURTROOM NODES ==========
    
    def _pro_opening(self, state: GraphState) -> GraphState:
        """Pro agent presents opening statement with Web Search reinforcement"""
        state["current_phase"] = "pro_opening"
        
        # DEBUG: Log document count entering pro_opening
        logger.info(f"🟢 Pro Opening: Got {len(state.get('retrieved_documents', []))} documents in state")
        
        # Web Search (Bullish)
        web_context = ""
        try:
            search_query = f"{state['company_name']} {state['ticker']} bullish growth revenue news"
            web_context = self._run_search(search_query)
        except Exception as e:
            logger.warning(f"Pro Agent search failed: {e}")
            
        if not web_context: web_context = "No relevant news found."
        
        # 1. Sub-Agent: Strategy Formulation
        strategy = {}
        try:
            strat_prompt = PRO_STRATEGY_PROMPT.format(
                web_context=web_context,
                financial_data=json.dumps(state.get("financial_metrics", {}), indent=2),
                company_name=state["company_name"],
                ticker=state["ticker"]
            )
            # Self-reflection call (using same model or planner model for variety)
            strat_response = self._get_llm("pro_agent").invoke([HumanMessage(content=strat_prompt)])
            strategy = self._parse_response(strat_response.content)
            logger.info("💡 Pro Agent Strategy Formulated")
        except Exception as e:
            logger.warning(f"Pro Strategy failed: {e}")
            strategy = {"strategy_angles": ["General Bullishness"]}

        # 2. Final Opening Statement
        prompt = PRO_OPENING_PROMPT.format(
            company_name=state["company_name"],
            ticker=state["ticker"],
            query=state["query"],
            context=self._format_context(state["retrieved_documents"], state["document_sources"]),
            web_context=web_context,
            strategy=json.dumps(strategy, indent=2)
        )
        try:
            response = self._get_llm("pro_agent").invoke([HumanMessage(content=prompt)])
            state["pro_opening"] = self._parse_response(response.content)
            state["pro_opening"]["strategy_used"] = strategy # Save strategy for context
            logger.info("🟢 Pro Agent delivered opening statement (with strategy & search)")
        except Exception as e:
            state["pro_opening"] = {"error": str(e)}
            state["errors"].append(f"Pro Opening: {e}")
        
        # RETURN ONLY UPDATES to avoid parallel conflict
        return {"pro_opening": state["pro_opening"]}
    
    def _against_opening(self, state: GraphState) -> GraphState:
        """Against agent presents opening statement with Web Search reinforcement"""
        state["current_phase"] = "against_opening"
        
        # Web Search (Bearish)
        web_context = ""
        try:
            search_query = f"{state['company_name']} {state['ticker']} bearish risks scandal controversy lawsuits"
            web_context = self._run_search(search_query)
        except Exception as e:
            logger.warning(f"Against Agent search failed: {e}")
            
        if not web_context: web_context = "No relevant news found."
        
        # 1. Sub-Agent: Strategy Formulation
        strategy = {}
        try:
            strat_prompt = AGAINST_STRATEGY_PROMPT.format(
                web_context=web_context,
                financial_data=json.dumps(state.get("financial_metrics", {}), indent=2),
                company_name=state["company_name"],
                ticker=state["ticker"]
            )
            strat_response = self._get_llm("against_agent").invoke([HumanMessage(content=strat_prompt)])
            strategy = self._parse_response(strat_response.content)
            logger.info("💡 Against Agent Strategy Formulated")
        except Exception as e:
            logger.warning(f"Against Strategy failed: {e}")
            strategy = {"strategy_angles": ["General Skepticism"]}
            
        # 2. Final Opening Statement
        prompt = AGAINST_OPENING_PROMPT.format(
            company_name=state["company_name"],
            ticker=state["ticker"],
            query=state["query"],
            context=self._format_context(state["retrieved_documents"], state["document_sources"]),
            web_context=web_context,
            strategy=json.dumps(strategy, indent=2)
        )
        try:
            response = self._get_llm("against_agent").invoke([HumanMessage(content=prompt)])
            state["against_opening"] = self._parse_response(response.content)
            state["against_opening"]["strategy_used"] = strategy # Save strategy
            logger.info("🔴 Against Agent delivered opening statement (with strategy & search)")
        except Exception as e:
            state["against_opening"] = {"error": str(e)}
            state["errors"].append(f"Against Opening: {e}")
        
        # RETURN ONLY UPDATES to avoid parallel conflict
        return {"against_opening": state["against_opening"]}
    
    def _cross_examination(self, state: GraphState) -> GraphState:
        """Cross-examination round"""
        state["current_phase"] = "cross_examination"
        
        # Pro rebuts Against
        pro_prompt = CROSS_EXAMINATION_PROMPT.format(
            agent_role="PRO",
            opponent_opening=json.dumps(state["against_opening"], indent=2)
        )
        try:
            response = self._get_llm("pro_agent").invoke([HumanMessage(content=pro_prompt)])
            state["pro_rebuttal"] = self._parse_response(response.content)
        except Exception as e:
            state["pro_rebuttal"] = {"error": str(e)}
        
        # Against rebuts Pro
        against_prompt = CROSS_EXAMINATION_PROMPT.format(
            agent_role="AGAINST",
            opponent_opening=json.dumps(state["pro_opening"], indent=2)
        )
        try:
            response = self._get_llm("against_agent").invoke([HumanMessage(content=against_prompt)])
            state["against_rebuttal"] = self._parse_response(response.content)
        except Exception as e:
            state["against_rebuttal"] = {"error": str(e)}
        
        logger.info("⚔️ Cross-Examination complete")
        return state
    
    def _pro_closing(self, state: GraphState) -> GraphState:
        """Pro agent closing statement"""
        state["current_phase"] = "pro_closing"
        prompt = CLOSING_STATEMENT_PROMPT.format(
            agent_role="PRO",
            own_opening=json.dumps(state["pro_opening"]),
            cross_exam=json.dumps(state["pro_rebuttal"]),
            opponent_arguments=json.dumps(state["against_opening"])
        )
        try:
            response = self._get_llm("pro_agent").invoke([HumanMessage(content=prompt)])
            state["pro_closing"] = self._parse_response(response.content)
            logger.info("🟢 Pro Closing Statement delivered")
        except Exception as e:
            state["pro_closing"] = {"error": str(e)}
        return state
    
    def _against_closing(self, state: GraphState) -> GraphState:
        """Against agent closing statement"""
        state["current_phase"] = "against_closing"
        prompt = CLOSING_STATEMENT_PROMPT.format(
            agent_role="AGAINST",
            own_opening=json.dumps(state["against_opening"]),
            cross_exam=json.dumps(state["against_rebuttal"]),
            opponent_arguments=json.dumps(state["pro_opening"])
        )
        try:
            response = self._get_llm("against_agent").invoke([HumanMessage(content=prompt)])
            state["against_closing"] = self._parse_response(response.content)
            logger.info("🔴 Against Closing Statement delivered")
        except Exception as e:
            state["against_closing"] = {"error": str(e)}
        return state
    
    # ========== JURY NODES ==========
    
    def _jury_observe(self, state: GraphState) -> GraphState:
        """Jury observes the debate"""
        state["current_phase"] = "jury_observation"
        specialties = ["fundamentals", "risk", "esg", "sentiment"]
        context = self._format_context(state["retrieved_documents"], state["document_sources"])
        
        state["jury_observations"] = {}
        for specialty in specialties:
            prompt = JURY_OBSERVATION_PROMPT.format(
                specialty=specialty,
                pro_opening=json.dumps(state["pro_opening"]),
                against_opening=json.dumps(state["against_opening"]),
                cross_exam=json.dumps({"pro": state["pro_rebuttal"], "against": state["against_rebuttal"]}),
                context=context,
                financial_data=json.dumps(state.get("financial_metrics", {}), indent=2)
            )
            try:
                response = self._get_llm(f"jury_{specialty}").invoke([HumanMessage(content=prompt)])
                state["jury_observations"][specialty] = self._parse_response(response.content)
            except Exception as e:
                state["jury_observations"][specialty] = {"error": str(e)}
        
        logger.info("👥 Jury Observations complete")
        return state
    
    def _jury_deliberate(self, state: GraphState) -> GraphState:
        """Jury final deliberation"""
        state["current_phase"] = "jury_deliberation"
        specialties = ["fundamentals", "risk", "esg", "sentiment"]
        
        state["jury_deliberations"] = {}
        for specialty in specialties:
            prompt = JURY_DELIBERATION_PROMPT.format(
                specialty=specialty,
                pro_opening=json.dumps(state["pro_opening"]),
                against_opening=json.dumps(state["against_opening"]),
                cross_exam=json.dumps({"pro": state["pro_rebuttal"], "against": state["against_rebuttal"]}),
                pro_closing=json.dumps(state["pro_closing"]),
                against_closing=json.dumps(state["against_closing"]),
                observations=json.dumps(state["jury_observations"].get(specialty, {})),
                financial_data=json.dumps(state.get("financial_metrics", {}), indent=2)
            )
            try:
                response = self._get_llm(f"jury_{specialty}").invoke([HumanMessage(content=prompt)])
                state["jury_deliberations"][specialty] = self._parse_response(response.content)
            except Exception as e:
                state["jury_deliberations"][specialty] = {"error": str(e)}
        
        logger.info("⚖️ Jury Deliberation complete")
        return state
    
    # ========== JUDGE NODE (DUAL LLM) ==========
    
    def _judge_verdict(self, state: GraphState) -> GraphState:
        """Judge renders verdict using dual-LLM reasoning for harder deliberation"""
        state["current_phase"] = "judge_verdict"
        
        # Build evidence summary prompt
        prompt = JUDGE_VERDICT_PROMPT.format(
            company_name=state["company_name"],
            ticker=state["ticker"],
            query=state["query"],
            pro_opening=json.dumps(state["pro_opening"]),
            pro_rebuttal=json.dumps(state["pro_rebuttal"]),
            pro_closing=json.dumps(state["pro_closing"]),
            against_opening=json.dumps(state["against_opening"]),
            against_rebuttal=json.dumps(state["against_rebuttal"]),
            against_closing=json.dumps(state["against_closing"]),
            jury_deliberations=json.dumps(state["jury_deliberations"], indent=2),
            financial_data=json.dumps(state.get("financial_metrics", {}), indent=2)
        )
        
        try:
            # Step 1: Primary Judge - Initial verdict
            logger.info("⚖️ Judge Primary: Rendering initial verdict...")
            primary_response = self._get_llm("judge_agent").invoke([HumanMessage(content=prompt)])
            primary_verdict = self._parse_response(primary_response.content)
            
            # Step 2: Sub-Agent - Validate and challenge
            logger.info("⚖️ Judge Sub-Agent: Validating verdict...")
            validation_prompt = f"""You are a senior legal advisor reviewing a judge's verdict.

ORIGINAL VERDICT:
{json.dumps(primary_verdict, indent=2)}

Your task:
1. Identify any logical flaws or overlooked evidence
2. Confirm or challenge the decision
3. Provide your own DECISION (INVEST or NOT_TO_INVEST) with CONFIDENCE (0-100)

Respond in JSON:
{{"validation": "CONFIRMED|CHALLENGED", "concerns": [...], "DECISION": "...", "CONFIDENCE": ...}}"""
            
            sub_response = self._get_llm("judge_sub_agent").invoke([HumanMessage(content=validation_prompt)])
            sub_verdict = self._parse_response(sub_response.content)
            
            # Step 3: Synthesize - If both agree, high confidence; if disagree, note concern
            final_decision = primary_verdict.get("DECISION", "HOLD")
            final_confidence = self._to_percent(primary_verdict.get("CONFIDENCE"), 50)
            
            if sub_verdict.get("validation") == "CHALLENGED":
                # Average confidence if challenged
                sub_conf = self._to_percent(sub_verdict.get("CONFIDENCE"), 50)
                final_confidence = min(100, max(0, int((final_confidence + sub_conf) // 2)))
                if sub_verdict.get("DECISION") != final_decision:
                    final_decision = "HOLD"  # Disagreement = cautious approach
            
            state["judge_verdict"] = {
                "primary_verdict": primary_verdict,
                "sub_agent_validation": sub_verdict,
                "DECISION": final_decision,
                "CONFIDENCE": final_confidence,
                "dual_llm_consensus": sub_verdict.get("validation", "UNKNOWN"),
                "REASONING": primary_verdict.get("REASONING", primary_verdict.get("Reasoning", primary_verdict.get("reasoning", "N/A"))),
                "KEY_CONSIDERATIONS": primary_verdict.get("KEY_CONSIDERATIONS", []),
                "DISSENTING_VIEWS": primary_verdict.get("DISSENTING_VIEWS", [])
            }
            logger.info(f"👨‍⚖️ Judge Verdict: {final_decision} (Confidence: {final_confidence}%, Consensus: {sub_verdict.get('validation', 'N/A')})")
            
        except Exception as e:
            state["judge_verdict"] = {"error": str(e), "DECISION": "HOLD", "CONFIDENCE": 0}
            state["errors"].append(f"Judge: {e}")
        
        return state
    
    # ========== CRITIQUE NODE (MEDIA) ==========
    
    def _critique_agent(self, state: GraphState) -> GraphState:
        """Critique agent (Media) provides external accountability with Web Search"""
        state["current_phase"] = "critique"
        
        # 1. Perform Web Search for latest news/controversies
        web_context = "No search results available."
        try:
            search_query = f"{state['company_name']} {state['ticker']} financial controversy news risks stock performance"
            web_context = self._run_search(search_query)
            logger.info(f"📰 Critique Agent found {len(web_context.splitlines())} news items")
        except Exception as e:
            logger.warning(f"⚠️ Web search failed: {e}")
        
        prompt = CRITIQUE_PROMPT.format(
            judge_verdict=json.dumps(state["judge_verdict"], indent=2),
            pro_summary=json.dumps(state.get("pro_closing", {})),
            against_summary=json.dumps(state.get("against_closing", {})),
            jury_summary=json.dumps(state["jury_deliberations"]),
            web_context=web_context
        )
        try:
            response = self._get_llm("critique_agent").invoke([HumanMessage(content=prompt)])
            state["critique_report"] = self._parse_response(response.content)
            
            # Check if verdict passes critique
            confidence = self._to_ratio(
                state["critique_report"].get(
                    "confidence_in_verdict",
                    state["critique_report"].get("verdict_fairness", 0.7)
                ),
                0.7
            )
            state["critique_report"]["confidence_in_verdict"] = confidence
            if "verdict_fairness" in state["critique_report"]:
                state["critique_report"]["verdict_fairness"] = self._to_ratio(
                    state["critique_report"].get("verdict_fairness"),
                    confidence
                )
            recommendation = state["critique_report"].get("recommendation", "ACCEPT")
            state["critique_passed"] = recommendation == "ACCEPT" and confidence >= self.config.critique_confidence_threshold
            
            logger.info(f"🔍 Critique (Media) report: {recommendation}")
        except Exception as e:
            state["critique_report"] = {"error": str(e)}
            state["critique_passed"] = True
            state["errors"].append(f"Critique: {e}")
        
        return state

    # ========== KING AGENT NODE (DUAL LLM) ==========
    
    def _king_agent(self, state: GraphState) -> GraphState:
        """King Agent validates everything using dual-LLM for royal deliberation"""
        state["current_phase"] = "king_agent"
        
        prompt = KING_AGENT_PROMPT.format(
            query=state["query"],
            judge_verdict=json.dumps(state["judge_verdict"], indent=2),
            critique_report=json.dumps(state["critique_report"], indent=2)
        )
        
        try:
            # Step 1: Primary King - Initial royal verdict
            logger.info("👑 King Primary: Deliberating initial verdict...")
            primary_response = self._get_llm("king_agent").invoke([HumanMessage(content=prompt)])
            primary_verdict = self._parse_response(primary_response.content)
            
            # Step 2: Royal Advisor Sub-Agent - Second opinion
            logger.info("👑 King Sub-Agent: Royal advisor deliberation...")
            advisor_prompt = f"""You are a senior royal investment advisor to the King.

The King has made an initial verdict on this investment decision:
{json.dumps(primary_verdict, indent=2)}

Judge's Assessment:
{json.dumps(state["judge_verdict"], indent=2)}

Media Critique:
{json.dumps(state["critique_report"], indent=2)}

Your role:
1. Review the King's verdict with fresh eyes
2. Challenge any weak reasoning
3. Consider long-term implications
4. Provide your own FINAL_DECISION (INVEST or NOT_TO_INVEST)

Respond in JSON:
{{"advisor_opinion": "AGREE|DISAGREE", "reasoning": "...", "FINAL_DECISION": "...", "CONFIDENCE": 0-100, "CONDITIONS": [...]}}"""

            advisor_response = self._get_llm("king_sub_agent").invoke([HumanMessage(content=advisor_prompt)])
            advisor_verdict = self._parse_response(advisor_response.content)
            
            # Step 3: Royal Synthesis - King considers advisor's counsel
            final_decision = primary_verdict.get(
                "FINAL_DECISION",
                primary_verdict.get("OFFICIAL_VERDICT", "HOLD")
            )
            final_confidence = self._to_percent(
                primary_verdict.get("CONFIDENCE", primary_verdict.get("FINAL_CONFIDENCE")),
                50
            )
            
            # If advisor disagrees, King reconsiders
            if advisor_verdict.get("advisor_opinion") == "DISAGREE":
                advisor_decision = advisor_verdict.get("FINAL_DECISION", final_decision)
                advisor_conf = self._to_percent(advisor_verdict.get("CONFIDENCE"), 50)
                
                # If high confidence disagreement, defer to advisor
                if advisor_conf > final_confidence:
                    final_decision = advisor_decision
                    final_confidence = (final_confidence + advisor_conf) // 2
                else:
                    # Reduce confidence due to disagreement
                    final_confidence = max(30, final_confidence - 20)
            else:
                # Agreement boosts confidence
                advisor_conf = self._to_percent(advisor_verdict.get("CONFIDENCE"), 50)
                # Ensure confidence stays within 0-100 range and is an integer
                final_confidence = min(100, max(0, int((final_confidence + advisor_conf) // 2 + 10)))
            
            state["final_verdict"] = {
                "king_primary": primary_verdict,
                "royal_advisor": advisor_verdict,
                "FINAL_DECISION": final_decision,
                "CONFIDENCE": final_confidence,
                "CONDITIONS": advisor_verdict.get("CONDITIONS", primary_verdict.get("CONDITIONS", [])),
                "dual_llm_consensus": "ALIGNED" if advisor_verdict.get("advisor_opinion") == "AGREE" else "CONTESTED",
                "OFFICIAL_VERDICT": final_decision,
                "FINAL_CONFIDENCE": final_confidence,
                "VALIDATION_STATUS": "VERIFIED" if advisor_verdict.get("advisor_opinion") == "AGREE" else "CONTESTED",
                "EXECUTIVE_SUMMARY": primary_verdict.get("EXECUTIVE_SUMMARY", "N/A"),
                "KEY_DRIVERS": primary_verdict.get("KEY_DRIVERS", []),
                "ACTIONABLE_ADVICE": primary_verdict.get("ACTIONABLE_ADVICE", "N/A")
            }
            logger.info(f"👑 Royal Verdict: {final_decision} (Confidence: {final_confidence}%, Consensus: {state['final_verdict']['dual_llm_consensus']})")
            
        except Exception as e:
            state["final_verdict"] = {
                "error": str(e),
                "FINAL_DECISION": "HOLD",
                "OFFICIAL_VERDICT": "HOLD",
                "CONFIDENCE": 0,
                "FINAL_CONFIDENCE": 0,
            }
            state["errors"].append(f"King Agent: {e}")
        
        return state
    
    def _debate_join(self, state: GraphState) -> GraphState:
        """Join node to synchronize parallel Pro/Against agents"""
        # Ensure both sides have spoken
        if not state.get("pro_opening") or not state.get("against_opening"):
            logger.warning("⚠️ Debate Join: One side is missing!")
        
        state["current_phase"] = "debate_join"
        logger.info("🤝 Debate Joined: Pro and Against have spoken.")
        return state

    # ========== GRAPH BUILDING ==========
    
    def _build_graph(self) -> StateGraph:
        """Build the 12-agent workflow graph"""
        workflow = StateGraph(GraphState)
        
        # Add all nodes
        workflow.add_node("query_understanding", self._query_understanding)
        workflow.add_node("planner", self._planner)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("pro_opening", self._pro_opening)
        workflow.add_node("against_opening", self._against_opening)
        workflow.add_node("debate_join", self._debate_join)  # NEW JOIN NODE
        workflow.add_node("cross_examination", self._cross_examination)
        workflow.add_node("pro_closing", self._pro_closing)
        workflow.add_node("against_closing", self._against_closing)
        workflow.add_node("jury_observe", self._jury_observe)
        workflow.add_node("jury_deliberate", self._jury_deliberate)
        workflow.add_node("judge_verdict", self._judge_verdict)
        workflow.add_node("critique_agent", self._critique_agent)
        workflow.add_node("king_agent", self._king_agent)
        
        # Define edges - Orchestration flow
        workflow.set_entry_point("query_understanding")
        workflow.add_edge("query_understanding", "planner")
        workflow.add_edge("planner", "retrieve_documents")
        
        # Courtroom flow - Opening statements (PARALLEL EXECUTION)
        workflow.add_edge("retrieve_documents", "pro_opening")
        workflow.add_edge("retrieve_documents", "against_opening")
        
        # Join parallel branches after both opening statements complete
        workflow.add_edge(["pro_opening", "against_opening"], "debate_join")
        
        # Cross-examination
        workflow.add_edge("debate_join", "jury_observe")
        workflow.add_edge("jury_observe", "cross_examination")
        
        # Closing statements
        workflow.add_edge("cross_examination", "pro_closing")
        workflow.add_edge("pro_closing", "against_closing")
        
        # Jury deliberation
        workflow.add_edge("against_closing", "jury_deliberate")
        
        # Judge verdict
        workflow.add_edge("jury_deliberate", "judge_verdict")
        
        # Critique (Media)
        workflow.add_edge("judge_verdict", "critique_agent")
        
        # King Agent (Final Validator)
        workflow.add_edge("critique_agent", "king_agent")
        workflow.add_edge("king_agent", END)
        
        return workflow.compile()
    
    # ========== PUBLIC API ==========
    
    def _compute_rl_evaluation(self, state: GraphState) -> Dict:
        """
        Compute RL evaluation metrics for feedback loop.
        
        Metrics:
        1. Judge-King Disagreement: Penalty if Judge and King disagree
        2. Jury Confidence: Average confidence from jury members
        3. Media Sentiment Delta: Difference between verdict and media critique
        
        Returns a score 0-100 where higher = better decision quality
        """
        rl_metrics = {
            "judge_king_disagreement": 0.0,
            "jury_confidence": 50.0,
            "media_sentiment_delta": 0.0,
            "overall_score": 50.0,  # Baseline
            "feedback_signals": []
        }
        
        try:
            # 1. Judge-King Disagreement
            judge_verdict = state.get("judge_verdict", {})
            final_verdict = state.get("final_verdict", {})
            
            judge_decision = judge_verdict.get("DECISION", "HOLD")
            king_decision = final_verdict.get("FINAL_DECISION", "HOLD")
            
            if judge_decision == king_decision:
                rl_metrics["judge_king_disagreement"] = 0.0  # Agreement = good
                rl_metrics["feedback_signals"].append("✅ Judge-King Aligned")
            else:
                rl_metrics["judge_king_disagreement"] = -20.0  # Disagreement = penalty
                rl_metrics["feedback_signals"].append(f"⚠️ Judge-King Disagreement: {judge_decision} vs {king_decision}")
            
            # Check dual-LLM consensus within each agent
            judge_consensus = judge_verdict.get("dual_llm_consensus", "UNKNOWN")
            king_consensus = final_verdict.get("dual_llm_consensus", "UNKNOWN")
            
            if judge_consensus == "CHALLENGED":
                rl_metrics["judge_king_disagreement"] -= 10
                rl_metrics["feedback_signals"].append("⚠️ Judge sub-agent challenged primary")
            if king_consensus == "CONTESTED":
                rl_metrics["judge_king_disagreement"] -= 10
                rl_metrics["feedback_signals"].append("⚠️ King advisor disagreed")
            
            # 2. Jury Confidence (average across all jury members)
            jury = state.get("jury_deliberations", {})
            jury_confidences = []
            for member, vote in jury.items():
                if isinstance(vote, dict):
                    confidence = vote.get("CONFIDENCE", vote.get("confidence"))
                    if confidence is not None:
                        jury_confidences.append(self._to_percent(confidence, 50))
            
            if jury_confidences:
                avg_jury_conf = sum(jury_confidences) / len(jury_confidences)
                rl_metrics["jury_confidence"] = avg_jury_conf
                
                # Bonus for high jury confidence, penalty for low
                if avg_jury_conf >= 70:
                    rl_metrics["feedback_signals"].append(f"✅ High Jury Confidence: {avg_jury_conf:.0f}%")
                elif avg_jury_conf < 40:
                    rl_metrics["feedback_signals"].append(f"⚠️ Low Jury Confidence: {avg_jury_conf:.0f}%")
            
            # 3. Media Sentiment Delta
            critique = state.get("critique_report", {})
            media_confidence = self._to_percent(
                critique.get("confidence_in_verdict", critique.get("verdict_fairness")),
                50
            )
            judge_confidence = self._to_percent(judge_verdict.get("CONFIDENCE"), 50)
            
            delta = abs(media_confidence - judge_confidence)
            rl_metrics["media_sentiment_delta"] = delta
            
            if delta > 30:
                rl_metrics["feedback_signals"].append(f"⚠️ Media-Judge Delta: {delta}pts (Media skeptical)")
            elif delta < 10:
                rl_metrics["feedback_signals"].append("✅ Media validates verdict")
            
            # Compute overall RL score
            base_score = 50.0
            base_score += (rl_metrics["jury_confidence"] - 50) * 0.5  # Scale jury contribution
            base_score += rl_metrics["judge_king_disagreement"]  # Disagreement penalties
            base_score -= rl_metrics["media_sentiment_delta"] * 0.3  # Media delta penalty
            
            rl_metrics["overall_score"] = max(0, min(100, base_score))
            
            logger.info(f"🎯 RL Evaluation Score: {rl_metrics['overall_score']:.1f}/100")
            
        except Exception as e:
            logger.error(f"RL Evaluation failed: {e}")
            rl_metrics["error"] = str(e)
        
        return rl_metrics
    
    def load_documents(self, pdf_paths: List[str] = None) -> int:
        """Load PDF documents"""
        return self.rag.load_documents(pdf_paths)
    
    def analyze(self, query: str, ticker: str, company_name: str) -> Dict:
        """Run full courtroom analysis"""
        logger.info(f"\n{'='*60}")
        logger.info(f"⚖️ COURTROOM ANALYSIS: {company_name} ({ticker})")
        logger.info(f"{'='*60}\n")
        
        initial_state: GraphState = {
            "query": query,
            "ticker": ticker,
            "company_name": company_name,
            "parsed_query": None,
            "execution_plan": None,
            "retrieved_documents": [],
            "document_sources": [],
            "pro_opening": None,
            "against_opening": None,
            "pro_rebuttal": None,
            "against_rebuttal": None,
            "pro_closing": None,
            "against_closing": None,
            "jury_observations": {},
            "jury_deliberations": {},
            "judge_verdict": None,
            "critique_report": None,

            "critique_passed": False,
            "final_verdict": None,
            "current_phase": "start",
            "iteration_count": 0,
            "errors": []
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # Compute RL evaluation metrics
        rl_metrics = self._compute_rl_evaluation(final_state)
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "query": query,
            "financial_metrics": final_state.get("financial_metrics", {}),
            "decision": final_state.get("judge_verdict"),
            "pro_case": {
                "opening": final_state.get("pro_opening"),
                "rebuttal": final_state.get("pro_rebuttal"),
                "closing": final_state.get("pro_closing")
            },
            "against_case": {
                "opening": final_state.get("against_opening"),
                "rebuttal": final_state.get("against_rebuttal"),
                "closing": final_state.get("against_closing")
            },
            "jury_verdicts": final_state.get("jury_deliberations"),
            "critique_report": final_state.get("critique_report"),
            "critique_passed": final_state.get("critique_passed"),
            "final_verdict": final_state.get("final_verdict"),
            "documents_analyzed": len(final_state.get("retrieved_documents", [])),
            "errors": final_state.get("errors", []),
            "rl_evaluation": rl_metrics,  # RL feedback loop metrics
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# REPORT GENERATOR
# ============================================================================

def generate_report(result: Dict) -> str:
    """Generate formatted investment report"""
    report = []
    report.append("=" * 70)
    report.append("⚖️ COURTROOM INVESTMENT ANALYSIS REPORT")
    report.append(f"Company: {result['company_name']} ({result['ticker']})")
    report.append(f"Generated: {result['timestamp']}")
    report.append("=" * 70)
    report.append("")
    
    # Decision
    decision = result.get("decision", {})
    if decision and not decision.get("error"):
        report.append("📊 FINAL VERDICT")
        report.append("-" * 40)
        report.append(f"Decision: {decision.get('DECISION', 'N/A')}")
        report.append(f"Confidence: {decision.get('CONFIDENCE', 'N/A')}")
        if decision.get('REASONING'):
            report.append(f"\nReasoning:\n{decision['REASONING']}")
        report.append("")
    
    # Critique (Media) Report
    critique = result.get("critique_report", {})
    if critique and not critique.get("error"):
        report.append("📰 MEDIA CRITIQUE")
        report.append("-" * 40)
        report.append(f"Headline: {critique.get('headline', 'N/A')}")
        report.append(f"Verdict Confidence: {critique.get('confidence_in_verdict', 'N/A')}")
        report.append(f"Recommendation: {critique.get('recommendation', 'N/A')}")
        if critique.get("overlooked_factors"):
            report.append("Overlooked Factors:")
            for f in critique["overlooked_factors"]:
                report.append(f"  • {f}")
        report.append("")
    
    report.append("=" * 70)
    return "\n".join(report)
