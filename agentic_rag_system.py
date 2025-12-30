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
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

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
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "allenai/olmo-3.1-32b-think:free"))
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


# Model mapping for all 12 agents
# Using 8 FREE models: 3 new (kimi-k2, glm-4.5-air, deepseek-r1t-chimera) + 5 original
# Model mapping for all 13 agents (added Super Agent)
# Using 10 FREE models: 2 new (devstral, deepseek-v3.1) + 8 previous
AGENT_MODEL_MAPPING = {
    # Orchestration Layer
    "query_understanding": "moonshotai/kimi-k2:free",
    "planner": "moonshotai/kimi-k2:free",
    
    # Courtroom Agents
    "pro_agent": "moonshotai/kimi-k2:free",
    "against_agent": "tngtech/deepseek-r1t-chimera:free",
    "judge_agent": "z-ai/glm-4.5-air:free",
    
    # Jury Specialists
    "jury_fundamentals": "tngtech/deepseek-r1t-chimera:free",
    "jury_risk": "nvidia/nemotron-3-nano-30b-a3b:free",
    "jury_esg": "allenai/olmo-3.1-32b-think:free",
    "jury_sentiment": "xiaomi/mimo-v2-flash:free",
    
    # Media/Critique
    "critique_agent": "z-ai/glm-4.5-air:free",
    
    # Super Agent (Final Validator) - Devstral 2512
    "super_agent": "mistralai/devstral-2512:free",
}

# All available FREE models (10 total)
FREE_MODELS = {
    # Newest additions
    "Mistral Devstral 2512": "mistralai/devstral-2512:free",
    "DeepSeek V3.1 Nex": "nex-agi/deepseek-v3.1-nex-n1:free",
    # Previous models
    "Kimi K2": "moonshotai/kimi-k2:free",
    "GLM 4.5 Air": "z-ai/glm-4.5-air:free",
    "DeepSeek R1T Chimera": "tngtech/deepseek-r1t-chimera:free",
    "OLMo 3.1 32B Think": "allenai/olmo-3.1-32b-think:free",
    "OLMo 3 32B Think": "allenai/olmo-3-32b-think:free",
    "Nvidia Nemotron 30B": "nvidia/nemotron-3-nano-30b-a3b:free",
    "Xiaomi MiMo V2 Flash": "xiaomi/mimo-v2-flash:free",
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
        """Count tokens using tiktoken"""
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
        self.vectorstore = None
        self.documents_loaded = False
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        
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
                
                logger.info(f"  → {len([c for c in all_chunks if c.metadata.get('source_file') == str(pdf_path)])} semantic chunks")
            except Exception as e:
                logger.error(f"Error loading {pdf_path}: {e}")
        
        if all_chunks:
            self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            self.documents_loaded = True
            logger.info(f"✅ Indexed {len(all_chunks)} semantic chunks")
        
        return len(all_chunks)
    
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

QUERY_UNDERSTANDING_PROMPT = """You are the Query Understanding Agent. Parse the user's investment query.

USER QUERY: {query}

Extract and return JSON:
{{
    "intent": "investment_analysis|comparison|general_question",
    "ticker": "extracted ticker or null",
    "company_name": "extracted company name or null",
    "time_horizon": "short|medium|long|unspecified",
    "risk_tolerance": "low|medium|high|unspecified",
    "specific_focus": ["list of specific areas to analyze"],
    "requires_rag": true/false
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

PRO_OPENING_PROMPT = """You are the PRO AGENT (Bullish Advocate) in an investment courtroom.

COMPANY: {company_name} ({ticker})
QUERY: {query}

EVIDENCE FROM DOCUMENTS:
{context}

Deliver your OPENING STATEMENT. Build the strongest bullish case:

Return JSON:
{{
    "opening_statement": "Your 2-3 paragraph opening argument",
    "key_bullish_points": ["Point 1", "Point 2", "Point 3"],
    "evidence_cited": ["specific evidence from documents"],
    "growth_catalysts": ["upcoming positive events"],
    "bull_case_confidence": 0.0-1.0
}}"""

AGAINST_OPENING_PROMPT = """You are the AGAINST AGENT (Bearish Advocate) in an investment courtroom.

COMPANY: {company_name} ({ticker})
QUERY: {query}

EVIDENCE FROM DOCUMENTS:
{context}

Deliver your OPENING STATEMENT. Present all risks and concerns:

Return JSON:
{{
    "opening_statement": "Your 2-3 paragraph opening argument",
    "key_bearish_points": ["Risk 1", "Risk 2", "Risk 3"],
    "risk_evidence": ["specific concerns from documents"],
    "downside_scenarios": ["what could go wrong"],
    "bear_case_confidence": 0.0-1.0
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

Take observation notes focusing on {specialty}:

Return JSON:
{{
    "observations": ["key observations from debate"],
    "pro_strength": 0.0-1.0,
    "against_strength": 0.0-1.0,
    "{specialty}_concerns": ["specific concerns in your area"],
    "preliminary_lean": "pro|against|neutral"
}}"""

JURY_DELIBERATION_PROMPT = """You are a JURY SPECIALIST ({specialty}) in final deliberation.

FULL DEBATE TRANSCRIPT:
- Pro Opening: {pro_opening}
- Against Opening: {against_opening}
- Cross-Examination: {cross_exam}
- Pro Closing: {pro_closing}
- Against Closing: {against_closing}

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

COMPANY: {company_name} ({ticker})
ORIGINAL QUERY: {query}

Weigh all evidence and render your FINAL VERDICT:

Return JSON:
{{
    "DECISION": "BUY|SELL|HOLD",
    "CONFIDENCE": 0.0-1.0,
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

Provide your media critique:

Return JSON:
{{
    "headline": "Your headline summarizing the verdict",
    "critique_summary": "2-3 paragraph analysis of the trial",
    "verdict_fairness": 0.0-1.0,
    "potential_biases_detected": ["any biases in reasoning"],
    "overlooked_factors": ["important factors not considered"],
    "public_accountability_notes": ["what investors should know"],
    "recommendation": "ACCEPT|REVISE|REJECT"
}}"""

SUPER_AGENT_PROMPT = """You are the SUPER AGENT (FINAL VALIDATOR).
Your role is to review the entire case, validate the process, and publish the OFFICIAL VERDICT.

FULL CASE HISTORY:
1. QUERY: {query}
2. JUDGE'S VERDICT: {judge_verdict}
3. MEDIA CRITIQUE: {critique_report}

Validate the consistency of the verdict with the evidence and critique.
Make the final authoritative decision.

Return JSON:
{{
    "OFFICIAL_VERDICT": "BUY|SELL|HOLD",
    "VALIDATION_STATUS": "VALIDATED|CORRECTED",
    "FINAL_CONFIDENCE": 0.0-1.0,
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
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response"""
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            return json.loads(json_str.strip())
        except:
            return {"raw_response": response, "parse_error": True}
    
    def _format_context(self, docs: List[str], sources: List[str]) -> str:
        """Format retrieved documents"""
        if not docs:
            return "No documents available."
        parts = []
        for i, (doc, src) in enumerate(zip(docs, sources), 1):
            parts.append(f"[Doc {i} - {Path(src).name}]\n{doc}\n")
        return "\n".join(parts)
    
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
        try:
            results = self.rag.multi_query_retrieve(
                state["query"], 
                state["company_name"], 
                state["ticker"]
            )
            state["retrieved_documents"] = [r["content"] for r in results]
            state["document_sources"] = [r["source"] for r in results]
            logger.info(f"📚 Retrieved {len(results)} documents")
        except Exception as e:
            state["retrieved_documents"] = []
            state["document_sources"] = []
            state["errors"].append(f"RAG: {e}")
        return state
    
    # ========== COURTROOM NODES ==========
    
    def _pro_opening(self, state: GraphState) -> GraphState:
        """Pro agent opening statement"""
        state["current_phase"] = "pro_opening"
        context = self._format_context(state["retrieved_documents"], state["document_sources"])
        prompt = PRO_OPENING_PROMPT.format(
            company_name=state["company_name"],
            ticker=state["ticker"],
            query=state["query"],
            context=context
        )
        try:
            response = self._get_llm("pro_agent").invoke([HumanMessage(content=prompt)])
            state["pro_opening"] = self._parse_response(response.content)
            logger.info("🟢 Pro Opening Statement delivered")
        except Exception as e:
            state["pro_opening"] = {"error": str(e)}
            state["errors"].append(f"Pro Opening: {e}")
        return state
    
    def _against_opening(self, state: GraphState) -> GraphState:
        """Against agent opening statement"""
        state["current_phase"] = "against_opening"
        context = self._format_context(state["retrieved_documents"], state["document_sources"])
        prompt = AGAINST_OPENING_PROMPT.format(
            company_name=state["company_name"],
            ticker=state["ticker"],
            query=state["query"],
            context=context
        )
        try:
            response = self._get_llm("against_agent").invoke([HumanMessage(content=prompt)])
            state["against_opening"] = self._parse_response(response.content)
            logger.info("🔴 Against Opening Statement delivered")
        except Exception as e:
            state["against_opening"] = {"error": str(e)}
            state["errors"].append(f"Against Opening: {e}")
        return state
    
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
                context=context
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
                observations=json.dumps(state["jury_observations"].get(specialty, {}))
            )
            try:
                response = self._get_llm(f"jury_{specialty}").invoke([HumanMessage(content=prompt)])
                state["jury_deliberations"][specialty] = self._parse_response(response.content)
            except Exception as e:
                state["jury_deliberations"][specialty] = {"error": str(e)}
        
        logger.info("⚖️ Jury Deliberation complete")
        return state
    
    # ========== JUDGE NODE ==========
    
    def _judge_verdict(self, state: GraphState) -> GraphState:
        """Judge renders final verdict"""
        state["current_phase"] = "judge_verdict"
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
            jury_deliberations=json.dumps(state["jury_deliberations"], indent=2)
        )
        try:
            response = self._get_llm("judge_agent").invoke([HumanMessage(content=prompt)])
            state["judge_verdict"] = self._parse_response(response.content)
            logger.info("👨‍⚖️ Judge Verdict rendered")
        except Exception as e:
            state["judge_verdict"] = {"error": str(e)}
            state["errors"].append(f"Judge: {e}")
        return state
    
    # ========== CRITIQUE NODE (MEDIA) ==========
    
    def _critique_agent(self, state: GraphState) -> GraphState:
        """Critique agent (Media) provides external accountability"""
        state["current_phase"] = "critique"
        
        prompt = CRITIQUE_PROMPT.format(
            judge_verdict=json.dumps(state["judge_verdict"], indent=2),
            pro_summary=json.dumps(state.get("pro_closing", {})),
            against_summary=json.dumps(state.get("against_closing", {})),
            jury_summary=json.dumps(state["jury_deliberations"])
        )
        try:
            response = self._get_llm("critique_agent").invoke([HumanMessage(content=prompt)])
            state["critique_report"] = self._parse_response(response.content)
            
            # Check if verdict passes critique
            confidence = state["critique_report"].get("confidence_in_verdict", 0.7)
            recommendation = state["critique_report"].get("recommendation", "ACCEPT")
            state["critique_passed"] = recommendation == "ACCEPT" and confidence >= self.config.critique_confidence_threshold
            
            logger.info(f"🔍 Critique (Media) report: {recommendation}")
        except Exception as e:
            state["critique_report"] = {"error": str(e)}
            state["critique_passed"] = True
            state["errors"].append(f"Critique: {e}")
        
        return state

    # ========== SUPER AGENT NODE ==========
    
    def _super_agent(self, state: GraphState) -> GraphState:
        """Super Agent validates everything and publishes final verdict"""
        state["current_phase"] = "super_agent"
        
        prompt = SUPER_AGENT_PROMPT.format(
            query=state["query"],
            judge_verdict=json.dumps(state["judge_verdict"], indent=2),
            critique_report=json.dumps(state["critique_report"], indent=2)
        )
        try:
            response = self._get_llm("super_agent").invoke([HumanMessage(content=prompt)])
            state["final_verdict"] = self._parse_response(response.content)
            logger.info("🏆 Super Agent published final verdict")
        except Exception as e:
            state["final_verdict"] = {"error": str(e)}
            state["errors"].append(f"Super Agent: {e}")
        
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
        workflow.add_node("cross_examination", self._cross_examination)
        workflow.add_node("pro_closing", self._pro_closing)
        workflow.add_node("against_closing", self._against_closing)
        workflow.add_node("jury_observe", self._jury_observe)
        workflow.add_node("jury_deliberate", self._jury_deliberate)
        workflow.add_node("judge_verdict", self._judge_verdict)
        workflow.add_node("critique_agent", self._critique_agent)
        workflow.add_node("super_agent", self._super_agent)
        
        # Define edges - Orchestration flow
        workflow.set_entry_point("query_understanding")
        workflow.add_edge("query_understanding", "planner")
        workflow.add_edge("planner", "retrieve_documents")
        
        # Courtroom flow - Opening statements (parallel conceptually)
        workflow.add_edge("retrieve_documents", "pro_opening")
        workflow.add_edge("pro_opening", "against_opening")
        
        # Cross-examination
        workflow.add_edge("against_opening", "jury_observe")
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
        
        # Super Agent (Final Validator)
        workflow.add_edge("critique_agent", "super_agent")
        workflow.add_edge("super_agent", END)
        
        return workflow.compile()
    
    # ========== PUBLIC API ==========
    
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
            "judge_verdict": None,
            "critique_report": None,
            "critique_passed": False,
            "final_verdict": None,
            "current_phase": "start",
            "iteration_count": 0,
            "errors": []
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "query": query,
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
