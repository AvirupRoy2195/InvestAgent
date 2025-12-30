# 🔧 Advanced Patterns

> Production patterns for extending the Investment Agent System

---

## Table of Contents

1. [Async Processing](#async-processing)
2. [Custom Jury Specialists](#custom-jury-specialists)
3. [Batch Analysis](#batch-analysis)
4. [Vector Database Options](#vector-database-options)
5. [Backtesting Framework](#backtesting-framework)
6. [Real-Time Data Integration](#real-time-data-integration)
7. [Multi-Company Comparison](#multi-company-comparison)

---

## Async Processing

For faster analysis, run agents in parallel using async:

```python
import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI

class AsyncInvestmentSystem:
    """Async version for parallel agent execution"""
    
    def __init__(self, config):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.default_model,
            openai_api_key=config.openrouter_api_key,
            openai_api_base=config.openrouter_base_url,
        )
    
    async def _run_agent_async(self, prompt: str) -> str:
        """Run single agent asynchronously"""
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
    
    async def run_debate_parallel(self, context: str, query: str) -> Dict:
        """Run Pro and Against agents in parallel"""
        pro_prompt = self._format_pro_prompt(context, query)
        against_prompt = self._format_against_prompt(context, query)
        
        # Run both simultaneously
        pro_task = asyncio.create_task(self._run_agent_async(pro_prompt))
        against_task = asyncio.create_task(self._run_agent_async(against_prompt))
        
        pro_result, against_result = await asyncio.gather(pro_task, against_task)
        
        return {
            "pro": pro_result,
            "against": against_result
        }
    
    async def run_jury_parallel(self, context: str) -> Dict:
        """Run all 4 jury specialists in parallel"""
        specialists = ["fundamentals", "risk", "esg", "sentiment"]
        
        tasks = [
            asyncio.create_task(
                self._run_agent_async(self._format_jury_prompt(spec, context))
            )
            for spec in specialists
        ]
        
        results = await asyncio.gather(*tasks)
        
        return dict(zip(specialists, results))


# Usage
async def main():
    config = Config()
    system = AsyncInvestmentSystem(config)
    
    context = "Retrieved document content..."
    query = "Should I invest?"
    
    debate = await system.run_debate_parallel(context, query)
    jury = await system.run_jury_parallel(context)
    
    print(f"Pro: {debate['pro'][:100]}...")
    print(f"Against: {debate['against'][:100]}...")

asyncio.run(main())
```

**Performance improvement:** 40-60% faster analysis

---

## Custom Jury Specialists

### Add Macro Analyst

```python
JURY_MACRO_PROMPT = """You are a MACRO ANALYST on the investment jury.

Analyze {company_name} ({ticker}) in the context of:
- Interest rate environment and central bank policy
- Currency fluctuations (INR/USD impacts)
- Commodity price trends (oil, natural gas)
- Global trade conditions
- Inflation trajectory
- GDP growth outlook
- Sector-specific government policies

CONTEXT FROM DOCUMENTS:
{context}

Provide a macro assessment:
1. MACRO_SCORE (0.0 to 1.0, higher = favorable environment)
2. KEY TAILWINDS (macro factors helping the company)
3. KEY HEADWINDS (macro factors hurting the company)
4. RATE_SENSITIVITY (how sensitive to interest rate changes)
5. CURRENCY_EXPOSURE (net exporter/importer impact)
6. POLICY_OUTLOOK (government/regulatory direction)

Format as structured JSON."""


def _jury_macro(self, state: GraphState) -> GraphState:
    """Jury macro specialist"""
    return self._run_jury_agent(state, "macro", JURY_MACRO_PROMPT)


# Add to _build_graph():
workflow.add_node("jury_macro", self._jury_macro)
workflow.add_edge("jury_sentiment", "jury_macro")
workflow.add_edge("jury_macro", "judge_agent")
```

### Add Technical Analyst

```python
JURY_TECHNICAL_PROMPT = """You are a TECHNICAL ANALYST on the investment jury.

While you can't see charts, analyze {company_name} ({ticker}) for:
- Historical price performance (if mentioned)
- Volume trends
- Support/resistance levels mentioned
- Relative performance vs sector/index
- Momentum indicators referenced
- Any technical patterns discussed

CONTEXT FROM DOCUMENTS:
{context}

Provide technical insights:
1. TECHNICAL_SCORE (0.0 to 1.0)
2. TREND_DIRECTION (bullish/neutral/bearish)
3. KEY_LEVELS (support, resistance if available)
4. MOMENTUM (improving/steady/weakening)
5. RELATIVE_STRENGTH (vs sector)

Format as structured JSON."""
```

### Add Competitive Analyst

```python
JURY_COMPETITIVE_PROMPT = """You are a COMPETITIVE ANALYST on the investment jury.

Analyze {company_name} ({ticker})'s competitive position:
- Market share and trends
- Key competitors and their strengths
- Barriers to entry
- Switching costs for customers
- Pricing power
- Innovation/R&D positioning
- Supplier and buyer power

CONTEXT FROM DOCUMENTS:
{context}

Provide competitive assessment:
1. COMPETITIVE_SCORE (0.0 to 1.0, higher = stronger moat)
2. MARKET_POSITION (leader/challenger/follower)
3. KEY_COMPETITORS (and their threats)
4. MOAT_STRENGTH (wide/narrow/none)
5. MOAT_DURABILITY (sustainable for how many years?)
6. DISRUPTION_RISK (high/medium/low)

Format as structured JSON."""
```

---

## Batch Analysis

Analyze multiple companies efficiently:

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List
import pandas as pd


class BatchAnalyzer:
    """Batch process multiple companies"""
    
    def __init__(self, config: Config):
        self.system = InvestmentAgentSystem(config)
        self.system.load_documents()
    
    def analyze_portfolio(
        self, 
        companies: List[Dict],
        max_workers: int = 3
    ) -> pd.DataFrame:
        """Analyze multiple companies in parallel
        
        Args:
            companies: List of {"ticker": "X", "name": "Company X"}
            max_workers: Parallel threads (respect rate limits)
        """
        results = []
        
        def analyze_one(company):
            try:
                result = self.system.analyze(
                    query="Full investment analysis with buy/sell/hold recommendation",
                    ticker=company["ticker"],
                    company_name=company["name"]
                )
                decision = result.get("decision", {})
                return {
                    "ticker": company["ticker"],
                    "company": company["name"],
                    "decision": decision.get("DECISION", "ERROR"),
                    "confidence": decision.get("CONFIDENCE", 0),
                    "fundamentals": result.get("jury_verdicts", {}).get("fundamentals", {}).get("FUNDAMENTALS_SCORE", 0),
                    "risk": result.get("jury_verdicts", {}).get("risk", {}).get("RISK_SCORE", 0),
                    "esg": result.get("jury_verdicts", {}).get("esg", {}).get("ESG_SCORE", 0),
                    "sentiment": result.get("jury_verdicts", {}).get("sentiment", {}).get("SENTIMENT_SCORE", 0),
                    "errors": len(result.get("errors", [])),
                }
            except Exception as e:
                return {
                    "ticker": company["ticker"],
                    "company": company["name"],
                    "decision": "ERROR",
                    "confidence": 0,
                    "errors": str(e)
                }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(analyze_one, companies))
        
        df = pd.DataFrame(results)
        df = df.sort_values("confidence", ascending=False)
        
        return df
    
    def screen_and_rank(self, portfolios: List[Dict]) -> pd.DataFrame:
        """Screen and rank companies by investment merit"""
        results = self.analyze_portfolio(portfolios)
        
        # Calculate composite score
        results["composite_score"] = (
            results["confidence"] * 0.3 +
            results["fundamentals"] * 0.3 +
            (1 - results["risk"]) * 0.2 +  # Invert risk (lower is better)
            results["esg"] * 0.1 +
            results["sentiment"] * 0.1
        )
        
        results = results.sort_values("composite_score", ascending=False)
        results["rank"] = range(1, len(results) + 1)
        
        return results


# Usage
companies = [
    {"ticker": "RIL", "name": "Reliance Industries"},
    {"ticker": "TCS", "name": "Tata Consultancy Services"},
    {"ticker": "HDFC", "name": "HDFC Bank"},
    {"ticker": "INFY", "name": "Infosys"},
]

analyzer = BatchAnalyzer(Config())
rankings = analyzer.screen_and_rank(companies)
print(rankings[["rank", "ticker", "decision", "composite_score"]])
```

---

## Vector Database Options

### Option 1: Pinecone (Cloud, Scalable)

```python
from langchain_pinecone import PineconeVectorStore
import pinecone

class PineconeRAG:
    """Cloud-based vector store for production"""
    
    def __init__(self, config):
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model
        )
        
        self.vectorstore = PineconeVectorStore(
            index_name="investment-docs",
            embedding=self.embeddings
        )
    
    def add_documents(self, documents):
        """Add documents to Pinecone index"""
        self.vectorstore.add_documents(documents)
    
    def retrieve(self, query: str, k: int = 5):
        """Retrieve from Pinecone"""
        return self.vectorstore.similarity_search(query, k=k)
```

### Option 2: ChromaDB (Local, Persistent)

```python
from langchain_chroma import Chroma

class ChromaRAG:
    """Local persistent vector store"""
    
    def __init__(self, config, persist_dir: str = "./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model
        )
        
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
    
    def load_documents(self, pdf_paths):
        """Load and persist documents"""
        chunks = []
        for path in pdf_paths:
            loader = PyPDFLoader(str(path))
            chunks.extend(loader.load_and_split())
        
        self.vectorstore.add_documents(chunks)
        # Persists automatically
```

### Option 3: Qdrant (Self-hosted or Cloud)

```python
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

class QdrantRAG:
    """High-performance vector store"""
    
    def __init__(self, config, url: str = "http://localhost:6333"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model
        )
        
        self.client = QdrantClient(url=url)
        
        self.vectorstore = Qdrant(
            client=self.client,
            collection_name="investment_docs",
            embeddings=self.embeddings
        )
```

---

## Backtesting Framework

Test the system against historical decisions:

```python
from datetime import datetime, timedelta
import yfinance as yf


class BacktestFramework:
    """Backtest investment decisions against actual returns"""
    
    def __init__(self, system: InvestmentAgentSystem):
        self.system = system
    
    def get_historical_return(
        self, 
        ticker: str, 
        decision_date: str,
        holding_period_days: int = 90
    ) -> Dict:
        """Calculate actual return after decision"""
        try:
            stock = yf.Ticker(ticker + ".NS")  # Add .NS for NSE
            
            start = datetime.fromisoformat(decision_date)
            end = start + timedelta(days=holding_period_days + 5)
            
            hist = stock.history(start=start, end=end)
            
            if len(hist) < 2:
                return {"error": "Insufficient data"}
            
            entry_price = hist.iloc[0]["Close"]
            exit_price = hist.iloc[min(holding_period_days, len(hist)-1)]["Close"]
            
            return_pct = (exit_price - entry_price) / entry_price * 100
            
            return {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": return_pct,
                "holding_days": min(holding_period_days, len(hist)-1)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_decision(
        self,
        decision: str,
        confidence: float,
        actual_return: float
    ) -> Dict:
        """Evaluate if decision was correct"""
        # Decision was correct if:
        # - BUY and return > 0
        # - SELL and return < 0
        # - HOLD irrelevant (skip)
        
        if decision == "HOLD":
            return {
                "correct": None,
                "category": "HOLD",
                "comment": "HOLD decisions not evaluated"
            }
        
        correct = (
            (decision == "BUY" and actual_return > 0) or
            (decision == "SELL" and actual_return < 0)
        )
        
        # Risk-adjusted score
        confidence_calibration = abs(actual_return) / 100 if correct else -abs(actual_return) / 100
        
        return {
            "correct": correct,
            "decision": decision,
            "actual_return": actual_return,
            "confidence": confidence,
            "confidence_calibration": confidence_calibration,
            "category": "CORRECT" if correct else "INCORRECT"
        }
    
    def run_backtest(
        self,
        historical_analyses: List[Dict],
        holding_period: int = 90
    ) -> Dict:
        """Run full backtest on historical analyses"""
        results = []
        
        for analysis in historical_analyses:
            actual = self.get_historical_return(
                ticker=analysis["ticker"],
                decision_date=analysis["date"],
                holding_period_days=holding_period
            )
            
            if "error" not in actual:
                evaluation = self.evaluate_decision(
                    decision=analysis["decision"],
                    confidence=analysis["confidence"],
                    actual_return=actual["return_pct"]
                )
                evaluation["ticker"] = analysis["ticker"]
                evaluation["date"] = analysis["date"]
                results.append(evaluation)
        
        # Calculate metrics
        evaluated = [r for r in results if r["correct"] is not None]
        correct_count = sum(1 for r in evaluated if r["correct"])
        
        return {
            "total_evaluated": len(evaluated),
            "correct": correct_count,
            "accuracy": correct_count / len(evaluated) if evaluated else 0,
            "details": results
        }


# Usage
backtest = BacktestFramework(system)

# Historical decisions (you'd store these from past analyses)
historical = [
    {"ticker": "RIL", "date": "2024-01-15", "decision": "BUY", "confidence": 0.78},
    {"ticker": "TCS", "date": "2024-02-01", "decision": "HOLD", "confidence": 0.65},
    {"ticker": "INFY", "date": "2024-03-10", "decision": "SELL", "confidence": 0.72},
]

results = backtest.run_backtest(historical, holding_period=90)
print(f"Accuracy: {results['accuracy']:.1%}")
```

---

## Real-Time Data Integration

Add live market data to enhance analysis:

```python
import yfinance as yf
from datetime import datetime


class RealTimeEnhancer:
    """Enhance analysis with real-time market data"""
    
    def get_live_metrics(self, ticker: str) -> Dict:
        """Fetch current market data"""
        try:
            stock = yf.Ticker(ticker + ".NS")  # NSE suffix
            info = stock.info
            
            return {
                "current_price": info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "beta": info.get("beta"),
                "eps": info.get("trailingEps"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margins": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_recent_news(self, ticker: str) -> List[Dict]:
        """Fetch recent news headlines"""
        try:
            stock = yf.Ticker(ticker + ".NS")
            news = stock.news
            
            return [
                {
                    "title": n.get("title"),
                    "publisher": n.get("publisher"),
                    "link": n.get("link"),
                    "published": n.get("providerPublishTime")
                }
                for n in news[:5]  # Last 5 headlines
            ]
        except Exception as e:
            return [{"error": str(e)}]
    
    def enhance_prompt(self, base_prompt: str, ticker: str) -> str:
        """Add real-time data to analysis prompt"""
        metrics = self.get_live_metrics(ticker)
        news = self.get_recent_news(ticker)
        
        enhancement = f"""

REAL-TIME MARKET DATA (as of {metrics.get('last_updated', 'N/A')}):
- Current Price: ₹{metrics.get('current_price', 'N/A')}
- Market Cap: ₹{metrics.get('market_cap', 'N/A'):,.0f}
- P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
- P/B Ratio: {metrics.get('pb_ratio', 'N/A')}
- 52-Week Range: ₹{metrics.get('52_week_low', 'N/A')} - ₹{metrics.get('52_week_high', 'N/A')}
- ROE: {metrics.get('roe', 'N/A'):.1%}
- Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}

RECENT NEWS HEADLINES:
"""
        for n in news:
            if "error" not in n:
                enhancement += f"- {n.get('title', 'N/A')}\n"
        
        return base_prompt + enhancement


# Usage: Integrate into agent prompts
enhancer = RealTimeEnhancer()
enhanced_prompt = enhancer.enhance_prompt(PRO_AGENT_PROMPT, "RIL")
```

---

## Multi-Company Comparison

Compare multiple companies head-to-head:

```python
class ComparisonAnalyzer:
    """Compare multiple companies side-by-side"""
    
    def __init__(self, system: InvestmentAgentSystem):
        self.system = system
    
    def compare(
        self, 
        companies: List[Dict],
        comparison_query: str = "Compare these companies for investment"
    ) -> Dict:
        """Analyze multiple companies and compare"""
        
        # Individual analyses
        analyses = {}
        for company in companies:
            result = self.system.analyze(
                query="Complete investment analysis",
                ticker=company["ticker"],
                company_name=company["name"]
            )
            analyses[company["ticker"]] = result
        
        # Comparison summary
        comparison = {
            "companies": [],
            "ranking": [],
            "best_buy": None,
            "best_hold": None,
            "avoid": []
        }
        
        for ticker, result in analyses.items():
            decision = result.get("decision", {})
            jury = result.get("jury_verdicts", {})
            
            company_summary = {
                "ticker": ticker,
                "decision": decision.get("DECISION"),
                "confidence": decision.get("CONFIDENCE", 0),
                "fundamentals": jury.get("fundamentals", {}).get("FUNDAMENTALS_SCORE", 0),
                "risk": jury.get("risk", {}).get("RISK_SCORE", 0),
                "esg": jury.get("esg", {}).get("ESG_SCORE", 0),
            }
            
            # Composite score
            company_summary["composite"] = (
                company_summary["confidence"] * 0.4 +
                company_summary["fundamentals"] * 0.3 +
                (1 - company_summary["risk"]) * 0.2 +
                company_summary["esg"] * 0.1
            )
            
            comparison["companies"].append(company_summary)
        
        # Rank by composite score
        comparison["companies"].sort(key=lambda x: x["composite"], reverse=True)
        comparison["ranking"] = [c["ticker"] for c in comparison["companies"]]
        
        # Find best buys and avoids
        for c in comparison["companies"]:
            if c["decision"] == "BUY" and c["confidence"] > 0.7:
                if comparison["best_buy"] is None:
                    comparison["best_buy"] = c["ticker"]
            elif c["decision"] == "SELL":
                comparison["avoid"].append(c["ticker"])
        
        return comparison
    
    def generate_comparison_report(self, comparison: Dict) -> str:
        """Generate human-readable comparison report"""
        lines = []
        lines.append("=" * 60)
        lines.append("MULTI-COMPANY COMPARISON REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        lines.append("📊 RANKING (Best to Worst):")
        lines.append("-" * 40)
        for i, company in enumerate(comparison["companies"], 1):
            lines.append(
                f"  {i}. {company['ticker']}: {company['decision']} "
                f"(Confidence: {company['confidence']:.0%}, "
                f"Composite: {company['composite']:.2f})"
            )
        
        lines.append("")
        if comparison["best_buy"]:
            lines.append(f"🟢 BEST BUY: {comparison['best_buy']}")
        
        if comparison["avoid"]:
            lines.append(f"🔴 AVOID: {', '.join(comparison['avoid'])}")
        
        return "\n".join(lines)


# Usage
comparator = ComparisonAnalyzer(system)

companies = [
    {"ticker": "RIL", "name": "Reliance Industries"},
    {"ticker": "ONGC", "name": "Oil and Natural Gas Corporation"},
    {"ticker": "BPCL", "name": "Bharat Petroleum"},
]

comparison = comparator.compare(companies)
print(comparator.generate_comparison_report(comparison))
```

---

## Best Practices Summary

| Pattern | Use Case | Benefit |
|---------|----------|---------|
| Async Processing | High throughput | 40-60% faster |
| Custom Jury | Domain expertise | More accurate |
| Batch Analysis | Portfolio screening | Efficiency |
| Cloud Vector DB | Large document sets | Scalability |
| Backtesting | Validate accuracy | Trust building |
| Real-Time Data | Current context | Better decisions |
| Comparison | Portfolio allocation | Clear ranking |

---

*For real-world examples and case studies, see `REAL_WORLD_EXAMPLES.md`.*
