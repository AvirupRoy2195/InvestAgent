"""
Configuration Management Module
================================
Secure, validated configuration handling for the Investment Agent System.

Author: Investment AI Team
Version: 2.0.1 (Security Hardened)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
import logging

from security_utils import SecurityValidator, config_handler

logger = logging.getLogger(__name__)


# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """System configuration with security validation and defaults"""
    
    # OpenRouter Configuration (with validation)
    openrouter_api_key: str = field(default_factory=lambda: config_handler.load_api_key())
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Model Configuration
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "google/gemma-4-26b-a4b-it:free")
    )
    temperature: float = field(default=0.3)
    max_tokens: int = field(default=4096)
    
    # Semantic RAG Configuration
    semantic_chunk_size: int = field(default=1500)
    min_chunk_size: int = field(default=200)
    chunk_overlap: int = field(default=200)
    use_sentence_splitting: bool = field(default=True)
    embedding_model: str = field(default="sentence-transformers/all-MiniLM-L6-v2")
    top_k_retrieval: int = field(default=8)
    max_retrieval_tokens: int = field(default=6000)
    
    # Document paths
    documents_dir: str = field(
        default_factory=lambda: os.getenv("DOCUMENTS_DIR", "./")
    )
    
    # Analysis settings
    confidence_threshold: float = field(default=0.6)
    require_unanimous_jury: bool = field(default=False)
    
    # Courtroom settings
    enable_cross_examination: bool = field(default=True)
    debate_rounds: int = field(default=1)
    
    # Critique settings
    critique_confidence_threshold: float = field(default=0.7)
    max_critique_loops: int = field(default=2)
    
    # Security settings (NEW)
    max_file_size_mb: int = field(default=50)
    max_response_length: int = field(default=50000)
    enforce_input_validation: bool = field(default=True)
    log_security_events: bool = field(default=True)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate all configuration values"""
        errors = []
        
        # Validate temperature (0.0-1.0)
        if not 0.0 <= self.temperature <= 1.0:
            errors.append(f"Temperature must be 0.0-1.0, got {self.temperature}")
        
        # Validate max_tokens (positive integer)
        if self.max_tokens <= 0:
            errors.append(f"max_tokens must be positive, got {self.max_tokens}")
        
        # Validate chunk sizes
        if self.semantic_chunk_size <= 0:
            errors.append(f"semantic_chunk_size must be positive")
        
        if self.min_chunk_size <= 0:
            errors.append(f"min_chunk_size must be positive")
        
        if self.semantic_chunk_size < self.min_chunk_size:
            errors.append(f"semantic_chunk_size must be >= min_chunk_size")
        
        # Validate retrieval settings
        if self.top_k_retrieval <= 0:
            errors.append(f"top_k_retrieval must be positive")
        
        if self.max_retrieval_tokens <= 0:
            errors.append(f"max_retrieval_tokens must be positive")
        
        # Validate thresholds
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append(f"confidence_threshold must be 0.0-1.0")
        
        if not 0.0 <= self.critique_confidence_threshold <= 1.0:
            errors.append(f"critique_confidence_threshold must be 0.0-1.0")
        
        # Validate documents directory
        if self.documents_dir and not Path(self.documents_dir).exists():
            logger.warning(f"⚠️ Documents directory does not exist: {self.documents_dir}")
        
        # Report errors
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("✅ Configuration validated successfully")
    
    def to_dict(self, include_secrets: bool = False) -> Dict:
        """Convert config to dictionary (optionally exclude secrets)"""
        d = {
            "openrouter_base_url": self.openrouter_base_url,
            "default_model": self.default_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "semantic_chunk_size": self.semantic_chunk_size,
            "top_k_retrieval": self.top_k_retrieval,
            "documents_dir": self.documents_dir,
            "max_file_size_mb": self.max_file_size_mb,
        }
        
        if include_secrets:
            d["openrouter_api_key"] = self.openrouter_api_key
        else:
            d["openrouter_api_key"] = f"***{self.openrouter_api_key[-8:]}"
        
        return d


# Model mapping for all agents (now with versioning)
# Using FREE models from OpenRouter
AGENT_MODEL_MAPPING = {
    # Orchestration Layer
    "query_understanding": "google/gemma-4-26b-a4b-it:free",
    "planner": "minimax/minimax-m2.5:free",
    
    # Debate Layer
    "pro_strategy": "z-ai/glm-4.5-air:free",
    "against_strategy": "minimax/minimax-m2.5:free",
    "pro_opening": "z-ai/glm-4.5-air:free",
    "against_opening": "minimax/minimax-m2.5:free",
    "pro_cross": "z-ai/glm-4.5-air:free",
    "against_cross": "minimax/minimax-m2.5:free",
    "pro_closing": "z-ai/glm-4.5-air:free",
    "against_closing": "minimax/minimax-m2.5:free",
    
    # Jury Layer
    "jury_observation_fundamentals": "google/gemma-4-26b-a4b-it:free",
    "jury_observation_risk": "minimax/minimax-m2.5:free",
    "jury_observation_esg": "z-ai/glm-4.5-air:free",
    "jury_observation_sentiment": "qwen/qwen3-coder:free",
    
    # Verdict Layer
    "judge": "google/gemma-4-26b-a4b-it:free",
    "critique": "z-ai/glm-4.5-air:free",
}


# Available free models for user selection
FREE_MODELS = {
    "Google Gemma 4 26B A4B IT": "google/gemma-4-26b-a4b-it:free",
    "MiniMax M2.5": "minimax/minimax-m2.5:free",
    "GLM 4.5 Air": "z-ai/glm-4.5-air:free",
    "Qwen3 Coder": "qwen/qwen3-coder:free",
    "NVIDIA Llama Nemotron Embed VL 1B V2": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
}


def get_config() -> Config:
    """Get validated configuration instance"""
    try:
        return Config()
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        raise


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("✅ Configuration loaded successfully")
    print(f"API Base URL: {config.openrouter_base_url}")
    print(f"Temperature: {config.temperature}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Security Events Logging: {config.log_security_events}")
