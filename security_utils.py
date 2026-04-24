"""
Security Utilities for Investment Agent System
==============================================
Provides validated input handling, secure configuration, and safe LLM interactions.

Author: Security Team
Version: 1.0.0
"""

import os
import re
import json
import html
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Input validation and security checks"""
    
    # Regex patterns for validation
    TICKER_PATTERN = r"^[A-Z0-9]{1,10}$"
    COMPANY_NAME_PATTERN = r"^[A-Za-z0-9\s\-&.,()]{1,200}$"
    QUERY_PATTERN = r"^[A-Za-z0-9\s\-?.,;:()&'\"]{1,5000}$"
    
    # Max lengths to prevent buffer overflow / DOS
    MAX_TICKER_LENGTH = 10
    MAX_COMPANY_LENGTH = 200
    MAX_QUERY_LENGTH = 5000
    MAX_LLM_RESPONSE_LENGTH = 50000
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate OpenRouter API key format"""
        if not api_key or not isinstance(api_key, str):
            return False
        
        # OpenRouter keys start with sk-or-v1-
        if not api_key.startswith("sk-or-v1-"):
            logger.warning("❌ Invalid API key format (should start with 'sk-or-v1-')")
            return False
        
        # Should be reasonably long (>20 chars total)
        if len(api_key) < 30:
            logger.warning("❌ API key appears too short")
            return False
        
        return True
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate stock ticker symbol"""
        if not ticker or not isinstance(ticker, str):
            return False
        
        ticker = ticker.strip().upper()
        
        if len(ticker) > SecurityValidator.MAX_TICKER_LENGTH:
            logger.warning(f"❌ Ticker too long (>{SecurityValidator.MAX_TICKER_LENGTH})")
            return False
        
        if not re.match(SecurityValidator.TICKER_PATTERN, ticker):
            logger.warning(f"❌ Ticker contains invalid characters: {ticker}")
            return False
        
        return True
    
    @staticmethod
    def validate_company_name(company: str) -> bool:
        """Validate company name"""
        if not company or not isinstance(company, str):
            return False
        
        company = company.strip()
        
        if len(company) > SecurityValidator.MAX_COMPANY_LENGTH:
            logger.warning(f"❌ Company name too long (>{SecurityValidator.MAX_COMPANY_LENGTH})")
            return False
        
        if not re.match(SecurityValidator.COMPANY_NAME_PATTERN, company):
            logger.warning(f"❌ Company name contains invalid characters")
            return False
        
        return True
    
    @staticmethod
    def validate_query(query: str) -> bool:
        """Validate user query for prompt injection attempts"""
        if not query or not isinstance(query, str):
            return False
        
        query = query.strip()
        
        if len(query) > SecurityValidator.MAX_QUERY_LENGTH:
            logger.warning(f"❌ Query too long (>{SecurityValidator.MAX_QUERY_LENGTH})")
            return False
        
        if not re.match(SecurityValidator.QUERY_PATTERN, query):
            logger.warning(f"❌ Query contains invalid characters (possible injection attempt)")
            return False
        
        # Additional prompt injection checks
        dangerous_patterns = [
            r"ignore.*instructions",
            r"system.*prompt",
            r"bypass",
            r"override",
        ]
        
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"❌ Potential prompt injection detected: {pattern}")
                return False
        
        return True
    
    @staticmethod
    def validate_file_upload(file_path: Path, max_size_mb: int = 50) -> bool:
        """Validate uploaded file safety"""
        if not file_path.exists():
            logger.warning("❌ File does not exist")
            return False
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            logger.warning(f"❌ File too large ({file_size_mb:.1f}MB > {max_size_mb}MB)")
            return False
        
        # Check file extension (PDFs only)
        if file_path.suffix.lower() != ".pdf":
            logger.warning(f"❌ Invalid file type: {file_path.suffix} (PDF only)")
            return False
        
        # Check for path traversal attempts
        if ".." in str(file_path):
            logger.warning("❌ Path traversal detected in file path")
            return False
        
        return True


class SecureResponseHandler:
    """Safe handling of LLM responses"""
    
    @staticmethod
    def sanitize_html(text: str, max_length: int = 50000) -> str:
        """Escape HTML to prevent XSS"""
        if not text or not isinstance(text, str):
            return ""
        
        # Enforce max length
        text = text[:max_length]
        
        # Escape all HTML characters
        return html.escape(text)
    
    @staticmethod
    def parse_json_safely(response: str, max_length: int = 50000) -> Dict[str, Any]:
        """Safely parse JSON response with strict validation"""
        if not response or not isinstance(response, str):
            return {"error": "Empty response", "parse_error": True}
        
        # Enforce response length limit
        if len(response) > max_length:
            logger.warning(f"⚠️ Response too long ({len(response)} > {max_length}), truncating")
            response = response[:max_length]
        
        try:
            # Try standard JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try Markdown code block extraction
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 3:
                    json_str = parts[1].strip()
                    # Remove language identifier (e.g., "python")
                    lines = json_str.split('\n')
                    json_str = '\n'.join(lines[1:] if not lines[0][0] == '{' else lines)
                    return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        except (IndexError, AttributeError):
            pass
        
        # Last resort: try non-greedy regex (safer than greedy)
        try:
            # Use non-greedy matching with length limit
            match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', response)
            if match:
                json_str = match.group(0)
                if len(json_str) < max_length:
                    return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # All parsing failed - return error
        logger.warning("❌ Failed to parse LLM response as JSON")
        return {
            "error": "JSON parsing failed",
            "parse_error": True,
            "response_preview": response[:200]  # Safe preview only
        }
    
    @staticmethod
    def validate_json_structure(data: Dict, required_keys: List[str]) -> bool:
        """Validate that JSON has required structure"""
        if not isinstance(data, dict):
            return False
        
        for key in required_keys:
            if key not in data:
                logger.warning(f"❌ Missing required key in response: {key}")
                return False
        
        return True


class SecureConfigHandler:
    """Secure configuration management"""
    
    @staticmethod
    def load_api_key(env_var: str = "OPENROUTER_API_KEY") -> str:
        """Load and validate API key from environment"""
        api_key = os.getenv(env_var, "").strip()
        
        if not api_key:
            raise ValueError(
                f"❌ {env_var} not found in environment. "
                "Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-..."
            )
        
        if not SecurityValidator.validate_api_key(api_key):
            raise ValueError(
                f"❌ {env_var} has invalid format. "
                "Should start with 'sk-or-v1-' and be at least 30 characters."
            )
        
        logger.info(f"✅ API key validated (length: {len(api_key)} chars)")
        return api_key
    
    @staticmethod
    def get_safe_config_dict() -> Dict[str, Any]:
        """Get configuration dictionary without exposing secrets"""
        return {
            "openrouter_base_url": "https://openrouter.ai/api/v1",
            "temperature": float(os.getenv("TEMPERATURE", "0.3")),
            "max_tokens": int(os.getenv("MAX_TOKENS", "4096")),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "1500")),
            "top_k_retrieval": int(os.getenv("TOP_K_RETRIEVAL", "8")),
        }


class SecurityEventLogger:
    """Audit logging for security events"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = Path(log_file)
        self.logger = logging.getLogger("security")
        
        # Create file handler for security events
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_validation_failure(self, field: str, reason: str):
        """Log input validation failure"""
        self.logger.warning(f"VALIDATION_FAILURE: {field} - {reason}")
    
    def log_injection_attempt(self, field: str, pattern: str):
        """Log suspected injection attempt"""
        self.logger.error(f"INJECTION_ATTEMPT: {field} matched {pattern}")
    
    def log_api_key_usage(self, agent: str, success: bool):
        """Log API key usage (without exposing key)"""
        status = "SUCCESS" if success else "FAILURE"
        self.logger.info(f"API_CALL: agent={agent} status={status}")
    
    def log_file_access(self, file_path: str, action: str):
        """Log file access"""
        self.logger.info(f"FILE_ACCESS: {action} - {file_path}")


# Convenience instances
security_validator = SecurityValidator()
response_handler = SecureResponseHandler()
config_handler = SecureConfigHandler()
event_logger = SecurityEventLogger()
