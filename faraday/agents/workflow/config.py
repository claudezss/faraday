"""
Configuration for the workflow components.
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI

# Default LLM configuration
_default_llm = ChatOpenAI(
    base_url=os.environ.get("OPENAI_API_BASE") or "http://localhost:11434/v1/",
    api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
    model=os.environ.get("OPENAI_MODEL") or "qwen3:32b",
)

# Current active LLM instance (can be overridden)
_current_llm: Optional[ChatOpenAI] = None


def get_llm():
    """Get the current LLM instance."""
    return _current_llm if _current_llm is not None else _default_llm


def set_llm(llm_instance: ChatOpenAI):
    """Set a custom LLM instance for the workflow."""
    global _current_llm
    _current_llm = llm_instance


def reset_llm():
    """Reset to the default LLM configuration."""
    global _current_llm
    _current_llm = None


# Keep the old interface for backward compatibility
llm = _default_llm
