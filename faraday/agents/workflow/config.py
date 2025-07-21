"""
Configuration for the workflow components.
"""

import os
from langchain_openai import ChatOpenAI

# Initialize the language model
llm = ChatOpenAI(
    base_url=os.environ.get("OPENAI_API_BASE") or "http://localhost:11434/v1/",
    api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
    model=os.environ.get("OPENAI_MODEL") or "qwen3:32b",
)
