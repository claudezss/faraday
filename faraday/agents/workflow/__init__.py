"""
Workflow package for Faraday agents.

This package contains the LangGraph workflow definition and all workflow nodes
organized for better maintainability and separation of concerns.
"""

from .graph import get_workflow

__all__ = ["get_workflow"]
