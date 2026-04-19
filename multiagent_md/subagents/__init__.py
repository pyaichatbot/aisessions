"""Subagent registry. All roles live as .md files in this directory."""
from .loader import SubagentDef, load_subagent, list_subagents

__all__ = ["SubagentDef", "load_subagent", "list_subagents"]
