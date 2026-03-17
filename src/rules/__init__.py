"""
Rule-based system for agent behavior.

Provides comprehensive rule files (10k+ lines) that agents load
to improve extraction quality and reduce hallucination.
"""

from .rule_loader import RuleLoader, load_rules_for_agent

__all__ = ["RuleLoader", "load_rules_for_agent"]
