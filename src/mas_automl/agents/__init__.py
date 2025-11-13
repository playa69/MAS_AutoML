"""Пакет агентов мультиагентной системы."""

from .base import Agent, AgentContext, AgentMessage, KnowledgeBase
from .executor import ExecutorAgent
from .planner import PlannerAgent
from .researcher import ResearchAgent
from .paper_research import PaperResearchAgent

__all__ = [
    "Agent",
    "AgentContext",
    "AgentMessage",
    "KnowledgeBase",
    "ExecutorAgent",
    "PaperResearchAgent",
    "PlannerAgent",
    "ResearchAgent",
]

