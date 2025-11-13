"""Агент-исследователь новых моделей и подходов."""

from __future__ import annotations

from typing import Any

from .base import Agent, AgentMessage


class ResearchAgent(Agent):
    """Находит новые модели и консолидирует знания для расширения пайплайнов."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("researcher", *args, **kwargs)

    async def handle(self, message: AgentMessage) -> AgentMessage:
        query = message.payload.get("query", "")
        findings = await self._search_literature(query)
        return AgentMessage(
            sender=self.name,
            content="research_summary",
            payload={"findings": findings},
        )

    async def _search_literature(self, query: str) -> list[dict[str, Any]]:
        """Заготовка под интеграцию с LLM и веб-скрейпингом."""
        kb = self.context.knowledge_base
        summary = ""
        if kb is not None:
            summary = await kb.query(
                f"Summarise recent AutoML papers relevant to: {query}",
                context={"agent": self.name},
            )
        return [
            {
                "title": "Adaptive AutoML Ensembles",
                "source": "arXiv",
                "summary": summary or "Требуется реализовать подключение к источникам.",
            }
        ]

