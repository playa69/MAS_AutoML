"""Агент-планировщик пайплайнов AutoML."""

from __future__ import annotations

from typing import Any

from .base import Agent, AgentMessage


class PlannerAgent(Agent):
    """Формирует стратегию гибридного пайплайна и распределяет задачи по агентам."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("planner", *args, **kwargs)

    async def handle(self, message: AgentMessage) -> AgentMessage:
        plan = self._build_plan(message)
        return AgentMessage(
            sender=self.name,
            content="plan_ready",
            payload={"plan": plan},
        )

    def _build_plan(self, message: AgentMessage) -> list[dict[str, Any]]:
        """Простейшая демо-логика формирования пайплайна."""
        objective = message.payload.get("objective", "classification")
        dataset = message.payload.get("dataset", "unknown")
        return [
            {
                "step": "preprocess",
                "framework": "autogluon",
                "preset": "medium_quality_faster_train",
                "dataset": dataset,
            },
            {
                "step": "train",
                "framework": "autosklearn",
                "preset": "automl",
                "objective": objective,
            },
            {
                "step": "ensemble",
                "framework": "fedot",
                "preset": "auto",
            },
        ]

