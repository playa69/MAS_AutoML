"""Агент-исполнитель, оркестрирующий вызовы AutoML-фреймворков."""

from __future__ import annotations

from typing import Any

from .base import Agent, AgentMessage
from ..pipelines.registry import HybridPipeline, PipelineStep
from ..services.orchestrator import OrchestrationResult, PipelineOrchestrator


class ExecutorAgent(Agent):
    """Запускает гибридные пайплайны и собирает результаты."""

    def __init__(self, orchestrator: PipelineOrchestrator, *args: Any, **kwargs: Any) -> None:
        super().__init__("executor", *args, **kwargs)
        self._orchestrator = orchestrator

    async def handle(self, message: AgentMessage) -> AgentMessage:
        plan_payload = message.payload.get("plan")
        if not plan_payload:
            raise ValueError("ExecutorAgent получил пустой план.")

        pipeline = HybridPipeline(
            name=plan_payload.get("name", "hybrid_pipeline"),
            steps=[PipelineStep(**step) for step in plan_payload.get("steps", [])],
        )

        result = await self._orchestrator.run_pipeline(pipeline)
        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            content="execution_result",
            payload={"result": result.model_dump()},
        )

