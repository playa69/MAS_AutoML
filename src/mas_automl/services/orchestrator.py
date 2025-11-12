"""Оркестратор выполнения гибридных пайплайнов."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict

from pydantic import BaseModel, Field

from ..pipelines.registry import HybridPipeline, PipelineStep


class FrameworkResult(BaseModel):
    framework: str
    step: str
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts_path: str | None = None


class OrchestrationResult(BaseModel):
    pipeline_name: str
    results: list[FrameworkResult]
    combined_metrics: dict[str, float] = Field(default_factory=dict)


class FrameworkAdapter(BaseModel):
    """Спецификация адаптера под AutoML-фреймворк."""

    name: str
    execute: Callable[[PipelineStep], Awaitable[FrameworkResult]]

    class Config:
        arbitrary_types_allowed = True


@dataclass
class PipelineOrchestrator:
    adapters: Dict[str, FrameworkAdapter] = field(default_factory=dict)

    def register_adapter(self, adapter: FrameworkAdapter) -> None:
        self.adapters[adapter.name] = adapter

    async def run_pipeline(self, pipeline: HybridPipeline) -> OrchestrationResult:
        tasks = []
        for step in pipeline.steps:
            adapter = self.adapters.get(step.framework)
            if adapter is None:
                raise KeyError(f"Не найден адаптер для фреймворка {step.framework}.")
            tasks.append(adapter.execute(step))

        results = await asyncio.gather(*tasks)
        combined_metrics = self._combine_metrics(results)
        return OrchestrationResult(
            pipeline_name=pipeline.name,
            results=list(results),
            combined_metrics=combined_metrics,
        )

    def _combine_metrics(self, results: list[FrameworkResult]) -> dict[str, float]:
        """Простейший способ объединения метрик."""
        aggregated: dict[str, list[float]] = {}
        for result in results:
            for metric, value in result.metrics.items():
                aggregated.setdefault(metric, []).append(value)
        return {metric: sum(values) / len(values) for metric, values in aggregated.items()}


async def dummy_execute(step: PipelineStep) -> FrameworkResult:
    """Заглушка для демонстраций и unit-тестов."""
    await asyncio.sleep(0.1)
    return FrameworkResult(
        framework=step.framework,
        step=step.step,
        metrics={"score": 0.5},
        artifacts_path=f"artifacts/{step.framework}/{step.step}",
    )

