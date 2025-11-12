"""Регистрация и описание гибридных пайплайнов."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional


@dataclass(slots=True)
class PipelineStep:
    """Описывает шаг пайплайна AutoML."""

    step: str
    framework: str
    preset: str | None = None
    dataset: str | None = None
    objective: str | None = None
    params: dict[str, str | float | int] = field(default_factory=dict)


@dataclass(slots=True)
class HybridPipeline:
    """Составной пайплайн из нескольких AutoML-фреймворков."""

    name: str
    steps: List[PipelineStep] = field(default_factory=list)

    def add_step(self, step: PipelineStep) -> None:
        self.steps.append(step)


class PipelineRegistry:
    """Реестр заранее определённых пайплайнов."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[[], HybridPipeline]] = {}

    def register(self, name: str, factory: Callable[[], HybridPipeline]) -> None:
        if name in self._registry:
            raise ValueError(f"Пайплайн с именем {name} уже зарегистрирован.")
        self._registry[name] = factory

    def get(self, name: str) -> HybridPipeline:
        if name not in self._registry:
            raise KeyError(f"Пайплайн {name} не найден.")
        return self._registry[name]()

    def list(self) -> Iterable[str]:
        return self._registry.keys()


registry = PipelineRegistry()


def default_hybrid_pipeline() -> HybridPipeline:
    pipeline = HybridPipeline(name="default_hybrid")
    pipeline.add_step(
        PipelineStep(step="preprocess", framework="autogluon", preset="medium_quality")
    )
    pipeline.add_step(PipelineStep(step="train", framework="autosklearn", objective="auto"))
    pipeline.add_step(PipelineStep(step="blend", framework="fedot", preset="auto"))
    return pipeline


registry.register("default", default_hybrid_pipeline)

