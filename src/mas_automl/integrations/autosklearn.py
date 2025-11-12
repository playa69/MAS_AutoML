"""Адаптер для Auto-sklearn 2.0."""

from __future__ import annotations

from typing import Awaitable, Callable

from ..pipelines.registry import PipelineStep
from ..services.orchestrator import FrameworkAdapter, FrameworkResult, dummy_execute
__all__ = ["AutoGluonAdapter", "AutoSklearnAdapter", "FedotAdapter"]


def autosklearn_adapter() -> FrameworkAdapter:
    execute: Callable[[PipelineStep], Awaitable[FrameworkResult]] = dummy_execute
    try:
        import autosklearn.classification  # type: ignore

        async def run(step: PipelineStep) -> FrameworkResult:
            _ = autosklearn.classification
            return await dummy_execute(step)

        execute = run
    except Exception:
        pass

    return FrameworkAdapter(name="autosklearn", execute=execute)

