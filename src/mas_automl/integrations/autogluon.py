"""Адаптер для AutoGluon."""

from __future__ import annotations

from typing import Awaitable, Callable

from ..pipelines.registry import PipelineStep
from ..services.orchestrator import FrameworkAdapter, FrameworkResult, dummy_execute


def autogluon_adapter() -> FrameworkAdapter:
    execute: Callable[[PipelineStep], Awaitable[FrameworkResult]] = dummy_execute
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore

        async def run(step: PipelineStep) -> FrameworkResult:
            # TODO: добавить реальную интеграцию с AutoGluon
            _ = TabularPredictor  # чтобы линтер не ругался
            return await dummy_execute(step)

        execute = run
    except Exception:
        pass

    return FrameworkAdapter(name="autogluon", execute=execute)

