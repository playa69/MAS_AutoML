"""Адаптер для FEDOT."""

from __future__ import annotations

from typing import Awaitable, Callable

from ..pipelines.registry import PipelineStep
from ..services.orchestrator import FrameworkAdapter, FrameworkResult, dummy_execute


def fedot_adapter() -> FrameworkAdapter:
    execute: Callable[[PipelineStep], Awaitable[FrameworkResult]] = dummy_execute
    try:
        from fedot.api.main import Fedot  # type: ignore

        async def run(step: PipelineStep) -> FrameworkResult:
            _ = Fedot
            return await dummy_execute(step)

        execute = run
    except Exception:
        pass

    return FrameworkAdapter(name="fedot", execute=execute)

