"""Сервисы и оркестраторы мультиагентной системы."""

from .orchestrator import (
    FrameworkAdapter,
    FrameworkResult,
    OrchestrationResult,
    PipelineOrchestrator,
    dummy_execute,
)

__all__ = [
    "FrameworkAdapter",
    "FrameworkResult",
    "OrchestrationResult",
    "PipelineOrchestrator",
    "dummy_execute",
]

