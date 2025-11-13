"""Базовые типы и абстракции для агентов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class AgentMessage:
    """Сообщение, которым обмениваются агенты."""

    sender: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)


class KnowledgeBase(Protocol):
    """Интерфейс доступа к внешним знаниям (LLM, векторные БД, веб)."""

    async def query(self, question: str, *, context: dict[str, Any] | None = None) -> str: ...

    async def ingest(self, content: str, *, metadata: dict[str, Any] | None = None) -> None: ...


@dataclass(slots=True)
class AgentContext:
    """Контекст, который разделяют агенты."""

    knowledge_base: KnowledgeBase | None = None
    shared_state: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Абстрактный агент с асинхронным API."""

    name: str

    def __init__(self, name: str, context: AgentContext | None = None) -> None:
        self.name = name
        self.context = context or AgentContext()

    @abstractmethod
    async def handle(self, message: AgentMessage) -> AgentMessage:
        """Обработать входящее сообщение и вернуть ответ."""
        raise NotImplementedError

    async def publish(self, message: AgentMessage) -> None:
        """По умолчанию агенты не публикуют сообщения напрямую."""
        _ = message  # для совместимости

