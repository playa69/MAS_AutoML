"""Конфигурационные модели и загрузка настроек проекта."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AutoMLFrameworkConfig(BaseSettings):
    """Настройки для отдельного AutoML-фреймворка."""

    name: str
    preset: str | None = None
    enabled: bool = True
    timeout_minutes: int = 60
    extra_params: dict[str, str | int | float] = Field(default_factory=dict)


class BenchmarkConfig(BaseSettings):
    """Описывает набор задач для проверки пайплайнов."""

    suites: Sequence[str] = ("classification", "regression")
    aml_benchmark_root: Path = Path("data/amlb")
    max_concurrency: int = 1


class LLMConfig(BaseSettings):
    """Настройки для подключения LLM."""

    provider: Literal["openai", "azure", "anthropic", "local"] = "openai"
    model: str = "gpt-4.1"
    temperature: float = 0.2
    max_tokens: int = 2048
    api_key: str | None = None


class MASSettings(BaseSettings):
    """Высокоуровневые настройки мультиагентной системы."""

    model_config = SettingsConfigDict(env_prefix="MAS_", env_file=".env", env_nested_delimiter="__")

    frameworks: list[AutoMLFrameworkConfig] = Field(
        default_factory=lambda: [
            AutoMLFrameworkConfig(name="autogluon", preset="best_quality"),
            AutoMLFrameworkConfig(name="autosklearn", preset="automl"),
            AutoMLFrameworkConfig(name="fedot", preset="auto"),
        ]
    )
    benchmark: BenchmarkConfig = BenchmarkConfig()
    llm: LLMConfig = LLMConfig()
    workspace_root: Path = Path(".").resolve()
    cache_dir: Path = workspace_root / ".cache"


settings = MASSettings()

