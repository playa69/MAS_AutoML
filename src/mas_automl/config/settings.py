"""Конфигурационные модели и загрузка настроек проекта."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root(start_path: Path | None = None) -> Path:
    """Найти корень проекта, ища файлы-маркеры (pyproject.toml, .git, README.md).
    
    Args:
        start_path: Начальный путь для поиска. Если None, используется текущий файл.
    
    Returns:
        Путь к корню проекта.
    """
    if start_path is None:
        # Используем путь к текущему файлу как отправную точку
        # Из src/mas_automl/config/settings.py поднимаемся до корня проекта
        start_path = Path(__file__).parent.parent.parent.parent
    
    current = Path(start_path).resolve()
    
    # Маркеры корня проекта
    markers = ["pyproject.toml", ".git", "README.md", "setup.py"]
    
    # Поднимаемся вверх по директориям, пока не найдём маркер
    for parent in [current] + list(current.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent
    
    # Если маркер не найден, возвращаем текущую директорию
    return current


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


class LLMConfig(BaseModel):
    """Конфигурация для подключения LLM. Может быть использована для каждого агента отдельно."""

    provider: Literal["openai", "azure", "anthropic", "local", "openrouter"] = "openai"
    model: str = "google/gemini-2.0-flash-001"
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls, model: str | None = None) -> LLMConfig:
        """Создать конфигурацию из переменных окружения.
        
        Args:
            model: Модель для использования. Если не указана, используется дефолтная.
        """
        import os
        return cls(
            provider="openrouter",
            model=model or "google/gemini-2.0-flash-001",
            api_key=os.getenv("API_KEY") or os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("BASE_URL") or "https://openrouter.ai/api/v1",
        )


class AgentsConfig(BaseModel):
    """Конфигурация LLM для агентов. Каждый агент может иметь свою модель."""

    paper_researcher: LLMConfig = Field(
        default_factory=lambda: LLMConfig(model="google/gemini-2.0-flash-001")
    )


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
    llm: LLMConfig = Field(default_factory=LLMConfig.from_env)
    agents: AgentsConfig = AgentsConfig()
    workspace_root: Path = Field(default_factory=lambda: find_project_root())
    cache_dir: Path = Field(default_factory=lambda: find_project_root() / ".cache")


settings = MASSettings()

