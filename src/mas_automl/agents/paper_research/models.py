"""Pydantic модели для PaperResearchAgent."""

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


def generate_slug(name: str) -> str:
    """
    Преобразует имя фреймворка в slug:
    - нижний регистр
    - пробелы и спецсимволы → "-"
    - удаляет повторяющиеся дефисы
    """
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug


class PaperInfo(BaseModel):
    """Информация о научной статье."""

    title: str
    authors: list[str]
    abstract: str
    url: str
    source: Literal["arxiv"]
    pdf_url: str | None = None
    published_date: str | None = None


class FrameworkInfo(BaseModel):
    """
    Детальная информация об AutoML-фреймворке.
    С auto-generated slug.
    """

    name: str = Field(
        ...,
        description="Название фреймворка"
    )
    slug: str = Field(
        default="",
        description="Автоматически генерируемый идентификатор (URL-friendly)"
    )
    description: str = Field(
        ...,
        description="Описание фреймворка, его назначение и основные возможности"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Сильные стороны фреймворка"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Слабые стороны и ограничения фреймворка"
    )
    categories: List[str] = Field(
        default_factory=list,
        description=(
            "Категории: 'tabular', 'time-series', 'multimodal', 'NAS', 'HPO', "
            "'pipeline-automation', 'LLM-based AutoML' и т.п."
        )
    )
    architecture: List[str] = Field(
        default_factory=list,
        description=(
            "Архитектурные компоненты: 'searcher', 'evaluator', 'meta-learner', "
            "'feature-engineering', 'pipeline-generator' и т.п."
        )
    )
    benchmarks: List[str] = Field(
        default_factory=list,
        description="Список бенчмарков и результатов"
    )
    code: List[str] = Field(
        default_factory=list,
        description="Примеры использования фреймворка (фрагменты кода)"
    )
    repository_urls: List[str] = Field(
        default_factory=list,
        description="Ссылки на репозитории (GitHub, GitLab и другие ссылки на исходный код)"
    )
    paper_url: str = Field(
        default="",
        description="URL статьи, из которой извлечена информация"
    )

    @model_validator(mode="after")
    def auto_slug(self):
        """
        Генерирует slug, если не был передан, либо нормализует существующий.
        """
        if not self.slug and self.name:
            self.slug = generate_slug(self.name)
        elif self.slug:
            self.slug = generate_slug(self.slug)
        return self


class FrameworksList(BaseModel):
    """Модель для списка AutoML фреймворков с детальной информацией."""

    frameworks: list[FrameworkInfo] = Field(
        description="Список AutoML фреймворков с детальной информацией о каждом"
    )


class AutoMLRelevance(BaseModel):
    """Модель для определения релевантности статьи запросу об AutoML-фреймворках."""

    label: Literal["ДА", "НЕТ"] = Field(
        ...,
        description="Релевантна ли статья запросу об AutoML-фреймворках."
    )
    reason: Optional[str] = Field(
        None,
        description="Краткое объяснение (1–2 предложения), почему статья не подходит, если label == 'НЕТ'."
    )


class VerificationFeedback(BaseModel):
    """Модель для структурированного фидбека от верификатора качества извлечения."""

    quality_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Оценка качества извлечения от 0 до 100"
    )
    need_rerun: bool = Field(
        ...,
        description="Нужно ли повторить извлечение"
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Краткие причины, почему результат плохой или неполный"
    )
    instructions_for_extraction: str = Field(
        default="",
        description="Конкретные инструкции для улучшения следующей итерации извлечения"
    )

