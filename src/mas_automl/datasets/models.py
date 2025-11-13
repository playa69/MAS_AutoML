"""Модели данных для работы с датасетами."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DatasetType(str, Enum):
    """Тип задачи машинного обучения."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    UNKNOWN = "unknown"


class DatasetFormat(str, Enum):
    """Формат файла датасета."""

    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    EXCEL = "excel"
    HDF5 = "hdf5"
    ARFF = "arff"
    UNKNOWN = "unknown"


class DatasetMetadata(BaseModel):
    """Метаданные о датасете (локальные файлы)."""

    name: str = Field(..., description="Уникальное имя датасета")
    description: str | None = Field(None, description="Описание датасета")
    path: Path = Field(..., description="Путь к файлу датасета")
    format: DatasetFormat = Field(..., description="Формат файла")
    dataset_type: DatasetType = Field(..., description="Тип задачи ML")
    target_column: str | None = Field(None, description="Название целевой колонки")
    feature_columns: list[str] = Field(default_factory=list, description="Список признаков")
    num_rows: int | None = Field(None, description="Количество строк")
    num_features: int | None = Field(None, description="Количество признаков")
    num_classes: int | None = Field(None, description="Количество классов (для classification)")
    is_processed: bool = Field(False, description="Обработан ли датасет")
    tags: list[str] = Field(default_factory=list, description="Теги для категоризации")
    extra_info: dict[str, Any] = Field(default_factory=dict, description="Дополнительная информация")


class OpenMLDatasetMetadata(BaseModel):
    """Полные метаданные датасета из OpenML."""

    # Основные идентификаторы
    dataset_id: int = Field(..., description="OpenML ID датасета")
    name: str = Field(..., description="Название датасета")
    version: int = Field(..., description="Версия датасета")
    version_label: str | None = Field(None, description="Метка версии")

    # Описание
    description: str | None = Field(None, description="Описание датасета")
    citation: str | None = Field(None, description="Цитирование")
    paper_url: str | None = Field(None, description="URL публикации")
    paper_reference: str | None = Field(None, description="Ссылка на публикацию")

    # Авторы и источники
    creator: str | None = Field(None, description="Создатель датасета")
    contributor: str | None = Field(None, description="Вкладчик")
    collection_date: str | None = Field(None, description="Дата сбора данных")
    upload_date: datetime | None = Field(None, description="Дата загрузки в OpenML")

    # Локализация и лицензия
    language: str | None = Field(None, description="Язык данных")
    licence: str | None = Field(None, description="Лицензия")

    # URL и пути
    url: str | None = Field(None, description="URL датасета")
    original_data_url: str | None = Field(None, description="URL исходных данных")
    minio_url: str | None = Field(None, description="MinIO URL")

    # Формат и структура
    format: str | None = Field(None, description="Формат данных (ARFF, CSV и т.д.)")
    file_id: int | None = Field(None, description="ID файла в OpenML")

    # Целевая переменная и атрибуты
    default_target_attribute: str | None = Field(None, description="Целевая переменная по умолчанию")
    ignore_attribute: list[str] | None = Field(None, description="Атрибуты для игнорирования")
    row_id_attribute: str | None = Field(None, description="Атрибут ID строки")

    # Статистика и качество
    num_rows: int | None = Field(None, description="Количество строк")
    num_features: int | None = Field(None, description="Количество признаков")
    num_classes: int | None = Field(None, description="Количество классов")
    num_missing_values: int | None = Field(None, description="Количество пропущенных значений")
    quality: dict[str, float | str] = Field(default_factory=dict, description="Метрики качества")

    # Классификация задачи
    dataset_type: DatasetType | None = Field(None, description="Тип задачи ML")

    # Метаданные
    tags: list[str] = Field(default_factory=list, description="Теги")
    status: str | None = Field(None, description="Статус датасета")
    visibility: str | None = Field(None, description="Видимость датасета")

    # Дополнительная информация
    extra_info: dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")

    # Локальный путь (если датасет был скачан)
    local_path: Path | None = Field(None, description="Локальный путь к файлу датасета")

