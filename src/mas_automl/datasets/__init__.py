"""Модуль для работы с датасетами."""

from __future__ import annotations

from .manager import DatasetManager
from .models import (
    DatasetFormat,
    DatasetMetadata,
    DatasetType,
    OpenMLDatasetMetadata,
)

__all__ = [
    "DatasetManager",
    "DatasetMetadata",
    "OpenMLDatasetMetadata",
    "DatasetType",
    "DatasetFormat",
]

