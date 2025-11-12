"""Пакет мультиагентной системы для оркестрации AutoML-пайплайнов."""

from importlib.metadata import version


def get_version() -> str:
    """Вернуть текущую версию пакета."""
    try:
        return version("mas-automl")
    except Exception:
        return "0.1.0"


__all__ = ["get_version"]

