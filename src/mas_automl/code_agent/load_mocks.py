from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
MOCKS_DIR = ROOT_DIR / "data" / "mocks"


@lru_cache(maxsize=None)
def _load_json(filename: str) -> Dict[str, Any]:
    path = MOCKS_DIR / filename
    if not path.exists():
        available = ", ".join(sorted(p.name for p in MOCKS_DIR.glob("*.json")))
        raise FileNotFoundError(f"Не найден файл {path}. Доступные моки: {available}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_mock_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Загружает описания данных, метаданные и реестр фреймворков."""
    data_analyze = _load_json("data_analyze.json")
    metadata_analyze = _load_json("metadata_analyze.json")
    framework_registry = _load_json("framework_registry.json")
    return data_analyze, metadata_analyze, framework_registry


__all__ = ["load_mock_inputs"]
