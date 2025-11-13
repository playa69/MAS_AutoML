from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config.settings import settings


_TARGET_GUESSES = ("target", "class", "label", "y")


@dataclass(slots=True)
class DatasetManifest:
    dataset_id: str
    source_url: str
    commit: str | None
    local_path: Path
    shape: tuple[int, int]
    target_column: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_url": self.source_url,
            "commit": self.commit,
            "local_path": str(self.local_path),
            "shape": list(self.shape),
            "target_column": self.target_column,
        }


def _find_candidate_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)


def _guess_target_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for guess in _TARGET_GUESSES:
        for col in cols:
            if col.lower() == guess:
                return col
    # heuristic: last column named like outcome or ending with _target/_label
    regexes = (r"(outcome|response)$", r"(_target|_label)$")
    for rx in regexes:
        for col in cols:
            if re.search(rx, col.lower()):
                return col
    # fallback: last column
    return cols[-1]


def load_amlb_dataset(dataset_id: str, *, target: str | None = None) -> DatasetManifest:
    """Resolve AMLB dataset by id under configured root; pick the largest CSV."""
    root = (settings.benchmark.aml_benchmark_root / dataset_id).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' was not found under {root.parent}")
    csvs = _find_candidate_csvs(root)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found for dataset '{dataset_id}' in {root}")
    data_path = csvs[0]
    df = pd.read_csv(data_path)
    tgt = target or _guess_target_column(df)
    if tgt not in df.columns:
        raise ValueError(f"Target column '{tgt}' not found in dataset '{dataset_id}'")
    return DatasetManifest(
        dataset_id=dataset_id,
        source_url=data_path.as_uri(),
        commit=None,
        local_path=data_path,
        shape=(df.shape[0], df.shape[1]),
        target_column=tgt,
    )


