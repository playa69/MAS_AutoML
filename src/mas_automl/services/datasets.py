from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

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


def _parse_openml_id_from_url(url: str) -> int | None:
    try:
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(url)
        if "openml.org" not in parsed.netloc:
            return None
        # common patterns: ?id=31 or path /data/31 or /d/31 etc. We support query param first.
        q = parse_qs(parsed.query)
        if "id" in q and len(q["id"]) > 0:
            return int(q["id"][0])
        # try to grab last integer segment
        parts = [p for p in parsed.path.split("/") if p.strip()]
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
    except Exception:
        return None
    return None


def _load_openml_by_id(openml_id: int, *, target: str | None = None, source_url: str | None = None) -> DatasetManifest:
    ds = fetch_openml(data_id=openml_id, as_frame=True)
    X = ds.data
    y = ds.target
    # harmonize target name
    tgt_name = target or (y.name if hasattr(y, "name") and y.name else "target")
    if getattr(y, "name", None) != tgt_name:
        y = y.rename(tgt_name)
    df = pd.concat([X, y], axis=1)
    # persist under aml_benchmark_root
    root = (settings.benchmark.aml_benchmark_root / f"openml_{openml_id}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "data.csv"
    df.to_csv(data_path, index=False)
    return DatasetManifest(
        dataset_id=f"openml_{openml_id}",
        source_url=source_url or f"openml:data_id={openml_id}",
        commit=None,
        local_path=data_path,
        shape=(df.shape[0], df.shape[1]),
        target_column=tgt_name,
    )


def load_amlb_dataset(dataset_id: str, *, target: str | None = None) -> DatasetManifest:
    """Resolve AMLB dataset by id:
    - http(s) OpenML URL → fetch via sklearn and cache under data/amlb/openml_<id>/
    - identifiers: openml:<id>, openml/<id>, numeric '<id>' → fetch via sklearn
    - otherwise: treat as local folder under aml_benchmark_root and pick the largest CSV
    """
    # URL case
    if dataset_id.startswith("http://") or dataset_id.startswith("https://"):
        maybe_id = _parse_openml_id_from_url(dataset_id)
        if maybe_id is not None:
            return _load_openml_by_id(maybe_id, target=target, source_url=dataset_id)
        raise ValueError(f"Unsupported dataset URL: {dataset_id}")
    # openml:<id> or openml/<id>
    m = re.match(r"^openml[:/](\d+)$", dataset_id)
    if m:
        return _load_openml_by_id(int(m.group(1)), target=target)
    # pure numeric → assume OpenML id
    if dataset_id.isdigit():
        return _load_openml_by_id(int(dataset_id), target=target)

    # Local folder
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


