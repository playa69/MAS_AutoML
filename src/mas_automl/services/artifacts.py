from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config.settings import settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ArtifactLocation:
    root: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def uri(self, *parts: str) -> str:
        return self.path(*parts).as_uri()


class ArtifactStore:
    """Simple local artifact manager. Returns file:// URIs."""

    def __init__(self, base_dir: Path | None = None) -> None:
        workspace = settings.workspace_root
        self.base_dir = (base_dir or (workspace / "data" / "processed")).resolve()

    def create_run_location(self, namespace: str, dataset_id: str) -> ArtifactLocation:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_dir / namespace / dataset_id / ts
        loc = ArtifactLocation(run_dir)
        loc.ensure()
        return loc

    # Writers
    def save_json(self, location: ArtifactLocation, rel_path: str, payload: dict[str, Any]) -> str:
        path = location.path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path.as_uri()

    def save_dataframe_csv(self, location: ArtifactLocation, rel_path: str, df: pd.DataFrame) -> str:
        path = location.path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path.as_uri()

    def materialize_text(self, location: ArtifactLocation, rel_path: str, content: str) -> str:
        path = location.path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path.as_uri()

    # Helpers
    def new_run_metadata(self, dataset_id: str) -> dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "created_at": _now_iso(),
            "workspace_root": str(settings.workspace_root),
        }


