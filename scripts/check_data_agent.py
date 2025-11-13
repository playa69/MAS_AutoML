from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# allow running without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mas_automl.agents import DataAgent, AgentMessage


def _ensure_demo_dataset() -> str:
    """Create a small demo dataset under data/amlb/demo_agent if not present."""
    root = Path("data/amlb/demo_agent")
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "data.csv"
    if not csv_path.exists():
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(size=n)
        x2 = rng.normal(loc=2.0, scale=3.0, size=n)
        cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
        # high-cardinality column (~0.6 unique ratio)
        user_ids = np.array([f"u{int(i)}" for i in rng.choice(800, size=n, replace=False)])
        # timestamp column
        ts = pd.to_datetime("2021-01-01") + pd.to_timedelta(np.arange(n), unit="h")
        # mildly imbalanced target
        y = (rng.random(size=n) > 0.92).astype(int)
        # some text column
        text = np.array([f"free text sample number {i} with value {x1_i:.3f}" for i, x1_i in enumerate(x1)])
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "cat": cat,
                "user_id": user_ids,
                "timestamp": ts,
                "text": text,
                "target": y,
            }
        )
        # introduce some missingness
        df.loc[df.sample(frac=0.1, random_state=42).index, "x2"] = np.nan
        df.to_csv(csv_path, index=False)
    return "demo_agent"


async def _run(dataset_id: str, target: str | None = None) -> dict[str, Any]:
    agent = DataAgent()
    msg = AgentMessage(
        sender="script",
        recipient="AetherML",
        content="prepare",
        payload={"dataset_id": dataset_id, "target": target},
    )
    res = await agent.handle(msg)
    return res.payload


def main() -> None:
    dataset_id = _ensure_demo_dataset()
    payload = asyncio.run(_run(dataset_id))
    print("\n=== DataAgent Result (keys) ===")
    print(sorted(payload.keys()))
    print("\nmanifest:", json.dumps(payload["manifest"], indent=2))
    print("\nvalidation_report_url:", payload["validation_report_url"])
    print("split_metadata_url:", payload["split_metadata_url"])
    print("metafeatures_url:", payload["metafeatures_url"])
    print("run_metadata_url:", payload["run_metadata_url"])
    print("code_agent_recommendation_url:", payload.get("code_agent_recommendation_url"))
    print("\n=== Code Agent Recommendation (summary) ===")
    print(payload["code_agent_recommendation"]["summary"])
    print("\nconfidence:", payload["code_agent_recommendation"]["confidence"])
    print("\nexample pipeline snippet:\n")
    print(payload["code_agent_recommendation"]["example_pipeline_snippet"])


if __name__ == "__main__":
    main()


