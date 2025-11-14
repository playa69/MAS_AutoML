from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# allow running without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mas_automl.services.llm import try_generate_code_agent_recommendation_llm, llm_healthcheck


def main() -> None:
    if not (os.getenv("GIGACHAT_API_KEY")):
        print("GIGACHAT_API_KEY not set; export it or add to .env")
        raise SystemExit(2)

    hc = llm_healthcheck()
    print("Healthcheck:", json.dumps(hc, ensure_ascii=False, indent=2))
    if not hc.get("chat_ok"):
        print("Minimal chat failed; fix connectivity/model before recipe generation.")
        raise SystemExit(1)

    # minimal payloads
    validation_report = {
        "dataset": {"rows": 100, "cols": 5, "target": "y", "task_type": "classification", "duplicates": 0},
        "feature_types": {"numeric": ["x1", "x2"], "categorical": ["c1"], "datetime": [], "text": []},
        "missing_per_column": {"x1": 0.0, "x2": 0.0, "c1": 0.0},
        "high_cardinality": [],
        "temporal_columns": [],
        "leakage_suspects": [],
        "warnings": {"high_missing_cols": [], "imbalance": None, "outliers": None},
    }
    metafeatures = {
        "dataset": {"n_rows": 100, "n_cols": 5, "feature_types": validation_report["feature_types"]},
        "features": {
            "x1": {"dtype": "float64", "n_unique": 95, "missing_rate": 0.0, "entropy": 3.0},
            "x2": {"dtype": "float64", "n_unique": 90, "missing_rate": 0.0, "entropy": 3.0},
            "c1": {"dtype": "object", "n_unique": 5, "missing_rate": 0.0, "entropy": 2.0},
        },
    }

    recipe = try_generate_code_agent_recommendation_llm(validation_report, metafeatures)
    if recipe is None:
        print("LLM FAIL (returned None). Check provider/base_url/key and network.")
        raise SystemExit(1)
    print("LLM OK. Keys:", sorted(recipe.keys()))
    print("\nSummary:", recipe.get("summary", "")[:200])
    print("\nConfidence:", recipe.get("confidence"))
    print("\nSnippet (first 200 chars):\n", (recipe.get("example_pipeline_snippet") or "")[:200])
    # Optionally dump JSON to stdout:
    # print(json.dumps(recipe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


