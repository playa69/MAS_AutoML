import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mas_automl.integrations.autogluon import autogluon_adapter
from mas_automl.integrations.autosklearn import autosklearn_adapter
from mas_automl.integrations.fedot import fedot_adapter
from mas_automl.pipelines.registry import PipelineStep


def _build_classification_dataset() -> dict[str, pd.DataFrame]:
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1.5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    columns = [f"f{i}" for i in range(X.shape[1])]
    train_df = pd.DataFrame(X_train, columns=columns)
    train_df["target"] = y_train

    test_df = pd.DataFrame(X_test, columns=columns)
    test_df["target"] = y_test

    return {
        "train": train_df,
        "test": test_df,
        "label": "target",
        "problem_type": "classification",
    }


@pytest.mark.asyncio
async def test_autosklearn_adapter_runs_on_synthetic_data() -> None:
    pytest.importorskip("autosklearn.classification")

    dataset = _build_classification_dataset()
    step = PipelineStep(
        step="train",
        framework="autosklearn",
        params={
            "dataset": dataset,
            "time_left_for_this_task": 20,
            "per_run_time_limit": 5,
        },
    )

    adapter = autosklearn_adapter()
    result = await adapter.execute(step)

    assert result.framework == "autosklearn"
    assert "primary_metric" in result.metrics


@pytest.mark.asyncio
async def test_autogluon_adapter_runs_on_synthetic_data() -> None:
    pytest.importorskip("autogluon.tabular")

    dataset = _build_classification_dataset()
    step = PipelineStep(
        step="train",
        framework="autogluon",
        preset="medium_quality",
        params={
            "dataset": dataset,
            "time_limit": 20,
        },
    )

    adapter = autogluon_adapter()
    result = await adapter.execute(step)

    assert result.framework == "autogluon"
    assert result.metrics


@pytest.mark.asyncio
async def test_fedot_adapter_runs_on_synthetic_data() -> None:
    pytest.importorskip("fedot")

    dataset = _build_classification_dataset()
    step = PipelineStep(
        step="train",
        framework="fedot",
        preset="light_tun",
        params={
            "dataset": dataset,
            "timeout": 0.5,
        },
    )

    adapter = fedot_adapter()
    result = await adapter.execute(step)

    assert result.framework == "fedot"
    assert "primary_metric" in result.metrics


