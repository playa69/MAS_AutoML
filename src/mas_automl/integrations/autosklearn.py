"""Адаптер для Auto-sklearn 2.0."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Any

from ..pipelines.registry import PipelineStep
from ..services.orchestrator import FrameworkAdapter, FrameworkResult, dummy_execute

__all__ = ["autosklearn_adapter"]


def autosklearn_adapter() -> FrameworkAdapter:
    execute: Callable[[PipelineStep], Awaitable[FrameworkResult]] = dummy_execute
    try:
        import numpy as np
        import pandas as pd
        from autosklearn.classification import AutoSklearnClassifier  # type: ignore
        from autosklearn.metrics import accuracy as ask_accuracy  # type: ignore
        from autosklearn.regression import AutoSklearnRegressor  # type: ignore
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score

        async def run(step: PipelineStep) -> FrameworkResult:
            dataset: dict[str, Any] | None = step.params.get("dataset")  # type: ignore[assignment]
            if dataset is None:
                raise ValueError("Для запуска Auto-sklearn требуется dataset в step.params['dataset'].")

            try:
                train_df = dataset["train"]
                test_df = dataset["test"]
            except KeyError as exc:  # pragma: no cover
                raise ValueError("dataset должен содержать ключи 'train' и 'test'.") from exc

            if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
                raise TypeError("train и test в dataset должны быть pandas.DataFrame.")

            label_column = dataset.get("label") or dataset.get("target")
            if label_column is None:
                raise ValueError("dataset должен содержать имя столбца с целевой переменной ('label').")

            if label_column not in train_df.columns or label_column not in test_df.columns:
                raise ValueError(f"Столбец {label_column!r} отсутствует в train/test данных.")

            problem_type = dataset.get("problem_type") or step.objective or "classification"
            time_left = int(step.params.get("time_left_for_this_task", 30))
            per_run_time_limit = int(step.params.get("per_run_time_limit", max(5, time_left // 2 or 1)))
            artifact_path = step.params.get("artifacts_path")
            metric_name = step.params.get("metric")

            def _fit_and_evaluate() -> FrameworkResult:
                X_train = train_df.drop(columns=[label_column]).to_numpy()
                y_train = train_df[label_column].to_numpy()
                X_test = test_df.drop(columns=[label_column]).to_numpy()
                y_test = test_df[label_column].to_numpy()

                if problem_type not in {"classification", "regression"}:
                    raise ValueError("problem_type должен быть 'classification' или 'regression'.")

                if problem_type == "regression":
                    model = AutoSklearnRegressor(
                        time_left_for_this_task=time_left,
                        per_run_time_limit=per_run_time_limit,
                        n_jobs=1,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    metrics = {
                        "r2": float(r2_score(y_test, y_pred)),
                        "rmse": float(np.sqrt(mse)),
                    }
                    primary_metric = metric_name or "r2"
                else:
                    automl_metric = ask_accuracy
                    model = AutoSklearnClassifier(
                        time_left_for_this_task=time_left,
                        per_run_time_limit=per_run_time_limit,
                        metric=automl_metric,
                        n_jobs=1,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = {
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    }
                    primary_metric = metric_name or "accuracy"

                metrics["primary_metric"] = float(metrics.get(primary_metric, next(iter(metrics.values()))))

                return FrameworkResult(
                    framework="autosklearn",
                    step=step.step,
                    metrics=metrics,
                    artifacts_path=artifact_path,
                )

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _fit_and_evaluate)

        execute = run
    except Exception:
        pass

    return FrameworkAdapter(name="autosklearn", execute=execute)

