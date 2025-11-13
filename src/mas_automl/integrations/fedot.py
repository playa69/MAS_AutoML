"""Адаптер для FEDOT."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Any

from ..pipelines.registry import PipelineStep
from ..services.orchestrator import FrameworkAdapter, FrameworkResult, dummy_execute


def fedot_adapter() -> FrameworkAdapter:
    execute: Callable[[PipelineStep], Awaitable[FrameworkResult]] = dummy_execute
    try:
        import numpy as np
        import pandas as pd
        from fedot.api.main import Fedot  # type: ignore
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

        async def run(step: PipelineStep) -> FrameworkResult:
            dataset: dict[str, Any] | None = step.params.get("dataset")  # type: ignore[assignment]
            if dataset is None:
                raise ValueError("Для запуска FEDOT требуется dataset в step.params['dataset'].")

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
            timeout = float(step.params.get("timeout", 0.25))  # timeout в минутах
            preset = step.preset or dataset.get("preset") or "light_tun"
            artifact_path = step.params.get("artifacts_path")
            metric_name = step.params.get("metric")

            def _fit_and_evaluate() -> FrameworkResult:
                features_train = train_df.drop(columns=[label_column])
                target_train = train_df[label_column]
                features_test = test_df.drop(columns=[label_column])
                target_test = test_df[label_column]

                model = Fedot(
                    problem=problem_type,
                    timeout=timeout,
                    preset=preset,
                    verbose_level=0,
                )
                model.fit(features=features_train, target=target_train)
                prediction = model.predict(features=features_test)
                y_pred = np.ravel(prediction.predict)

                if problem_type == "regression":
                    mse = mean_squared_error(target_test, y_pred)
                    metrics = {
                        "r2": float(r2_score(target_test, y_pred)),
                        "rmse": float(np.sqrt(mse)),
                    }
                    primary = metric_name or "r2"
                else:
                    metrics = {
                        "accuracy": float(accuracy_score(target_test, y_pred)),
                        "f1": float(f1_score(target_test, y_pred, average="weighted")),
                    }
                    primary = metric_name or "accuracy"

                metrics["primary_metric"] = float(metrics.get(primary, next(iter(metrics.values()))))

                return FrameworkResult(
                    framework="fedot",
                    step=step.step,
                    metrics=metrics,
                    artifacts_path=artifact_path,
                )

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _fit_and_evaluate)

        execute = run
    except Exception:
        pass

    return FrameworkAdapter(name="fedot", execute=execute)

