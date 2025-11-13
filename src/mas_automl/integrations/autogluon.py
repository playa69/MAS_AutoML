"""Адаптер для AutoGluon."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Any

from ..pipelines.registry import PipelineStep
from ..services.orchestrator import FrameworkAdapter, FrameworkResult, dummy_execute


def autogluon_adapter() -> FrameworkAdapter:
    execute: Callable[[PipelineStep], Awaitable[FrameworkResult]] = dummy_execute
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore
        import pandas as pd  # type: ignore

        async def run(step: PipelineStep) -> FrameworkResult:
            dataset: dict[str, Any] | None = step.params.get("dataset")  # type: ignore[assignment]
            if dataset is None:
                raise ValueError("Для запуска AutoGluon требуется dataset в step.params['dataset'].")

            try:
                train_df = dataset["train"]
                test_df = dataset["test"]
            except KeyError as exc:  # pragma: no cover - защита от неправильного словаря
                raise ValueError("dataset должен содержать ключи 'train' и 'test'.") from exc

            if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
                raise TypeError("train и test в dataset должны быть pandas.DataFrame.")

            label_column = dataset.get("label") or dataset.get("target")
            if label_column is None:
                raise ValueError("dataset должен содержать имя столбца с целевой переменной ('label').")

            if label_column not in train_df.columns or label_column not in test_df.columns:
                raise ValueError(f"Столбец {label_column!r} отсутствует в train/test данных.")

            # Определяем problem_type: AutoGluon ожидает "binary" или "multiclass" (а не "classification")
            provided_problem_type: str | None = (dataset.get("problem_type") or step.objective)  # type: ignore[assignment]
            # Если явно передали "classification", маппируем на binary/multiclass по числу классов
            if provided_problem_type in (None, "classification"):
                num_classes = int(train_df[label_column].nunique())
                problem_type = "binary" if num_classes == 2 else "multiclass"
            else:
                problem_type = provided_problem_type
            presets = step.preset or dataset.get("presets") or step.params.get("presets") or "medium_quality"
            time_limit = int(step.params.get("time_limit", 30))
            eval_metric = step.params.get("eval_metric")
            artifact_path = step.params.get("artifacts_path")

            def _fit_and_evaluate() -> FrameworkResult:
                predictor = TabularPredictor(
                    label=label_column,
                    problem_type=problem_type,
                    eval_metric=eval_metric,
                )
                predictor.fit(
                    train_data=train_df,
                    presets=presets,
                    time_limit=time_limit,
                    verbosity=0,
                )
                score = predictor.evaluate(test_df, silent=True)
                metric_name = predictor.eval_metric or "score"
                metrics = {str(metric_name): float(score)}
                return FrameworkResult(
                    framework="autogluon",
                    step=step.step,
                    metrics=metrics,
                    artifacts_path=artifact_path,
                )

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _fit_and_evaluate)

        execute = run
    except Exception:
        pass

    return FrameworkAdapter(name="autogluon", execute=execute)

