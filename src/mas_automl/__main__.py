"""Консольный интерфейс AetherML."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .integrations import autogluon_adapter, autosklearn_adapter, fedot_adapter
from .pipelines import HybridPipeline, registry
from .services import PipelineOrchestrator

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def list_pipelines() -> None:
    """Показать доступные пайплайны."""
    table = Table(title="Зарегистрированные пайплайны")
    table.add_column("Название", style="cyan")
    for name in registry.list():
        table.add_row(name)
    console.print(table)


@app.command()
def run(
    name: str = typer.Argument("default", help="Имя пайплайна в реестре."),
    dataset: Optional[str] = typer.Option(None, help="Путь или алиас датасета."),
) -> None:
    """Запустить гибридный пайплайн."""
    pipeline = registry.get(name)
    if dataset is not None:
        for step in pipeline.steps:
            if step.dataset is None:
                step.dataset = dataset
    asyncio.run(_execute(pipeline))


async def _execute(pipeline: HybridPipeline) -> None:
    orchestrator = PipelineOrchestrator()
    orchestrator.register_adapter(autogluon_adapter())
    orchestrator.register_adapter(autosklearn_adapter())
    orchestrator.register_adapter(fedot_adapter())

    result = await orchestrator.run_pipeline(pipeline)
    table = Table(title=f"Результаты пайплайна {result.pipeline_name}")
    table.add_column("Шаг", style="magenta")
    table.add_column("Фреймворк", style="cyan")
    table.add_column("Метрики", style="green")
    for framework_result in result.results:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in framework_result.metrics.items())
        table.add_row(framework_result.step, framework_result.framework, metrics_str or "-")

    console.print(table)
    if result.combined_metrics:
        console.print("[bold]Сводные метрики:[/bold]", result.combined_metrics)


if __name__ == "__main__":
    app()


