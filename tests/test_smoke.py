import asyncio

from mas_automl.pipelines import registry
from mas_automl.services import PipelineOrchestrator, dummy_execute, FrameworkAdapter


async def _run_default_pipeline() -> None:
    pipeline = registry.get("default")
    orchestrator = PipelineOrchestrator()
    orchestrator.register_adapter(FrameworkAdapter(name="autogluon", execute=dummy_execute))
    orchestrator.register_adapter(FrameworkAdapter(name="autosklearn", execute=dummy_execute))
    orchestrator.register_adapter(FrameworkAdapter(name="fedot", execute=dummy_execute))

    result = await orchestrator.run_pipeline(pipeline)
    assert result.pipeline_name == pipeline.name
    assert len(result.results) == len(pipeline.steps)


def test_pipeline_smoke() -> None:
    asyncio.run(_run_default_pipeline())

