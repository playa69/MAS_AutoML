# MAS_AutoML — Дата-агент (AetherML)

Дата-агент загружает наборы данных AMLB (локальные или OpenML), валидирует схему/качество, выполняет разбиение на обучающую/тестовую выборки, считает метапризнаки, регистрирует запуск (MLflow) и генерирует рецепт препроцессинга/фичеинжиниринга для Код-агента.

## Быстрый старт

- Самый быстрый путь (без установки пакета, из исходников):

```bash
cd /Users/sergeykudriashov/itmo_autods/MAS_AutoML
PYTHONPATH=src python3 scripts/check_data_agent.py
```

Скрипт создаст синтетический датасет в `data/amlb/demo_agent/` и выполнит агент end-to-end.

## Запуск на реальном датасете AMLB

- Положите CSV в `data/amlb/<dataset_id>/` или передайте OpenML ID/URL:

```bash
# Локальная папка с данными
PYTHONPATH=src python3 scripts/check_data_agent.py --dataset-id adult --target income

# OpenML по URL (таргет берётся из OpenML автоматически, если указан в наборе)
PYTHONPATH=src python3 scripts/check_data_agent.py --dataset-id "https://www.openml.org/search?type=data&sort=runs&status=active&id=31"

# OpenML по ID или схеме
PYTHONPATH=src python3 scripts/check_data_agent.py --dataset-id 31
PYTHONPATH=src python3 scripts/check_data_agent.py --dataset-id openml:31
```

Примечания:

- `--target` опционален. Для OpenML таргет берётся из набора; для локальных CSV он угадывается по имени (`target`, `class`, `label`, `y`, суффиксы), иначе берётся последний столбец.
- Наборы, скачанные с OpenML, кэшируются в `data/amlb/openml_<id>/data.csv`.

## Результаты (артефакты)

Артефакты сохраняются в:

- `data/processed/aetherml/<dataset_id>/<timestamp>/`
  - `validation_report.json`
  - `split_metadata.json` (+ `splits/train.csv`, `splits/test.csv`)
  - `metafeatures.json`
  - `preprocessing_recipe.json` — машинно/человеко‑читаемые шаги для Код‑агента
  - `code_agent_recommendation.json` — единый JSON-бандл с манифестом, валидацией, метапризнаками, рецептом и метаданными запуска
  - `run_metadata.json`

Все артефакты также логируются в MLflow: `mlruns/0/<run_id>/artifacts/`.

## Программное использование

```python
import asyncio
from mas_automl.agents import DataAgent, AgentMessage

async def run(dataset_id: str, target: str | None = None):
    agent = DataAgent()
    msg = AgentMessage(sender="user", recipient="AetherML", content="prepare",
                       payload={"dataset_id": dataset_id, "target": target})
    res = await agent.handle(msg)
    return res.payload  # dict с URL-ами и code_agent_recommendation

# Пример:
# asyncio.run(run("openml:31"))
```

## Интеграция в MAS

- Дата-агент доступен как `mas_automl.agents.DataAgent`.
- Типовой сценарий:
  1. Планировщик/Исследователь выбирает `dataset_id`.
  2. Исполнитель вызывает `DataAgent.handle(...)` с payload `{"dataset_id": "...", "target": optional}`.
  3. Используем поля ответа:
     - `preprocessing_recipe_url` и/или `code_agent_recommendation_url` для Код‑агента.
     - `split_metadata_url`, `validation_report_url`, `metafeatures_url` для адаптеров AutoML‑фреймворков.

Минимальный пример интеграции:

```python
from mas_automl.agents import DataAgent, AgentMessage

async def prepare_dataset_for_code_agent(dataset_id: str):
    agent = DataAgent()
    res = await agent.handle(AgentMessage(sender="orchestrator", recipient="AetherML",
                                          content="prepare", payload={"dataset_id": dataset_id}))
    return {
        "preprocessing_recipe_url": res.payload["preprocessing_recipe_url"],
        "code_agent_recommendation_url": res.payload["code_agent_recommendation_url"],
        "train_test_split": res.payload["split_metadata_url"],
    }
```

## Конфигурация

- Настройки из `src/mas_automl/config/settings.py`:
  - `benchmark.aml_benchmark_root`: корневая папка AMLB (по умолчанию `data/amlb`)
  - Пороговые значения по умолчанию: пропуски 0.6, high-cardinality 0.5, train_size 0.8, seed 42
- Можно задавать через переменные окружения с префиксом `MAS_` (pydantic-settings).

## Зависимости

- Базовые: pandas, numpy, scikit-learn, pydantic, pydantic-settings, mlflow
- Опциональные: category_encoders, featuretools (рекомендуются рецептом; не обязательны для запуска агента)

Установка через PDM:

```bash
pdm install
pdm run python scripts/check_data_agent.py --dataset-id openml:31
```
