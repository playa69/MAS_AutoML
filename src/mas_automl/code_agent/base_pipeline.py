from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from mas_automl.code_agent.load_mocks import load_mock_inputs  # type: ignore
    from mas_automl.code_agent.openai_wraper import LLMClient, LLMConfig  # type: ignore
else:
    from .load_mocks import load_mock_inputs
    from .openai_wraper import LLMClient, LLMConfig

DEFAULT_MAX_ITERATIONS = 3

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


@dataclass
class PipelineResult:
    framework: str
    reason: str
    code: str
    tests_passed: bool
    iterations: int
    feedback: str


def run_pipeline(max_iterations: int = DEFAULT_MAX_ITERATIONS, llm: LLMClient | None = None) -> PipelineResult:
    """Основной сценарий: загрузка моков → выбор фреймворка → кодогенерация → проверка."""
    data_analysis, metadata, registry = load_mock_inputs()
    llm = llm or LLMClient(LLMConfig())

    framework, reason = choose_framework(data_analysis, metadata, registry, llm)

    feedback = ""
    code = ""
    tests_passed = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        code = generate_code(framework, llm, iteration, feedback)
        tests_passed, feedback = evaluate_code(code, framework)
        if tests_passed:
            break

    return PipelineResult(
        framework=framework,
        reason=reason,
        code=code,
        tests_passed=tests_passed,
        iterations=iteration,
        feedback=feedback,
    )


def choose_framework(
    data_analysis: Dict[str, Any],
    metadata: Dict[str, Any],
    registry: Dict[str, str],
    llm: LLMClient,
) -> Tuple[str, str]:
    """
    Расширенный выбор лучшего AutoML-фреймворка с помощью ChatPromptTemplate.
    """

    # --- Дефолтные значения и fallback ---
    fallback_framework = _fallback_framework_choice(metadata, registry)
    fallback = json.dumps(
        {"framework": fallback_framework, "reason": "Эвристика на основе типа задачи и размера выборки."},
        ensure_ascii=False,
    )

    # --- Форматируем данные для контекста ---
    frameworks_list = "\n".join(
                            f"\n\n###### {name} ---> {desc} ######"
                            for name, desc in registry.items()
    )

    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    analysis_json = json.dumps(data_analysis, ensure_ascii=False, indent=2)

    # --- Prompt шаблон ---
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Ты — эксперт по AutoML и ML-инженер. "
            "Твоя задача — выбрать наиболее подходящий AutoML-фреймворк для табличных данных. "
            "Оцени качество данных, размер, тип задачи, ограничения и предложи лучший вариант из списка."
        ),
        HumanMessagePromptTemplate.from_template(
            "Вот анализ датасета и метаданные:\n\n"
            "### 📊 Data Analysis\n{analysis_json}\n\n"
            "### 🧾 Metadata\n{metadata_json}\n\n"
            "### ⚙️ Доступные AutoML фреймворки\n{frameworks_list}\n\n"
            "Поясни свой выбор кратко, но содержательно. "
            "Если несколько подходят, выбери наиболее универсальный и стабильный вариант. "
            "Ответ верни в формате JSON:\n"
            "{{\"framework\": \"...\", \"reason\": \"...\"}}"
        )
    ])
  
    # --- Формируем финальный текст промпта ---
    formatted_prompt = prompt.format_messages(
        analysis_json=analysis_json,
        metadata_json=metadata_json,
        frameworks_list=frameworks_list,
    )

    # --- Вызов LLM ---
    try:
        llm_response = llm._client.invoke(formatted_prompt)
        content = getattr(llm_response, "content", "").strip()
        parsed = json.loads(content)
        framework = parsed.get("framework", fallback_framework)
        reason = parsed.get("reason", "Без объяснения.")
    except Exception as e:
        print("⚠️ Ошибка при вызове LLM или парсинге:", repr(e))
        framework, reason = fallback_framework, "Ошибка в ответе LLM; использована эвристика."

    # --- Проверка валидности ---
    if framework not in registry:
        framework = fallback_framework
        reason = f"LLM предложил неизвестный фреймворк. Эвристика: {reason}"

    return framework, reason, prompt


def _fallback_framework_choice(metadata: Dict[str, Any], registry: Dict[str, str]) -> str:
    dataset_type = (metadata.get("dataset_type") or "").lower()
    num_rows = metadata.get("num_rows") or 0

    if dataset_type in {"classification", "binary"} and num_rows <= 50_000 and "AutoGluon" in registry:
        return "AutoGluon"
    if num_rows > 100_000 and "H2O AutoML" in registry:
        return "H2O AutoML"
    if "LightAutoML" in registry:
        return "LightAutoML"
    return next(iter(registry.keys()))

def generate_code(framework: str, llm: LLMClient, iteration: int, feedback: str) -> str:
    prompt = (
        f"Итерация {iteration}. Напиши компактную функцию train_model(...) для фреймворка {framework}. "
        "Функция должна принимать train_df, test_df и label, вызывать обучение и возвращать обученную сущность. "
        "Если есть обратная связь от тестов, учти её.\n"
        f"Обратная связь: {feedback or 'нет'}\n"
        "Верни только код функции и необходимые импорты (без пояснений)."
    )
    fallback_code = _fallback_code_template(framework)
    raw_code = llm.chat(prompt, fallback=fallback_code)
    return _extract_code(raw_code) or fallback_code


def evaluate_code(code: str, framework: str) -> Tuple[bool, str]:
    """Простейшие эвристические проверки вместо реальных тестов."""
    code_lower = code.lower()
    framework_token = framework.split()[0].lower()

    if framework_token not in code_lower:
        return False, f"В коде должно быть упоминание {framework}."
    if "def train_model" not in code:
        return False, "Ожидается функция train_model(...)."
    if "fit" not in code_lower and "train" not in code_lower:
        return False, "Код должен вызывать обучение (fit/train)."

    return True, "Проверки пройдены."


def _fallback_code_template(framework: str) -> str:
    name = framework.lower()
    if "autogluon" in name:
        return (
            "import pandas as pd\n"
            "from autogluon.tabular import TabularPredictor\n\n"
            "def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str) -> TabularPredictor:\n"
            "    predictor = TabularPredictor(label=label)\n"
            "    predictor.fit(train_data=train_df, presets='medium_quality', time_limit=60)\n"
            "    predictor.evaluate(test_df, silent=True)\n"
            "    return predictor\n"
        )
    if "lightautoml" in name:
        return (
            "import pandas as pd\n"
            "from lightautoml.automl.presets.tabular_presets import TabularAutoML\n"
            "from lightautoml.tasks import Task\n\n"
            "def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str):\n"
            "    task = Task('binary')\n"
            "    automl = TabularAutoML(task=task, timeout=300)\n"
            "    automl.fit_predict(train_df, roles={'target': label})\n"
            "    return automl\n"
        )
    if "h2o" in name:
        return (
            "import h2o\n"
            "from h2o.automl import H2OAutoML\n\n"
            "def train_model(train_df, test_df, label):\n"
            "    h2o.init()\n"
            "    train_hf = h2o.H2OFrame(train_df)\n"
            "    aml = H2OAutoML(max_runtime_secs=120)\n"
            "    aml.train(y=label, training_frame=train_hf)\n"
            "    return aml\n"
        )
    return (
        "def train_model(train_df, test_df, label):\n"
        f"    raise NotImplementedError('Неизвестный фреймворк: {framework}')\n"
    )


def _extract_code(raw_output: str) -> str:
    if "```" not in raw_output:
        return raw_output.strip()
    parts = raw_output.split("```")
    if len(parts) < 3:
        return raw_output.strip()
    code_block = parts[1]
    if code_block.startswith(("python", "py")):
        code_block = code_block.split("\n", 1)[1]
    return code_block.strip()


__all__ = [
    "PipelineResult",
    "run_pipeline",
    "choose_framework",
    "generate_code",
    "evaluate_code",
]


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))

