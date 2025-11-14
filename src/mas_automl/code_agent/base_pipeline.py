from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from mas_automl.code_agent.load_mocks import load_mock_inputs  # type: ignore
    from mas_automl.code_agent.openai_wraper import LLMClient, LLMConfig  # type: ignore
    from mas_automl.code_agent.execnet_gateway import PythonSandboxClient, SandboxResult  # type: ignore
else:
    from .load_mocks import load_mock_inputs
    from .openai_wraper import LLMClient, LLMConfig
    from .execnet_gateway import PythonSandboxClient, SandboxResult

DEFAULT_MAX_ITERATIONS = 3

@dataclass
class PipelineResult:
    framework: str
    reason: str
    code: str
    tests_passed: bool
    iterations: int
    feedback: str
    predict_path: str | None = None


def run_pipeline(
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    llm: LLMClient | None = None,
    csv_path: str | None = None,
    output_dir: str | None = None,
) -> PipelineResult:
    """Основной сценарий: загрузка моков → выбор фреймворка → кодогенерация → проверка."""
    data_analysis, metadata, registry, final_data = load_mock_inputs()
    llm = llm or LLMClient(LLMConfig())

    # Получаем путь к CSV файлу
    if csv_path is None:
        csv_path = final_data.get("manifest", {}).get("local_path")
        if csv_path is None:
            csv_path = metadata.get("local_path")
    
    if csv_path is None:
        raise ValueError("Не указан путь к CSV файлу и не найден в метаданных")

    # Создаем директорию для сохранения предиктов
    if output_dir is None:
        output_dir = str(Path(csv_path).parent / "predictions")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    framework, reason = choose_framework(data_analysis, metadata, registry, llm)

    feedback = ""
    code = ""
    tests_passed = False
    iteration = 0
    predict_path = None

    for iteration in range(1, max_iterations + 1):
        code = generate_code(framework, llm, iteration, feedback, final_data)
        tests_passed, feedback, predict_path = evaluate_code(
            code, framework, csv_path, output_dir, iteration, final_data
        )
        if tests_passed:
            break

    return PipelineResult(
        framework=framework,
        reason=reason,
        code=code,
        tests_passed=tests_passed,
        iterations=iteration,
        feedback=feedback,
        predict_path=predict_path,
    )


def choose_framework(
    data_analysis: Dict[str, Any],
    metadata: Dict[str, Any],
    registry: Dict[str, str],
    llm: LLMClient,
) -> Tuple[str, str]:
    dataset_summary = data_analysis.get("summary", "Нет описания.")
    metadata_preview = json.dumps(
        {k: metadata.get(k) for k in ("name", "dataset_type", "num_rows", "num_features")},
        ensure_ascii=False,
        indent=2,
    )
    frameworks_list = "\n".join(f"- {name}" for name in registry)

    prompt = (
        "Даны описание датасета, основные метаданные и список AutoML-фреймворков. "
        "Выбери лучший фреймворк и объясни выбор.\n"
        f"Описание датасета: {dataset_summary}\n"
        f"Метаданные: {metadata_preview}\n"
        f"Доступные фреймворки:\n{frameworks_list}\n\n"
        'Ответ верни в JSON: {"framework": "...", "reason": "..."}'
    )

    fallback_framework = _fallback_framework_choice(metadata, registry)
    fallback = json.dumps(
        {"framework": fallback_framework, "reason": "Эвристика на основе типа задачи и размера выборки."},
        ensure_ascii=False,
    )

    raw_response = llm.chat(prompt, fallback=fallback)
    try:
        parsed = json.loads(raw_response)
        framework_name = parsed["framework"]
        reason = parsed.get("reason", "Без объяснения.")
    except Exception:
        framework_name = fallback_framework
        reason = "Не удалось распарсить ответ LLM; использована эвристика."

    if framework_name not in registry:
        framework_name = fallback_framework
        reason = f"LLM предложил неизвестный фреймворк. Эвристика: {reason}"

    return framework_name, reason


def generate_code(
    framework: str, llm: LLMClient, iteration: int, feedback: str, final_data: Dict[str, Any] | None = None
) -> str:
    """Генерирует код для обучения модели."""
    preprocessing_info = ""
    if final_data:
        preprocessing_recipe = final_data.get("preprocessing_recipe", {})
        if preprocessing_recipe:
            preprocessing_info = (
                f"\n\nИнформация о препроцессинге:\n"
                f"- Числовые колонки: {preprocessing_recipe.get('numeric_columns', [])}\n"
                f"- Категориальные колонки: {preprocessing_recipe.get('categorical_columns', [])}\n"
                f"- Тип задачи: {preprocessing_recipe.get('task_type', 'classification')}\n"
            )
    
    prompt = f"""
    Итерация {iteration}.
    Ты — опытный Data Scientist. Нужно написать компактную корректную функцию train_model(...),
    которая использует МОЙ AutoML (он уже импортирован; НЕ нужно определять класс AutoML).

    Функция train_model должна:

    1) Принимать:
        - train_df : pandas.DataFrame
        - test_df  : pandas.DataFrame
        - label : str

    2) Правильно разделять данные:
        X = train_df.drop(columns=[label])
        y = train_df[label]

    3) Инициализировать AutoML строго такими параметрами:
        automl = AutoML(
            task='classification',
            use_preprocessing_pipeline=False,
            feature_selector_type=None,
            use_val_test_pipeline=False,
            auto_models_init_kwargs={
                "metric": "roc_auc",
                "time_series": False,
                "models_list": ["linear", "forests", "boostings"],
                "blend": True,
                "stack": True,
                "n_splits": 10,
            },
            n_jobs=3,
            random_state=0,
        )

    4) Обучить модель:
        automl = automl.fit(
            X, y,
            auto_model_fit_kwargs={"tuning_timeout": 10}
        )

        — НЕ использовать sklearn в любом виде
        — НЕ выполнять вручную препроцессинг, кодирование, скейлинг

    5) Получить предсказания:
        preds = automl.predict(test_df)

    6) Вычислить:
        score = automl.auto_model.best_score
        test_predictions = preds[:, 1]

    7) Вернуть:
        return automl, score, test_predictions

    Если передано preprocessing_recipe — учитывай его только как информационный блок,
    но НЕ применяй его в коде.

    Информация о препроцессинге:
    {preprocessing_info}

    ПРИМЕР КАК ИСПОЛЬЗУЕТСЯ МОЙ AutoML (НЕ копировать, только ориентир):

        automl = AutoML(...)
        automl = automl.fit(X, y, auto_model_fit_kwargs={"tuning_timeout": 10})
        score = automl.auto_model.best_score
        test_predictions = automl.predict(X)[:, 1]

    Требования к выводу:
    - Вернуть СТРОГО: только код функции + необходимые импорты.
    - Никакого текста, комментариев, markdown.
    - Функция должна называться train_model.
    - Переменные: score и test_predictions — обязательны.

    Обратная связь:
    {feedback or "нет"}
    """

    fallback_code = _fallback_code_template_sklearn(final_data)
    raw_code = llm.chat(prompt, fallback=fallback_code)
    return _extract_code(raw_code) or fallback_code


def evaluate_code(
    code: str,
    framework: str,
    csv_path: str,
    output_dir: str,
    iteration: int,
    final_data: Dict[str, Any] | None = None,
) -> Tuple[bool, str, str | None]:
    """
    Выполняет код в песочнице, тестирует функцию train_model и сохраняет предикты.
    Возвращает (tests_passed, feedback, predict_path).
    """
    sandbox = PythonSandboxClient.get()
    
    # Получаем информацию о датасете
    target_column = "class"
    if final_data:
        target_column = final_data.get("manifest", {}).get("target_column", "class")
    print("1 - start testing")
    # Подготавливаем код для выполнения
    test_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Загружаем данные
df = pd.read_csv(r'{csv_path}')
print(f"Загружено строк: {{len(df)}}")

# Разделяем на train и test
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['{target_column}'])
print(f"Train: {{len(train_df)}}, Test: {{len(test_df)}}")

# Генерированный код пользователя
{code}

# Выполняем функцию
try:
    model = train_model(train_df, test_df, '{target_column}')
    print("Модель обучена успешно")
    
    # Делаем предсказания
    X_test = test_df.drop(columns=['{target_column}'])
    predictions = model.predict(X_test)
    print(f"Предсказания получены, форма: {{predictions.shape}}")
    
    # Сохраняем предикты
    output_path = Path(r'{output_dir}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predict_file = output_path / f'predictions_iter{iteration}_{{timestamp}}.csv'
    
    predict_df = pd.DataFrame({{
        'prediction': predictions,
        'true_label': test_df['{target_column}'].values
    }})
    predict_df.to_csv(predict_file, index=False)
    print(f"Предикты сохранены в: {{predict_file}}")
    
    # Простые проверки
    errors = []
    if len(predictions) != len(test_df):
        errors.append(f"Количество предсказаний {{len(predictions)}} не совпадает с размером test {{len(test_df)}}")
    
    if not hasattr(model, 'predict'):
        errors.append("Модель не имеет метода predict")
    
    if errors and any(errors):
        raise ValueError("; ".join([e for e in errors if e]))
    
    result = {{"ok": True, "predict_path": str(predict_file), "message": "Все проверки пройдены"}}
    
except Exception as e:
    import traceback
    result = {{"ok": False, "predict_path": None, "message": str(e), "traceback": traceback.format_exc()}}

# Выводим результат в формате JSON для парсинга
print("RESULT_START")
print(json.dumps(result, ensure_ascii=False))
print("RESULT_END")
"""
    
    # Выполняем код в песочнице
    result = sandbox.run(test_code)
    print("2 - end testing")
  
    if not result.ok:
        feedback = f"Ошибка выполнения: {result.stderr}\n{result.stdout}"
        return False, feedback, None
    print("3 - try gateway")

    # Парсим результат из stdout
    try:
        stdout = result.stdout
        # Ищем маркеры RESULT_START и RESULT_END
        if "RESULT_START" in stdout and "RESULT_END" in stdout:
            start_idx = stdout.find("RESULT_START") + len("RESULT_START")
            end_idx = stdout.find("RESULT_END")
            result_json = stdout[start_idx:end_idx].strip()
            result_dict = json.loads(result_json)
        else:
            # Fallback: пытаемся найти JSON в stdout
            import re
            json_match = re.search(r'\{[^{}]*"ok"[^{}]*\}', stdout)
            if json_match:
                result_dict = json.loads(json_match.group())
            else:
                result_dict = {"ok": False, "message": "Не удалось найти результат в stdout"}
    except Exception as e:
        feedback = f"Ошибка парсинга результата: {e}\nStdout: {result.stdout}\nStderr: {result.stderr}"
        return False, feedback, None
    print("4 - end gateway")

    if result_dict.get("ok", False):
        predict_path = result_dict.get("predict_path")
        message = result_dict.get("message", "Проверки пройдены")
        return True, message, predict_path
    else:
        error_msg = result_dict.get("message", "Неизвестная ошибка")
        traceback_info = result_dict.get("traceback", "")
        feedback = f"Тесты не пройдены: {error_msg}\n{traceback_info}"
        return False, feedback, None


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


def _fallback_code_template_sklearn(final_data: Dict[str, Any] | None = None) -> str:
    """Шаблон кода на scikit-learn для fallback."""
    numeric_cols = ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"]
    categorical_cols = ["checking_status", "credit_history", "purpose", "savings_status", "employment", "personal_status", "other_parties", "property_magnitude", "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]
    
    if final_data:
        preprocessing_recipe = final_data.get("preprocessing_recipe", {})
        if preprocessing_recipe:
            numeric_cols = preprocessing_recipe.get("numeric_columns", numeric_cols)
            categorical_cols = preprocessing_recipe.get("categorical_columns", categorical_cols)
    
    return (
        "import pandas as pd\n"
        "import numpy as np\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.ensemble import RandomForestClassifier\n\n"
        f"numeric_cols = {numeric_cols}\n"
        f"categorical_cols = {categorical_cols}\n\n"
        "def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str):\n"
        "    # Подготовка признаков\n"
        "    numeric_transformer = Pipeline(steps=[\n"
        "        ('imputer', SimpleImputer(strategy='median')),\n"
        "        ('scaler', StandardScaler())\n"
        "    ])\n"
        "    categorical_transformer = Pipeline(steps=[\n"
        "        ('imputer', SimpleImputer(strategy='most_frequent')),\n"
        "        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n"
        "    ])\n"
        "    preprocessor = ColumnTransformer(\n"
        "        transformers=[\n"
        "            ('num', numeric_transformer, numeric_cols),\n"
        "            ('cat', categorical_transformer, categorical_cols)\n"
        "        ]\n"
        "    )\n"
        "    \n"
        "    # Подготовка данных\n"
        "    X_train = train_df.drop(columns=[label])\n"
        "    y_train = train_df[label]\n"
        "    \n"
        "    # Создание пайплайна\n"
        "    model = Pipeline(steps=[\n"
        "        ('preprocessor', preprocessor),\n"
        "        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))\n"
        "    ])\n"
        "    \n"
        "    # Обучение\n"
        "    model.fit(X_train, y_train)\n"
        "    \n"
        "    return model\n"
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
    "PythonSandboxClient",
]


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))

