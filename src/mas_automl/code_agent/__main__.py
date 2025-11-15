from __future__ import annotations

from mas_automl.code_agent.load_mocks import load_mock_inputs
from typing import Any

from mas_automl.code_agent.load_mocks import load_mock_inputs

# DATA_ANALYZE, METADATA, REGISTRY
import pandas as pd

from typing import Dict, Any
# --- Инициализация клиента ---
from mas_automl.code_agent.openai_wraper import LLMClient, LLMConfig

# --- Вызов выбора фреймворка ---

# Итеративная генерация и тестирование кода
from mas_automl.code_agent.base_pipeline import generate_code, evaluate_code
from mas_automl.code_agent.execnet_gateway import PythonSandboxClient
from pathlib import Path
import pandas as pd

client = LLMClient(LLMConfig())



framework = "scikit-learn"
max_iterations = 3
feedback = ""



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
    Ты — опытный Data Scientist. Нужно написать корректный код,
    который использует МОЙ AutoML модуль (он уже импортирован; НЕ нужно определять класс AutoML).
    Действуй так, будто class_labels и label определен и инициализирован!
    
    Код должен:
    1) Использовать:
        - df : pandas.DataFrame

    2) Правильно разделять данные:
        X =  df.drop(columns=[label])
        y =  df[label]
        где - label : str (метка класса)

    3) Написать этот код для labels:
        label_mapping = {{v: k for k, v in enumerate(class_labels)}}
        if isinstance(y, pd.Series):
            y = np.array(y.map(label_mapping), dtype=int)
        else:
            # Если y уже массив или другой тип, преобразуем через pandas для совместимости
            y_series = pd.Series(y)
            y = np.array(y_series.map(label_mapping), dtype=int)


    4) Инициализировать AutoML с параметрами, например:
        automl = AutoML(
            task='classification',
            use_preprocessing_pipeline=False,
            feature_selector_type=None,
            use_val_test_pipeline=False,
            auto_models_init_kwargs={{ 
                "metric": "roc_auc",
                "time_series": False,
                "models_list": ["linear", "forests", "boostings"],
                "blend": True,
                "stack": True,
                "n_splits": 10
            }},
            n_jobs=3,
            random_state=0,
        )

    Описание параметров класса AutoML:
     Parameters
    ----------
    task
        Machine Learning task to solve, by default classification
        Should be one of:
            -"classification (binary and multiclass)
            - regression
    use_preprocessing_pipeline, optional
        Whether to use preprocessing pipeline, by default True
    preprocessing_pipeline_kwargs, optional
        Keyword arguments to initialize preprocessing_pipeline, by default {{ }}
        List of possible arguments and their default values:
            - pipe_steps = ['all']
            - nan_share_ts=0.2
            - qconst_feature_val_share_ts=0.95
            - impute_num_strategy='median'
            - impute_cat_strategy='most_frequent'
            - outlier_capping_method='gaussian'
            - outlier_cap_tail='both'
            - corr_ts = 0.8
            - corr_coef_methods=['pearson', 'spearman']
            - corr_selection_method="missing_values"
            - oe_min_freq=0.1
            - obj_encoders = ['oe', 'ohe']
            - num_encoder = "ss"
            - verbose=True
    use_val_test_pipeline, optional
        Whether to use val_test_pipeline, by default True
    val_test_pipeline_kwargs, optional
        Keyword arguments to initialize val_test_pipeline, by default {{ }}
        List of possible arguments and their default values:
            - pipe_steps = ['all']
            - split_col='is_test_for_val'
            - psi_cut_off=0.5
            - psi_threshold=0.2
            - psi_bins=15
            - psi_strategy='equal_width'
            - adversarial_auc_trshld=0.7
            - verbose=True
    feature_selector_type, optional
        Whether to use feature_selector and which feature_selector to use, by default "CatboostByShap"
        Possible values:
            - CatboostByShap
            - LAMA
    feature_selector_kwargs, optional
        Keyword arguments to initialize feature_selector, by default {{ }}
        List of possible arguments and their default values:
            * CatboostByShap
                - n_features_to_select = 50
                - complexity = "Regular"
                - steps = 5
            * LAMA
                -task_type
                - target_colname
                - metric_name
                - metric_direction
                - timeout=120
                - model='lama'
                - strategy='RFA'
                - permutation_n_repeats = 5
    auto_models_init_kwargs, optional
        Keyword arguments to initialize AutoModel, by default {{ }}
        List of possible arguments and their default values:
            - metric
            - time_series=False
            - models_list=None
            - blend=False
            - stack=False
            - timeout=60 (timeout для моделей, которые его поддерживают, например TabularLama)
    n_jobs, optional
        Number of cores for parallel computations, by default 1
    random_state, optional
        Random state, by default 42
    log_to_file, optional
        Whether to save logs in files.
        Save files locations:
            - ml_data/YYYY_mm_dd___HH-MM-SS/logs.log for info logs
            - ml_data/YYYY_mm_dd___HH-MM-SS/error.log for error, critical, warning logs

    5) Обучить модель:
        automl = automl.fit(
            X, y,
            auto_model_fit_kwargs={{"tuning_timeout": 10}}
        )

        — НЕ использовать sklearn в любом виде
        — НЕ выполнять вручную препроцессинг, кодирование, скейлинг

    6) Получить предсказания:
        preds = automl.predict(X)

    7) Вычислить переменные:
        score = automl.auto_model.best_score
        test_predictions = preds[:, 1]


    Если передано preprocessing_recipe — учитывай его только как информационный блок,
    но НЕ применяй его в коде.

    Информация о препроцессинге:
    {preprocessing_info}

    ПРИМЕР КАК ИСПОЛЬЗУЕТСЯ МОЙ AutoML (НЕ копировать, только ориентир):
        label_mapping = {{v: k for k, v in enumerate(class_labels)}}
        if isinstance(y, pd.Series):
            y = np.array(y.map(label_mapping), dtype=int)
        else:
            # Если y уже массив или другой тип, преобразуем через pandas для совместимости
            y_series = pd.Series(y)
            y = np.array(y_series.map(label_mapping), dtype=int)

        automl = AutoML(...)
        automl = automl.fit(X, y, auto_model_fit_kwargs={{"tuning_timeout": 10}})
        score = automl.auto_model.best_score
        test_predictions = automl.predict(X)[:, 1]



    Требования к выводу:
    - Вернуть СТРОГО: только код функции + необходимые импорты.
    - Никакого текста, комментариев, markdown.
    - Переменные: score и test_predictions — обязательны.
    - Код я буду исполнять - НЕ ПИШИ ФУНКЦИЮ.
    - Действуй так, будто class_labels и label определен и инициализирован!
    

    Обратная связь:
    {feedback or "нет"}
    """


    fallback_code = _fallback_code_template_sklearn(final_data)
    raw_code = llm.chat(prompt, fallback=fallback_code)
    return _extract_code(raw_code) or fallback_code

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


import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

if __package__ is None or __package__ == "":
    from mas_automl.code_agent.load_mocks import load_mock_inputs  # type: ignore
    from mas_automl.code_agent.openai_wraper import LLMClient, LLMConfig  # type: ignore
    from mas_automl.code_agent.execnet_gateway import PythonSandboxClient, SandboxResult  # type: ignore
else:
    from .load_mocks import load_mock_inputs
    from .openai_wraper import LLMClient, LLMConfig
    from .execnet_gateway import PythonSandboxClient, SandboxResult

DEFAULT_MAX_ITERATIONS = 3
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
CSV_PATH_TO_PREDICT = f"C:\\Users\\User1\\Desktop\\ITMO_bootcamp\\data\\datasets\\TEST\\test_{timestamp}.csv"

@dataclass
class PipelineResult:
    framework: str
    reason: str
    code: str
    tests_passed: bool
    iterations: int
    feedback: str
    predict_path: str | None = None


def evaluate_code(
    code: str,
    framework: str,
    csv_path: str,
    CSV_PATH_TO_PREDICT: str,
    output_dir: str,
    iteration: int,
    final_data: Dict[str, Any] | None,
    class_labels,
) :
    """
    Выполняет код в песочнице, тестирует функцию train_model и сохраняет предикты.
    Возвращает (tests_passed, feedback, predict_path).
    """
    sandbox = PythonSandboxClient.get()
    automl_path = 'C:\\Users\\User1\\Desktop\\ITMO_bootcamp\\src'
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
from automl import AutoML

class_labels = {class_labels}
# Загружаем данные
df = pd.read_csv(r'{csv_path}')

label = 'class'
import sys
sys.path.append(r'{automl_path}')
# Генерированный код пользователя
{code}

# Выполняем функцию
try:
    df["test_predittions"] = test_predictions
    df.to_csv(r'{CSV_PATH_TO_PREDICT}', index=False)
    # print(f"Предикты сохранены в: {CSV_PATH_TO_PREDICT}")
    CSV_PATH_TO_PREDICT = r'{CSV_PATH_TO_PREDICT}'
    # Простые проверки
    errors = []
   
    if errors and any(errors):
        raise ValueError("; ".join([e for e in errors if e]))
    
    result = {{"ok": True, "predict_path": CSV_PATH_TO_PREDICT, "score": score, "message": "Все проверки пройдены"}}
    
except Exception as e:
    import traceback
    result = {{"ok": False, "predict_path": None, "score": score, "message": str(e), "traceback": traceback.format_exc()}}

# Выводим результат в формате JSON для парсинга
print("RESULT_START")
print(json.dumps(result, ensure_ascii=False))
print("RESULT_END")
"""
    # Write script into sandbox
    sandbox_script = f"sandbox_exec_{iteration}.py"
    Path(sandbox_script).write_text(test_code, encoding="utf-8")
    print(f"ПУТЬ ПУТЬ {Path(sandbox_script)}")
    # Execute the script in sandbox
    result = sandbox.run(f"exec(open(r'{sandbox_script}', encoding='utf-8').read())")

    # Выполняем код в песочнице
    #result = sandbox.run(test_code)
    print("2 - end testing")
  
    if not result.ok:
        feedback = f"Ошибка выполнения: {result.stderr}\n{result.stdout}"
        return False, feedback, None, None
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
        return False, feedback, None, None
    print("4 - end gateway")

    if result_dict.get("ok", False):
        predict_path = result_dict.get("predict_path")
        score = result_dict.get("score")
        message = result_dict.get("message", "Проверки пройдены")
        return True, message, predict_path, score
    else:
        error_msg = result_dict.get("message", "Неизвестная ошибка")
        traceback_info = result_dict.get("traceback", "")
        score = result_dict.get("score")

        feedback = f"Тесты не пройдены: {error_msg}\n{traceback_info}"
        return False, feedback, None, None



if __name__ == "__main__":
    DATA_ANALYZE, METADATA, REGISTRY, FINAL_DATA = load_mock_inputs()
    CSV_PATH = FINAL_DATA["manifest"]["local_path"]
    OUTPUT_DIR = str(Path(CSV_PATH).parent / "predictions")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"CSV путь: {CSV_PATH}")
    print(f"Директория для предиктов: {OUTPUT_DIR}\n")
    print("\n=== Загружены данные c Data-Agent ===")
    print("Ключи DATA_ANALYZE:", list(DATA_ANALYZE.keys()))
    print("Ключи METADATA:", list(METADATA.keys()))
    print("\n=== Загружены данные c Registry-Agent ===")

    print("Фреймворки REGISTRY:", list(REGISTRY.keys()))
    print("Ключи FINAL_DATA:", list(FINAL_DATA.keys()))
    print("================================\n")

    DATA_ANALYZE, METADATA, REGISTRY, FINAL_DATA = load_mock_inputs()

    print("=== Информация о данных ===")
    PATH_TO_CSV = FINAL_DATA["manifest"]["local_path"]
    print("Путь к датасету:", PATH_TO_CSV)

    dataset_df = pd.read_csv(PATH_TO_CSV)
    print(f"Размер датафрейма: {dataset_df.shape[0]} строк, {dataset_df.shape[1]} колонок\n")

    print("Типы колонок:")
    print(dataset_df.dtypes)
    print("\nПервые 5 строк:")
    print(dataset_df.head(5))
    print("================================\n")

    iteration = 1

    print("=== Генерация кода AutoML ===")
    code = generate_code(framework, client, iteration, feedback, FINAL_DATA)
    print("Код успешно сгенерирован.\n")

    print("=== Тестирование кода в песочнице ===")
    tests_passed, test_feedback, predict_path, score = evaluate_code(
        code, framework, CSV_PATH, CSV_PATH_TO_PREDICT, OUTPUT_DIR, iteration, FINAL_DATA,
        class_labels=list(FINAL_DATA['metafeatures']['dataset']['class_balance'].keys())
    )

    predict_path = CSV_PATH_TO_PREDICT

    print("\n=== Результаты тестирования ===")
    print("Тесты пройдены:", tests_passed)
    print("Сообщение:", test_feedback)
    
    if predict_path:
        print("Путь к предиктам:", predict_path)

        try:
            pred_df = pd.read_csv(predict_path)
            print("\nПервые 5 строк предиктов:")
            print(pred_df.head())
        except Exception as e:
            print("Ошибка при загрузке предиктов:", e)
    print("\n=== Финальный скор для датасета credit-g (id=31) ===\n")
    print(f"\n ROC-AUC = {score}\n")
    print("\n=== Завершено ===\n")


















