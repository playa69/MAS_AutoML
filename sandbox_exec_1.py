
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from automl import AutoML

class_labels = ['good', 'bad']
# Загружаем данные
df = pd.read_csv(r'C:\Users\User1\Desktop\ITMO_bootcamp\data\datasets\openml_31_credit-g.csv')

label = 'class'
import sys
sys.path.append(r'C:\Users\User1\Desktop\ITMO_bootcamp\src')
# Генерированный код пользователя
import pandas as pd
import numpy as np

X = df.drop(columns=[label])
y = df[label]

label_mapping = {v: k for k, v in enumerate(class_labels)}
if isinstance(y, pd.Series):
    y = np.array(y.map(label_mapping), dtype=int)
else:
    y_series = pd.Series(y)
    y = np.array(y_series.map(label_mapping), dtype=int)

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
        "n_splits": 10
    },
    n_jobs=3,
    random_state=0,
)

automl = automl.fit(
    X, y,
    auto_model_fit_kwargs={"tuning_timeout": 10}
)

preds = automl.predict(X)
score = automl.auto_model.best_score
test_predictions = preds[:, 1]

# Выполняем функцию
try:
    df["test_predittions"] = test_predictions
    df.to_csv(r'C:\Users\User1\Desktop\ITMO_bootcamp\data\datasets\TEST\test_20251115_133613.csv', index=False)
    # print(f"Предикты сохранены в: C:\Users\User1\Desktop\ITMO_bootcamp\data\datasets\TEST\test_20251115_133613.csv")
    CSV_PATH_TO_PREDICT = r'C:\Users\User1\Desktop\ITMO_bootcamp\data\datasets\TEST\test_20251115_133613.csv'
    # Простые проверки
    errors = []
   
    if errors and any(errors):
        raise ValueError("; ".join([e for e in errors if e]))
    
    result = {"ok": True, "predict_path": CSV_PATH_TO_PREDICT, "score": score, "message": "Все проверки пройдены"}
    
except Exception as e:
    import traceback
    result = {"ok": False, "predict_path": None, "score": score, "message": str(e), "traceback": traceback.format_exc()}

# Выводим результат в формате JSON для парсинга
print("RESULT_START")
print(json.dumps(result, ensure_ascii=False))
print("RESULT_END")
