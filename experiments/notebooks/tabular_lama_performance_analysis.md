# Анализ производительности TabularLama

## Проблема
TabularLama занимает ~11.4 секунды для обучения (с 18:47:31,949 до 18:47:43,364), что может быть медленнее других моделей в пайплайне.

## Текущие параметры из notebook
- `n_splits: 10` - количество фолдов для cross-validation
- `tuning_timeout: 10` - таймаут для тюнинга
- `n_jobs: 3` - количество параллельных процессов
- `timeout: 60` (по умолчанию) - таймаут для TabularAutoML

## Основные причины медленной работы

### 1. **Большое количество фолдов (n_splits=10)**
**Проблема:**
- В `default_lama.py:91` параметр `cv: self.n_folds` передается в `reader_params`
- При `n_splits=10` TabularAutoML выполняет 10-fold cross-validation
- Каждый алгоритм внутри LightAutoML обучается на 10 фолдах
- Это увеличивает время обучения в ~10 раз по сравнению с 5 фолдами

**Код:**
```python
"reader_params": {
    "n_jobs": 1,
    "cv": self.n_folds,  # = 10
    "random_state": self.random_state,
},
```

**Рекомендация:** Уменьшить `n_splits` до 5 для TabularLama или использовать отдельный параметр.

### 2. **TabularAutoML использует все алгоритмы по умолчанию**
**Проблема:**
- Когда `use_algos=None` (по умолчанию), LightAutoML пробует множество алгоритмов:
  - `linear_l2` (линейные модели)
  - `lgb` (LightGBM)
  - `cb` (CatBoost)
  - `nn` (нейронные сети)
  - И другие...
- Каждый алгоритм обучается отдельно, что занимает время

**Код:**
```python
# default_lama.py:57
self.use_algos = use_algos  # None по умолчанию - используются все алгоритмы LightAutoML

# default_lama.py:97-98
if self.use_algos is not None:
    model_kwargs["general_params"] = {"use_algos": self.use_algos}
```

**Рекомендация:** Ограничить набор алгоритмов через параметр `use_algos`, например:
```python
use_algos=[["lgb"], ["cb"]]  # Только LightGBM и CatBoost
```

### 3. **Ограниченная параллелизация в reader_params**
**Проблема:**
- `reader_params` имеет `n_jobs=1`, что ограничивает параллелизацию чтения и обработки данных
- При этом `cpu_limit=self.n_jobs` (3) используется для основного процесса

**Код:**
```python
"reader_params": {
    "n_jobs": 1,  # Ограничение параллелизации
    "cv": self.n_folds,
    "random_state": self.random_state,
},
"cpu_limit": self.n_jobs,  # = 3
```

**Рекомендация:** Увеличить `n_jobs` в `reader_params` до `self.n_jobs` или использовать меньшее значение для баланса.

### 4. **Timeout использует tuning_timeout из fit()**
**Как работает:**
- `tuning_timeout` передается в метод `fit()` AutoModel (по умолчанию 60 секунд)
- В `fit()` вызывается `model.tune(timeout=tuning_timeout)`
- В `tune()` для TabularLama устанавливается `self.timeout = timeout`
- Затем в `fit()` TabularLama использует `self.timeout` для создания TabularAutoML

**Код:**
```python
# main.py:204-206
model.tune(
    x_train_iter, y,
    timeout=tuning_timeout,  # tuning_timeout передается в timeout
    ...
)

# default_lama.py:138
def tune(self, X, y, timeout: int = 60, ...):
    self.timeout = timeout  # Устанавливается из tuning_timeout

# default_lama.py:87
"timeout": self.timeout,  # Используется в TabularAutoML
```

**Использование:**
```python
automl.fit(
    X, y,
    auto_model_fit_kwargs={
        "tuning_timeout": 30,  # Это значение будет использовано как timeout для TabularLama
    }
)
```

### 5. **TabularAutoML - комплексный AutoML фреймворк**
**Проблема:**
- TabularAutoML не просто обучает одну модель
- Он выполняет:
  - Feature engineering
  - Feature selection
  - Обучение нескольких алгоритмов
  - Blending/Stacking моделей
  - Cross-validation для каждого алгоритма
- Все это занимает время, особенно при большом количестве фолдов

## Рекомендации по оптимизации

### Вариант 1: Уменьшить количество фолдов для TabularLama
```python
# В models_lists.py или при инициализации
TabularLamaClassification(n_splits=5, ...)  # Вместо 10
```

### Вариант 2: Ограничить набор алгоритмов
```python
# При инициализации TabularLama
TabularLamaClassification(
    use_algos=[["lgb"], ["cb"]],  # Только LightGBM и CatBoost
    ...
)
```

### Вариант 3: Увеличить параллелизацию
```python
# В default_lama.py:89-90
"reader_params": {
    "n_jobs": self.n_jobs,  # Вместо 1
    "cv": self.n_folds,
    ...
},
```

### Вариант 4: Использовать TabularUtilizedAutoML
- `TabularUtilizedAutoML` может быть быстрее, так как использует меньше алгоритмов
- Уже есть класс `TabularLamaUtilizedClassification`

### Вариант 5: Увеличить timeout для TabularLama
```python
# В tune() или при инициализации
TabularLamaClassification(timeout=30, ...)  # Вместо 60
# Или в tune()
model.tune(..., timeout=30)  # Для TabularLama отдельно
```

## Сравнение с другими моделями

Другие модели (LightGBM, CatBoost, XGBoost) обучаются быстрее, потому что:
1. Они обучают только одну модель, а не несколько алгоритмов
2. Они используют меньше фолдов (если не переопределено)
3. Они не делают feature engineering внутри себя
4. Они более оптимизированы для быстрого обучения

## Дополнительные наблюдения

### 6. **seed_everything вызывается при каждом fit**
**Проблема:**
- `seed_everything()` вызывается в `_prepare()` при каждом вызове `fit()`
- Это устанавливает seed для random, numpy, torch, что может занимать небольшое время
- Но это не критично для производительности

**Код:**
```python
# default_lama.py:63
def _prepare(self, X, y=None, ...):
    seed_everything(self.random_state)  # Вызывается каждый раз
    ...
```

### 7. **Отсутствие кэширования подготовленных данных**
**Проблема:**
- Данные подготавливаются заново при каждом вызове `fit()`
- `_prepare()` вызывается и в `tune()`, и в `fit()`

## Выводы

Основные причины медленной работы TabularLama:
1. **n_splits=10** - главная причина (увеличивает время в ~10 раз по сравнению с 5 фолдами)
2. **Использование всех алгоритмов LightAutoML** - пробует много моделей (linear_l2, lgb, cb, nn и др.)
3. **Ограниченная параллелизация** - `n_jobs=1` в reader_params
4. **Комплексность TabularAutoML** - делает feature engineering, selection, обучение нескольких алгоритмов
5. **Timeout=10 секунд** - может быть недостаточно для полного обучения всех алгоритмов

**Приоритетные действия (по важности):**
1. **Уменьшить `n_splits` до 5 для TabularLama** - даст наибольший прирост скорости (~2x)
2. **Ограничить `use_algos` до 1-2 алгоритмов** - например, только `[["lgb"]]` или `[["lgb"], ["cb"]]`
3. **Увеличить `n_jobs` в `reader_params`** - с 1 до `self.n_jobs` (3) для лучшей параллелизации
4. ✅ **Использовать `tuning_timeout`** - timeout для TabularLama берется из `tuning_timeout` в `auto_model_fit_kwargs`

## Конкретные рекомендации по коду

### Рекомендация 1: Добавить отдельный параметр n_splits_lama
Модифицировать `default_lama.py`:
```python
def __init__(
    self,
    ...
    n_splits: int = 5,
    n_splits_lama: Optional[int] = None,  # Новый параметр
    ...
):
    ...
    self.n_folds = n_splits_lama if n_splits_lama is not None else min(n_splits, 5)  # Ограничить до 5
```

### Рекомендация 2: Ограничить алгоритмы по умолчанию
Модифицировать `default_lama.py`:
```python
def __init__(
    self,
    ...
    use_algos: Optional[List[List[str]]] = None,
    ...
):
    ...
    # По умолчанию использовать только быстрые алгоритмы
    if use_algos is None:
        self.use_algos = [["lgb"], ["cb"]]  # Только LightGBM и CatBoost
    else:
        self.use_algos = use_algos
```

### Рекомендация 3: Улучшить параллелизацию
Модифицировать `default_lama.py:89-90`:
```python
"reader_params": {
    "n_jobs": min(self.n_jobs, 4),  # Вместо 1, но ограничить до 4
    "cv": self.n_folds,
    "random_state": self.random_state,
},
```

### Рекомендация 4: Использовать TabularUtilizedAutoML по умолчанию
`TabularUtilizedAutoML` может быть быстрее, так как использует меньше алгоритмов. Уже есть класс `TabularLamaUtilizedClassification`.

### Рекомендация 5: Использовать tuning_timeout (РЕАЛИЗОВАНО)
Timeout для TabularLama берется из `tuning_timeout`, который передается в `auto_model_fit_kwargs`:
```python
automl.fit(
    X, y,
    auto_model_fit_kwargs={
        "tuning_timeout": 30,  # Это значение будет использовано как timeout для TabularLama
    }
)
```

Это позволяет контролировать время обучения TabularLama через `tuning_timeout`, который также используется для других моделей при тюнинге.

