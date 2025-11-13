"""Менеджер для работы с датасетами из OpenML и локальными файлами.

Модуль предоставляет простой интерфейс для:
- Загрузки метаданных датасетов из OpenML
- Загрузки данных датасетов в pandas DataFrame
- Автоматического кэширования метаданных и данных локально
- Работы с локальными датасетами
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import openml
except ImportError:
    openml = None  # type: ignore

from ..config.settings import settings
from .models import (
    DatasetFormat,
    DatasetMetadata,
    DatasetType,
    OpenMLDatasetMetadata,
)


class DatasetManager:
    """Менеджер для загрузки и управления датасетами из OpenML и локальных файлов."""

    def __init__(
        self,
        raw_data_dir: Path | None = None,
        metadata_dir: Path | None = None,
        cache_openml: bool = True,
    ) -> None:
        """
        Инициализация менеджера датасетов.

        Args:
            raw_data_dir: Директория с сырыми датасетами
            metadata_dir: Директория для хранения метаданных
            cache_openml: Кэшировать ли датасеты из OpenML локально
        """
        workspace_root = settings.workspace_root
        self.raw_data_dir = raw_data_dir or workspace_root / "data" / "datasets"
        self.metadata_dir = metadata_dir or workspace_root / "data" / "datasets" / ".metadata"
        self.cache_openml = cache_openml

        # Создаём директории, если их нет
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self._local_registry: dict[str, DatasetMetadata] = {}
        self._openml_registry: dict[int, OpenMLDatasetMetadata] = {}
        self._name_to_openml_id: dict[str, int] = {}  # Маппинг имени на OpenML ID
        self._last_metadata: OpenMLDatasetMetadata | None = None  # Метаданные последнего загруженного датасета
        self._load_registry()

    def _load_registry(self) -> None:
        """Загрузить реестр датасетов из сохранённых метаданных.

        Загружает метаданные из JSON файлов в директории метаданных.
        Если файлы повреждены или отсутствуют, реестр остаётся пустым.
        """
        # Загрузка локальных датасетов
        local_metadata_file = self.metadata_dir / "local_registry.json"
        if local_metadata_file.exists():
            try:
                with open(local_metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError(f"Ожидался словарь, получен {type(data).__name__}")
                    for name, metadata_dict in data.items():
                        if not isinstance(metadata_dict, dict):
                            print(f"Предупреждение: пропущен некорректный элемент '{name}' в локальном реестре (ожидался словарь, получен {type(metadata_dict).__name__})")
                            continue
                        if "path" not in metadata_dict:
                            print(f"Предупреждение: пропущен элемент '{name}' без поля 'path' в локальном реестре")
                            continue
                        metadata_dict["path"] = Path(metadata_dict["path"])
                        self._local_registry[name] = DatasetMetadata(**metadata_dict)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                print(f"Предупреждение: не удалось загрузить локальный реестр: {e}")

        # Загрузка OpenML датасетов
        openml_metadata_file = self.metadata_dir / "openml_registry.json"
        if openml_metadata_file.exists():
            try:
                with open(openml_metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError(f"Ожидался словарь, получен {type(data).__name__}")
                    for dataset_id_str, metadata_dict in data.items():
                        if not isinstance(metadata_dict, dict):
                            print(f"Предупреждение: пропущен некорректный элемент '{dataset_id_str}' в OpenML реестре (ожидался словарь, получен {type(metadata_dict).__name__})")
                            continue
                        try:
                            dataset_id = int(dataset_id_str)
                        except ValueError:
                            print(f"Предупреждение: пропущен элемент с некорректным ID '{dataset_id_str}' в OpenML реестре")
                            continue
                        # Преобразуем datetime строки обратно в datetime объекты
                        if "upload_date" in metadata_dict and metadata_dict["upload_date"]:
                            metadata_dict["upload_date"] = datetime.fromisoformat(
                                metadata_dict["upload_date"]
                            )
                        # Преобразуем строки путей обратно в Path объекты
                        if "local_path" in metadata_dict and metadata_dict["local_path"]:
                            metadata_dict["local_path"] = Path(metadata_dict["local_path"])
                        self._openml_registry[dataset_id] = OpenMLDatasetMetadata(**metadata_dict)
                        if "name" in metadata_dict:
                            self._name_to_openml_id[metadata_dict["name"]] = dataset_id
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                print(f"Предупреждение: не удалось загрузить OpenML реестр: {e}")

    def _save_registry(self) -> None:
        """Сохранить реестр датасетов в JSON файлы.

        Сохраняет метаданные локальных и OpenML датасетов в отдельные JSON файлы
        для последующей загрузки при следующем запуске.
        """
        # Сохранение локальных датасетов
        local_metadata_file = self.metadata_dir / "local_registry.json"
        data = {
            name: metadata.model_dump(mode="json") for name, metadata in self._local_registry.items()
        }
        # Преобразуем Path в строку для JSON
        for name in data:
            data[name]["path"] = str(data[name]["path"])

        with open(local_metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Сохранение OpenML датасетов
        openml_metadata_file = self.metadata_dir / "openml_registry.json"
        data = {
            str(dataset_id): metadata.model_dump(mode="json")
            for dataset_id, metadata in self._openml_registry.items()
        }
        # Преобразуем datetime и Path в строки для JSON
        for dataset_id_str in data:
            if "upload_date" in data[dataset_id_str] and data[dataset_id_str]["upload_date"]:
                upload_date = data[dataset_id_str]["upload_date"]
                if isinstance(upload_date, datetime):
                    data[dataset_id_str]["upload_date"] = upload_date.isoformat()
            if "local_path" in data[dataset_id_str] and data[dataset_id_str]["local_path"]:
                local_path = data[dataset_id_str]["local_path"]
                if isinstance(local_path, Path):
                    data[dataset_id_str]["local_path"] = str(local_path)

        with open(openml_metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_openml_dataset(
        self, identifier: int | str, *, download: bool = False, cache: bool | None = None
    ) -> pd.DataFrame | None:
        """
        Получить данные датасета из OpenML по ID или имени.

        Args:
            identifier: OpenML ID (int) или имя датасета (str)
            download: Скачать ли данные датасета. Если False, возвращает None, но метаданные сохраняются
            cache: Кэшировать ли датасет (если None, используется self.cache_openml)

        Returns:
            DataFrame с данными датасета, если download=True, иначе None.
            Метаданные доступны через метод get_metadata() или get_dataset_info()
        """
        if openml is None:
            raise ImportError("Библиотека openml не установлена. Установите её: pip install openml")

        # Получаем метаданные (они всегда сохраняются в реестре)
        metadata = self._get_or_load_metadata(identifier)
        self._last_metadata = metadata  # Сохраняем для доступа через get_metadata()

        if not download:
            return None

        # Загружаем данные датасета
        dataset_id = metadata.dataset_id

        # Проверяем, есть ли уже скачанный файл локально
        if metadata.local_path and metadata.local_path.exists():
            return pd.read_csv(metadata.local_path)

        # Загружаем данные напрямую из OpenML
        openml_dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, _ = openml_dataset.get_data(target=openml_dataset.default_target_attribute)

        # Формируем DataFrame с признаками и целевой переменной
        df = pd.DataFrame(X)
        if y is not None:
            target_col = openml_dataset.default_target_attribute or "target"
            df[target_col] = y

        # Кэшируем данные локально, если включено кэширование
        should_cache = cache if cache is not None else self.cache_openml
        if should_cache:
            # Используем путь из метаданных (уже определён при создании)
            local_path = metadata.local_path
            if local_path:
                df.to_csv(local_path, index=False)
                # Обновляем метаданные в реестре
                self._openml_registry[dataset_id] = metadata
                self._save_registry()

        return df

    def _get_or_load_metadata(self, identifier: int | str) -> OpenMLDatasetMetadata:
        """Внутренний метод для получения или загрузки метаданных из OpenML.

        Сначала проверяет кэш, если метаданные не найдены - загружает из OpenML.

        Args:
            identifier: OpenML ID (int) или имя датасета (str)

        Returns:
            Метаданные датасета из OpenML

        Raises:
            ImportError: Если библиотека openml не установлена
            ValueError: Если датасет не найден в OpenML
        """
        if openml is None:
            raise ImportError("Библиотека openml не установлена.")

        # Определяем ID датасета
        if isinstance(identifier, str):
            # Поиск по имени в кэше
            if identifier in self._name_to_openml_id:
                dataset_id = self._name_to_openml_id[identifier]
            else:
                # Поиск в OpenML по имени
                datasets = openml.datasets.list_datasets(data_name=identifier, output_format="dataframe")
                if datasets.empty:
                    raise ValueError(f"Датасет с именем '{identifier}' не найден в OpenML.")
                dataset_id = int(datasets.iloc[0]["did"])
        else:
            dataset_id = identifier

        # Проверяем кэш - если метаданные уже загружены, возвращаем их
        if dataset_id in self._openml_registry:
            metadata = self._openml_registry[dataset_id]
            # Если local_path не установлен (старые метаданные), устанавливаем его
            if metadata.local_path is None:
                safe_name = metadata.name.replace("/", "_").replace("\\", "_").replace(":", "_")
                metadata.local_path = self.raw_data_dir / f"openml_{dataset_id}_{safe_name}.csv"
                self._openml_registry[dataset_id] = metadata
                self._save_registry()
            return metadata

        # Загружаем метаданные из OpenML (без скачивания данных)
        openml_dataset = openml.datasets.get_dataset(dataset_id, download_data=False)

        # Преобразуем в нашу модель метаданных
        metadata = self._convert_openml_to_metadata(openml_dataset)

        # Определяем тип задачи из качества или тегов, если не определён
        if metadata.dataset_type is None:
            metadata.dataset_type = self._infer_dataset_type(metadata)

        # Сохраняем в реестр для последующего использования
        self._openml_registry[dataset_id] = metadata
        self._name_to_openml_id[metadata.name] = dataset_id
        self._save_registry()

        return metadata

    def _convert_openml_to_metadata(self, openml_dataset: Any) -> OpenMLDatasetMetadata:
        """Преобразовать объект OpenML датасета в модель метаданных.

        Извлекает все доступные метаданные из объекта OpenML и преобразует
        их в структурированный формат OpenMLDatasetMetadata.

        Args:
            openml_dataset: Объект датасета из библиотеки openml

        Returns:
            Метаданные датасета в формате OpenMLDatasetMetadata
        """
        # Извлекаем метрики качества
        quality = {}
        if hasattr(openml_dataset, "qualities") and openml_dataset.qualities:
            quality = dict(openml_dataset.qualities)

        # Определяем тип задачи из метрик качества
        dataset_type = None
        if "NumberOfClasses" in quality:
            num_classes = quality.get("NumberOfClasses")
            if num_classes and isinstance(num_classes, (int, float)) and num_classes > 0:
                dataset_type = DatasetType.CLASSIFICATION
        elif "NumberOfNumericFeatures" in quality or "NumberOfSymbolicFeatures" in quality:
            # Если нет информации о классах, проверяем теги
            tags = getattr(openml_dataset, "tags", []) or []
            tags_lower = [t.lower() for t in tags]
            if "classification" in tags_lower:
                dataset_type = DatasetType.CLASSIFICATION
            elif "regression" in tags_lower:
                dataset_type = DatasetType.REGRESSION

        # Определяем путь к локальному файлу (даже если он еще не скачан)
        # Используем безопасное имя файла (заменяем недопустимые символы)
        safe_name = openml_dataset.name.replace("/", "_").replace("\\", "_").replace(":", "_")
        local_path = self.raw_data_dir / f"openml_{openml_dataset.dataset_id}_{safe_name}.csv"

        return OpenMLDatasetMetadata(
            dataset_id=openml_dataset.dataset_id,
            name=openml_dataset.name,
            version=openml_dataset.version,
            version_label=getattr(openml_dataset, "version_label", None),
            description=getattr(openml_dataset, "description", None),
            citation=getattr(openml_dataset, "citation", None),
            paper_url=getattr(openml_dataset, "paper_url", None),
            paper_reference=getattr(openml_dataset, "paper_reference", None),
            creator=getattr(openml_dataset, "creator", None),
            contributor=getattr(openml_dataset, "contributor", None),
            collection_date=getattr(openml_dataset, "collection_date", None),
            upload_date=getattr(openml_dataset, "upload_date", None),
            language=getattr(openml_dataset, "language", None),
            licence=getattr(openml_dataset, "licence", None),
            url=getattr(openml_dataset, "url", None),
            original_data_url=getattr(openml_dataset, "original_data_url", None),
            minio_url=getattr(openml_dataset, "minio_url", None),
            format=getattr(openml_dataset, "format", None),
            file_id=getattr(openml_dataset, "file_id", None),
            default_target_attribute=openml_dataset.default_target_attribute,
            ignore_attribute=openml_dataset.ignore_attribute,
            row_id_attribute=openml_dataset.row_id_attribute,
            num_rows=quality.get("NumberOfInstances"),
            num_features=quality.get("NumberOfFeatures"),
            num_classes=quality.get("NumberOfClasses"),
            num_missing_values=quality.get("NumberOfMissingValues"),
            quality=quality,
            dataset_type=dataset_type,
            tags=getattr(openml_dataset, "tags", []) or [],
            status=getattr(openml_dataset, "status", None),
            visibility=getattr(openml_dataset, "visibility", None),
            local_path=local_path,
        )

    def _infer_dataset_type(self, metadata: OpenMLDatasetMetadata) -> DatasetType:
        """Определить тип задачи ML из метаданных датасета.

        Использует эвристики на основе количества классов и тегов датасета.

        Args:
            metadata: Метаданные датасета

        Returns:
            Определённый тип задачи или UNKNOWN, если определить не удалось
        """
        # Если есть классы - это классификация
        if metadata.num_classes and metadata.num_classes > 0:
            return DatasetType.CLASSIFICATION

        # Проверяем теги для определения типа задачи
        tags_lower = [t.lower() for t in metadata.tags]
        if "classification" in tags_lower:
            return DatasetType.CLASSIFICATION
        elif "regression" in tags_lower:
            return DatasetType.REGRESSION
        elif "timeseries" in tags_lower or "time_series" in tags_lower:
            return DatasetType.TIME_SERIES

        return DatasetType.UNKNOWN

    def get_dataset_info(self, identifier: int | str | None = None) -> OpenMLDatasetMetadata:
        """
        Получить полные метаданные датасета из OpenML.

        Args:
            identifier: OpenML ID (int) или имя датасета (str). Если None, возвращает метаданные последнего загруженного датасета.

        Returns:
            Полные метаданные датасета из OpenML (OpenMLDatasetMetadata)
        """
        if identifier is None:
            if self._last_metadata is None:
                raise ValueError("Нет загруженных метаданных. Сначала вызовите get_openml_dataset() или укажите identifier.")
            return self._last_metadata
        return self._get_or_load_metadata(identifier)

    def get_metadata(self, identifier: int | str | None = None) -> OpenMLDatasetMetadata:
        """
        Получить метаданные датасета (алиас для get_dataset_info).

        Args:
            identifier: OpenML ID (int) или имя датасета (str). Если None, возвращает метаданные последнего загруженного датасета.

        Returns:
            Полные метаданные датасета из OpenML (OpenMLDatasetMetadata)
        """
        return self.get_dataset_info(identifier)

    @property
    def metadata(self) -> OpenMLDatasetMetadata:
        """
        Получить метаданные последнего загруженного датасета.

        Returns:
            Полные метаданные датасета из OpenML (OpenMLDatasetMetadata)

        Raises:
            ValueError: Если нет загруженных метаданных
        """
        if self._last_metadata is None:
            raise ValueError("Нет загруженных метаданных. Сначала вызовите get_openml_dataset().")
        return self._last_metadata

    # Методы для работы с локальными датасетами

    def _detect_format(self, file_path: Path) -> DatasetFormat:
        """Определить формат файла по расширению."""
        suffix = file_path.suffix.lower()
        format_mapping = {
            ".csv": DatasetFormat.CSV,
            ".parquet": DatasetFormat.PARQUET,
            ".json": DatasetFormat.JSON,
            ".xlsx": DatasetFormat.EXCEL,
            ".xls": DatasetFormat.EXCEL,
            ".h5": DatasetFormat.HDF5,
            ".hdf5": DatasetFormat.HDF5,
        }
        return format_mapping.get(suffix, DatasetFormat.UNKNOWN)

    def _load_dataframe(self, file_path: Path, file_format: DatasetFormat) -> pd.DataFrame:
        """Загрузить датасет из файла в pandas DataFrame.

        Поддерживает различные форматы файлов: CSV, Parquet, JSON, Excel, HDF5.

        Args:
            file_path: Путь к файлу датасета
            file_format: Формат файла

        Returns:
            DataFrame с данными датасета

        Raises:
            ValueError: Если формат не поддерживается
        """
        format_loaders = {
            DatasetFormat.CSV: pd.read_csv,
            DatasetFormat.PARQUET: pd.read_parquet,
            DatasetFormat.JSON: pd.read_json,
            DatasetFormat.EXCEL: pd.read_excel,
            DatasetFormat.HDF5: pd.read_hdf,
        }

        loader = format_loaders.get(file_format)
        if loader is None:
            raise ValueError(f"Неподдерживаемый формат: {file_format}")

        return loader(file_path)

    def register_dataset(
        self,
        name: str,
        file_path: Path | str,
        *,
        description: str | None = None,
        dataset_type: DatasetType | None = None,
        target_column: str | None = None,
        tags: list[str] | None = None,
        extra_info: dict[str, Any] | None = None,
        auto_detect: bool = True,
    ) -> DatasetMetadata:
        """
        Зарегистрировать новый локальный датасет.

        Args:
            name: Уникальное имя датасета
            file_path: Путь к файлу датасета
            description: Описание датасета
            dataset_type: Тип задачи ML (если None, будет определён автоматически)
            target_column: Название целевой колонки
            tags: Теги для категоризации
            extra_info: Дополнительная информация
            auto_detect: Автоматически определить метаданные из данных

        Returns:
            Метаданные зарегистрированного датасета
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл датасета не найден: {file_path}")

        if name in self._local_registry:
            raise ValueError(f"Датасет с именем '{name}' уже зарегистрирован.")

        file_format = self._detect_format(file_path)
        if file_format == DatasetFormat.UNKNOWN:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")

        metadata = DatasetMetadata(
            name=name,
            description=description,
            path=file_path,
            format=file_format,
            dataset_type=dataset_type or DatasetType.UNKNOWN,
            target_column=target_column,
            tags=tags or [],
            extra_info=extra_info or {},
        )

        # Автоматическое определение метаданных из данных
        if auto_detect:
            try:
                df = self._load_dataframe(file_path, file_format)
                metadata.num_rows = len(df)
                metadata.num_features = len(df.columns) - (1 if target_column else 0)
                metadata.feature_columns = [col for col in df.columns if col != target_column]

                # Определяем тип задачи на основе типа данных целевой переменной
                if dataset_type is None and target_column:
                    if df[target_column].dtype in ["object", "category"]:
                        metadata.dataset_type = DatasetType.CLASSIFICATION
                        metadata.num_classes = df[target_column].nunique()
                    else:
                        metadata.dataset_type = DatasetType.REGRESSION
            except (ValueError, KeyError, AttributeError, pd.errors.EmptyDataError) as e:
                print(f"Предупреждение: не удалось автоматически определить метаданные: {e}")

        self._local_registry[name] = metadata
        self._save_registry()
        return metadata

