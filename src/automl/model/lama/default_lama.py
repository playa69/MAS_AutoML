import warnings

import numpy as np
import pandas as pd
from torch import set_num_threads as set_num_threads_torch

from ...loggers import get_logger
from ..base import BaseModel
from ..type_hints import FeaturesType, TargetType
from ..utils import CatchLamaLogs, convert_to_pandas, seed_everything

from typing import Optional, Callable, List, Union
warnings.filterwarnings("ignore")


log = get_logger(__name__)

class TabularLamaBase(BaseModel):
    def __init__(
        self,
        model_type: str,
        random_state: int = 42,
        time_series: bool = False,
        timeout: int = 60,
        verbose: int = -1,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Callable]] = None,
        use_algos: Optional[List[List[str]]] = None,
        **kwargs,
    ):
        super().__init__(
            model_type=model_type,
            random_state=random_state,
            time_series=time_series,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            )
        
        if self.model_type == 'classification':
            self.task: str = 'binary'
        elif self.model_type == 'regression':
            self.task = "reg"
            
        self.timeout = timeout
        self.n_folds = n_splits
        self.use_algos = use_algos  # None по умолчанию - используются все алгоритмы LightAutoML. Можно ограничить, например: [["linear_l2"]] или [["linear_l2"], ["lgb"]]
        
        set_num_threads_torch(self.n_jobs)
        self._not_inner_model_params += ['timeout', 'task', 'n_folds', 'model_class', 'use_algos']
        
    def _prepare(self, X: FeaturesType, y: Optional[TargetType] = None, categorical_feature: Optional[List[Union[str, int]]] = None):
        seed_everything(self.random_state)
        X, y = self._prepare_data(X, y, categorical_feature)
        
        if y is not None:
            X = X.assign(target=y)
            if self.model_type == "classification":
                self.num_class = np.unique(y).shape[0]
                # correct objective based on the number of classes
                if self.model_type == 'classification' and self.num_class > 2:
                    self.task = "multiclass"
        
        # Convert CategoricalDtype columns to object or numeric types
        # LightAutoML cannot handle CategoricalDtype directly
        # Do this after adding target so we also convert target if needed
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if pd.api.types.is_categorical_dtype(X[col]):
                    # Convert categorical to object (string) type
                    # This preserves the values but changes the dtype
                    X[col] = X[col].astype(str)
        
        if y is not None:
            return X, y
        return X
        
    def fit(self, X: FeaturesType, y: TargetType, categorical_feature: Optional[List[Union[str, int]]] = None):
        # Lazy import to avoid log_calls compatibility issues with object class
        from lightautoml.tasks import Task
        from lightautoml.validation.np_iterators import TimeSeriesIterator
        
        log.info(f"Fitting {self.name}", msg_type="start")

        data, _ = self._prepare(X, y, categorical_feature)

        # Подготовка параметров для TabularAutoML
        model_kwargs = {
            "task": Task(
                name=self.task,
                # metric=self.scorer.score,
                # greater_is_better=self.scorer.greater_is_better,
            ),
            "timeout": self.timeout,  # Убрано умножение на 2 для более точного контроля времени
            "cpu_limit": self.n_jobs,
            "reader_params": {
                "n_jobs": 1,
                "cv": self.n_folds,
                "random_state": self.random_state,
            },
        }
        
        # Добавляем general_params только если указаны алгоритмы для ограничения
        if self.use_algos is not None:
            model_kwargs["general_params"] = {"use_algos": self.use_algos}

        model = self.model_class(**model_kwargs)

        roles = {"target": "target"}
        if len(self.categorical_feature) > 0:
            roles["category"] = self.categorical_feature

        with CatchLamaLogs(log):
            if self.time_series:
                artificial_index = np.arange(X.shape[0])
                oof_preds = model.fit_predict(
                    data,
                    roles=roles,
                    verbose=self.verbose,
                    cv_iter=TimeSeriesIterator(artificial_index),
                ).data
            else:
                oof_preds = model.fit_predict(
                    data, roles=roles, verbose=self.verbose
                ).data

        # flatten the output in regression case
        # and add 0 class probabilities in binary case
        if self.task == "reg":
            oof_preds = oof_preds.reshape(oof_preds.shape[0])
        elif self.task == "binary":
            oof_preds = np.hstack((1 - oof_preds, oof_preds))

        self.model = model

        log.info(f"Fitting {self.name}", msg_type="end")
        return oof_preds
    
    def tune(
        self, 
        X: FeaturesType, y: TargetType, 
        timeout: int = 60, 
        categorical_feature: Optional[List[str]] = None,
        ):
        self.timeout = timeout
        self._prepare_categorical_features(X, categorical_feature)
        
    def _predict(self, X):
        X = self._prepare(X, categorical_feature=self.categorical_feature)
        y_pred = self.model.predict(X).data

        # flatten the output in regression case
        # and add 0 class probabilities in binary case
        if self.task == "reg":
            y_pred = y_pred.reshape(y_pred.shape[0])
        elif self.task == "binary":
            y_pred = np.hstack((1 - y_pred, y_pred))

        return y_pred
    
    @property
    def inner_params(self):
        return {
        }


class TabularLamaClassification(TabularLamaBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        timeout: int = 60,
        verbose: int = -1,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Callable]] = None,
        **kwargs,
    ):
        # Lazy import to avoid log_calls compatibility issues with object class
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        
        super().__init__(
            model_type='classification',
            random_state=random_state,
            time_series=time_series,
            timeout=timeout,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            **kwargs,
            )
        self.name: str = "TabularLama"
        self.model_class = TabularAutoML
        
class TabularLamaRegression(TabularLamaBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        timeout: int = 60,
        verbose: int = -1,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Callable]] = None,
        **kwargs,
    ):
        # Lazy import to avoid log_calls compatibility issues with object class
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        
        super().__init__(
            model_type='regression',
            random_state=random_state,
            time_series=time_series,
            timeout=timeout,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            **kwargs,
            )
        self.name: str = "TabularLama"
        self.model_class = TabularAutoML

class TabularLamaUtilizedClassification(TabularLamaBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        timeout: int = 60,
        verbose: int = -1,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Callable]] = None,
        **kwargs,
    ):
        # Lazy import to avoid log_calls compatibility issues with object class
        from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
        
        super().__init__(
            model_type='classification',
            random_state=random_state,
            time_series=time_series,
            timeout=timeout,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            **kwargs,
            )
        self.name: str = "TabularLamaUtilized"
        self.model_class = TabularUtilizedAutoML
        

class TabularLamaUtilizedRegression(TabularLamaBase):
    def __init__(
        self,
        random_state: int = 42,
        time_series: bool = False,
        timeout: int = 60,
        verbose: int = -1,
        device_type: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        eval_metric: Optional[Union[str, Callable]] = None,
        **kwargs,
    ):
        # Lazy import to avoid log_calls compatibility issues with object class
        from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
        
        super().__init__(
            model_type='regression',
            random_state=random_state,
            time_series=time_series,
            timeout=timeout,
            verbose=verbose,
            device_type=device_type,
            n_jobs=n_jobs,
            n_splits=n_splits,
            eval_metric=eval_metric,
            **kwargs,
            )
        self.name: str = "TabularLamaUtilized"
        self.model_class = TabularUtilizedAutoML