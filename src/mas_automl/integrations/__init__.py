"""Адаптеры подключения AutoML-фреймворков и внешних сервисов."""

from .autogluon import AutoGluonAdapter
from .autosklearn import AutoSklearnAdapter
from .fedot import FedotAdapter

__all__ = ["AutoGluonAdapter", "AutoSklearnAdapter", "FedotAdapter"]

