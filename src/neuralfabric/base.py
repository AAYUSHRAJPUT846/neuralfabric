from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self


class BaseEstimator:
    def get_params(self) -> dict[str, Any]:
        return {
            key: value for key, value in self.__dict__.items() if not key.endswith("_")
        }

    def set_params(self, **params: Any) -> Self:
        for key, value in params.items():
            setattr(self, key, value)

        return self


class RegressorMixin(ABC):
    @abstractmethod
    def predict(self, X): ...

    def score(self, X, y) -> float:
        from neuralfabric.metrics import r2_score

        return r2_score(y, self.predict(X))


class ClassifierMixin(ABC):
    @abstractmethod
    def predict(self, X): ...

    def score(self, X, y) -> float:
        from neuralfabric.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


class TransformerMixin(ABC):
    @abstractmethod
    def fit(self, X, y=None): ...

    @abstractmethod
    def transform(self, X): ...

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
