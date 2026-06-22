from __future__ import annotations

from neuralfabric.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from neuralfabric.core.tensor import Tensor


class DummyEstimator(BaseEstimator):
    def __init__(self) -> None:
        self.alpha = 1.0
        self.beta = 2.0
        self.fitted_ = True


class DummyRegressor(RegressorMixin):
    def predict(self, X):
        return Tensor([1.0, 2.0, 3.0])


class DummyClassifier(ClassifierMixin):
    def predict(self, X):
        return Tensor([1, 0, 1])


class DummyTransformer(TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def transform(self, X):
        return Tensor([x * 2 for x in X])


def test_get_params() -> None:
    estimator = DummyEstimator()

    assert estimator.get_params() == {
        "alpha": 1.0,
        "beta": 2.0,
    }


def test_set_params() -> None:
    estimator = DummyEstimator()

    result = estimator.set_params(
        alpha=10.0,
        gamma=3.0,
    )

    assert result is estimator
    assert estimator.alpha == 10.0
    assert estimator.gamma == 3.0


def test_regressor_score() -> None:
    model = DummyRegressor()

    score = model.score(
        None,
        Tensor([1.0, 2.0, 3.0]),
    )

    assert score == 1.0


def test_classifier_score() -> None:
    model = DummyClassifier()

    score = model.score(
        None,
        Tensor([1, 0, 1]),
    )

    assert score == 1.0


def test_fit_transform() -> None:
    transformer = DummyTransformer()

    result = transformer.fit_transform(
        [1, 2, 3],
    )

    assert transformer.is_fitted is True
    assert result.data.tolist() == [2, 4, 6]
