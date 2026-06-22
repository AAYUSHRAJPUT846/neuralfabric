from __future__ import annotations

import pytest

from neuralfabric.core.tensor import Tensor
from neuralfabric.linear_model import LogisticRegression


def test_fit_learns_and_gate() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression(
        lr=0.1,
        epochs=5000,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert (predictions.data == y.data).all()


def test_predict_before_fit_raises_error() -> None:
    model = LogisticRegression()

    with pytest.raises(
        RuntimeError,
        match="must be fitted before prediction",
    ):
        model.predict(Tensor([[0.0, 0.0]]))


def test_parameters_before_fit() -> None:
    model = LogisticRegression()

    assert model.parameters() == []


def test_parameters_exist_after_fit() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression()

    model.fit(X, y)

    params = model.parameters()

    assert len(params) == 2
    assert params[0] is model.weight
    assert params[1] is model.bias


def test_prediction_shape() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression(
        lr=0.1,
        epochs=5000,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape


def test_score_returns_high_accuracy() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression(
        lr=0.1,
        epochs=5000,
    )

    model.fit(X, y)

    assert model.score(X, y) == 1.0


def test_predict_proba_returns_values_between_zero_and_one() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression(
        lr=0.1,
        epochs=3000,
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert (probabilities.data >= 0.0).all()
    assert (probabilities.data <= 1.0).all()


def test_invalid_learning_rate() -> None:
    with pytest.raises(
        ValueError,
        match="lr must be positive",
    ):
        LogisticRegression(lr=0.0)


def test_invalid_epochs() -> None:
    with pytest.raises(
        ValueError,
        match="epochs must be positive",
    ):
        LogisticRegression(epochs=0)


def test_fit_requires_2d_input() -> None:
    X = Tensor([0.0, 1.0, 0.0])

    y = Tensor(
        [
            [0.0],
            [1.0],
            [0.0],
        ]
    )

    model = LogisticRegression()

    with pytest.raises(
        ValueError,
        match="X must be a 2D tensor",
    ):
        model.fit(X, y)


def test_fit_requires_matching_samples() -> None:
    X = Tensor(
        [
            [0.0],
            [1.0],
            [0.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression()

    with pytest.raises(
        ValueError,
        match="same number of samples",
    ):
        model.fit(X, y)


def test_invalid_threshold() -> None:
    X = Tensor(
        [
            [0.0],
            [1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression()

    model.fit(X, y)

    with pytest.raises(
        ValueError,
        match="threshold must be between 0 and 1",
    ):
        model.predict(
            X,
            threshold=1.5,
        )


def test_repr() -> None:
    model = LogisticRegression(
        lr=0.1,
        epochs=100,
    )

    assert repr(model) == "LogisticRegression(lr=0.1, epochs=100)"
