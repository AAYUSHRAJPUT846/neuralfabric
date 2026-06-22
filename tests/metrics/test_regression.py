from __future__ import annotations

from neuralfabric.core.tensor import Tensor
from neuralfabric.metrics import r2_score


def test_r2_score_perfect_fit() -> None:
    y_true = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]
    )

    y_pred = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]
    )

    assert r2_score(y_true, y_pred) == 1.0


def test_r2_score_less_than_one_for_imperfect_fit() -> None:
    y_true = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]
    )

    y_pred = Tensor(
        [
            [1.2],
            [1.8],
            [3.1],
            [3.9],
        ]
    )

    score = r2_score(
        y_true,
        y_pred,
    )

    assert score < 1.0


def test_r2_score_zero_total_variance() -> None:
    y_true = Tensor(
        [
            [5.0],
            [5.0],
            [5.0],
        ]
    )

    y_pred = Tensor(
        [
            [5.0],
            [5.0],
            [5.0],
        ]
    )

    assert r2_score(y_true, y_pred) == 0.0
