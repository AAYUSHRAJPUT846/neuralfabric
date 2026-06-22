from __future__ import annotations

import numpy as np
import pytest

from neuralfabric.core.tensor import Tensor


def test_tensor_properties() -> None:
    x = Tensor([[1.0, 2.0]])

    assert x.shape == (1, 2)
    assert x.ndim == 2
    assert isinstance(x.dtype, np.dtype)


def test_tensor_from_tensor() -> None:
    x = Tensor([1.0, 2.0])
    y = Tensor(x)

    assert np.array_equal(x.data, y.data)


def test_integer_input_casts_to_float() -> None:
    x = Tensor([1, 2, 3])

    assert np.issubdtype(x.dtype, np.floating)


def test_item() -> None:
    assert Tensor(42.0).item() == 42.0


def test_numpy() -> None:
    assert isinstance(Tensor([1.0]).numpy(), np.ndarray)


def test_detach() -> None:
    x = Tensor([1.0], requires_grad=True)
    y = x.detach()

    assert y.requires_grad is False
    assert np.array_equal(x.data, y.data)


def test_requires_grad_inplace() -> None:
    x = Tensor([1.0])

    result = x.requires_grad_(True)

    assert result is x
    assert x.requires_grad is True


def test_repr() -> None:
    x = Tensor([1.0], requires_grad=True)

    assert repr(x) == "Tensor([1.], requires_grad=True)"


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (Tensor([1.0]) + Tensor([2.0]), 3.0),
        (Tensor([5.0]) - Tensor([2.0]), 3.0),
        (Tensor([3.0]) * Tensor([2.0]), 6.0),
        (Tensor([8.0]) / 2.0, 4.0),
        (2.0 + Tensor([1.0]), 3.0),
        (5.0 - Tensor([2.0]), 3.0),
        (2.0 * Tensor([3.0]), 6.0),
        (8.0 / Tensor([2.0]), 4.0),
        (Tensor([3.0]) ** 2, 9.0),
        (-Tensor([2.0]), -2.0),
    ],
)
def test_arithmetic_operations(
    result: Tensor,
    expected: float,
) -> None:
    assert result.item() == pytest.approx(expected)


def test_power_requires_scalar() -> None:
    with pytest.raises(
        AssertionError,
        match="only scalar powers are supported",
    ):
        Tensor([2.0]) ** Tensor([2.0])


def test_matmul() -> None:
    x = Tensor([[1.0, 2.0]])
    y = Tensor([[3.0], [4.0]])

    assert (x @ y).item() == 11.0


def test_transpose() -> None:
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    y = x.T

    assert y.shape == (2, 2)

    assert np.array_equal(
        y.data,
        np.array(
            [
                [1.0, 3.0],
                [2.0, 4.0],
            ]
        ),
    )


def test_reshape() -> None:
    x = Tensor([1.0, 2.0, 3.0, 4.0])

    assert x.reshape(2, 2).shape == (2, 2)
    assert x.reshape((2, 2)).shape == (2, 2)


def test_sum() -> None:
    x = Tensor([1.0, 2.0, 3.0])

    assert x.sum().item() == 6.0


def test_sum_keepdims() -> None:
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    y = x.sum(axis=0, keepdims=True)

    assert y.shape == (1, 2)

    assert np.array_equal(
        y.data,
        np.array([[4.0, 6.0]]),
    )


def test_mean() -> None:
    x = Tensor([1.0, 2.0, 3.0])

    assert x.mean().item() == 2.0


def test_mean_axis() -> None:
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    y = x.mean(axis=0)

    assert np.array_equal(
        y.data,
        np.array([2.0, 3.0]),
    )


def test_mean_keepdims() -> None:
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    y = x.mean(axis=0, keepdims=True)

    assert np.array_equal(
        y.data,
        np.array([[2.0, 3.0]]),
    )


def test_exp() -> None:
    assert Tensor([0.0]).exp().item() == pytest.approx(1.0)


def test_log() -> None:
    assert Tensor([1.0]).log().item() == pytest.approx(0.0)


def test_relu() -> None:
    x = Tensor([-1.0, 0.0, 2.0])

    assert np.array_equal(
        x.relu().data,
        np.array([0.0, 0.0, 2.0]),
    )


def test_sigmoid() -> None:
    assert Tensor([0.0]).sigmoid().item() == pytest.approx(0.5)


def test_tanh() -> None:
    assert Tensor([0.0]).tanh().item() == pytest.approx(0.0)


def test_clip() -> None:
    x = Tensor([-1.0, 0.5, 2.0])

    y = x.clip(0.0, 1.0)

    assert np.array_equal(
        y.data,
        np.array([0.0, 0.5, 1.0]),
    )


def test_backward_scalar_gradient() -> None:
    x = Tensor([2.0], requires_grad=True)

    y = x * x

    y.backward()

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(4.0)


def test_backward_with_explicit_gradient() -> None:
    x = Tensor([2.0], requires_grad=True)

    y = x * x

    y.backward(np.array([3.0]))

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(12.0)


def test_gradient_accumulates() -> None:
    x = Tensor([2.0], requires_grad=True)

    (x + x).backward()

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(2.0)


def test_zero_grad() -> None:
    x = Tensor([2.0], requires_grad=True)

    (x * x).backward()

    x.zero_grad()

    assert x.grad is None


def test_broadcast_add_backward() -> None:
    x = Tensor(
        [[1.0], [2.0]],
        requires_grad=True,
    )

    y = Tensor(
        [[1.0, 2.0]],
        requires_grad=True,
    )

    z = (x + y).sum()

    z.backward()

    assert np.array_equal(
        x.grad,
        np.array([[2.0], [2.0]]),
    )

    assert np.array_equal(
        y.grad,
        np.array([[2.0, 2.0]]),
    )


def test_exp_backward() -> None:
    x = Tensor([1.0], requires_grad=True)

    y = x.exp()

    y.backward()

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(np.e)


def test_log_backward() -> None:
    x = Tensor([1.0], requires_grad=True)

    y = x.log()

    y.backward()

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(1.0)


def test_sigmoid_backward() -> None:
    x = Tensor([0.0], requires_grad=True)

    y = x.sigmoid()

    y.backward()

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(0.25)


def test_tanh_backward() -> None:
    x = Tensor([0.0], requires_grad=True)

    y = x.tanh()

    y.backward()

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(1.0)


def test_relu_backward() -> None:
    x = Tensor(
        [-1.0, 0.0, 2.0],
        requires_grad=True,
    )

    x.relu().sum().backward()

    assert np.array_equal(
        x.grad,
        np.array([0.0, 0.0, 1.0]),
    )


def test_clip_backward() -> None:
    x = Tensor(
        [-1.0, 0.5, 2.0],
        requires_grad=True,
    )

    x.clip(0.0, 1.0).sum().backward()

    assert np.array_equal(
        x.grad,
        np.array([0.0, 1.0, 0.0]),
    )


def test_transpose_backward() -> None:
    x = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )

    x.T.sum().backward()

    assert np.array_equal(
        x.grad,
        np.ones_like(x.data),
    )


def test_reshape_backward() -> None:
    x = Tensor(
        [1.0, 2.0, 3.0, 4.0],
        requires_grad=True,
    )

    x.reshape(2, 2).sum().backward()

    assert np.array_equal(
        x.grad,
        np.ones_like(x.data),
    )


def test_matmul_backward_both_operands() -> None:
    x = Tensor(
        [[1.0, 2.0]],
        requires_grad=True,
    )

    y = Tensor(
        [[3.0], [4.0]],
        requires_grad=True,
    )

    z = (x @ y).sum()

    z.backward()

    assert np.array_equal(
        x.grad,
        np.array([[3.0, 4.0]]),
    )

    assert np.array_equal(
        y.grad,
        np.array([[1.0], [2.0]]),
    )


def test_sum_axis_backward() -> None:
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    y = x.sum(axis=0)

    y.backward(np.ones_like(y.data))

    assert np.array_equal(
        x.grad,
        np.ones_like(x.data),
    )
