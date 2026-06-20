from neuralfabric.core.tensor import Tensor
from neuralfabric.linear_model import LinearRegression
from neuralfabric.model_selection import train_test_split


def main() -> None:
    """
    House Price Prediction

    Features:
        - Area (sq ft)
        - Bedrooms

    Target:
        - House Price ($)
    """

    X = Tensor([
        [800, 2],
        [1000, 2],
        [1200, 3],
        [1500, 3],
        [1800, 4],
        [2000, 4],
        [2200, 5],
        [2500, 5],
    ])

    y = Tensor([
        [120000],
        [150000],
        [180000],
        [220000],
        [260000],
        [290000],
        [320000],
        [370000],
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = LinearRegression(
        lr=0.00000001,
        epochs=10000,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTest Predictions")
    print("----------------")
    print(predictions.data)

    print("\nActual Values")
    print("-------------")
    print(y_test.data)

    print("\nModel Parameters")
    print("----------------")
    print("Weights:")
    print(model.weight.data)

    print("\nBias:")
    print(model.bias.data)

    print("\nR² Score")
    print("--------")
    print(model.score(X_test, y_test))

    new_houses = Tensor([
        [1700, 3],
        [2400, 5],
    ])

    new_predictions = model.predict(new_houses)

    print("\nNew House Predictions")
    print("---------------------")

    for house, price in zip(
        new_houses.data,
        new_predictions.data.flatten(),
    ):
        area, bedrooms = house

        print(
            f"Area: {area:.0f} sq ft, "
            f"Bedrooms: {bedrooms:.0f} "
            f"-> Predicted Price: ${price:,.2f}"
        )


if __name__ == "__main__":
    main()