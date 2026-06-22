# Contributing to NeuralFabric

Thank you for your interest in contributing to NeuralFabric. We welcome contributions of all sizes, including bug fixes, documentation improvements, new machine learning algorithms, performance enhancements, and test coverage improvements.

## Getting Started

Clone the repository and install the development dependencies:

```bash
git clone https://github.com/aryanap07/neuralfabric.git
cd neuralfabric

make dev
pre-commit install
```

## Development Workflow

1. Fork the repository.
2. Create a feature branch from `main`.

```bash
git checkout -b feature/my-feature
```

3. Implement your changes.
4. Add or update tests.
5. Run all quality checks.

```bash
make format
make lint
make test
```

6. Commit your changes using a clear and descriptive commit message.

```bash
git commit -m "feat: add decision tree classifier"
```

7. Push your branch and open a Pull Request.

## Coding Standards

* Follow PEP 8 guidelines.
* Use type annotations whenever appropriate.
* Write clear, maintainable, and modular code.
* Keep functions and classes focused on a single responsibility.
* Prefer descriptive variable and function names.
* Avoid unnecessary dependencies.

## Project Structure

### Machine Learning Models

Place estimators in the appropriate package:

```text
src/neuralfabric/

├── linear_model/     # Linear Regression, Logistic Regression
├── tree/             # Decision Trees
├── ensemble/         # Random Forests, Gradient Boosting
├── svm/              # Support Vector Machines
├── cluster/          # Clustering Algorithms
└── nn/               # Neural Network Components
```

All estimators should implement a consistent API:

```python
model.fit(X, y)
model.predict(X)
```

### Neural Network Components

Neural network implementations should be placed under:

```text
src/neuralfabric/nn/
```

Core building blocks include:

```python
Tensor
Module
Parameter
```

### Transformer Components

Transformer-related implementations belong in:

```text
src/neuralfabric/transformer/
```

## Testing

Every new feature or bug fix must include appropriate tests.

Example:

```text
src/neuralfabric/linear_model/logistic_regression.py
tests/linear_model/test_logistic_regression.py
```

Run the test suite with:

```bash
make test
```

### Test Guidelines

* Write unit tests for new functionality.
* Cover edge cases whenever possible.
* Ensure existing tests continue to pass.
* Maintain or improve overall test coverage.

## Documentation

When adding new features:

* Update relevant documentation.
* Include usage examples when appropriate.
* Keep docstrings concise and informative.

Example:

```python
def predict(X):
    """Predict target values for input samples."""
```

## Pull Request Guidelines

Before submitting a Pull Request, ensure:

* All tests pass.
* Code is formatted correctly.
* Linting passes without errors.
* Documentation has been updated if necessary.
* The Pull Request includes a clear description of the changes.

### Pull Request Checklist

* [ ] Code follows project conventions
* [ ] Tests added or updated
* [ ] Documentation updated
* [ ] Quality checks pass
* [ ] Ready for review

## Reporting Issues

When creating an issue, please provide:

* A clear description of the problem
* Steps to reproduce
* Expected behavior
* Actual behavior
* Python version
* Operating system
* Relevant error messages or logs

## Feature Requests

Feature requests are welcome. Please describe:

* The problem you are trying to solve
* The proposed solution
* Any alternative approaches considered

## License

By contributing to NeuralFabric, you agree that your contributions will be licensed under the MIT License.
