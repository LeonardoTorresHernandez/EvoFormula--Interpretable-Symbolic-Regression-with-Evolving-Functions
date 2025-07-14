# Symbolic Regression with Automatically Defined Functions (ADFs)

**Author:** Leonardo Torres Hernández

## Overview
This project implements a powerful symbolic regression engine using genetic programming with support for Automatically Defined Functions (ADFs). The system is designed to evolve interpretable mathematical expressions that fit data, leveraging advanced techniques such as:
- Evolving and mining reusable ADFs from the best sub-expressions in the population
- Complexity control via parsimony pressure (complexity penalty)
- Adaptive mutation and diversity injection to avoid premature convergence
- Full support for custom operators, including power (**), trigonometric, and exponential/logarithmic functions
- Sklearn-like `SymbolicRegressor` API for easy integration and experimentation

## Features
- **Global, evolving ADF pool**: The system automatically discovers and reuses the most useful sub-expressions as ADFs, improving search efficiency and model interpretability.
- **Complexity control**: Penalize or reward model complexity using the `complexity_weight` hyperparameter.
- **Parallelized evaluation**: Fast fitness evaluation and genetic operations using joblib.
- **Flexible operator set**: Easily extendable to new operators and function sets.
- **Comprehensive test suite**: Demonstrates the model's ability to fit sinusoids, polynomials, and composite functions.

## Installation
Clone the repository and install the required dependencies:
```bash
pip install -r symbolic_regression_project/requirements.txt
```

## Usage
### Example: Fitting a Function
```python
from symbolic_regression.operators import SymbolicRegressor
import numpy as np

x = np.linspace(-2, 2, 200)
y = np.sin(x) * np.cos(x) + x**2
model = SymbolicRegressor(StartTrees=500, MaxRanTreeDepth=5, max_generations=10, n_adfs=2, complexity_weight=0.01)
model.fit(x, y)
y_pred = model.predict(x)
```

### Running the Test Suite
To run all tests and save results to the `results/` directory:
```bash
python symbolic_regression_project/symbolic_regression/tests/test_symbolic_regressor.py
```

## Results
- All test outputs (plots and metrics) are saved in the `results/` directory.
- Each test demonstrates the model's ability to recover the underlying function and balance accuracy with simplicity.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
This project is licensed under the MIT License.

## Author
**Leonardo Torres Hernández**
