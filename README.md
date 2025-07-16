# EvoFormula: Interpretable Symbolic Regression with Evolving Functions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Built with NumPy](https://img.shields.io/badge/built%20with-NumPy-blue.svg)](https://numpy.org/)

**EvoFormula** is a symbolic regression engine that uses genetic programming to discover and evolve interpretable mathematical formulas that fit your data.
---

## Why EvoFormula?

While many machine learning models are "black boxes," symbolic regression aims to find the underlying mathematical equation that describes a dataset. **EvoFormula** enhances this process by automatically discovering and reusing common sub-expressions, just like a human mathematician would factor out and reuse parts of an equation. This leads to simpler, more intuitive models.

* 🧠 **Automated Function Discovery (ADFs):** Automatically identifies and reuses the most useful parts of equations (like `sin(x) * x`) as new building blocks, leading to more efficient discovery of elegant solutions.
* ⚖️ **Tunable Complexity:** You can control the simplicity of the final formula using a `complexity_weight`, balancing the trade-off between accuracy and interpretability.
* 🚀 **Parallelized Performance:** Leverages `joblib` for fast, multi-core fitness evaluations during evolution.
* 🧩 **Extensible Operator Set:** Easily add your own custom mathematical operators to the genetic programming engine.
* sklearn **-like API:** Integrates smoothly into data science workflows with a familiar `.fit()` and `.predict()` interface.

---

## How It Works 🧬

The core of **EvoFormula** is a **genetic programming** algorithm.

1.  **Initialization:** It starts with a large population of random mathematical formulas (trees).
2.  **Evaluation:** Each formula is evaluated on how well it fits the training data.
3.  **Selection & Evolution:** The best-performing formulas are selected to "breed." They are combined (**crossover**) and randomly changed (**mutation**) to create a new generation of formulas.
4.  **ADF Mining:** The system analyzes the best formulas and extracts useful sub-trees to create **Automatically Defined Functions (ADFs)**, which become new, powerful building blocks for future generations.
5.  **Termination:** The process repeats for many generations, a time limit is reached, or a target fitness is achieved, hopefully resulting in a formula that is both accurate and simple.

---

## Installation

Clone the repository and install the required dependencies. It's recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/LeonardoTorresHernandez/EvoFormula--Interpretable-Symbolic-Regression-with-Evolving-Functions.git
cd EvoFormula--Interpretable-Symbolic-Regression-with-Evolving-Functions

# Install dependencies
pip install -r requirements.txt
```

-----

## Quick Start

Using the `SymbolicRegressor` is designed to be straightforward.

```python
import numpy as np
from evo_formula.regressor import SymbolicRegressor
import matplotlib.pyplot as plt

# 1. Generate a complex target function
x = np.linspace(-5, 5, 400)
y = np.sin(x**2) * np.cos(x) - 1 + np.random.normal(0, 0.1, size=x.shape)

# 2. Instantiate and configure the regressor
#    - MaxRanTreeDepth: Initial depth of random trees
#    - n_adfs: Number of automatically defined functions to mine
#    - complexity_weight: Penalizes overly complex formulas
model = SymbolicRegressor(
    MaxRanTreeDepth=4,
    n_adfs=2,
    complexity_weight=0.005,
    max_generations=20
)

# 3. Fit the model to the data
model.fit(x, y)

# 4. Predict and visualize the results
y_pred = model.predict(x)

print(f"\nDiscovered Formula: {model.best_program}")
print(f"Final Fitness (MSE): {model.best_fitness_:.4f}")

# Plotting
plt.figure(figsize=(12, 7))
plt.scatter(x, y, alpha=0.4, label='Noisy Data')
plt.plot(x, y_pred, 'r-', linewidth=2.5, label='EvoFormula Fit')
plt.title('EvoFormula Prediction vs. Data', fontsize=16)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

-----

## Running Tests

The test suite includes several canonical problems for symbolic regression. To run all tests and save the resulting plots and metrics:

```bash
python evo_formula/tests/test_symbolic_regressor.py
```

All outputs will be saved in the `results/` directory.

-----

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines on how to submit pull requests, report issues, or suggest enhancements.

## License

This project is licensed under the MIT License.
