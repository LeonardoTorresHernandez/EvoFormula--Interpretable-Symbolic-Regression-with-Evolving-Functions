# Contributing to Symbolic Regression with AFD

Thank you for your interest in contributing to this project! We welcome contributions that uphold the highest standards of mathematical rigor, code quality, and scientific reproducibility.

## Table of Contents
- [Project Philosophy](#project-philosophy)
- [Getting Started](#getting-started)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Requests](#pull-requests)
- [Code Review](#code-review)
- [Attribution and Licensing](#attribution-and-licensing)

---

## Project Philosophy
This project is built on the principles of:
- **Mathematical rigor**: All algorithms and code should be grounded in sound mathematical reasoning.
- **Reproducibility**: Results and experiments must be reproducible from code and data.
- **Transparency**: Code, documentation, and results should be clear and accessible.

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Symbolic-regression-with-AFD.git
   cd Symbolic-regression-with-AFD
   ```
2. **Set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r symbolic_regression_project/requirements.txt
   ```
4. **(Optional) Install additional tools for development:**
   - `pytest` for testing
   - `flake8` or `black` for code style

## Coding Standards
- Use **absolute imports** from the project root.
- Place all test/demo code under `if __name__ == "__main__":`.
- Include type annotations and docstrings for all public functions and classes.
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style.
- Avoid unnecessary `sys.path` manipulations; standardize import patterns.
- Use clear, descriptive variable and function names, especially for mathematical objects.
- Include debug prints (e.g., `print(__name__, tr)`) to confirm module identity when relevant.

## Testing
- All new features and bug fixes must include appropriate tests.
- Place tests in the `symbolic_regression_project/symbolic_regression/tests/` directory.
- Tests should be deterministic and reproducible (set random seeds where appropriate).
- Save all test outputs and results to the `results/` directory.
- Use `pytest` or the built-in `unittest` framework.

## Documentation
- Update the `README.md` with any major changes to features or usage.
- All public classes and functions must have clear docstrings explaining their purpose, parameters, and return values.
- If you add new modules, include a module-level docstring describing its purpose.

## Pull Requests
- Fork the repository and create a new branch for your feature or bugfix.
- Ensure your branch is up to date with `main` before submitting a PR.
- Write a clear, descriptive PR message outlining the motivation and changes.
- Reference any related issues or discussions.
- Ensure all tests pass before submitting.

## Code Review
- All code will be reviewed for correctness, clarity, and adherence to project philosophy.
- Be prepared to discuss mathematical and algorithmic choices.
- Address all review comments before merging.

## Attribution and Licensing
- Ensure the author is correctly attributed as **Leonardo Torres Hernández** in all relevant files and metadata.
- All contributions are subject to the project’s [LICENSE](../LICENSE).

---

By contributing, you agree to abide by these guidelines and uphold the standards of the project. Thank you for helping advance open, rigorous scientific software! 