# Author: Leonardo Torres Hernández
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- BEGIN: Add parent directory to sys.path if needed ---
try:
    from symbolic_regression.operators import genetic_algorithm
    from symbolic_regression import tree as tr
except ImportError:
    # If running directly, add parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from symbolic_regression.operators import genetic_algorithm
        from symbolic_regression import tree as tr
    except ImportError as e:
        print("Failed to import symbolic_regression modules even after sys.path adjustment.\n"
              "Make sure you are running this script from the project root or that the package structure is correct.\n"
              f"Error: {e}")
        raise
# --- END: Add parent directory to sys.path if needed ---

def test():
    # 1. Generate the dataset (univariate)
    np.random.seed(42)
    n_samples = 200
    x = np.random.uniform(-2, 2, size=n_samples)
    # True function
    y = np.sin(x)**2 * np.cos(x)+ x**2

    # 2. Build ADF arities and ADFs dicts (matching SymbolicRegressor defaults)
    adf_arities = {"ADF0": 1, "ADF1": 2, "ADF2": 1, "ADF3": 2}
    adfs = {
        "ADF0": tr.tree(tr.RanADFTree(max_depth=5, arity=1), adf_arities=adf_arities, adfs=None)
    }

    # 3. Direct call to genetic_algorithm
    best_tree, best_fitness, generation, fitness_history, significant_trees = genetic_algorithm(
        x, y,
        StartTrees=10000,
        Percentage=10,
        MaxRanTreeDepth=3,
        mutation_rate=0.9,
        max_generations=10,
        target_fitness=0.99,
        adf_arities=adf_arities,
        adfs=adfs,
        n_adfs=1,
        complexity_weight=0.1
    )
    pred_direct = best_tree.evaluate(x)

    # 4. Print and compare results
    print("First 5 predictions (genetic_algorithm):", pred_direct[:5])
    print("First 5 true values:", y[:5])


    # Overlay plot: x vs true and x vs predicted
    plt.figure(figsize=(8, 6))
    sort_idx = np.argsort(x)
    plt.plot(x[sort_idx], y[sort_idx], label='True y', color='blue')
    plt.plot(x[sort_idx], pred_direct[sort_idx], label='Predicted y', color='orange', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Symbolic Regression: True and Predicted Functions')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()
