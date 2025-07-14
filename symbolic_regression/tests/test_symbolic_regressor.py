# Author: Leonardo Torres Hernández
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import matplotlib.pyplot as plt
from symbolic_regression.operators import SymbolicRegressor

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_plot(x, y_true, y_pred, name):
    plt.figure(figsize=(8, 6))
    sort_idx = np.argsort(x)
    plt.plot(x[sort_idx], y_true[sort_idx], label='True y', color='blue')
    plt.plot(x[sort_idx], y_pred[sort_idx], label='Predicted y', color='orange', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Symbolic Regression: {name}')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f'{name}_plot.png')
    plt.savefig(out_path)
    plt.close()
    return out_path

def save_metrics(y_true, y_pred, name):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    out_path = os.path.join(RESULTS_DIR, f'{name}_metrics.txt')
    with open(out_path, 'w') as f:
        f.write(f'MSE: {mse}\nMAE: {mae}\nR2: {r2}\n')
    return out_path

def test_sin():
    np.random.seed(0)
    x = np.linspace(-3, 3, 200)
    y = np.sin(x)
    model = SymbolicRegressor(StartTrees=500, MaxRanTreeDepth=4, max_generations=8, n_adfs=2, complexity_weight=0.01)
    model.fit(x, y)
    y_pred = model.predict(x)
    save_plot(x, y, y_pred, 'sin')
    save_metrics(y, y_pred, 'sin')

def test_poly():
    np.random.seed(1)
    x = np.linspace(-2, 2, 200)
    y = 2 * x**3 - 3 * x**2 + x + 5
    model = SymbolicRegressor(StartTrees=500, MaxRanTreeDepth=4, max_generations=8, n_adfs=2, complexity_weight=0.01)
    model.fit(x, y)
    y_pred = model.predict(x)
    save_plot(x, y, y_pred, 'poly')
    save_metrics(y, y_pred, 'poly')

def test_composite():
    np.random.seed(2)
    x = np.linspace(-2, 2, 200)
    y = np.sin(x) * np.cos(x) + x**2
    model = SymbolicRegressor(StartTrees=500, MaxRanTreeDepth=5, max_generations=10, n_adfs=2, complexity_weight=0.01)
    model.fit(x, y)
    y_pred = model.predict(x)
    save_plot(x, y, y_pred, 'composite')
    save_metrics(y, y_pred, 'composite')

def run_all():
    test_sin()
    test_poly()
    test_composite()
    print('All tests completed. Results saved to:', RESULTS_DIR)

if __name__ == '__main__':
    run_all() 