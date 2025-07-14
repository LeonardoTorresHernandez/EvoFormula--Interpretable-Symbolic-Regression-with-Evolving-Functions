# Author: Leonardo Torres Hernández
# Import the trees.py
from math import nan
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tree as tr

# Define the fitness function without caching
def fitness(Tree:tr.tree,X:np.ndarray,y:np.ndarray, debug=False, complexity_weight=0.0):
    """
    Calculate fitness of a tree without caching.
    Always evaluates the tree fresh to ensure accuracy.
    """
    # Vectorized calculation of predictions
    try:
        ObserY = Tree.evaluate(X)
    except Exception as e:
        if debug:
            print(f"DEBUG: Error in tree evaluation: {e}")
        return 0
    
    # Convert to numpy array (in case output is not already)
    ObserY = np.array(ObserY)
    
    # Only print debug info if requested
    if debug:
        print(f"\n=== FITNESS DEBUGGING ===")
        print(f"Tree representation: {Tree.rep()}")
        print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"ObserY range: [{ObserY.min():.6f}, {ObserY.max():.6f}]")
        print(f"ObserY has NaN: {np.any(np.isnan(ObserY))}")
        print(f"ObserY has inf: {np.any(np.isinf(ObserY))}")
        print(f"First 5 X values: {X[:5]}")
        print(f"First 5 y values: {y[:5]}")
        print(f"First 5 ObserY values: {ObserY[:5]}")
    
    # Check for numerical issues
    if np.any(np.isnan(ObserY)) or np.any(np.isinf(ObserY)):
        if debug:
            print("WARNING: ObserY contains NaN or inf values!")
        return 0
    
    # Calculate fitness using multiple metrics for robustness
    SE = np.sum((ObserY - y) ** 2)
    MSE = SE / len(y)
    y_var = np.var(y)
    
    # Use absolute error instead of squared error for better sensitivity to small errors
    MAE = np.mean(np.abs(ObserY - y))
    y_range = np.max(y) - np.min(y)
    normalized_mae = MAE / y_range
    
    # Calculate R-squared for additional insight
    y_mean = np.mean(y)
    SS_tot = np.sum((y - y_mean) ** 2)
    SS_res = SE
    R_squared = 1 - (SS_res / SS_tot) if SS_tot > 0 else 0
    
    # Use a combination of metrics for more robust fitness
    # Penalize both large errors and constant predictions
    if np.allclose(ObserY, ObserY[0]):
        # Heavy penalty for constant predictions
        constant_penalty = 0.1
    else:
        constant_penalty = 1.0
    
    # Calculate fitness using multiple metrics
    fitness_mae = 1 / (1 + normalized_mae)
    fitness_mse = 1 / (1 + MSE / y_var) if y_var > 0 else 0
    fitness_r2 = max(0, R_squared)  # Ensure non-negative
    
    # Combine metrics with weights
    final_fitness = (0.4 * fitness_mae + 0.4 * fitness_mse + 0.2 * fitness_r2) * constant_penalty
    
    # --- BEGIN: Complexity penalty ---
    def count_nodes(node):
        return 1 + sum(count_nodes(child) for child in getattr(node, 'child', []))
    complexity = count_nodes(Tree.root)
    if complexity_weight > 0.0:
        if debug:
            print(f"Complexity penalty: {complexity_weight} * {complexity} = {complexity_weight * complexity}")
        final_fitness -= complexity_weight * complexity
    # --- END: Complexity penalty ---
    
    if debug:
        print(f"Sum of squared errors (SE): {SE:.6f}")
        print(f"Mean squared error (MSE): {MSE:.6f}")
        print(f"Mean absolute error (MAE): {MAE:.6f}")
        print(f"Variance of y (y_var): {y_var:.6f}")
        print(f"Normalized MAE: {normalized_mae:.6f}")
        print(f"R-squared: {R_squared:.6f}")
        print(f"Constant penalty: {constant_penalty:.6f}")
        print(f"Fitness components - MAE: {fitness_mae:.6f}, MSE: {fitness_mse:.6f}, R2: {fitness_r2:.6f}")
        print(f"Final fitness: {final_fitness:.6f}")
        print(f"=== END DEBUGGING ===\n")
    
    return final_fitness