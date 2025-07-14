# Author: Leonardo Torres Hernández
from cProfile import label
from hmac import new
from math import sin
import sys
import os
import copy
import numpy as np
# Use only absolute imports from the project root
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Remove or comment out
from symbolic_regression import tree as tr
from symbolic_regression.algorithm import fitness
import random
from sklearn.base import BaseEstimator, RegressorMixin
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm



# Define the functions for selection, crossover and mutation
def selection(X,y,StartTrees: int,Percentage:float,MaxRanTreeDepth: int = 4, adf_arities=None, adfs=None):

    # First, I must generate random trees and calculate the fitness
    RanTrees = []
    for i in range(StartTrees):
        GenTree = tr.RanTree(MaxRanTreeDepth, adf_arities=adf_arities, adfs=adfs)
        RanTrees.append(GenTree)
    # Parallel fitness evaluation
    Fitnesses = Parallel(n_jobs=-1)(delayed(fitness)(tree, X, y) for tree in RanTrees)
    # If a fitness of 1 is found, return the tree
    for GenTree, CurrentFit in zip(RanTrees, Fitnesses):
        if CurrentFit == 1:
            return [GenTree]
    # Order the trees from the higher to the lower fitness
    combined = list(zip(Fitnesses, RanTrees))
    # Remove any entries with None fitness
    combined = [(f, t) for f, t in combined if f is not None]
    if not combined:
        return []
    Ordered = sorted(combined, reverse=True, key=lambda x: float(x[0]))
    if Ordered:
        Fitnesses, RanTrees = zip(*Ordered)
        # Choose the best trees.
        Best = RanTrees[:int(StartTrees * (Percentage / 100))]
        return Best
    else:
        return []

def crossover(tree_1: tr.tree, tree_2: tr.tree):
    # Choose two random nodes in tree_1 and tree_2, then swap them.
    RanNode_1, Path1 = tree_1.select_random_node()
    RanNode_2, Path_2 = tree_2.select_random_node()

    # Copy the trees, preserving adfs and adf_arities
    def deepcopy_with_adfs(tree):
        new_tree = copy.deepcopy(tree)
        new_tree.adfs = copy.deepcopy(tree.adfs)
        new_tree.adf_arities = copy.deepcopy(tree.adf_arities)
        return new_tree

    Copy_1 = deepcopy_with_adfs(tree_1)
    Copy_2 = deepcopy_with_adfs(tree_2)

    # Swap the nodes
    try:
        Copy_1.replace_node_by_path(Path1, RanNode_2)
        Copy_2.replace_node_by_path(Path_2, RanNode_1)
    except Exception as e:
        print(f"Error in crossover: {e}")
        print(f"Path1: {Path1}, Path2: {Path_2}")
        print(f"RanNode_1: {RanNode_1}, RanNode_2: {RanNode_2}")
        raise e

    # Ensure each child contains all ADFs present in its own parent
    def get_adfs_in_tree(tree_obj):
        adfs_found = set()
        def visit(node):
            if isinstance(node.value, str) and node.value in tree_obj.adfs:
                adfs_found.add(node.value)
            for child in getattr(node, 'child', []):
                visit(child)
        visit(tree_obj.root)
        return adfs_found

    parent1_adfs = get_adfs_in_tree(tree_1)
    parent2_adfs = get_adfs_in_tree(tree_2)
    child1_adfs = get_adfs_in_tree(Copy_1)
    child2_adfs = get_adfs_in_tree(Copy_2)
    # For each child, insert any missing ADFs from its own parent
    for adf in parent1_adfs - child1_adfs:
        tr.insert_random_adf_call(Copy_1, adf_arities=Copy_1.adf_arities, adfs=Copy_1.adfs)
    for adf in parent2_adfs - child2_adfs:
        tr.insert_random_adf_call(Copy_2, adf_arities=Copy_2.adf_arities, adfs=Copy_2.adfs)

    return Copy_1, Copy_2


def perform_crossover_on_population(trees):
    # Parallel crossover
    def crossover_pair(i):
        parent1 = trees[2 * i]
        parent2 = trees[2 * i + 1]
        return crossover(parent1, parent2)
    num_pairs = len(trees) // 2
    results = list(Parallel(n_jobs=-1)(delayed(crossover_pair)(i) for i in range(num_pairs)))
    # Filter out None results (in case of error)
    results = [pair for pair in results if pair is not None]
    new_population = [child for pair in results for child in pair]
    if len(trees) % 2 == 1:
        # Copy the last tree, preserving adfs/adf_arities
        last_tree = trees[-1]
        new_tree = copy.deepcopy(last_tree)
        new_tree.adfs = copy.deepcopy(last_tree.adfs)
        new_tree.adf_arities = copy.deepcopy(last_tree.adf_arities)
        new_population.append(new_tree)
    return new_population

def mutation(tree: tr.tree):
    """
    Perform mutation on a tree by randomly changing a node's value.
    """
    # Select a random node
    node, path = tree.select_random_node()
    
    # Define possible values for different node types
    operators = tr.Operators
    functions = tr.Functions
    terminals = tr.Terminal
    ADFs = tr.ADFs
    # Determine what type of node this is and choose appropriate replacement
    if isinstance(node.value, (int, float)):
        # It's a constant - change it slightly
        node.value += random.uniform(-2, 2)
    elif node.value in operators:
        # It's an operator - replace with different operator
        node.value = random.choice(operators)
    elif node.value in functions:
        # It's a function - replace with different function
        node.value = random.choice(functions)
    elif node.value == "X":
        # It's X - replace with constant or keep X
        if random.random() < 0.5:
            node.value = random.uniform(-10, 10)
    elif node.value in ADFs:
        # Modify the ADF: mutate the corresponding ADF tree
        mutate_adf(node.value, ADFs)
    else:
        # Unknown type - replace with random value
        all_values = operators + functions + terminals
        node.value = random.choice(all_values)
    
    return tree

def mutate_adf(adf_name, ADFs, max_depth=3):
    """
    Mutate the ADF tree for the given adf_name in the ADFs dict.
    Mutation is performed by selecting a random node in the ADF tree and mutating it.
    """
    adf_tree = ADFs[adf_name]
    # Deepcopy to avoid side effects
    adf_tree = copy.deepcopy(adf_tree)
    node, path = adf_tree.select_random_node()
    # Define possible values for different node types
    operators = tr.Operators
    functions = tr.Functions
    arity = len([k for k in tr.ADF_arities if k == adf_name and tr.ADF_arities[k] is not None and tr.ADF_arities[k] > 0])
    terminals = [f"ARG{i}" for i in range(tr.ADF_arities[adf_name])] + ["Cons"]
    # Mutate node value
    if isinstance(node.value, (int, float)):
        node.value += random.uniform(-2, 2)
    elif node.value in operators:
        node.value = random.choice(operators)
    elif node.value in functions:
        node.value = random.choice(functions)
    elif isinstance(node.value, str) and node.value.startswith("ARG"):
        # Change to another argument or constant
        if random.random() < 0.5:
            node.value = random.choice(terminals)
    elif node.value == "Cons":
        node.value = random.uniform(-10, 10)
    else:
        # Unknown type - replace with random value
        all_values = operators + functions + terminals
        node.value = random.choice(all_values)
    # Update the ADFs dict
    ADFs[adf_name] = adf_tree
    return adf_tree

def apply_mutation_to_population(trees, mutation_rate=0.1):
    def maybe_mutate(tree):
        if random.random() < mutation_rate:
            mutated_tree = copy.deepcopy(tree)
            mutation(mutated_tree)
            return mutated_tree
        else:
            return copy.deepcopy(tree)
    mutated_trees = list(Parallel(n_jobs=-1)(delayed(maybe_mutate)(tree) for tree in trees))
    return mutated_trees

def structure_preserving_mutation(tree):
    """
    Mutate a tree while preserving its mathematical structure.
    Identifies patterns like c * g(x) and randomizes c and g while keeping the form.
    """
    def find_constant_function_pattern(node):
        """Find patterns like (constant * function(X)) or similar structures."""
        if not hasattr(node, 'child') or not node.child:
            return None
        
        # Look for multiplication patterns
        if node.value == "*" and len(node.child) == 2:
            left, right = node.child[0], node.child[1]
            
            # Check if one child is a constant and the other is a function
            if isinstance(left.value, (int, float)) and right.value in tr.Functions:
                return ("const_func", left, right)
            elif isinstance(right.value, (int, float)) and left.value in tr.Functions:
                return ("const_func", right, left)
            
            # Check for nested function patterns
            if left.value in tr.Functions and hasattr(left, 'child') and left.child:
                if left.child[0].value == "X":
                    return ("const_func", node, left)
            if right.value in tr.Functions and hasattr(right, 'child') and right.child:
                if right.child[0].value == "X":
                    return ("const_func", node, right)
        
        # Recursively search children
        for child in node.child:
            result = find_constant_function_pattern(child)
            if result:
                return result
        return None
    
    # Find a pattern to mutate
    pattern = find_constant_function_pattern(tree.root)
    if pattern:
        pattern_type, const_node, func_node = pattern
        
        if pattern_type == "const_func":
            # Randomize the constant
            if isinstance(const_node.value, (int, float)):
                const_node.value = random.uniform(-10, 10)
            
            # Randomize the function
            if func_node.value in tr.Functions:
                func_node.value = random.choice(tr.Functions)
        
        tree._update_hash()
    return tree

def node_preserving_rearrangement(tree):
    """
    Rearrange a tree while preserving its nodes (constants, functions, variables).
    Creates new structures using the same building blocks.
    """
    def collect_nodes(node):
        """Collect all nodes in the tree."""
        nodes = [node]
        if hasattr(node, 'child'):
            for child in node.child:
                nodes.extend(collect_nodes(child))
        return nodes
    
    def is_terminal(node):
        """Check if a node is a terminal (constant, variable, or function with X)."""
        if isinstance(node.value, (int, float)) or node.value == "X":
            return True
        if node.value in tr.Functions and hasattr(node, 'child') and node.child:
            return node.child[0].value == "X"
        return False
    
    # Collect all nodes
    all_nodes = collect_nodes(tree.root)
    
    # Separate terminals and non-terminals
    terminals = [n for n in all_nodes if is_terminal(n)]
    non_terminals = [n for n in all_nodes if not is_terminal(n)]
    
    if len(terminals) < 2:
        return tree  # Not enough nodes to rearrange
    
    # Create a new simple structure using the terminals
    if len(terminals) >= 2:
        # Create a binary operation with two terminals
        op = random.choice(tr.Operators)
        left = copy.deepcopy(random.choice(terminals))
        right = copy.deepcopy(random.choice(terminals))
        
        # Ensure we don't pick the same terminal twice
        if len(terminals) > 1:
            while right.value == left.value and isinstance(left.value, (int, float)):
                right = copy.deepcopy(random.choice(terminals))
        
        new_root = tr.node(op, [left, right])
        tree.root = new_root
        tree._update_hash()
    
    return tree

def apply_advanced_operators_to_population(trees, advanced_rate=0.2):
    """
    Apply advanced operators (structure preserving and node preserving) to a population.
    """
    def apply_advanced_operator(tree):
        if random.random() < advanced_rate:
            tree_copy = copy.deepcopy(tree)
            if random.random() < 0.5:
                return structure_preserving_mutation(tree_copy)
            else:
                return node_preserving_rearrangement(tree_copy)
        else:
            return copy.deepcopy(tree)
    
    advanced_trees = list(Parallel(n_jobs=-1)(delayed(apply_advanced_operator)(tree) for tree in trees))
    return advanced_trees

# --- BEGIN: Local search operator ---
def local_search(tree, perturbation_scale=0.5):
    """
    Simple local search: perturb all constants in the tree by a small random amount.
    """
    def perturb_constants(node):
        if isinstance(node.value, (int, float)):
            node.value += np.random.uniform(-perturbation_scale, perturbation_scale)
        for child in getattr(node, 'child', []):
            perturb_constants(child)
    tree_copy = copy.deepcopy(tree)
    perturb_constants(tree_copy.root)
    tree_copy._update_hash()
    return tree_copy
# --- END: Local search operator ---

# --- BEGIN: ADF mining and contribution scoring ---
def extract_subtrees(tree, min_size=2):
    """Extract all unique subtrees of at least min_size nodes from a tree. Returns dict rep->node."""
    subtrees = {}
    def collect(node):
        nodes = [node]
        for child in getattr(node, 'child', []):
            nodes.extend(collect(child))
        return nodes
    all_nodes = collect(tree.root)
    for n in all_nodes:
        if hasattr(n, 'child') and len(collect(n)) >= min_size:
            rep = n.rep()
            if rep not in subtrees:
                subtrees[rep] = n
    return subtrees

def subtree_contribution(tree, subtree_rep, X, y, fitness_func):
    """Estimate the contribution of a subtree by replacing it with a constant and measuring fitness drop."""
    import copy
    # Find all nodes matching subtree_rep
    def find_and_replace(node):
        if node.rep() == subtree_rep:
            # Replace with a constant node
            return tr.node(0.0)
        if hasattr(node, 'child'):
            new_children = [find_and_replace(child) for child in node.child]
            return tr.node(node.value, new_children)
        return tr.node(node.value)
    # Replace all occurrences in the tree
    new_tree = tr.tree(find_and_replace(tree.root), adf_arities=tree.adf_arities, adfs=tree.adfs)
    # Compute fitness drop
    original_fitness = fitness_func(tree, X, y)
    replaced_fitness = fitness_func(new_tree, X, y)
    return original_fitness - replaced_fitness

def mine_best_adfs(population, X, y, fitness_func, n_adfs=4, top_k=10, min_size=2):
    """Extract and score candidate ADFs from the top_k individuals, return top n_adfs as (rep, node) tuples."""
    import copy
    scored_pop = [(fitness_func(tree, X, y), tree) for tree in population]
    scored_pop.sort(reverse=True, key=lambda x: x[0])
    top_individuals = [tree for _, tree in scored_pop[:top_k]]
    candidate_subtrees = {}
    for tree in top_individuals:
        candidate_subtrees.update(extract_subtrees(tree, min_size=min_size))
    contributions = []
    for rep, node in candidate_subtrees.items():
        contribs = []
        for tree in top_individuals:
            if rep in extract_subtrees(tree, min_size=min_size):
                contrib = subtree_contribution(tree, rep, X, y, fitness_func)
                contribs.append(contrib)
        if contribs:
            avg_contrib = np.mean(contribs)
            contributions.append((avg_contrib, rep, node))
    contributions.sort(reverse=True, key=lambda x: x[0])
    best_adfs = [(rep, node) for _, rep, node in contributions[:n_adfs]]
    return best_adfs
# --- END: ADF mining and contribution scoring ---

def genetic_algorithm(X, y, StartTrees=30, Percentage=10, MaxRanTreeDepth=4, 
                     mutation_rate=0.1, max_generations=10, target_fitness=0.9, adf_arities=None, adfs=None, n_adfs=4, complexity_weight=0.0):
    import time
    X = np.array(X)
    y = np.array(y)
    top_k_adf = 10
    min_adf_size = 2
    # Initialize ADFs as random trees
    adf_arities = adf_arities if adf_arities is not None else {f"ADF{i}": 1 for i in range(n_adfs)}
    adfs = adfs if adfs is not None else {name: tr.tree(tr.RanADFTree(max_depth=3, arity=arity), adf_arities=adf_arities, adfs=None) for name, arity in adf_arities.items()}
    current_population = [tr.RanTree(MaxRanTreeDepth, adf_arities=adf_arities, adfs=adfs) for _ in range(StartTrees)]
    best_tree = None
    best_fitness = 0.0
    generation = 0
    fitness_history = []
    significant_trees = []
    base_mutation_rate = mutation_rate
    current_mutation_rate = mutation_rate
    stagnation_count = 0
    max_stagnation_before_adaptation = 10
    max_adaptive_mutation_rate = 0.8
    base_population_size = StartTrees
    current_population_size = StartTrees
    max_population_size = StartTrees * 3
    stuck_generations = 5
    high_mutation_rate = 0.8
    high_mutation_active = False
    pbar = tqdm(total=max_generations, desc=f"Genetic Algorithm", ncols=100)
    for gen in range(max_generations):
        generation = gen + 1
        t0 = time.time()
        fitnesses = []
        for tree in current_population:
            if tree is not None:
                fit = fitness(tree, X, y, complexity_weight=complexity_weight)
                y_pred = tree.evaluate(X)
                if np.all(np.isnan(y_pred)) or np.all(np.isinf(y_pred)):
                    fit = -1e6
                fitnesses.append(fit)
            else:
                fitnesses.append(-1e6)
        t1 = time.time()
        previous_best_fitness = best_fitness
        for tree, fit in zip(current_population, fitnesses):
            if tree is not None and fit is not None and fit > best_fitness:
                best_fitness = fit
                best_tree = tree
                if len(significant_trees) < 4 or fit > significant_trees[-1]['fitness']:
                    significant_trees.append({
                        'generation': generation,
                        'fitness': fit,
                        'tree': tree,
                        'tree_repr': tree.rep()
                    })
                    significant_trees.sort(key=lambda x: x['fitness'], reverse=True)
                    significant_trees = significant_trees[:4]
        fitness_history.append(best_fitness)
        pbar.set_description(f"Gen {generation}/{max_generations} | Best Fitness: {best_fitness:.4f}")
        pbar.update(1)
        print(f"Generation {generation} completed in {t1-t0:.2f} seconds. Best fitness: {best_fitness:.4f}")
        if best_fitness <= previous_best_fitness:
            stagnation_count += 1
        else:
            stagnation_count = 0
            high_mutation_active = False
            current_mutation_rate = base_mutation_rate
        if stagnation_count >= stuck_generations and not high_mutation_active:
            print(f"[DIVERSITY INJECTION] Stuck for {stagnation_count} generations. Applying high mutation rate: {high_mutation_rate}")
            current_mutation_rate = high_mutation_rate
            high_mutation_active = True
        elif high_mutation_active and stagnation_count == 0:
            print(f"[DIVERSITY INJECTION] Reverting mutation rate to normal: {base_mutation_rate}")
            current_mutation_rate = base_mutation_rate
            high_mutation_active = False
        if best_fitness >= target_fitness and best_tree is not None:
            print(f"\n🎉 Target fitness {target_fitness} reached in generation {generation}!")
            print(f"Best tree: {best_tree.rep()}")
            print(f"Best fitness: {best_fitness:.6f}")
            pbar.close()
            return best_tree, best_fitness, generation, fitness_history, significant_trees
        combined = list(zip(fitnesses, current_population))
        combined = [(f, t) for f, t in combined if t is not None and f is not None]
        if not combined:
            break
        ordered = sorted(combined, reverse=True, key=lambda x: float(x[0]) if x[0] is not None else -float('inf'))
        fitnesses, current_population = zip(*ordered)
        current_population = list(current_population)
        fitnesses = list(fitnesses)
        num_keep = max(2, int(current_population_size * (Percentage / 100)))
        current_population = list(current_population[:num_keep])
        fitnesses = list(fitnesses[:num_keep])
        elite_tree = copy.deepcopy(current_population[0])
        crossed_trees = perform_crossover_on_population(current_population)
        mutated_trees = apply_mutation_to_population(crossed_trees, current_mutation_rate)
        locally_searched_trees = [local_search(tree) for tree in mutated_trees]
        if stagnation_count > max_stagnation_before_adaptation + 10:
            locally_searched_trees = apply_advanced_operators_to_population(locally_searched_trees, advanced_rate=0.3)
        num_new = current_population_size - len(locally_searched_trees) - 1
        if stagnation_count > max_stagnation_before_adaptation:
            diversity_boost = min(0.8, stagnation_count * 0.15)
            num_new = int(num_new * (1 + diversity_boost))
            if stagnation_count > max_stagnation_before_adaptation + 5:
                depth_variations = [MaxRanTreeDepth - 1, MaxRanTreeDepth, MaxRanTreeDepth + 1]
                extra_trees = []
                for _ in range(min(50, num_new // 2)):
                    depth = random.choice(depth_variations)
                    extra_trees.append(tr.RanTree(depth, adf_arities=adf_arities, adfs=adfs))
                locally_searched_trees.extend(extra_trees)
                num_new -= len(extra_trees)
        if num_new > 0:
            new_trees = [tr.RanTree(MaxRanTreeDepth, adf_arities=adf_arities, adfs=adfs) for _ in range(num_new)]
            locally_searched_trees.extend(new_trees)
        next_population = [elite_tree] + locally_searched_trees
        # --- BEGIN: Evolve ADFs from best individuals ---
        best_adfs = mine_best_adfs(next_population, X, y, lambda t, X, y: fitness(t, X, y, complexity_weight=complexity_weight), n_adfs=n_adfs, top_k=top_k_adf, min_size=min_adf_size)
        adfs = {f"ADF{i}": tr.tree(copy.deepcopy(node)) for i, (rep, node) in enumerate(best_adfs)}
        adf_arities = {f"ADF{i}": 1 for i in range(n_adfs)}
        # --- END: Evolve ADFs from best individuals ---
        # Update ADFs in all trees for next generation
        for tree in next_population:
            tree.adfs = adfs
            tree.adf_arities = adf_arities
        current_population = next_population
    pbar.close()
    print(f"\n⚠️  Maximum generations ({max_generations}) reached without achieving target fitness.")
    print(f"Best fitness achieved: {best_fitness:.6f}")
    if best_tree is not None:
        print(f"Best tree: {best_tree.rep()}")
    print("\nFinal ADFs:")
    for adf_name, adf_tree in adfs.items():
        print(f"  {adf_name}: {adf_tree.rep()}")
    else:
        print("No valid tree found to plot.")
    if best_tree is None:
        best_tree = tr.RanTree(MaxRanTreeDepth, adf_arities=adf_arities, adfs=adfs)
    return best_tree, best_fitness, generation, fitness_history, significant_trees

def plot_fitness_evolution(fitness_history, significant_trees):
    """
    Plot the fitness evolution and the 4 most significant trees.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Fitness evolution
    generations = list(range(1, len(fitness_history) + 1))
    ax1.plot(generations, fitness_history, 'b-', linewidth=2, label='Best Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution Over Generations')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight significant improvements with vertical lines and annotations
    colors = ['red', 'orange', 'green', 'purple']
    for i, tree_info in enumerate(significant_trees):
        if i < 4:
            # Add vertical line at the generation where significant improvement occurred
            ax1.axvline(x=tree_info['generation'], color=colors[i], linestyle='--', alpha=0.7, 
                       label=f"Tree {i+1}: Gen {tree_info['generation']}")
            # Add annotation with fitness value
            ax1.annotate(f"{tree_info['fitness']:.4f}", 
                        xy=(tree_info['generation'], tree_info['fitness']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
                        fontsize=8)
    
    ax1.legend()
    
    # Plot 2: Significant trees
    ax2.set_title('4 Most Significant Trees Discovered')
    ax2.axis('off')
    
    # Create text boxes for each significant tree
    y_positions = [0.8, 0.6, 0.4, 0.2]
    
    for i, tree_info in enumerate(significant_trees):
        if i < 4:
            text = f"Tree {i+1} (Gen {tree_info['generation']}, Fitness: {tree_info['fitness']:.6f}):\n{tree_info['tree_repr']}"
            ax2.text(0.05, y_positions[i], text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=colors[i], alpha=0.3))
    
    plt.tight_layout()
    plt.show()

class SymbolicRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, StartTrees=1000, Percentage=10, MaxRanTreeDepth=4, mutation_rate=0.1, max_generations=100, target_fitness=0.95, verbose=True, adf_arities=None, n_adfs=4, complexity_weight=0.0):
        self.StartTrees = StartTrees
        self.Percentage = Percentage
        self.MaxRanTreeDepth = MaxRanTreeDepth
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.verbose = verbose
        self.n_adfs = n_adfs
        self.complexity_weight = complexity_weight
        # Default ADF arities if not provided
        if adf_arities is None:
            adf_arities = {f"ADF{i}": 1 for i in range(n_adfs)}
        self.adf_arities = adf_arities
        # Generate random ADF trees for each ADF
        self.adfs = {name: tr.tree(tr.RanADFTree(max_depth=3, arity=arity), adf_arities=self.adf_arities, adfs=None) for name, arity in self.adf_arities.items()}
        self.best_tree_ = None
        self.best_fitness_ = None
        self.generation_ = None
        self.fitness_history_ = None
        self.significant_trees_ = None

    def fit(self, X, y):
        import tree as tr
        X = np.asarray(X)
        y = np.asarray(y)
        if self.verbose:
            print(f"Fitting SymbolicRegressor with {len(X)} samples.")
        best_tree, best_fitness, generation, fitness_history, significant_trees = genetic_algorithm(
            X, y,
            StartTrees=self.StartTrees,
            Percentage=self.Percentage,
            MaxRanTreeDepth=self.MaxRanTreeDepth,
            mutation_rate=self.mutation_rate,
            max_generations=self.max_generations,
            target_fitness=self.target_fitness,
            adf_arities=self.adf_arities,
            adfs=self.adfs,
            n_adfs=self.n_adfs,
            complexity_weight=self.complexity_weight
        )
        self.best_tree_ = best_tree
        self.best_fitness_ = best_fitness
        self.generation_ = generation
        self.fitness_history_ = fitness_history
        self.significant_trees_ = significant_trees
        return self

    def predict(self, X):
        if self.best_tree_ is None:
            raise RuntimeError("You must fit the estimator before predicting.")
        X = np.asarray(X)
        y_pred = self.best_tree_.evaluate(X)
        return y_pred

    def get_params(self, deep=True):
        return {
            'StartTrees': self.StartTrees,
            'Percentage': self.Percentage,
            'MaxRanTreeDepth': self.MaxRanTreeDepth,
            'mutation_rate': self.mutation_rate,
            'max_generations': self.max_generations,
            'target_fitness': self.target_fitness,
            'verbose': self.verbose,
            'adf_arities': self.adf_arities,
            'n_adfs': self.n_adfs,
            'complexity_weight': self.complexity_weight
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

if __name__ == "__main__":
    # Example: run a quick test only if this file is executed directly
    def test():
        np.random.seed(42)
        n_samples = 200
        x = np.random.uniform(-2, 2, size=n_samples)
        y = np.sin(x) * np.cos(x) + x**2 - np.exp(x / (x + 1))
        adf_arities = {"ADF0": 1, "ADF1": 2, "ADF2": 1, "ADF3": 2}
        adfs = {
            "ADF0": tr.tree(tr.RanADFTree(max_depth=3, arity=1), adf_arities=adf_arities, adfs=None),
            "ADF1": tr.tree(tr.RanADFTree(max_depth=3, arity=2), adf_arities=adf_arities, adfs=None),
            "ADF2": tr.tree(tr.RanADFTree(max_depth=3, arity=1), adf_arities=adf_arities, adfs=None),
            "ADF3": tr.tree(tr.RanADFTree(max_depth=3, arity=2), adf_arities=adf_arities, adfs=None)
        }
        best_tree, best_fitness, generation, fitness_history, significant_trees = genetic_algorithm(
            x, y,
            StartTrees=30,
            Percentage=10,
            MaxRanTreeDepth=4,
            mutation_rate=0.1,
            max_generations=10,
            target_fitness=0.99,
            adf_arities=adf_arities,
            adfs=adfs,
            complexity_weight=0.0 # Pass the complexity_weight parameter
        )
        pred_direct = best_tree.evaluate(x)
        print("First 5 predictions (genetic_algorithm):", pred_direct[:5])
        print("First 5 true values:", y[:5])
        plt.figure(figsize=(8, 6))
        plt.scatter(y, pred_direct, alpha=0.6, label='Predicted (Direct)', marker='x')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')
        plt.xlabel('True y')
        plt.ylabel('Predicted y')
        plt.title('Symbolic Regression: True vs Predicted (Univariate)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    test()
