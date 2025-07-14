# Author: Leonardo Torres Hernández
import random
import math
import numpy as np
import hashlib
# Here, we will define the class of the tree, as well as the generate random and evaluate functions
# We will first define the node class 
# We will define the types of nodes
Operators = ["+","-","*","/","**"]
Functions = ["sin","cos","log","exp"]
Terminal = ["X"]


class node:
    def __init__(self, value, child: list = []):
        self.value = value
        self.child = child if child is not None else []

    def rep(self):
        if not self.child:  # Leaf node
            return str(self.value)
        else:  # Internal node
            children_reps = [c.rep() for c in self.child]
            # For binary operators, format as (left op right)
            if len(self.child) == 2:
                return f"({children_reps[0]} {self.value} {children_reps[1]})"
            # For unary functions, format as func(child)
            elif len(self.child) == 1:
                return f"{self.value}({children_reps[0]})"
            else:
                # Fallback for nodes with unexpected number of children
                return f"{self.value}({', '.join(children_reps)})"

    def addChild(self, node):
        self.child.append(node)

def RanNode():
    #Create a node with a random value.
    Values=["+","-","/","*","log","sin","cos","X","Cons"]
    #Choose a random value
    RanVal=random.choice(Values)
    if RanVal == "Cons":
        RanVal = random.uniform(-10,10)
    return node(RanVal)
# Define a class for a tree
class tree:
    def __init__ (self,root : node, adf_arities=None, adfs=None):
        self.root = root
        self.adf_arities = adf_arities if adf_arities is not None else {}
        self.adfs = adfs if adfs is not None else {}
        # Cache for tree evaluations
        self._eval_cache = {}
        self._tree_hash = None
        self._update_hash()
    
    def _update_hash(self):
        """Update the tree hash when the tree structure changes."""
        self._tree_hash = self._compute_tree_hash()
        # Clear evaluation cache when tree changes
        self._eval_cache.clear()
    
    def _compute_tree_hash(self):
        """Compute a hash of the tree structure for caching purposes."""
        def hash_node(n):
            if isinstance(n.value, (int, float)):
                return f"const_{n.value}"
            elif n.value == "X":
                return "X"
            else:
                # For operators and functions, include children hashes
                children_hashes = [hash_node(child) for child in n.child]
                return f"{n.value}({','.join(children_hashes)})"
        
        return hash_node(self.root)
    
    def get_tree_hash(self):
        """Get the current tree hash."""
        return self._tree_hash
    
    def rep(self):
        if not self.root.child:  # Leaf node
            return str(self.root.value)
        else:  # Internal node
            children_reps = [c.rep() for c in self.root.child]
            # For binary operators, format as (left op right)
            if len(self.root.child) == 2:
                return f"({children_reps[0]} {self.root.value} {children_reps[1]})"
            # For unary functions, format as func(child)
            elif len(self.root.child) == 1:
                return f"{self.root.value}({children_reps[0]})"
            else:
                # Fallback for nodes with unexpected number of children
                return f"{self.root.value}({', '.join(children_reps)})"

    def addChild(self, node):
        self.root.child.append(node)
        self._update_hash()
    
    def evaluate(self, x_value):
        """
        Evaluate the tree for a single value or a numpy array of values.
        If x_value is a numpy array, returns a numpy array of results.
        Uses caching to avoid recomputing the same tree on the same data.
        """
        # Create a cache key based on input data
        if isinstance(x_value, np.ndarray):
            # For arrays, use a hash of the array data
            x_hash = hashlib.md5(x_value.tobytes()).hexdigest()
        else:
            # For scalars, use the value directly
            x_hash = str(x_value)
        
        cache_key = f"{self._tree_hash}_{x_hash}"
        
        # Check if we have cached result
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        
        # Compute the result
        if isinstance(x_value, np.ndarray):
            result = self.evaluate_vectorized(x_value)
        else:
            result = self._eval_node(self.root, x_value)
        
        # Cache the result
        self._eval_cache[cache_key] = result
        return result

    def _eval_node(self, n, x_value, adf_args=None):
        if adf_args is None:
            adf_args = {}
        # If it's a constant (float/int)
        if isinstance(n.value, (int, float)):
            return n.value
        # If it's the variable "X"
        if n.value == "X":
            return x_value
        # If it's an argument terminal (for ADFs)
        if isinstance(n.value, str) and n.value.startswith("ARG"):
            arg_idx = int(n.value[3:])
            if arg_idx in adf_args:
                return adf_args[arg_idx]
            else:
                return float("nan")
        # If it's a binary operator
        if n.value in {"+", "-", "*", "/", "**"}:
            if not hasattr(n, 'child') or len(n.child) < 2:
                return float('nan')
            left = self._eval_node(n.child[0], x_value, adf_args)
            right = self._eval_node(n.child[1], x_value, adf_args)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                if n.value == "+":
                    result = left + right
                elif n.value == "-":
                    result = left - right
                elif n.value == "*":
                    result = left * right
                elif n.value == "/":
                    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                        result = left / right
                        if isinstance(result, np.ndarray):
                            result = np.where(right == 0, np.nan, result)
                    else:
                        if right == 0:
                            result = float("nan")
                        else:
                            result = left / right
                elif n.value == "**":
                    # Only allow real-valued powers for valid domains
                    try:
                        result = np.power(left, right)
                        # Set nan where base < 0 and exponent is not integer
                        if isinstance(result, np.ndarray):
                            mask = (left < 0) & (np.abs(right - np.round(right)) > 1e-8)
                            result = np.where(mask, np.nan, result)
                        elif left < 0 and abs(right - round(right)) > 1e-8:
                            result = float('nan')
                    except Exception:
                        result = float('nan')
                # After operation, replace non-finite with np.nan
                if isinstance(result, np.ndarray):
                    result = np.where(np.isfinite(result), result, np.nan)
                elif not np.isfinite(result):
                    result = float("nan")
                return result
        # If it's a unary function
        if n.value in {"sin", "cos", "log", "exp"}:
            if not hasattr(n, 'child') or len(n.child) < 1:
                return float('nan')
            arg = self._eval_node(n.child[0], x_value, adf_args)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                if n.value == "sin":
                    result = np.sin(arg)
                elif n.value == "cos":
                    result = np.cos(arg)
                elif n.value == "log":
                    # log only defined for arg > 0
                    result = np.log(arg)
                    if isinstance(result, np.ndarray):
                        result = np.where(arg <= 0, np.nan, result)
                    elif arg <= 0:
                        result = float("nan")
                elif n.value == "exp":
                    # Clip arg to avoid overflow
                    arg_clipped = np.clip(arg, -20, 20)
                    result = np.exp(arg_clipped)
                # After operation, replace non-finite with np.nan
                if isinstance(result, np.ndarray):
                    result = np.where(np.isfinite(result), result, np.nan)
                elif not np.isfinite(result):
                    result = float("nan")
                return result
        # If it's an ADF (automatic function)
        if n.value in self.adfs:
            adf_tree = self.adfs[n.value]
            adf_args_list = [self._eval_node(child, x_value, adf_args) for child in n.child]
            adf_args_dict = {i: arg for i, arg in enumerate(adf_args_list)}
            return adf_tree._eval_node(adf_tree.root, x_value, adf_args=adf_args_dict)
        # Fallback for unknown node types
        return float("nan")

    def evaluate_vectorized(self, x_array):
        """
        Evaluate the tree for a numpy array of x values, returning a numpy array of results.
        """
        x_array = np.asarray(x_array)
        result = self._eval_node(self.root, x_array)
        # If result is a scalar, broadcast to shape of x_array
        if np.isscalar(result) or (isinstance(result, np.ndarray) and result.shape == ()):  # handle 0-d arrays
            result = np.full_like(x_array, result, dtype=float)
        return result

    def select_random_node(self):
        """
        Select a random node from the tree using depth-first traversal.
        Returns the selected node and its path from root.
        """
        def collect_all_nodes(node, path=[]):
            """Recursively collect all nodes with their paths."""
            nodes_with_paths = [(node, path.copy())]
            
            for i, child in enumerate(node.child):
                child_path = path + [i]  # Path to this child
                nodes_with_paths.extend(collect_all_nodes(child, child_path))
            
            return nodes_with_paths
        
        # Collect all nodes with their paths
        all_nodes_with_paths = collect_all_nodes(self.root)
        
        # Select a random node
        selected_node, selected_path = random.choice(all_nodes_with_paths)
        
        return selected_node, selected_path

    def get_node_by_path(self, path):
        """
        Get a node at a specific path from the root.
        Path is a list of child indices.
        """
        current = self.root
        for child_index in path:
            current = current.child[child_index]
        return current

    def replace_node_by_path(self, path, new_node):
        """
        Replace a node at a specific path with a new node.
        """
        if not path:  # Root node
            self.root = new_node
        else:
            # Navigate to the parent of the target node
            current = self.root
            for child_index in path[:-1]:
                current = current.child[child_index]
            
            # Replace the target node
            current.child[path[-1]] = new_node
        
        # Update hash after tree modification
        self._update_hash()

def RanTree(max_depth: int, current_depth: int = 0, require_adf: bool = True, adf_force_depth_limit: int = 2, adf_arities=None, adfs=None):
    binary_ops = ["+", "-", "*", "/", "**"]
    unary_ops = ["sin", "cos", "log", "exp"]
    terminals = ["X", "Cons"]
    adf_arities = adf_arities if adf_arities is not None else {}
    adfs = adfs if adfs is not None else {}
    for adf_name, adf_tree in adfs.items():
        arity = len(adf_tree.root.child) if hasattr(adf_tree.root, 'child') else 0
        if adf_name in adf_arities:
            arity = adf_arities[adf_name]
        if arity == 1:
            unary_ops.append(adf_name)
        elif arity == 2:
            binary_ops.append(adf_name)
        elif arity == 0:
            terminals.append(adf_name)
    tree_obj = None
    if current_depth == max_depth:
        term = random.choice(terminals)
        if term == "Cons":
            value = random.uniform(-10, 10)
        else:
            value = term
        tree_obj = tree(node(value), adf_arities=adf_arities, adfs=adfs)
    else:
        prob_terminal = 0.3 + 0.7 * (current_depth / max_depth)
        if random.random() < prob_terminal:
            term = random.choice(terminals)
            if term == "Cons":
                value = random.uniform(-10, 10)
            else:
                value = term
            tree_obj = tree(node(value), adf_arities=adf_arities, adfs=adfs)
        else:
            op_type = random.choice(["binary", "unary"])
            if op_type == "binary" and binary_ops:
                op = random.choice(binary_ops)
                left = RanTree(max_depth, current_depth + 1, require_adf=require_adf, adf_force_depth_limit=adf_force_depth_limit, adf_arities=adf_arities, adfs=adfs).root
                right = RanTree(max_depth, current_depth + 1, require_adf=require_adf, adf_force_depth_limit=adf_force_depth_limit, adf_arities=adf_arities, adfs=adfs).root
                tree_obj = tree(node(op, [left, right]), adf_arities=adf_arities, adfs=adfs)
            elif op_type == "unary" and unary_ops:
                op = random.choice(unary_ops)
                child = RanTree(max_depth, current_depth + 1, require_adf=require_adf, adf_force_depth_limit=adf_force_depth_limit, adf_arities=adf_arities, adfs=adfs).root
                tree_obj = tree(node(op, [child]), adf_arities=adf_arities, adfs=adfs)
            else:
                term = random.choice(terminals)
                if term == "Cons":
                    value = random.uniform(-10, 10)
                else:
                    value = term
                tree_obj = tree(node(value), adf_arities=adf_arities, adfs=adfs)
    if require_adf and (current_depth <= adf_force_depth_limit) and not tree_contains_adf(tree_obj, adfs):
        insert_random_adf_call(tree_obj, adf_arities=adf_arities, adfs=adfs, adf_force_depth_limit=adf_force_depth_limit)
    return tree_obj

def RanADFTree(max_depth: int, arity: int, current_depth: int = 0):
    terminals = [f"ARG{i}" for i in range(arity)] + ["Cons"]
    binary_ops = ["+", "-", "*", "/", "**"]
    unary_ops = ["sin", "cos", "log", "exp"]
    if current_depth == max_depth:
        term = random.choice(terminals)
        if term == "Cons":
            value = random.uniform(-10, 10)
        else:
            value = term
        return node(value)
    prob_terminal = 0.3 + 0.7 * (current_depth / max_depth)
    if random.random() < prob_terminal:
        term = random.choice(terminals)
        if term == "Cons":
            value = random.uniform(-10, 10)
        else:
            value = term
        return node(value)
    op_type = random.choice(["binary", "unary"])
    if op_type == "binary":
        op = random.choice(binary_ops)
        left = RanADFTree(max_depth, arity, current_depth + 1)
        right = RanADFTree(max_depth, arity, current_depth + 1)
        return node(op, [left, right])
    else:
        op = random.choice(unary_ops)
        child = RanADFTree(max_depth, arity, current_depth + 1)
        return node(op, [child])

# We will define the  ADF settings
ADF_arities = {
    "ADF0": 1,  # Unary function: good for trig functions
    "ADF1": 2,  # Binary function: good for combinations
    "ADF2": 1,  # Power function: X^n patterns
    "ADF3": 2   # Complex combination: trig(X) * power(X)
}
ADFs = {}
# We will generate random start ADFs with the set arity
for adf_name, arity in ADF_arities.items():
    # You can choose a max_depth for the ADF trees, e.g., 3
    ADFs[adf_name] = tree(RanADFTree(max_depth=3, arity=arity))
    if arity == 1:
        Functions.append(adf_name)
    elif arity == 2:
        Operators.append(adf_name)
    else:
        Terminal.append(adf_name)

# Utility: Check if a tree contains any ADF call

def tree_contains_adf(tree_obj, adfs):
    """
    Returns True if the tree contains any ADF call (from global ADFs), else False.
    """
    def contains_adf_node(node):
        if isinstance(node.value, str) and node.value in adfs:
            return True
        return any(contains_adf_node(child) for child in getattr(node, 'child', []))
    return contains_adf_node(tree_obj.root)

# Utility: Insert a random ADF call at a random position in the tree

def insert_random_adf_call(tree_obj, max_arity=2, adf_force_depth_limit=2, adf_arities=None, adfs=None):
    adf_arities = adf_arities if adf_arities is not None else {}
    adfs = adfs if adfs is not None else {}
    adf_name = random.choice(list(adfs.keys()))
    arity = adf_arities.get(adf_name, 1)
    args = [RanTree(max_depth=2, require_adf=False, adf_force_depth_limit=adf_force_depth_limit, adf_arities=adf_arities, adfs=adfs).root for _ in range(arity)]
    adf_node = node(adf_name, args)
    all_nodes_with_paths = []
    def collect_nodes(n, path=[]):
        all_nodes_with_paths.append((n, path.copy()))
        for i, child in enumerate(getattr(n, 'child', [])):
            collect_nodes(child, path + [i])
    collect_nodes(tree_obj.root)
    _, path = random.choice(all_nodes_with_paths)
    tree_obj.replace_node_by_path(path, adf_node)
    tree_obj._update_hash()
    return tree_obj


