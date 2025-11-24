import sys
import os
import time
import numpy as np
import importlib.util

# Get workspace root
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)

# Add 'code' directory to sys.path
sys.path.append(os.path.join(workspace_root, 'code'))

# Helper to import modules
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import Algorithms
AlgoKMeans = import_module_from_path("AlgoKMeans", os.path.join(workspace_root, "code/algorithm/algo_kmeans.py"))
AlgoKruskal = import_module_from_path("AlgoKruskal", os.path.join(workspace_root, "code/algorithm/algo_kruskal.py"))
AlgoBKruskalGra = import_module_from_path("AlgoBKruskalGra", os.path.join(workspace_root, "code/algorithm/algo_bkruskal.py"))

def test_debug_bkruskal():
    print("=== Debugging BKruskalGra Discrepancy ===")
    
    # Small scale test
    n = 50
    M = 5
    WIDTH = 100
    
    # Fix random seed for reproducibility
    np.random.seed(42)
    points = np.random.uniform(0, WIDTH, (n, 2))
    
    print(f"Testing with N={n}, M={M}")
    
    # 1. Run KMeans (Reference)
    print("\n--- Running KMeans (Reference) ---")
    # We need to capture the internal best_tsp_value_route to see what happened
    # Since we can't easily modify the imported module's internal state without editing files,
    # we will rely on the return value which is min(best_tsp_value_route)
    res_kmeans = AlgoKMeans.run(points, M)
    print(f"KMeans Result: {res_kmeans}")

    # 2. Run Kruskal (Reference)
    print("\n--- Running Kruskal (Reference) ---")
    res_kruskal = AlgoKruskal.run(points, M)
    print(f"Kruskal Result: {res_kruskal}")

    # 3. Run BKruskalGra
    print("\n--- Running BKruskalGra ---")
    res_bkruskal = AlgoBKruskalGra.run(points, M)
    print(f"BKruskalGra Result: {res_bkruskal}")
    
    if abs(res_kmeans - res_kruskal) < 1e-5 and res_bkruskal > res_kmeans:
        print("\n[Observation] KMeans and Kruskal are identical, but BKruskalGra is worse.")
        print(f"Difference: {res_bkruskal - res_kmeans}")
    else:
        print("\n[Observation] Results are mixed.")

if __name__ == "__main__":
    test_debug_bkruskal()
