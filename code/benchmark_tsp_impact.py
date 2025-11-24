import sys
import os
import time
import numpy as np
import importlib

# Setup paths
# Assuming we run from workspace root
workspace_root = os.getcwd()
code_dir = os.path.join(workspace_root, 'code')
sys.path.append(code_dir)

# Import modules
# We need to import algo_kmeans. Since it is in code/algorithm, and code is in path:
import algorithm.algo_kmeans as algo_kmeans
import TSP.TSP_LKH as TSP_LKH
import TSP.TSP_Pointerformer as TSP_Pointerformer

def generate_points(n, width=100, height=100):
    return np.random.uniform(0, width, (n, 2))

def run_benchmark():
    N_VALUES = [50, 100, 150, 200]
    M = 5
    
    print(f"{'N':<5} | {'Method':<15} | {'Min Val':<10} | {'Time (s)':<10} | {'Improvement':<12}")
    print("-" * 65)
    
    for n in N_VALUES:
        points = generate_points(n)
        
        # 1. Original (Greedy / Pointerformer)
        # Ensure we are using the original
        algo_kmeans.TSP_Pointerformer = TSP_Pointerformer
        
        start_time = time.perf_counter()
        # algo_kmeans.run returns (min_val, best_k, all_vals)
        min_val_greedy, _, _ = algo_kmeans.run(points, M)
        time_greedy = time.perf_counter() - start_time
        
        print(f"{n:<5} | {'Greedy':<15} | {min_val_greedy:<10.2f} | {time_greedy:<10.4f} | {'-':<12}")
        
        # 2. LKH
        # Monkey patch
        algo_kmeans.TSP_Pointerformer = TSP_LKH
        
        start_time = time.perf_counter()
        min_val_lkh, _, _ = algo_kmeans.run(points, M)
        time_lkh = time.perf_counter() - start_time
        
        improvement = (min_val_greedy - min_val_lkh) / min_val_greedy * 100
        
        print(f"{n:<5} | {'LKH':<15} | {min_val_lkh:<10.2f} | {time_lkh:<10.4f} | {improvement:<10.2f}%")
        print("-" * 65)

if __name__ == "__main__":
    run_benchmark()
