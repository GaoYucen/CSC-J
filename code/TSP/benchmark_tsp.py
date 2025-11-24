import sys
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt

# Add the current directory to sys.path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add clustering directory to sys.path for TSP_LKH dependency
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clustering'))

import TSP_GD
import TSP_Pointerformer
import TSP_LKH

def generate_points(n, width=100, height=100):
    return np.random.uniform(0, width, (n, 2))

def run_benchmark():
    N_VALUES = [20, 50, 100]
    results = []

    print(f"{'Algorithm':<20} | {'N':<5} | {'Cost':<10} | {'Time (s)':<10}")
    print("-" * 55)

    for n in N_VALUES:
        points = generate_points(n)
        indices = list(range(n))
        
        # 1. TSP_GD (Greedy)
        start_time = time.perf_counter()
        cost_gd, _ = TSP_GD.tsp(points, indices)
        time_gd = time.perf_counter() - start_time
        print(f"{'TSP_GD':<20} | {n:<5} | {cost_gd:<10.2f} | {time_gd:<10.4f}")

        # 2. TSP_Pointerformer (Greedy)
        start_time = time.perf_counter()
        cost_pf, _ = TSP_Pointerformer.tsp(points, indices)
        time_pf = time.perf_counter() - start_time
        print(f"{'TSP_Pointerformer':<20} | {n:<5} | {cost_pf:<10.2f} | {time_pf:<10.4f}")

        # 3. TSP_LKH (LKH Solver)
        try:
            start_time = time.perf_counter()
            cost_lkh, _ = TSP_LKH.tsp(points, indices)
            time_lkh = time.perf_counter() - start_time
            print(f"{'TSP_LKH':<20} | {n:<5} | {cost_lkh:<10.2f} | {time_lkh:<10.4f}")
        except Exception as e:
            print(f"{'TSP_LKH':<20} | {n:<5} | {'FAILED':<10} | {0:<10.4f}")
            print(f"  Error: {e}")

        print("-" * 55)

if __name__ == "__main__":
    run_benchmark()
