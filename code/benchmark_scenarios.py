import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Get workspace root
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
sys.path.append(os.path.join(workspace_root, 'code'))
sys.path.append(os.path.join(workspace_root, 'code', 'baseline'))

# Import Data Generator
import data_generator

# Import Algorithms
try:
    from baseline.algorithm import Graph, CoCycle, OSweep, MinExpand
except ImportError:
    from algorithm import Graph, CoCycle, OSweep, MinExpand

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

AlgoKMeans = import_module_from_path("AlgoKMeans", os.path.join(workspace_root, "code/algorithm/algo_kmeans.py"))
AlgoKruskal = import_module_from_path("AlgoKruskal", os.path.join(workspace_root, "code/algorithm/algo_kruskal.py"))
AlgoBKruskalGra = import_module_from_path("AlgoBKruskalGra", os.path.join(workspace_root, "code/algorithm/algo_bkruskal.py"))

# Configuration
N = 100
M = 5
WIDTH = 100
HEIGHT = 100
SCENARIOS = {
    "Uniform": data_generator.generate_uniform,
    "Multi-Center": data_generator.generate_gaussian_mixture,
    "Unbalanced": data_generator.generate_unbalanced_mixture,
    "Manifold": data_generator.generate_linear_manifold
}

# Setup logging
log_dir = os.path.join(workspace_root, 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'CSC_scenarios_benchmark_n{N}_m{M}_{timestamp}.csv')

print(f"Starting Scenario Benchmark with N={N}, M={M}")

with open(log_filename, 'w') as fout:
    header = 'Scenario,CoCycle,OSweep,MinExpand,KMeans,Kruskal,BKruskalGra,Best_K_KMeans,Best_K_Kruskal,Best_K_BKruskal\n'
    fout.write(header)
    
    results = {}
    
    for scenario_name, generator_func in SCENARIOS.items():
        print(f"\nRunning Scenario: {scenario_name}...")
        results[scenario_name] = {}
        
        # Generate data
        if scenario_name == "Uniform":
            points = generator_func(N, width=WIDTH, height=HEIGHT)
        elif scenario_name == "Multi-Center":
            points = generator_func(N, M, width=WIDTH, height=HEIGHT)
        elif scenario_name == "Unbalanced":
            points = generator_func(N, M, width=WIDTH, height=HEIGHT)
        elif scenario_name == "Manifold":
            points = generator_func(N, M, width=WIDTH, height=HEIGHT)
        else:
            points = generator_func(N, WIDTH, HEIGHT)
            
        # Plot the scenario for verification
        plt.figure(figsize=(6, 6))
        plt.scatter(points[:, 0], points[:, 1], s=10)
        plt.title(f"Scenario: {scenario_name}")
        plt.savefig(os.path.join(log_dir, f"Scenario_{scenario_name}_{timestamp}.png"))
        plt.close()
        
        # Prepare Graph for Baselines
        graph = Graph(width=WIDTH, height=HEIGHT, n=N, points=points)
        
        # --- Baselines ---
        _, _, p_cocycle = CoCycle(graph, M)
        _, _, p_osweep = OSweep(graph, M)
        _, _, p_minexpand = MinExpand(graph, M)
        
        # --- New Algorithms ---
        # Note: These now use the Hybrid (Greedy+LKH) strategy we implemented
        p_kmeans, k_kmeans, _ = AlgoKMeans.run(points, M)
        p_kruskal, k_kruskal, _ = AlgoKruskal.run(points, M)
        p_bkruskal, k_bkruskal, _ = AlgoBKruskalGra.run(points, M)
        
        # Log
        line = f"{scenario_name},{p_cocycle},{p_osweep},{p_minexpand},{p_kmeans},{p_kruskal},{p_bkruskal},{k_kmeans},{k_kruskal},{k_bkruskal}\n"
        fout.write(line)
        
        print(f"  Baselines -> CoCycle: {p_cocycle:.2f}, OSweep: {p_osweep:.2f}, MinExpand: {p_minexpand:.2f}")
        print(f"  New Algos -> KMeans: {p_kmeans:.2f} (k={k_kmeans}), Kruskal: {p_kruskal:.2f} (k={k_kruskal}), BKruskalGra: {p_bkruskal:.2f} (k={k_bkruskal})")
        
        results[scenario_name] = {
            'CoCycle': p_cocycle, 'OSweep': p_osweep, 'MinExpand': p_minexpand,
            'KMeans': p_kmeans, 'Kruskal': p_kruskal, 'BKruskalGra': p_bkruskal
        }

    # Plotting
    scenario_names = list(results.keys())
    x = np.arange(len(scenario_names))
    width = 0.12

    fig, ax = plt.subplots(figsize=(14, 8))
    
    vals_cocycle = [results[s]['CoCycle'] for s in scenario_names]
    vals_osweep = [results[s]['OSweep'] for s in scenario_names]
    vals_minexpand = [results[s]['MinExpand'] for s in scenario_names]
    vals_kmeans = [results[s]['KMeans'] for s in scenario_names]
    vals_kruskal = [results[s]['Kruskal'] for s in scenario_names]
    vals_bkruskal = [results[s]['BKruskalGra'] for s in scenario_names]

    rects1 = ax.bar(x - 2.5*width, vals_cocycle, width, label='CoCycle')
    rects2 = ax.bar(x - 1.5*width, vals_osweep, width, label='OSweep')
    rects3 = ax.bar(x - 0.5*width, vals_minexpand, width, label='MinExpand')
    rects4 = ax.bar(x + 0.5*width, vals_kmeans, width, label='E-Cluster-KMeans')
    rects5 = ax.bar(x + 1.5*width, vals_kruskal, width, label='E-Cluster-Kruskal')
    rects6 = ax.bar(x + 2.5*width, vals_bkruskal, width, label='E-Cluster-BKruskal-Gra')

    ax.set_ylabel('Max Sweep Period')
    ax.set_title(f'Performance Comparison by Scenario (N={N}, m={M})')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names)
    ax.legend()
    ax.grid(True, axis='y')

    plot_filename = os.path.join(log_dir, f'CSC_scenarios_benchmark_n{N}_m{M}_{timestamp}.png')
    plt.savefig(plot_filename)
    print(f"Benchmark plot saved to {plot_filename}")
