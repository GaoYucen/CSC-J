import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Get workspace root (assuming this script is in code/)
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)

# Add 'code' directory to sys.path so that 'clustering' and 'TSP' packages can be found
sys.path.append(os.path.join(workspace_root, 'code'))

# Import Baseline Algorithms
# We need to make sure we can import from code/baseline
sys.path.append(os.path.join(workspace_root, 'code', 'baseline'))
try:
    from algorithm import Graph, CoCycle, OSweep, MinExpand, PDBA
except ImportError:
    # Fallback if running from a different context
    from baseline.algorithm import Graph, CoCycle, OSweep, MinExpand, PDBA

# Helper to import modules with hyphens
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import New Algorithms
# Using relative paths from workspace root
AlgoKMeans = import_module_from_path("AlgoKMeans", os.path.join(workspace_root, "code/algorithm/algo_kmeans.py"))
AlgoKruskal = import_module_from_path("AlgoKruskal", os.path.join(workspace_root, "code/algorithm/algo_kruskal.py"))
AlgoBKruskalGra = import_module_from_path("AlgoBKruskalGra", os.path.join(workspace_root, "code/algorithm/algo_bkruskal.py"))

# Configuration
N_VALUES = [50, 100, 150, 200, 250, 300]
M = 5
WIDTH = 100
HEIGHT = 100

# Setup logging
log_dir = os.path.join(workspace_root, 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'CSC_comparison_benchmark_m{M}_{timestamp}.csv')
plot_filename_period = os.path.join(log_dir, f'CSC_comparison_period_m{M}_{timestamp}.png')
plot_filename_time = os.path.join(log_dir, f'CSC_comparison_time_m{M}_{timestamp}.png')

results = {
    'n': [],
    'CoCycle_Period': [], 'CoCycle_Time': [],
    'OSweep_Period': [], 'OSweep_Time': [],
    'MinExpand_Period': [], 'MinExpand_Time': [],
    'PDBA_Period': [], 'PDBA_Time': [],
    'KMeans_Period': [], 'KMeans_Time': [],
    'Kruskal_Period': [], 'Kruskal_Time': [],
    'BKruskalGra_Period': [], 'BKruskalGra_Time': []
}

print(f"Starting benchmark with N={N_VALUES}, m={M}")

with open(log_filename, 'w') as fout:
    header = 'n,CoCycle_Period,OSweep_Period,MinExpand_Period,PDBA_Period,KMeans_Period,Kruskal_Period,BKruskalGra_Period,' \
             'CoCycle_Time,OSweep_Time,MinExpand_Time,PDBA_Time,KMeans_Time,Kruskal_Time,BKruskalGra_Time\n'
    fout.write(header)
    
    for n in N_VALUES:
        print(f"\nRunning for n={n}...")
        
        # Generate data: Uniform distribution
        points = np.random.uniform(0, WIDTH, (n, 2))
        
        # Prepare Graph for Baselines
        graph = Graph(width=WIDTH, height=HEIGHT, n=n, points=points)
        
        # --- Baselines ---
        # CoCycle
        t0 = time.perf_counter()
        _, _, p_cocycle = CoCycle(graph, M)
        t_cocycle = time.perf_counter() - t0
        
        # OSweep
        t0 = time.perf_counter()
        _, _, p_osweep = OSweep(graph, M)
        t_osweep = time.perf_counter() - t0
        
        # MinExpand
        t0 = time.perf_counter()
        _, _, p_minexpand = MinExpand(graph, M)
        t_minexpand = time.perf_counter() - t0
        
        # PDBA
        t0 = time.perf_counter()
        _, _, p_pdba = PDBA(graph, M)
        t_pdba = time.perf_counter() - t0
        
        # --- New Algorithms ---
        # KMeans
        t0 = time.perf_counter()
        p_kmeans, k_kmeans, vals_kmeans = AlgoKMeans.run(points, M)
        t_kmeans = time.perf_counter() - t0
        
        # Kruskal
        t0 = time.perf_counter()
        p_kruskal, k_kruskal, vals_kruskal = AlgoKruskal.run(points, M)
        t_kruskal = time.perf_counter() - t0
        
        # BKruskalGra
        t0 = time.perf_counter()
        p_bkruskal, k_bkruskal, vals_bkruskal = AlgoBKruskalGra.run(points, M)
        t_bkruskal = time.perf_counter() - t0
        
        # Record
        results['n'].append(n)
        results['CoCycle_Period'].append(p_cocycle)
        results['CoCycle_Time'].append(t_cocycle)
        results['OSweep_Period'].append(p_osweep)
        results['OSweep_Time'].append(t_osweep)
        results['MinExpand_Period'].append(p_minexpand)
        results['MinExpand_Time'].append(t_minexpand)
        results['PDBA_Period'].append(p_pdba)
        results['PDBA_Time'].append(t_pdba)
        results['KMeans_Period'].append(p_kmeans)
        results['KMeans_Time'].append(t_kmeans)
        results['Kruskal_Period'].append(p_kruskal)
        results['Kruskal_Time'].append(t_kruskal)
        results['BKruskalGra_Period'].append(p_bkruskal)
        results['BKruskalGra_Time'].append(t_bkruskal)
        
        line = f"{n},{p_cocycle},{p_osweep},{p_minexpand},{p_pdba},{p_kmeans},{p_kruskal},{p_bkruskal}," \
               f"{t_cocycle},{t_osweep},{t_minexpand},{t_pdba},{t_kmeans},{t_kruskal},{t_bkruskal}\n"
        fout.write(line)
        print(f"  Baselines -> CoCycle: {p_cocycle:.2f}, OSweep: {p_osweep:.2f}, MinExpand: {p_minexpand:.2f}, PDBA: {p_pdba:.2f}")
        print(f"  New Algos -> KMeans: {p_kmeans:.2f} (k={k_kmeans}), Kruskal: {p_kruskal:.2f} (k={k_kruskal}), BKruskalGra: {p_bkruskal:.2f} (k={k_bkruskal})")
        print(f"    KMeans Vals: {np.round(vals_kmeans, 2)}")
        print(f"    Kruskal Vals: {np.round(vals_kruskal, 2)}")
        print(f"    BKruskal Vals: {np.round(vals_bkruskal, 2)}")

# Plotting Period
plt.figure(figsize=(12, 8))
plt.plot(results['n'], results['CoCycle_Period'], marker='o', label='CoCycle')
plt.plot(results['n'], results['OSweep_Period'], marker='s', label='OSweep')
plt.plot(results['n'], results['MinExpand_Period'], marker='^', label='MinExpand')
# plt.plot(results['n'], results['PDBA_Period'], marker='x', label='PDBA')
plt.plot(results['n'], results['KMeans_Period'], marker='D', linestyle='--', label='E-Cluster-KMeans')
plt.plot(results['n'], results['Kruskal_Period'], marker='*', linestyle='--', label='E-Cluster-Kruskal')
plt.plot(results['n'], results['BKruskalGra_Period'], marker='p', linestyle='--', label='E-Cluster-BKruskal-Gra')

plt.xlabel('Number of Points (n)')
plt.ylabel('Max Sweep Period')
plt.title(f'Performance Comparison (Period) vs Scale (m={M})')
plt.legend()
plt.grid(True)
plt.savefig(plot_filename_period)
print(f"Period plot saved to {plot_filename_period}")

# Plotting Time
plt.figure(figsize=(12, 8))
plt.plot(results['n'], results['CoCycle_Time'], marker='o', label='CoCycle')
plt.plot(results['n'], results['OSweep_Time'], marker='s', label='OSweep')
plt.plot(results['n'], results['MinExpand_Time'], marker='^', label='MinExpand')
# plt.plot(results['n'], results['PDBA_Time'], marker='x', label='PDBA')
plt.plot(results['n'], results['KMeans_Time'], marker='D', linestyle='--', label='E-Cluster-KMeans')
plt.plot(results['n'], results['Kruskal_Time'], marker='*', linestyle='--', label='E-Cluster-Kruskal')
plt.plot(results['n'], results['BKruskalGra_Time'], marker='p', linestyle='--', label='E-Cluster-BKruskal-Gra')

plt.xlabel('Number of Points (n)')
plt.ylabel('Execution Time (s)')
plt.title(f'Execution Time Comparison vs Scale (m={M})')
plt.legend()
plt.grid(True)
plt.savefig(plot_filename_time)
print(f"Time plot saved to {plot_filename_time}")
