import numpy as np
import time
import os
import datetime
import matplotlib.pyplot as plt
from algorithm import *

# Configuration
N_VALUES = [50, 100, 150, 200, 250, 300]
M = 5
WIDTH = 100
HEIGHT = 100

# Setup logging
if not os.path.exists('log'):
    os.makedirs('log')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'log/CSC_scale_benchmark_m{M}_{timestamp}.csv'
plot_filename_period = f'log/CSC_scale_benchmark_period_m{M}_{timestamp}.png'
plot_filename_time = f'log/CSC_scale_benchmark_time_m{M}_{timestamp}.png'

results = {
    'n': [],
    'CoCycle_Period': [], 'CoCycle_Time': [],
    'OSweep_Period': [], 'OSweep_Time': [],
    'MinExpand_Period': [], 'MinExpand_Time': [],
    'PDBA_Period': [], 'PDBA_Time': []
}

print(f"Starting benchmark with N={N_VALUES}, m={M}")
print(f"Points distribution: Uniform random in [{WIDTH}x{HEIGHT}]")

with open(log_filename, 'w') as fout:
    fout.write('n,CoCycle_Period,OSweep_Period,MinExpand_Period,PDBA_Period,CoCycle_Time,OSweep_Time,MinExpand_Time,PDBA_Time\n')
    
    for n in N_VALUES:
        print(f"\nRunning for n={n}...")
        
        # Generate data: Uniform distribution
        points = np.random.uniform(0, WIDTH, (n, 2))
        
        graph = Graph(width=WIDTH, height=HEIGHT, n=n, points=points)
        
        # Run algorithms
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
        
        line = f"{n},{p_cocycle},{p_osweep},{p_minexpand},{p_pdba},{t_cocycle},{t_osweep},{t_minexpand},{t_pdba}\n"
        fout.write(line)
        print(f"  Done. CoCycle: {p_cocycle:.2f}, OSweep: {p_osweep:.2f}, MinExpand: {p_minexpand:.2f}, PDBA: {p_pdba:.2f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(results['n'], results['CoCycle_Period'], marker='o', label='CoCycle')
plt.plot(results['n'], results['OSweep_Period'], marker='s', label='OSweep')
plt.plot(results['n'], results['MinExpand_Period'], marker='^', label='MinExpand')
plt.plot(results['n'], results['PDBA_Period'], marker='x', label='PDBA')
plt.xlabel('Number of Points (n)')
plt.ylabel('Max Sweep Period')
plt.title(f'Performance Comparison (Period) vs Scale (m={M})')
plt.legend()
plt.grid(True)
plt.savefig(plot_filename_period)
print(f"Period plot saved to {plot_filename_period}")

plt.figure(figsize=(10, 6))
plt.plot(results['n'], results['CoCycle_Time'], marker='o', label='CoCycle')
plt.plot(results['n'], results['OSweep_Time'], marker='s', label='OSweep')
plt.plot(results['n'], results['MinExpand_Time'], marker='^', label='MinExpand')
plt.plot(results['n'], results['PDBA_Time'], marker='x', label='PDBA')
plt.xlabel('Number of Points (n)')
plt.ylabel('Execution Time (s)')
plt.title(f'Execution Time vs Scale (m={M})')
plt.legend()
plt.grid(True)
plt.savefig(plot_filename_time)
print(f"Time plot saved to {plot_filename_time}")
