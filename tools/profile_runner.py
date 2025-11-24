import os
import cProfile
import pstats
import io
import numpy as np
import sys

# Ensure repo root is on path to import modules
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from code.clustering import kruskal_clustering, kruskal_clustering_balance
from code.TSP import TSP_Pointerformer

DATA_PATH = os.path.join(REPO_ROOT, 'data', 'points.csv')

def load_points(n=None):
    pts = np.loadtxt(DATA_PATH)
    if n is not None:
        return pts[:n]
    return pts

def profile_func(func, *args, outname=None, run_name=None):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    out = s.getvalue()
    header = f"==== Profile ({run_name or func.__name__}) ===="
    print(header)
    print(out)
    if outname:
        with open(outname, 'w') as f:
            f.write(header + "\n" + out)
    return result

def run_all():
    print('Loading points...')
    points = load_points(200)  # moderate size for profiling

    print('\nProfiling kruskal_clustering.clustering (k=3)...')
    profile_func(kruskal_clustering.clustering, points, 3, outname='profile_kruskal.txt', run_name='kruskal_clustering')

    print('\nProfiling kruskal_clustering_balance.clustering (k=5)...')
    profile_func(kruskal_clustering_balance.clustering, points, 5, outname='profile_kruskal_balance.txt', run_name='kruskal_balance')

    print('\nPreparing data for TSP...')
    # TSP_Pointerformer expects reading CSV internally in its main; use its tsp function
    pts = load_points(80)  # smaller for TSP
    column_index = list(range(int(len(pts))))
    print('\nProfiling TSP_Pointerformer.tsp...')
    profile_func(TSP_Pointerformer.tsp, pts, column_index, outname='profile_tsp.txt', run_name='tsp')

if __name__ == '__main__':
    run_all()
