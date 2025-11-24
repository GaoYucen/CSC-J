import numpy as np
import math

def generate_uniform(n, width=100, height=100):
    """
    Scenario: Uniform distribution.
    Real-world: Agricultural monitoring of a flat, uniform field.
    """
    return np.random.uniform(0, width, (n, 2))

def generate_gaussian_mixture(n, m, width=100, height=100, std=5.0):
    """
    Scenario: Multi-center hotspots (Balanced).
    Real-world: Urban logistics with multiple distinct residential districts.
    """
    points = []
    # Fixed centers to ensure separation, or random
    # Let's use random centers but ensure they aren't too close
    centers = []
    while len(centers) < m:
        c = np.random.uniform(10, width-10, 2)
        if all(np.linalg.norm(c - existing) > 20 for existing in centers):
            centers.append(c)
    
    points_per_cluster = n // m
    remainder = n % m
    
    for i in range(m):
        count = points_per_cluster + (1 if i < remainder else 0)
        cluster_points = np.random.normal(centers[i], std, (count, 2))
        # Clip to boundaries
        cluster_points = np.clip(cluster_points, 0, width)
        points.append(cluster_points)
    
    return np.vstack(points)

def generate_unbalanced_mixture(n, m, width=100, height=100, std=5.0):
    """
    Scenario: Unbalanced hotspots.
    Real-world: City center (dense) vs Suburbs (sparse).
    """
    points = []
    centers = []
    while len(centers) < m:
        c = np.random.uniform(10, width-10, 2)
        if all(np.linalg.norm(c - existing) > 20 for existing in centers):
            centers.append(c)
            
    # 60% points in the first cluster
    n_large = int(n * 0.6)
    n_small = (n - n_large) // (m - 1)
    
    counts = [n_large] + [n_small] * (m - 2)
    counts.append(n - sum(counts))
    
    for i in range(m):
        # First cluster is denser (same std, more points)
        cluster_points = np.random.normal(centers[i], std, (counts[i], 2))
        cluster_points = np.clip(cluster_points, 0, width)
        points.append(cluster_points)
        
    return np.vstack(points)

def generate_linear_manifold(n, m, width=100, height=100):
    """
    Scenario: Linear/Curved structures.
    Real-world: Pipeline or coastline inspection.
    """
    points = []
    points_per_line = n // m
    remainder = n % m
    
    for i in range(m):
        count = points_per_line + (1 if i < remainder else 0)
        
        # Random start and end points
        start = np.random.uniform(0, width, 2)
        # End point somewhat far away
        angle = np.random.uniform(0, 2*math.pi)
        length = np.random.uniform(30, 60)
        end = start + np.array([math.cos(angle), math.sin(angle)]) * length
        
        # Interpolate points along the line with some noise
        t = np.random.uniform(0, 1, count)
        noise = np.random.normal(0, 2, (count, 2)) # Small noise width
        
        line_points = np.zeros((count, 2))
        for j in range(count):
            line_points[j] = start + t[j] * (end - start)
            
        line_points += noise
        line_points = np.clip(line_points, 0, width)
        points.append(line_points)
        
    return np.vstack(points)
