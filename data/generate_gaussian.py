import numpy as np
import matplotlib.pyplot as plt
import os

def generate_hybrid_points(n_samples=200, n_clusters=4, width=100, height=100, noise_ratio=0.2, random_seed=None):
    """
    生成混合分布的点：高斯聚类 + 均匀背景噪声
    模拟工业场景：大部分点集中在作业区（聚类），少部分点散布在全局（噪声）。
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n_noise = int(n_samples * noise_ratio)
    n_clustered = n_samples - n_noise
    
    points_list = []
    
    # 1. 生成聚类点 (模拟作业岛/货架区)
    if n_clustered > 0:
        samples_per_cluster = n_clustered // n_clusters
        # 让聚类中心尽量分布在中间区域，避免太靠边
        centers = np.random.uniform(width*0.1, width*0.9, (n_clusters, 2)) 
        # 聚类的紧凑程度
        std_devs = np.random.uniform(width/30, width/15, n_clusters)
        
        for i in range(n_clusters):
            count = samples_per_cluster
            # 补齐除不尽的余数
            if i == n_clusters - 1:
                count = n_clustered - (samples_per_cluster * (n_clusters - 1))
            
            cluster_points = np.random.normal(loc=centers[i], scale=std_devs[i], size=(count, 2))
            points_list.append(cluster_points)
            
    # 2. 生成噪声点 (模拟离散任务/均匀分布)
    if n_noise > 0:
        noise_points = np.random.uniform(0, min(width, height), (n_noise, 2))
        points_list.append(noise_points)
        
    points = np.vstack(points_list)
    np.random.shuffle(points) # 打乱顺序，避免聚类点在数组中扎堆
    
    # 裁剪坐标到 [0, width] 和 [0, height]
    points[:, 0] = np.clip(points[:, 0], 0, width)
    points[:, 1] = np.clip(points[:, 1], 0, height)
    
    return points

def save_and_plot(points, output_data_path, output_img_path):
    # 保存数据，格式为每行两个浮点数，空格分隔，保留6位小数
    np.savetxt(output_data_path, points, fmt='%.6f', delimiter=' ')
    print(f"[Success] Data saved to: {output_data_path}")
    
    # 画图展示
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.7, c='#1f77b4', edgecolors='k', s=40)
    plt.title(f"Gaussian Clustered Distribution (n={len(points)})\n(Clusters + 20% Noise)", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=150)
    print(f"[Success] Plot saved to: {output_img_path}")

if __name__ == "__main__":
    # 参数配置
    N = 200              # 总点数
    CLUSTERS = 4         # 聚类中心数量
    WIDTH = 100          # 区域宽度
    HEIGHT = 100         # 区域高度
    NOISE_RATIO = 0.2    # 噪声比例 (20%的点是均匀分布的)
    SEED = 42            # 随机种子，保证可复现
    
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DATA = os.path.join(current_dir, 'points_gaussian.csv')
    OUTPUT_IMG = os.path.join(current_dir, 'gaussian_distribution.png')
    
    print(f"Generating {N} points with {CLUSTERS} gaussian clusters and {NOISE_RATIO*100}% noise...")
    
    pts = generate_hybrid_points(N, CLUSTERS, WIDTH, HEIGHT, NOISE_RATIO, SEED)
    save_and_plot(pts, OUTPUT_DATA, OUTPUT_IMG)
    
    print("-" * 50)
    print("Tip: You can use this file in your benchmark by changing the input path.")
