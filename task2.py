import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from matplotlib.collections import LineCollection
import os

# 打开.pkl文件并加载数据
with open('./mnist_clustering_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

# 现在data变量包含了.pkl文件中的数据
images = np.array(data)
def compute_distance_matrix(X):
    """计算所有样本之间的两两距离矩阵"""
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(X[i] - X[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def single_linkage(dist_matrix, cluster1, cluster2):
    """单链接（Single Link）距离度量"""
    min_dist = np.inf
    for i in cluster1:
        for j in cluster2:
            if i < j:
                dist = dist_matrix[i, j]
            else:
                dist = dist_matrix[j, i]
            if dist < min_dist:
                min_dist = dist
    return min_dist

def complete_linkage(dist_matrix, cluster1, cluster2):
    """全链接（Complete Link）距离度量"""
    max_dist = -np.inf
    for i in cluster1:
        for j in cluster2:
            if i < j:
                dist = dist_matrix[i, j]
            else:
                dist = dist_matrix[j, i]
            if dist > max_dist:
                max_dist = dist
    return max_dist

def average_linkage(dist_matrix, cluster1, cluster2):
    """平均链接（Average Link）距离度量"""
    total_dist = 0
    count = 0
    for i in cluster1:
        for j in cluster2:
            if i < j:
                dist = dist_matrix[i, j]
            else:
                dist = dist_matrix[j, i]
            total_dist += dist
            count += 1
    return total_dist / count

def hierarchical_clustering(X, method='single', num_clusters=1):
    # 将数据展平为二维数组 (3000, 784)
    start_time_1 = time.time()
    X = X.reshape(X.shape[0], -1)
    
    n = X.shape[0]
    clusters = [[i] for i in range(n)]
    clusters_num = [i for i in range(n)]
    num = n
    distances = compute_distance_matrix(X)
    
    linkage_info = []
    
    while len(clusters) > num_clusters:
        min_dist = np.inf
        merge_i, merge_j = None, None
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if method == 'single':
                    dist = single_linkage(distances, clusters[i], clusters[j])
                elif method == 'complete':
                    dist = complete_linkage(distances, clusters[i], clusters[j])
                elif method == 'average':
                    dist = average_linkage(distances, clusters[i], clusters[j])
                
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = i, j
        
        # 合并最近的两个簇
        new_cluster = clusters[merge_i] + clusters[merge_j]
        del clusters[merge_j]
        del clusters[merge_i]
        
        
        # 更新距离矩阵
        for k in range(len(clusters)):
            if k != merge_i and k != merge_j:
                if method == 'single':
                    new_dist = single_linkage(distances, new_cluster, clusters[k])
                elif method == 'complete':
                    new_dist = complete_linkage(distances, new_cluster, clusters[k])
                elif method == 'average':
                    new_dist = average_linkage(distances, new_cluster, clusters[k])
                
                distances[clusters[k][0], new_cluster[0]] = new_dist
                distances[new_cluster[0], clusters[k][0]] = new_dist
        
        # 记录合并信息
        linkage_info.append((clusters_num[merge_i], clusters_num[merge_j], min_dist, len(new_cluster)))
        del clusters_num[merge_j]
        del clusters_num[merge_i]
        
        # 添加新簇
        clusters.append(new_cluster)
        clusters_num.append(num)
        num += 1
        end_time_1 = time.time()
        print(f"{method}：完成{(num - n) / n * 100:2f}% {num - n}/{n}，已花费: {end_time_1 - start_time_1:.4f} 秒")
    return linkage_info

def plot_dendrogram(linkage_info, title, filename, save_dir = "task2"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    Z = np.array(linkage_info)
    plt.figure(figsize=(10, 8))

    ddata = dendrogram(Z, truncate_mode='lastp', p=100, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=5, color_threshold=0)

    # set linewidths
    # 获取树状图的所有线条
    icoord = ddata['icoord']
    dcoord = ddata['dcoord']
    # 创建 LineCollection 对象
    line_segments = []
    for x, y in zip(icoord, dcoord):
        line_segments.append(list(zip(x, y)))
    lc = LineCollection(line_segments, linewidths = 0.3)
    # 将 LineCollection 添加到当前图形
    ax = plt.gca()
    ax.add_collection(lc)

    plt.ylim(np.min(Z[-100:, 2]) * 0.98, np.max(Z[-100:, 2]) * 1.02)
    plt.title(title)
    plt.xlabel('index')
    plt.ylabel('distance')
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# 生成示例数据
np.random.seed(1)  # 为了结果可重复
data = images[0:200, :, :, :]  # 示例数据，实际应替换为你的数据

# 设置不同的链接方法
methods = ['single', 'complete', 'average']

# 实验每种链接方法
for method in methods:
    start_time = time.time()
    linkage_info = hierarchical_clustering(data, method=method, num_clusters=1)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"方法: {method}")
    print(f"  计算时间: {elapsed_time:.4f} 秒")
    
    # 绘制从 1 簇到 100 簇的树状图
    plot_dendrogram(linkage_info, f"{method.capitalize()} Linkage Dendrogram (1 to 100 clusters)", f"{method.capitalize()} Linkage Dendrogram (1 to 100 clusters).png")