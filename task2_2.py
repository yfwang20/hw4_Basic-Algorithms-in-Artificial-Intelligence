import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

# 打开.pkl文件并加载数据
with open('./mnist_clustering_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

# 现在data变量包含了.pkl文件中的数据
images = np.array(data)

# 计算两点之间的欧氏距离
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# 初始化距离矩阵
def init_distance_matrix(data):
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            dist = euclidean_distance(data[i], data[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix
def hierarchical_clustering(data, method='single', max_clusters=100):
    start_time = time.time()
    
    n_samples = data.shape[0]
    clusters = [[i] for i in range(n_samples)]
    distance_matrix = init_distance_matrix(data)
    history = []
    
    while len(clusters) > max_clusters:
        # 找到最小距离
        if method == 'single':
            min_dist = np.min(distance_matrix + np.eye(distance_matrix.shape[0]) * np.inf)
            idx = np.where(distance_matrix == min_dist)
            i, j = idx[0][0], idx[1][0]
            
        elif method == 'complete':
            min_dist = np.min(distance_matrix + np.eye(distance_matrix.shape[0]) * np.inf)
            idx = np.where(distance_matrix == min_dist)
            i, j = idx[0][0], idx[1][0]
            
        elif method == 'average':
            avg_distances = np.array([[np.mean([euclidean_distance(data[x], data[y]) for x in clusters[i] for y in clusters[j]]) 
                                       for j in range(len(clusters))] for i in range(len(clusters))])
            min_dist = np.min(avg_distances + np.eye(avg_distances.shape[0]) * np.inf)
            idx = np.where(avg_distances == min_dist)
            i, j = idx[0][0], idx[1][0]
        
        # 合并簇
        new_cluster = clusters[i] + clusters[j]
        del clusters[max(i, j)]
        del clusters[min(i, j)]
        clusters.append(new_cluster)
        
        # 更新距离矩阵
        new_index = len(clusters) - 1
        new_distance_matrix = np.zeros((new_index, new_index))
        
        for k in range(new_index):
            if method == 'single':
                new_distance_matrix[k, new_index] = min(min(distance_matrix[k, i], distance_matrix[k, j]))
                new_distance_matrix[new_index, k] = new_distance_matrix[k, new_index]
            elif method == 'complete':
                new_distance_matrix[k, new_index] = max(max(distance_matrix[k, i], distance_matrix[k, j]))
                new_distance_matrix[new_index, k] = new_distance_matrix[k, new_index]
            elif method == 'average':
                new_distance_matrix[k, new_index] = np.mean([distance_matrix[k, x] for x in [i, j]])
                new_distance_matrix[new_index, k] = new_distance_matrix[k, new_index]
        
        # 删除旧的距离
        distance_matrix = new_distance_matrix
        
        # 记录历史
        history.append((len(clusters), clusters.copy()))
        
    end_time = time.time()
    print(f"Clustering with {method} link took {end_time - start_time:.2f} seconds.")
    return history


def plot_dendrogram(history, method):
    links = []
    for step, (num_clusters, clusters) in enumerate(history):
        for cluster in clusters:
            if len(cluster) > 1:
                links.append((step, num_clusters, cluster))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'Dendrogram for {method.capitalize()} Linkage')
    for step, num_clusters, cluster in links:
        if step > 0:
            prev_cluster = [c for c in history[step-1][1] if set(c).issubset(set(cluster))][0]
            ax.plot([step-1, step], [history[step-1][0], num_clusters], color='b')
            ax.scatter(step, num_clusters, color='r', s=10)
    
    plt.xlabel('Step')
    plt.ylabel('Number of Clusters')
    plt.show()



methods = ['single', 'complete', 'average']
for method in methods:
    history = hierarchical_clustering(images[0:300, :, :, :], method=method)
    plot_dendrogram(history, method)