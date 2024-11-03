import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
import time
import os

# 打开.pkl文件并加载数据
with open('./mnist_clustering_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

# 现在data变量包含了.pkl文件中的数据
data = np.array(data)

Z_single = np.load('./task2/Single Linkage Info.npy')
Z_complete = np.load('./task2/Complete Linkage Info.npy')
Z_average = np.load('./task2/Average Linkage Info.npy')

# 定义保存路径
save_dir = './task2/evaluate'
os.makedirs(save_dir, exist_ok=True)

# 定义评估函数
def evaluate_clustering(Z, method_name, max_clusters=60):
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        # 从链接矩阵中提取聚类标签
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # 计算轮廓系数
        silhouette = silhouette_score(data.reshape(data.shape[0], -1), labels)
        silhouette_scores.append(silhouette)
        
        # 计算 Calinski-Harabasz 指数
        ch_index = calinski_harabasz_score(data.reshape(data.shape[0], -1), labels)
        calinski_harabasz_scores.append(ch_index)
        
        # 计算 Davies-Bouldin 指数
        db_index = davies_bouldin_score(data.reshape(data.shape[0], -1), labels)
        davies_bouldin_scores.append(db_index)
        
        print(f"Method: {method_name}, Number of clusters: {n_clusters}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Calinski-Harabasz Score: {ch_index:.4f}")
        print(f"  Davies-Bouldin Score: {db_index:.4f}")
    
    return silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores

# 评估三种方法
methods = ['Single Link', 'Complete Link', 'Average Link']
linkage_matrices = [Z_single, Z_complete, Z_average]

results = {}
for method, Z in zip(methods, linkage_matrices):
    silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores = evaluate_clustering(Z, method)
    results[method] = {
        'silhouette_scores': silhouette_scores,
        'calinski_harabasz_scores': calinski_harabasz_scores,
        'davies_bouldin_scores': davies_bouldin_scores
    }

# 绘制各种指标随聚类数量的变化图
max_clusters = 60
plt.figure(figsize=(15, 5))

# 绘制轮廓系数图
plt.figure(figsize=(10, 5))
for method in methods:
    plt.plot(range(2, max_clusters + 1), results[method]['silhouette_scores'], marker='o', label=method)
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend()
plt.savefig(os.path.join(save_dir, "silhouette_scores.png"))
plt.close()

# 绘制 Calinski-Harabasz 指数图
plt.figure(figsize=(10, 5))
for method in methods:
    plt.plot(range(2, max_clusters + 1), results[method]['calinski_harabasz_scores'], marker='o', label=method)
plt.title("Calinski-Harabasz Score vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Score")
plt.legend()
plt.savefig(os.path.join(save_dir, "calinski_harabasz_scores.png"))
plt.close()

# 绘制 Davies-Bouldin 指数图
plt.figure(figsize=(10, 5))
for method in methods:
    plt.plot(range(2, max_clusters + 1), results[method]['davies_bouldin_scores'], marker='o', label=method)
plt.title("Davies-Bouldin Score vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Score")
plt.legend()
plt.savefig(os.path.join(save_dir, "davies_bouldin_scores.png"))
plt.close()


# # 选择最佳的聚类数量
# best_n_clusters = {}
# for method in methods:
#     best_n_clusters[method] = {
#         'silhouette': np.argmax(results[method]['silhouette_scores']) + 2,
#         'calinski_harabasz': np.argmax(results[method]['calinski_harabasz_scores']) + 2,
#         'davies_bouldin': np.argmin(results[method]['davies_bouldin_scores']) + 2
#     }
#     print(f"Best number of clusters for {method}:")
#     print(f"  Silhouette Score: {best_n_clusters[method]['silhouette']}")
#     print(f"  Calinski-Harabasz Score: {best_n_clusters[method]['calinski_harabasz']}")
#     print(f"  Davies-Bouldin Score: {best_n_clusters[method]['davies_bouldin']}")

# # 保存最佳 K 值的结果
# with open(os.path.join(save_dir, "optimal_k_values.txt"), 'w') as file:
#     for method in methods:
#         file.write(f"Best number of clusters for {method}:\n")
#         file.write(f"  Silhouette Score: {best_n_clusters[method]['silhouette']}\n")
#         file.write(f"  Calinski-Harabasz Score: {best_n_clusters[method]['calinski_harabasz']}\n")
#         file.write(f"  Davies-Bouldin Score: {best_n_clusters[method]['davies_bouldin']}\n")