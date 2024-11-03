import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import os
import pickle
import time


# 打开.pkl文件并加载数据
with open('./mnist_clustering_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

# 现在data变量包含了.pkl文件中的数据
data = np.array(data)


# 将数据展平为二维数组 [3000, 784]
data_flattened = data.reshape(data.shape[0], -1)

# 定义保存路径
save_dir = './task2/standard'
os.makedirs(save_dir, exist_ok=True)
max_clusters = 60


# 定义评估函数
def evaluate_clustering(Z, method_name, max_clusters=60):
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        # 从链接矩阵中提取聚类标签
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # 计算轮廓系数
        silhouette = silhouette_score(data_flattened, labels)
        silhouette_scores.append(silhouette)
        
        print(f"Method: {method_name}, Number of clusters: {n_clusters}")
        print(f"  Silhouette Score: {silhouette:.4f}")
    
    return silhouette_scores

# 计算三种链接矩阵
start_time = time.time()
Z_single = linkage(data_flattened, 'single')
Z_complete = linkage(data_flattened, 'complete')
Z_average = linkage(data_flattened, 'average')
end_time = time.time()
print(f"{end_time - start_time}秒")

# 评估三种方法
methods = ['Single Link', 'Complete Link', 'Average Link']
linkage_matrices = [Z_single, Z_complete, Z_average]

results = {}
for method, Z in zip(methods, linkage_matrices):
    silhouette_scores = evaluate_clustering(Z, method)
    results[method] = silhouette_scores

# 绘制轮廓系数图
plt.figure(figsize=(10, 5))
for method in methods:
    plt.plot(range(2, max_clusters + 1), results[method], marker='o', label=method)
plt.title("Silhouette Score vs Number of Clusters(Standard)")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend()
plt.savefig(os.path.join(save_dir, "silhouette_scores.png"))
plt.close()

print("图片已保存，继续执行其他计算...")