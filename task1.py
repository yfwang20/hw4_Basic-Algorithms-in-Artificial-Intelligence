import pickle
import numpy as np
import time
import matplotlib.pyplot as plt


# 打开.pkl文件并加载数据
with open('./mnist_clustering_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

# 现在data变量包含了.pkl文件中的数据
images = np.array(data)

def k_means(X, K, max_iters=100):
    # X: 数据集 (3000, 1, 28, 28)
    # K: 聚类数量
    # max_iters: 最大迭代次数
    
    # 将数据展平为二维数组 (3000, 784)
    X = X.reshape(X.shape[0], -1)
    
    # 随机选择初始质心
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    
    for _ in range(max_iters):
        # 计算每个样本到每个质心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis = -1)
        
        # 分配每个样本到最近的质心
        labels = np.argmin(distances, axis = 1)
        
        # 更新质心
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        
        # 如果质心没有变化，则停止迭代
        if np.all(centroids == new_centroids):
            print(f"OK, iters={_}")
            break
        
        centroids = new_centroids
    
    # 计算费用函数值
    cost = np.sum(np.min(distances, axis=1))
    
    return centroids, labels, cost

def visualize_and_save_all_centroids(centroids_list, K, save_dir):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num_inits = len(centroids_list)
    fig, axes = plt.subplots(num_inits, K, figsize=(2 * K, 2 * num_inits))
    for init, centroids in enumerate(centroids_list):
        for i, ax in enumerate(axes[init]):
            ax.imshow(centroids[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
            if init == 0:
                ax.set_title(f"Centroid {i+1}")
    
    for init in range(num_inits):
        axes[init, 0].set_ylabel(f"Init {init+1}", rotation=0, labelpad=50, va='center')
    plt.suptitle(f"K={K}")
    plt.savefig(os.path.join(save_dir, f"K_{K}_all_inits.png"))
    plt.close(fig)

np.random.seed(1)  # 为了结果可重复
data = images

# 设置不同的 K 值
K_values = [5, 10, 20, 30, 40, 50]
num_init = 5  # 每种 K 值下的随机初始化次数

results = {}

for K in K_values:
    results[K] = {
        'costs': [],
        'times': [],
        'centroids': []
    }

    for init in range(num_init):
        start_time = time.time()
        centroids, labels, cost = k_means(data, K)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        results[K]['costs'].append(cost)
        results[K]['times'].append(elapsed_time)
        results[K]['centroids'].append(centroids)
        
        # 报告每次实验的结果
        print(f"K={K}, 初始化 {init+1}/{num_init}:")
        print(f"  费用函数值: {cost:.4f}")
        print(f"  计算时间: {elapsed_time:.4f} 秒")
        
    # 可视化并保存所有初始化的簇中心
    visualize_and_save_all_centroids(results[K]['centroids'], K, "task1")


# 打印平均结果
for K in K_values:
    avg_cost = np.mean(results[K]['costs'])
    avg_time = np.mean(results[K]['times'])
    print(f"K={K} 的平均结果:")
    print(f"  平均费用函数值: {avg_cost:.4f}")
    print(f"  平均计算时间: {avg_time:.4f} 秒")

# # 输出结果
# print("Centroids:")
# print(centroids)
# print("Labels:")
# print(labels)