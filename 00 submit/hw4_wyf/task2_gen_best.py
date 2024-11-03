import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import os
import pickle

# 打开.pkl文件并加载数据
with open('./mnist_clustering_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

# 现在data变量包含了.pkl文件中的数据
data = np.array(data)

# 假设你已经有了链接矩阵
Z_average = np.load('./task2/Average Linkage Info.npy')

# 定义聚类数量
n_clusters = 10

# 从链接矩阵中提取聚类标签
labels = fcluster(Z_average, n_clusters, criterion='maxclust')

# 定义保存路径
save_dir = './task2/gen_best'
os.makedirs(save_dir, exist_ok=True)

# 计算每个聚类的平均图像
unique_labels = np.unique(labels)
average_images = []

for label in unique_labels:
    cluster_data = data[labels == label]
    average_image = np.mean(cluster_data, axis=0)
    average_images.append(average_image)

# 绘制每个聚类的平均图像
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Average Images of Each Cluster')

for ax, img, label in zip(axes.flatten(), average_images, unique_labels):
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Cluster {label}')
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(save_dir, "average_images.png"))
plt.close()

print("平均图像已保存，继续执行其他计算...")