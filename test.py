import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 示例数据
np.random.seed(1)  # 为了结果可重复
X = np.random.rand(10, 2)

# 计算链接矩阵
Z = linkage(X, 'ward')
# Z[0,0] = 20
print(Z)
# 绘制树状图
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='lastp', p=10, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=12)
# dendrogram(Z, truncate_mode=None, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=12)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()