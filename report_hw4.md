# 人工智能基础算法 第四次作业

## 1

实现了K均值聚类算法，其中K作为用户可调参数。代码进行K均值聚类的核心函数`k_means`如下

```python
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
```

函数返回质心节点向量`centroids`、质心对应的样本在数据集中的序号`labels`和费用函数值`cost`，在主函数中进行储存，之后进行输出和可视化。

## 2

实验 K=5，10，20，30，40，50六种情形下聚类结果，在每种情形下，随机选取初始化簇中心 5 次（为了让结果可重复，设置了随机数种子）。报告每次实验的K均值聚类费用函数值，计算时间，并可视化每个簇中心。

各次实验的费用函数值、计算时间如下表所示

<img src="/Users/wangyifeng/Library/Application Support/typora-user-images/image-20241028113320206.png" alt="image-20241028113320206" style="zoom:50%;" />

六种情形下费用函数和计算时间的平均值为

<img src="/Users/wangyifeng/Library/Application Support/typora-user-images/image-20241028113402912.png" alt="image-20241028113402912" style="zoom:50%;" />

各次实验得到的簇中心依次为

<img src="/Users/wangyifeng/Desktop/学习/人工智能基础算法/hw/hw4/Homework4/task1/K_5_all_inits.png" alt="K_5_all_inits" style="zoom:50%;" />

![K_10_all_inits](/Users/wangyifeng/Desktop/学习/人工智能基础算法/hw/hw4/Homework4/task1/K_10_all_inits.png)

![K_20_all_inits](/Users/wangyifeng/Desktop/学习/人工智能基础算法/hw/hw4/Homework4/task1/K_20_all_inits.png)

![K_30_all_inits](/Users/wangyifeng/Desktop/学习/人工智能基础算法/hw/hw4/Homework4/task1/K_30_all_inits.png)

![K_40_all_inits](/Users/wangyifeng/Desktop/学习/人工智能基础算法/hw/hw4/Homework4/task1/K_40_all_inits.png)

![K_50_all_inits](/Users/wangyifeng/Desktop/学习/人工智能基础算法/hw/hw4/Homework4/task1/K_50_all_inits.png)

## 3

