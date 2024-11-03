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

## 4

实现了分级聚类算法，并以三种簇与簇之间距离定义（Single Link，Complete Link和Average Link） 作为用户可选项。程序中进行分级聚类的核心函数`hierarchical_clustering`代码如下

```python
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
```

函数首先将每个样本单独作为一个簇，计算出各个簇之间的距离，储存在一个二维矩阵中，之后进行簇合并，每次删除原有的两个簇，并添加一个新簇，再更新距离矩阵，同时用一个伴随的数组来记录每个簇的序号，用于后续绘制树状图。函数将记录合并过程的数组`linkage_info`返回，用于绘制树状图。

## 5

在给定规模为3000的数据集上依次进行了Single Link，Complete Link和Average Link的计算，计算时间如下所示

|            | Single Link | Complete Link | Average Link |
| :--------: | :---------: | :-----------: | :----------: |
| 计算时间/s |   6220.41   |    6741.62    |   7471.46    |

三中距离定义对应的树状图（只画出从1簇到100簇）依次为

<img src="/Users/wangyifeng/Desktop/学习/人工智能基础算法/hw/hw4/Homework4/task2/Single Linkage Dendrogram (1 to 100 clusters).png" alt="Single Linkage Dendrogram (1 to 100 clusters)" style="zoom:67%;" />

<img src="/Users/wangyifeng/Desktop/学习/人工智能基础算法/hw/hw4/Homework4/task2/Complete Linkage Dendrogram (1 to 100 clusters).png" alt="Complete Linkage Dendrogram (1 to 100 clusters)" style="zoom:67%;" />

<img src="/Users/wangyifeng/Desktop/学习/人工智能基础算法/hw/hw4/Homework4/task2/Average Linkage Dendrogram (1 to 100 clusters).png" alt="Average Linkage Dendrogram (1 to 100 clusters)" style="zoom:67%;" />
