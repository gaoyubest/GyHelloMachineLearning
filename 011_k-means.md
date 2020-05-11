[TOC]
# 一、k-means聚类
k-means理解：
- 采用迭代式算法，非监督学习 
- 做在分类之前（物以类聚，人以群分）
- 缺点：容易收敛到局部最优解（多次聚类）
- K：把数据划分程多少个类别

# 二、k-means步骤：
1、随机在数据当中抽取三个样本，当做三个类别的中心（k1，k2，k3）
2、计算其余的点分别到这三个中心点的距离，每个一个样本有三个距离（a，b，c），从中选出距离最近的一个点作为自己的标记，形成三个族群。
3、分别计算这三个族群的平均值，把三个平均值与之前的三个旧中心值进行比较，若相同，结算聚类；若不同，把这个三个平均值当作新的中心点，重复第二步。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200324040647255.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

# 三、聚类Kmeans性能评估指标
**轮廓系数**：
**公式**：
$sc_i=\dfrac{b_i - a_i]}{max(b_i,a_i)}$ 
- 对于每个点i为已聚类数据中的样本
- $b_i$为i到其它族群的所有样本的距离最小值
- $a_i$为i到本身簇的距离平均值。
- 最终计算出所有样本点的轮廓系数平均值
**计算eg**：
1、计算蓝1到自身类别点的距离的平均值$a_i$
2、计算蓝1分别到红色类别、绿色类别所有的点的距离，求出平均值b1，b2，取其中最小的值当作$b_i$
3、蓝1的轮廓系数[-1,1]
极端：$b_i$>>$a_i$  完美；$a_i$>>$b_i$:最差

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200324044206731.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
