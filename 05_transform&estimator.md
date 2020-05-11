[TOC]
# 转换器&估计器：
## 转换器：
fit_transform()=fit()+StandardScaler()
fit将数据标准化，计算平均值、方差，StandardScaler按照fit计算出的标准进行转换。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200307234713486.png)
```python
"""
转换器
"""
s = StandardScaler()
data = [[1, 2, 3], [4, 5, 6]]
print("-"*70)
print("打印fit_transform值：")
print(s.fit_transform(data))
print("-"*70)

ss = StandardScaler()
print("打印fit()值：")
print(ss.fit(data))
print("-"*70)
print("打印transform()值：")
print(ss.transform(data))
print(ss.transform([[4, 5, 6], [7, 8, 9]]))
print("-"*70)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200307234234451.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

## 估计器(estimator):
估计器（estimator）是一类实现了算法的API。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200307235845914.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

1、用于分类的估计器
```python
sklearn.neighbors  # k-邻近算法
sklearn.naive_bayes # 贝叶斯
sklearn.linear_model.LogisticRegression # 逻辑回归 
sklearn.tree # 决策树与随机森林
```
2、用于回归的估计器
```python
sklearn.linear_model.LinearRegression # 线性回归
sklearn.linear_model.Ridge # 岭回归
```