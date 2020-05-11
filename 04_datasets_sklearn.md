[TOC]

# 数据集
# 一、数据分类
**训练集（train）**：用于训练，``构建模型（模型=算法+数据）``，75%
**测试集（test）**：在模型检验时使用，用于``评估模型``是否有效，25%

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200322001942561.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

|数据类型+变量名|特征值|目标值
|---|---|---|
|训练集|x_train|x_test|
|测试集|y_train|y_test|

# 二、sklearn数据集划分API：
**安装**：
```python
pip install scikit-learn
```
**导入**：
```python
from sklearn.* import *
```

```python
# 获取小规模数据集，数据包含在datasets里
datasets.load_*()  
# 获取大规模数据集，需要从网上下载，
# 函数的第一个参数是data_home，表示数据集下载的目录，默认是~/scikit_learn_data/
datasets.fetch_*(data_home=None) 
```

## 1、sklearn分类数据集
目标值是离散型
### （1）小数据集加载&返回
```python
# 加载并返回鸢尾花数据集
from sklearn.datasets import load_iris

# 加载获取流行数据集
li = load_iris()
print("获取特征值：")
print(li.data)  # [n_samples*n_feature]二维numpy.ndarray数组
print("-"*70)

print("打印目标值：")  # n_samples的一维numpy.ndarray数组
print(li.target)
print("-"*70)

print("打印数据描述：")
print(li.DESCR)
print("-"*70)


print("打印特征名：")
# 新闻数据，手写数字、回归数据集没有
print(li.feature_names)
print("-"*70)


print("打印标签名：")
print(li.target_names)

```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200307225250825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200307225258808.png)

### （2）数据集分割：

```python
# 加载并返回鸢尾花数据集
from sklearn.datasets import load_iris
# 数据集分割
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)

print("训练集的特征值+目标值：", x_train, x_test)
print("测试集的特征值+目标值：", y_train, y_test)
```
### （3）大数据集加载&返回
```python
"""
获取新闻数据
"""
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset="all", data_home="datasets")  
# 训练集train/测试集test/训练集+测试集all
print(news.data)
print(news.target)
```

## 2、sklearn回归数据集
目标值是连续型
```python
"""
加载并返回波士顿房价数据集
"""
from sklearn.datasets import load_boston

lb = load_boston()
print("获取特征值：")
print(lb.data)  # [n_samples*n_feature]二维numpy.ndarray数组
print("-"*70)

print("打印目标值：")  # n_samples的一维numpy.ndarray数组
print(lb.target)
print("-"*70)


print("打印数据描述：")
print(lb.DESCR)
print("-"*70)

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020030722524243.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)