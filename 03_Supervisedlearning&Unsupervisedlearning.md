[TOC]

# 机器学习算法
中文学习网站：https://sklearn.apachecn.org/#/
中文文档链接：https://github.com/apachecn/sklearn-doc-zh
英文学习网站：https://scikit-learn.org/stable/


# 数据类型：
 - **离散型数据**：由记录不同类别个体的数目所得到的数据，又称为计数数据，所有这些数据全部都是整数，而且不能再细分，也不能进一步提高他们的准确性。
 - **连续型数据**：变量可以在某个范围内取任一数，即变量的取值可以是连续的，如，长度、时间、质量值等，这类整数通常是非整数，含有小数部分。
 - **区别**：离散型``区间内不可分``，连续型``区间内可分``。

# 算法分类：
**按学习方式划分**：
| 学习方式  | 英文 | 输入数据 |eg|
| -------- | -----|---------|----------------- |
| 监督学习  | Supervised learning  |特征值+目标值|分类、回归|
| 非监督学习  | Unsupervised learning  |特征值|聚类|

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315222425194.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

**按学习任务划分**：
| 学习任务  | 英文 |目标值类型|包括|
| ----  | ------- |---- | ---- |
| 分类  | Classification  |离散型|k-邻近算法、贝叶斯分类、决策树与随机森林、逻辑回归、神经网|
| 回归  | Regression  |连续型|线性回归、岭回归|
| 聚类  | Clustering  |      |k-means|


# 一、监督学习（Supervised learning） [预测]
- **概念**：监督学习可以由数据中学到或建立一个模型，并依次模型推测新的结果。输入数据是由数据`特征值和目标值`组成。函数的输出可以是一个`连续的值（称回归）`或是输出是`有限个离散值（称为分类）`
- **分类Classification**（目标值离散型）：
  - k-邻近算法、贝叶斯分类、决策树与随机森林、逻辑回归、神经网络
  - 当输出变量取有限个离散值时，预测问题变成为分类问题，将数据特征分门别类。
  - 最基础的是二分类问题：即判断是非，从两个类别中选择一个作为预测结果；
  - eg：预测明天是阴晴雨，人脸识别
- **回归Regression**（目标值连续型）：线性回归、岭回归
  -eg：预测明天天气，预测房价 
- **标注**：隐马尔可夫模型（不做要求）

视频学习：https://morvanzhou.github.io/tutorials/machine-learning/sklearn/
- **基本功能**：分类，回归，聚类，数据降维，模型选择和数据预处理
![](https://img-blog.csdnimg.cn/20200302062231717.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)


# 二、非监督学习（Unsupervised learning）
- **概念**：非监督学习可以由数据中学到或建立一个模型，并依次模型推测新的结果。输入数据是由数据`特征值`组成。
- **聚类（Clustering）**：k-means


# 机器学习开发流程

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315222656963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

1、原始数据明确问题
2、数据基本处理：pandas处理数据（合并表，缺失值...）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315222704208.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
3、特征工程（特征处理）分类/回归
4、找到合适的算法进行预测
5、模型的评估，判定结果，若不合格，需换参数或算法/特征工程；若合格，上线使用，以API形式提供。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315222651179.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)



