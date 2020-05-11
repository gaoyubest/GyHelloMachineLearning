
[TOC]
# 一、逻辑回归
**公式**：
$h_θ(x) =g(θ^Tx) = \dfrac{1}{1+e^{-θ^Tx}}$
sifmoid函数：$g(z) = \dfrac{1}{1+e^{-z}}$
- e:一般为2.71
- Z：回归结果
- 输出：[0，1]区间的概率值，默认0.5作为阈值

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200324010638183.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

**分类**：
属于分类算法，解决二分类问题，通过梯度下降求解
softmax方法解决逻辑回归在多分类问题上的推广（神经网络 ）

# 二、逻辑回归的损失函数
**损失函数**：
1、**均方误差**（不存在多个局部最低点，只有一个最小值）
2、**对数似然损失**（多个局部最小值，目前解决不了的问题）改善方法：①多次随机初始化，多次比较最小值结果②求解过程中，调整学习率
## 1、对数似然损失函数：
$cost(h_θ(x),y) = 
           \begin{cases}
            -log(h_θ(x)) \ \ \ \ \ \ \ \ \ \  if\ \ y=1  
   \\       -log(1- h_θ(x) ) \ \ \ \  if\ \ y=0
    \end{cases}$ 

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200324012656951.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020032401270512.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

## 2、完整的损失函数：
$cost(h_θ(x)) = \sum\limits_{i=1}^{m}-y_ilog(h_θ(x))-(1-y_i)log(1-h_θ(x))$
- cost损失的值越小，那么预测的类型准确率更高

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200324012805839.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

# 三、逻辑回归预测进行癌症预测API
```python

import pandas as pd
import numpy as np
from datasets import load_data_breast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error



def Logistic_Regression():
    """
    逻辑回归做二分类进行癌症预测(根据细胞的属性特征)
    :return:
    """
    # https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
    # 构造列标签名字
    column = [
        'Sample code number',
        'Clump Thickness',
        'Uniformity of Cell Size',
        'Uniformity of Cell Shape',
        'Marginal Adhesion',
        'Single Epithelial Cell Size',
        'Bare Nuclei',
        'Bland Chromatin',
        'Normal Nucleoli',
        'Mitoses',
        'Class'
    ]
    # 读取数据
    data = pd.read_csv("./datasets/breast-cancer-wisconsin.csv", names=column)
    print(data)

    # 缺失值处理
    data = data.replace(to_replace="?", value= np.nan)
    data = data.dropna()

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)
    y_predict = lg.predict(x_test)
    print(lg.coef_)
    print("K值", lg.coef_)
    print("逻辑回归癌症预测准确率：", lg.score(x_test, y_test))
    print("逻辑回归癌症预测召回率：\n", classification_report(
        y_test, y_predict,
        labels=[2, 4],
        target_names=["良性", "恶性"]
    ))
    # print("逻辑回归癌症预测召回率：", classification_report(y_test, y_predict))
    print("逻辑回归癌症预测均方误差", mean_squared_error(y_test, y_predict))


if __name__ == '__main__':
    Logistic_Regression()


```

# 四、判别模型&生成模型
**生成模型**：需要进行计算先验概率P（C）。**判别模型**：不需要计算
||判别模型|生成模型|
|----|----|------|
|eg|逻辑回归|朴素贝叶斯|
|解决问题|二分类|多分类问题|
|应用场景|二分类需要概率时|文本分类|
|参数|正则化力度|没有|
|其他|k-近邻、决策树、随机森林、神经网络|隐马尔可夫模型|