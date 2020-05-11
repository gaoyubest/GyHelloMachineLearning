[TOC]
# 一、线性回归概念
## 1、线性回归
**定义**：线性回归通过一个或者多个自变量（特征）与因变量（目标）之间进行建模的回归分析。其中可以为一个或者多个自变量之间的线性组合（线性回归的一种）。
寻找一种预测的趋势
**一元线性回归**：涉及到的变量只有一个
**多元线性回归**：涉及到的变量两个或者两个以上

## 2、线性关系模型：
一个通过属性的线性组合来进行预测的函数。
$f(x) = b + \sum\limits_{i=1}^{d}{w_i⋅x_i}$
w：权重，b偏置项 
## 3、画线性关系图eg：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020031522084869.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter([60, 72, 75, 80, 83],[126, 151.2, 157.5, 168, 174.3])
plt.xlabel("房子面积")
plt.ylabel("房子价格")
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315221014966.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

## 4、矩阵运算：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200315221117150.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
```python
import numpy as np
a=[[1,2,3,4],[5,6,7,8],[2,3,4,5]]
b=[2,2,2,2]
np.multiply(a,b)
```
```bash
array([[ 2,  4,  6,  8],
       [10, 12, 14, 16],
       [ 4,  6,  8, 10]])
```
```python
import numpy as np
a=[[1,2,3,4],[5,6,7,8],[2,3,4,5]]
b=[[2],[2],[2],[2]]
np.dot(a,b)
```
```bash
array([[20],
       [52],
       [28]])
```
# 二、损失函数（误差大小）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321021811273.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

## 1、总损失定义（又称最小二乘法）
 $J($θ$) = (h_w(x_1)-y_1)^2+ h_w(x_1)-y_1)^2+...+h_w(x_m)-y_m)^2
         = \sum\limits_{i=1}^{m}{(h_w(x_i)-y_i)^2}$
- $y_i$ 为第i个训练样本的真实值
- $h_w(x_i)$为第i个训练样本特征值组合预测函数
- 相当于计算误差的平方和
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321021104246.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

## 2、最小二乘法之正规方程
**特点**：特征值过于复杂时，求解速度过慢
**计算**：$w = (x^TX)^{-1}x^Ty$
- x：特征值矩阵
- y：目标值矩阵
## 3、最小二乘法之梯度下降
**特点**：训练集数据规模庞大
$w_i = -w_i - α\dfrac{∂\left[cost(w_0 + w_i⋅x_i)\right]}{∂w_i}$
- $α$：学习速率，需要手动指定
- $\dfrac{∂\left[cost(w_0 + w_i⋅x_i)\right]}{∂w_i}$：方向
- 沿着函数下降的方向找，最后在山谷的最低点，更新w值。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321024048491.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200321024103418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

## 3、岭回归（Ridge）
具有L2正则化的线性最小二乘法
- 在存在病态数据较多时具有实用价值

# 三、正规方程预测波士顿房价                                                                                                                                                                  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200322012328192.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

```python

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# 获取数据
lb = load_boston()

# 分割数据集（测试集+训练集）
x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
# print(y_train, y_test)

# 标准化处理,特征值和目标值同时进行标准化
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.transform(x_test)

std_y = StandardScaler()
y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 将数据重塑为二维
y_test = std_y.transform(y_test.reshape(-1, 1))


def linear_regression():
    """
    正规方程预测房子价格
    :return:
    """
    l_r = LinearRegression()
    l_r.fit(x_train, y_train)

    # 预测测试集房子价格
    y_lr_predict= std_y.inverse_transform(l_r.predict(x_test))
    print("正规方程测试集中每个房子的预测价格：", y_lr_predict)
    print("正规方程参数：", l_r.coef_)
    print('正规方程误差: %.2f' % mean_squared_error(y_test, y_lr_predict))
    print("-"*80)

if __name__ == '__main__':
    linear_regression()

```
# 四、梯度下降预测波士顿房价
```python
from sklearn.linear_model import SGDRegressor

def sgd_regressor():
    """
    梯度下降预测波士顿房价
    :return:
    """
    # 梯度下降预测房价
    sgd_r = SGDRegressor()
    sgd_r.fit(x_train, y_train)

    # 预测测试集房子价格
    y_sgd_predict = std_y.inverse_transform(sgd_r.predict(x_test))
    print("梯度下降测试集中每个房子的预测价格：", y_sgd_predict)
    print("梯度下降参数：", sgd_r.coef_)
    print('正规方程误差: %.2f' % mean_squared_error(y_test, y_sgd_predict))


if __name__ == '__main__':
    sgd_regressor()
```
# 五、岭回归预测房价
```python
from sklearn.linear_model import  Ridge
def Ridge1():
    """
    岭回归预测房价
    :return:
    """
    # 岭回归预测房价
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)

    # 预测测试集房子价格
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    print("岭回归测试集中每个房子的预测价格：", y_rd_predict)
    print("岭回归参数：", rd.coef_)
    print('岭回归误差: %.2f' % mean_squared_error(y_test, y_rd_predict))

if __name__ == '__main__':
    Ridge1()
```



# 六、回归性能评估
均方误差（Mean Squared Error）评价机制：
$MSE = \frac{1}{m}\sum\limits_{i=1}^{d}{(y^i - \overline{y})^2}$
- $y^i$：预测值
- $\overline{y}$：真实值

```bash
mean_squared_error(y_test, y_sgd_predict))
```
# 七、梯度下降&正规方程对比
LinearRession不能解决拟合问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200322002629608.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

# 八、拟合&过拟合
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200322194745176.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200322195127445.png)

## 1、欠拟合（underfitting）：
- **定义**：一个假设在训练数据上不能获得更好的拟合，但是训练数据外的数据集上也不能很好地拟合数据，此时认为这个假设出现了欠拟合现象。
- **原因**：学习到的数据特征过少
- **解决**：增加数据的特征数量
## 2、过拟合（overfitting）：
- **定义**：一个假设在训练数据上能够获得比其他假设更好的拟合，但是在训练数据外的数据集上却不能很好地拟合数据，此时认为这个假设出现了过拟合的现象。
- **原因**：原始特征过多，存在一些嘈杂特征，模型过于复杂是因为模型尝试去兼顾各个测试数据点
- **解决**：
  1、进行特征选择（①过滤式，低方差特征②嵌入式：正则化，决策树，神经网络），消除关联性大的特征。
  2、交叉验证：根据验证结果现象判断。
  3、正则化：L2正则化（作用：可以使得W的每个元素都很小，都接近于0.优点：越小的参数说明模型越简单，越简单的模型越容易产生过拟合现象）

  # 九、模型保存与加载

**安装joblib**
```bash
pip install joblib
Requirement already satisfied: joblib in e:\python\python36\lib\site-packages (0.14.1)
```

**查看joblib**
```bash
pip list
Package                           Version
--------------------------------- -------
joblib                            0.14.1
```

```python

from sklearn.externals import joblib
import joblib

    # 保存训练好的模型（估计器对象，保存位置）
    joblib.dump(l_r, "./datasets/test.pkl")
    # 加载模型 预测房价结果，文件以pkl为扩招名
    model = joblib.load("./datasets/test.pkl")
    y_predict = std_y.inverse_transform(model.predict(x_test))
    print("保存的模型预测的结果", y_predict)
```

