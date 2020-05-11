
[TOC]

# 一、决策树(decision tree)
- 在分类问题中，表示基于特征对实例进行分类的过程，可以认为是if-then的集合。
- 从根节点开始，对实例的某一特征进行测试，根据测试结果将实例分配到其子节点，此时每个子节点对应着该特征的一个取值，如此递归的对实例进行测试并分配，直到到达叶节点，最后将实例分到叶节点的类中。
## 银行贷款数据分析eg：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311224111311.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311224119447.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
# 二、信息论基础
## 信息熵（shang）
**信息的单位**：比特
**公式**：
$H(X) = \sum\limits_{x∈X}^{}p(x) ⋅ logP(x)$

$H = \sum\limits_{i}^{}p_i ⋅ log(p_i^{-1})$
熵即不确定性，熵越大则不确定性越大，获取信息意味着消除熵。
**eg**：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311224505144.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311224455974.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
# 三、决策树的生成
# 决策树划分依据之一：信息增益
信息增益：表示当得知一个特征条件之后，减少的信息熵的大小
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311225551546.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
## 常见决策树使用的算法：
ID3、C4.5、CAST
回归树
分类树：基尼系数（sklearn默认系数）
# 四、泰坦尼克号乘客生存分类

```python

import pandas as pd
# 字典特征抽取
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# 导出决策树结构
from sklearn.tree import export_graphviz
import pydot


def decision_tree():
    """
    决策树对泰坦尼克号进行预测生死
    :return:
    """
    # 获取数据
    # http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt
    data_all = pd.read_csv("datasets/titanic.csv")

    # 处理数据，找出特征值和目标值
    data_feature = data_all[['pclass', 'age', 'sex']]
    data_target = data_all["survived"]
    print(data_feature)

    # 缺失值处理（直接操作原数据）
    data_feature['age'].fillna(data_feature['age'].mean(), inplace=True)

    # 分割数据集（训练集+测试集）
    x_train, x_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=0.25)

    # 进行处理（特征工程）特征-类别-one_hot编码
    # pd转换为字典，特征抽取(必须为字典)
    dic_vec = DictVectorizer(sparse=False)
    x_train = dic_vec.fit_transform(x_train.to_dict(orient="records"))
    x_test = dic_vec.transform(x_test.to_dict(orient="records"))
    print(dic_vec.get_feature_names())
    print(x_train)

    # 用决策树进行预测
    decision = DecisionTreeClassifier()
    """
    决策树分类器 criteria默认是gini系数，也可以选择信息增益的熵entropy
    max_depth树的深度大小
    random_state随机数种子
    decision_path返回决策树的路径
    """
    # dec = DecisionTreeClassifier(max_depth=5)
    decision.fit(x_train, y_train)

    # 预测准确率
    print("预测的准确率为：", decision.score(x_test, y_test))

    # 导出决策树结构，导出DOT格式
    data1 = export_graphviz(
                        decision,
                        out_file=".tree.dot",
                        feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']

                        )
    # graph = pydot.graph_from_dot_data(data1.getvalue())
    # graph[0].write_pdf("tree.pdf")


if __name__ == '__main__':
    decision_tree()

```




安装graphviz:
```bash
pip install graphviz

Collecting graphviz
  Downloading graphviz-0.13.2-py2.py3-none-any.whl (17 kB)
Installing collected packages: graphviz
Successfully installed graphviz-0.13.2
```
查看安装:
```bash
pip show graphviz

Name: graphviz
Version: 0.13.2
Summary: Simple Python interface for Graphviz
Home-page: https://github.com/xflr6/graphviz
Author: Sebastian Bank
Author-email: sebastian.bank@uni-leipzig.de
License: MIT
Location: e:\python\python36\lib\site-packages
Requires:
Required-by:
```


shell>bash>Cygwin---graphviz----dot
```bash
C:\Users\Gaoyu>bash
dot -version
dot - graphviz version 2.40.1 (20161225.0304)
libdir = "/usr/lib/graphviz-2.40"
Activated plugin library: cyggvplugin_dot_layout-6.dll
Using layout: dot:dot_layout
Activated plugin library: cyggvplugin_core-6.dll
Using render: dot:core
Using device: dot:dot:core
The plugin configuration file:
        /usr/lib/graphviz-2.40/config6
                was successfully loaded.
    render      :  cairo dot dot_json fig gd json json0 lasi map mp pic pov ps svg tk vml vrml xdot xdot_json
    layout      :  circo dot fdp neato nop nop1 nop2 osage patchwork sfdp twopi
    textlayout  :  textlayout
    device      :  bmp canon cmap cmapx cmapx_np dot dot_json eps fig gd gd2 gif gtk gv ico imap imap_np ismap jpe jpeg jpg json json0 mp pdf pic plain plain-ext png pov ps ps2 svg svgz tif tiff tk vml vmlz vrml wbmp x11 xdot xdot1.2 xdot1.4 xdot_json xlib
    loadimage   :  (lib) bmp eps gd gd2 gif ico jpe jpeg jpg pdf png ps svg xbm
```
将DOT文件转为png格式：。
```bash
C:\Users\Gaoyu>cd /d G:\phython\HelloMachineLearning

G:\phython\HelloMachineLearning>bash

Gaoyu@DESKTOP-PFK4IGG /cygdrive/g/phython/HelloMachineLearning
$ dot -Tpng tree.dot -o tree.png

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200314013213454.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
# 决策树优缺点：
优：
缺：过拟合
**改进**：
## 1、减枝cart算法
```python
 min_samples_split=min_samples_split,
 min_samples_leaf=min_samples_leaf,
```
## 2、随机森林
**集成学习**：通过建立几个模型组合来解决单一预测问题。它的工作原理是生成多个分类器/模型，各自独立地学习贺作出预测。这些预测最后结合成单预测，因此由于任何一个单分类做出预测。
**随机森林**：一个包含多个决策树的分类器，并且其输出的类别是由个别树输出的分类的众数而定。
**建立多个决策树的过程**：N个样本，M个特征
单个树建立过程：1、随机在N个样本当中选择一个样本，重复N次（样本有可能重复）2、随机在M个特征当中选出m个特征，m取值。
随机放回抽样（bootstrap）
**随机抽样训练集**：避免每棵树的训练集一样，训练出来的树分类结果也是完全一样的。
**优点**：具有极好的准确率，能够有效运行在大数据集上；处理具有高维特征的输入样本，而且不需要降维，能够评估各个特征在分类问题上的重要性
```python
from datasets import load_data_titanic
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_data_titanic()

    # 随机森林进行预测（超参数调优）
    rfc = RandomForestClassifier()

    # 网格搜索与交叉验证
    gcv = GridSearchCV(rfc,
                       param_grid={
                          "n_estimators": [120, 200, 300, 500, 800, 1200],
                          "max_depth":[5, 8, 15, 25, 30]},
                       cv=2)

    gcv.fit(X_train, Y_train)
    print("【网格搜索】")
    print("\t准确率", gcv.score(X_test, Y_test))
    print("\t最优模型", gcv.best_estimator_)
    print("\t最优准确率", gcv.best_score_)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200314002027418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)