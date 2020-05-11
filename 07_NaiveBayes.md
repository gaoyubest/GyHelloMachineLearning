
[TOC]
# 一、朴素贝叶斯算法原理
朴素贝叶斯算法在特征独立的前提下


**概率**：一件事情发生的可能性 
## 1、联合概率：
概念：包含多个条件，且所有条件同时成立的概率。
公式：P（A，B）= P（A）P（B）
 &nbsp;
## 2、条件概率：
概念：前提事件A、B相互独立，事件A在另一个事件B已经发生条件下的发生概率。
公式：$P(A|B)=\dfrac{P(AB)}{P(B)}$
特性：P(A1，A2|B)= P（A1|B）P（A2|B）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200308210528271.png)
 &nbsp;

## 3、朴素贝叶斯-贝叶斯公式：
**贝叶斯公式**：$P(Class|Feature) = \dfrac{P(Feature|Class) \cdot P(Class)}{P(Feature)}$

- $Class$：：文档类别（可以不同类别）。
- $feature_i$：给定文档的特征值（词频统计）。
- P(Class)每个文档类别的概率（某文档类别数/总文档数量）
- P(Feature|Class)给定类别下特征（被预测文档中出现的词）的概率
- P(F1,F2,...)预测文档中每个词的概率

**公式可以理解为**：
$P(Class|F1,F2,...) = \dfrac{P(F1,F2,...|Class) \cdot P(Class)}{P(F1,F2,...)}$
$P(F1|Class) = \dfrac{Ni}{N}$
- Ni为F1词在Class所有文档中出现的次数
- N为所属类别Class 下的文档所有词出现的次数和

**eg**：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200308215203265.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)

**拉普拉斯平滑系数**：$P(F_1|C) = \dfrac{N_i+α}{N+mα}$

- $α$：指定的系数，一般为1
- $m$：训练文档中统计出的特征词个数
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200308213514758.png)


# 二、朴素贝叶斯进行文本分析
MultinomialNB(alpha=1.0)
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def NavieBayes():
    """
    朴素贝叶斯文本分析
    :return:
    """
    news = fetch_20newsgroups(subset="all", data_home="datasets")
    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行数据抽取
    tf = TfidfVectorizer()
    # 以训练集当中的词的列表进行每篇文章重要性统计['a','b','c','d']
    x_train = tf.fit_transform(x_train)
    print((tf.get_feature_names()))
    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法预测
    mlt = MultinomialNB(alpha=1.0)
    
    print(x_train.toarray())
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    
    print("预测的文章类别是：", y_predict)
    print("准确率是：", mlt.score(x_test, y_test))


if __name__ == '__main__':
    NavieBayes()
```
# 三、优缺点
优：有稳定的分类效率；对缺失数据不太敏感，算法比较简单，常用于文本分类；分类准确度高，速度快。
缺：由于使用了样本属性独立性的假设，所以如果样本属性有关联时其效果不好。

# 混淆矩阵
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200309195359801.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)