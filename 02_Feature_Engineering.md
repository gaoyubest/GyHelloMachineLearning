<link rel='stylesheet' href='./style.css'>
<script src='./style.js'></script>

[TOC]


# 特征工程
- **概念**：将原始数据转换为更好地代表预测模型的潜在问题的特征的过程，从而提高对未知数据的预测准确性。
- **意义**：直接影响



## 一、特征抽取：
- **概念**：对文本等数据进行特征值化（转换为计算机理解的数字形式）

### 1、字典特征抽取
  - **概念**：对字典数据进行特征值化
  - **one hot编码**：是将类别变量转换为机器学习算法易于利用的一种形式的过程。使用one hot编码器对类别进行“二进制化”操作，然后将其作为模型训练的特征。
```python
# 字典特征抽取
from sklearn.feature_extraction import DictVectorizer
# 打印符号作为分隔符
NEXT_CHAPTER = '■'*60
print(NEXT_CHAPTER)
def extract_dict():
    """
    字典数据抽取
    :return:
    """
    # 实例化
    dv = DictVectorizer()  # 打印sparse矩阵格式
    # dv = DictVectorizer(sparse=False) # 打印ndarray格式
    print("字典特征抽取:")
    data_dict = [
        {'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}
    ]

    print("One-Hot编码")
    # 调用fit_transform
    value = dv.fit_transform(data_dict)
    names = dv.get_feature_names()
    print(value)
    print(names)
    print(NEXT_CHAPTER)
if __name__ == '__main__':
    extract_dict()
```

### 2、文本特征抽取
- **概念**：对文本数据进行特征值化。
- **内容**：
  - 统计所有文档中所有的词，重复的只看做一次，相当于词的列表。
  - 对每篇文章，在词的列表中进行统计每个词出现的次数，一一对应。注：单个字母不统计。
  - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228215036384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
  
##### （1）英文文档特征处理
```python
# 文本特征抽取
from sklearn.feature_extraction.text import CountVectorizer
def extract_text_count_alphabet():
    """
        文本特征抽取 字母频率统计
        :return:
        """
    print("文本特征抽取 频率统计")
    cv = CountVectorizer()
    data_text = ["life is short", "life is too long", "i love china"]
    value = cv.fit_transform(data_text)
    names = cv.get_feature_names()
    # print(value)
    # 将List转为数组
    print(value.toarray())
    print(names)
    print(NEXT_CHAPTER)
```

##### （2）中文文档特征处理
  - **下载jieba**：pip install jieba
  - **导入**：import jieba
```python
# 文本特征抽取
from sklearn.feature_extraction.text import CountVectorizer
# 汉字频率的统计
import jieba
def extract_text_count_hans():
    """
    文本特征抽取 汉字频率统计
    :return:
    """
    print("文本特征抽取 频率统计")
    data_text = ["中文对话特征抽取测试。", "中文对话", "特征抽取", "抽取测试"]
    # 分割字词
    data_after_text = []
    for i in data_text:
        print(i)
        # i代表循环的文章
        tmp = list(jieba.cut(i))
        data_after_text.append(' '.join(tmp))
    # end for
    data_text = data_after_text
    print(data_text)

    cv = CountVectorizer()
    value = cv.fit_transform(data_text)
    names = cv.get_feature_names()
    print(value.toarray())
    print(names)
    print(NEXT_CHAPTER)
```

##### （3）TF-IDF
   - **概念**：
     - **tf**：term frequency词的频率，指的是某一个给定的词语在该文件中出现的次数。
     - 公式：TFw = $\dfrac{在某一类中词条出现的次数}{该类中所有的词条数目}$
  &nbsp;
     - **idf**：inverse document frequency逆文档频率，
    &nbsp;
     - 公式：IDFw = $\log$$\dfrac{语料库的文档总数}{包含该词的文档数+1}$（分母+1，是为了避免分目为0）
    &nbsp;
     - log（总文档数量/该词出现的文档数量）
     - log（数值）：输入的数值越小，结果越小。
- **TF-IDF主要思想**：如果某词或短语在一篇文章中出现的频率高，并且在其他文章中很少出现，则认为此次或者短语具有较好的类别区分能力，适合用来分类。
- **TF_IDF作用**：用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
- **重要性程度公式**：TF-IDF = TF$\times$IDF
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228233256688.png)
```python
# 使用tf/idf
from sklearn.feature_extraction.text import TfidfVectorizer
def extract_text_tfidf_hans():
    """
    TF-IDF
    tf: term frequency 词频
    idf: inverse document frequency 逆文档频率
    :return:
    """
    print("文本特征抽取 tf-idf")
    data_text = ["i love china", "china is my hometown"]
    tv = TfidfVectorizer()
    value = tv.fit_transform(data_text)
    names = tv.get_feature_names()
    print(value.toarray())
    print(names)
    print(NEXT_CHAPTER)
```



## 二、特征预处理
- **概念**：对数据进行处理，通过特定的统计方法（数学方法）将数据转换成算法要求的数据。
- **特征处理的方法**：
  - 数值型数据：标准缩放：归一化、标准化、缺失值
  - 类别型数据：one-hot编码
  - 时间类型：时间的切分
### 1、归一化
- **特点**：通过对原始数据进行变换把数据映射到（默认为[0,1]）之间。
- **公式**：
  X' = $\dfrac{x - min}{max - min}$
&nbsp;
  X'' = x'$\times$(mx - min) + mi
&nbsp;
  注：公式作用于每一列，max为本列最大值，min为本列最小值，X''为最终值，mx，mi分别为指定区间值默认mx为1，mi为0。
**eg**：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302000142241.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200301235821288.png)

- **目的**：使某一特征值不会对最终结果造成巨大影响。
- **缺点**：max，min值易受异常点影响，公式计算结果影响巨大；鲁棒性较差，只适合精确小数据场景。
- **代码实现**：
```python
# 归一化
from sklearn.preprocessing import MinMaxScaler

# 打印符号作为分隔符
NEXT_CHAPTER = '■'*60
print(NEXT_CHAPTER)


def scale_minmax():
    """
    归一化处理数据
    :return: 
    """
    print("归一化（按列):")
    data = [
        [90, 2, 10, 40],
        [60, 4, 15, 45],
        [75, 3, 13, 46]
    ]
    print(data)

    # 默认feature_range=(0, 1)
    value1 = MinMaxScaler().fit_transform(data)
    print(value1)
    print()

    # 指定feature_range=(2, 3)
    value2 = MinMaxScaler(feature_range=(2, 3)).fit_transform(data)
    print(value2)

    print(NEXT_CHAPTER)


if __name__ == '__main__':
    scale_minmax()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302003307383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)


### 2、标准化
- **特点**：通过对原始数据的变换把数据变换到均值为0，标准差为1范围内。
- **公式**：X' = $\dfrac{（x1 - mean）}{σ}$
  &nbsp;
  var = $\dfrac{（x - mean）^2 + （x2 - mean）^2 + ...}{n（每个特征的样本数）}$
  &nbsp;
  $σ$ = $\sqrt{var}$
  作用于每一列，mean平均值，$σ$标准差，var方差（考量数据的稳定性）
-  **优点**：由于具有一定量的数据，少量异常点对于平均值影响不大，从而方差影响较小，在样本足够多的情况下比较稳定。
- **代码实现**：
``` python
# 标准化
from sklearn.preprocessing import StandardScaler

def scale_standard():
    """
    标准化处理数据
    """
    print("标准化（按列）：")
    std = StandardScaler()
    data = [
        [1, -1, 3],
        [2, 4, 2],
        [4, 6, -1]
    ]
    # 处理后每列数据都聚集在均值0附近，标准差为1
    value = std.fit_transform(data)
    # 每列特征的平均值
    mean = std.mean_
    # 每列特征的方差
    var = std.var_
    print(value)
    print(mean)
    print(var)
    print(NEXT_CHAPTER)
```

### 3、缺失值
- **处理方法**：
  - **删除**：若每列或行数据缺失值达到一定的比例，建立放弃整行或者整列。
  - **插补**：可以通过缺失值每行或者每列的平均、中位数来填充。
    - pandas处理数据：
```python
  data.dropna() # 对缺失的数据进行过滤
  data.fillna(0) # 用指定值或插值的方法填充缺失数据
```
```python
# 在版本0.20之前，请使用Imputer 类，版本0.22(包含0.22)之后的，则使用SimpleImputer
# from sklearn.preprocessing import Inputer
from sklearn.impute import SimpleImputer
import numpy as np
def scale_nan():
    """
    缺失值处理
    :return:
    """
    print("缺失值：")
    def scale_nan():
    """
    缺失值处理
    :return:
    """
    print("缺失值：")
    data = [[1, 2], [np.nan, 3], [7, 6]]
    """
    参数说明：
    missing_values ：指定何种占位符表示缺失值，可选 number ，string ，np.nan(default) ，None
    strategy ：插补策略，字符串，默认"mean"
        "mean" ：使用每列的平均值替换缺失值，只能与数字数据一起使用
        "median"：则使用每列的中位数替换缺失值，只能与数字数据一起使用
        "most_frequent" ：则使用每列中最常用的值替换缺失值，可以与字符串或数字数据一起使用
        "constant" ：则用 fill_value 替换缺失值。可以与字符串或数字数据一起使用
    fill_value ：字符串或数值，默认"None"，当strategy ==“constant”时，
        fill_value用于替换所有出现的missing_values。如果保留默认值，则在输入数字数据时fill_value将为0，
        对于字符串或对象数据类型则为“missing_value”。
    """
    #
    print("以1填补Nan:")
    value1 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=1).fit_transform(data)
    print(value1)
    print("以平均值填补Nan:")
    value2 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data)
    print(value2)
    print(NEXT_CHAPTER)
```
[SimpleImputer参数说明来自此链接：](https://blog.csdn.net/qq_38958113/article/details/98220246)




## 三、数据降维
- 概念：维度这里指特征的数量。
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302052343655.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1MzA1NTI0MDcz,size_16,color_FFFFFF,t_70)
### 1、特征选择 
- **概念**：单纯地提取到的所有特征中选择部分特征作为训练集特征。
- **原因**：
  - 冗余：部分特征的相关度高，容易消耗计算机性能。
  - 噪声：部分特征对预测结果有影响。
- **主要方法（三大武器）**：
  - Fillter（过滤式）VarianceThreshold
  - Embedded（嵌入式）：正则化、决策树
  - Wrapper（包裹式） 
  - 神经网络
#### Fillter（过滤式）
```python
# 数据降维
from sklearn.feature_selection import VarianceThreshold
def scale_Variance():
    print("删除低所有方差特征（按列）:")
    data = [
        [0, 2, 0, 3],
        [0, 1, 4, 3],
        [0, 1, 1, 3]
    ]
    # 删除方差为1.0的数据（默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征）
    value = VarianceThreshold(threshold=1.0).fit_transform(data)
    print(data)
    print(value)
    print(NEXT_CHAPTER)
# end def
```

### 2、主成分分析（PCA）
- **本质**：一种分析、简化数据集的技术。
- **目的**：使数据维数压缩，尽可能降低原数据的维度（复杂度），损失少量信息。
- **作用**：可以消减回归分析或者聚类分析中特征的数量。 
- 上百个数据式采用。
```python
# 主程序分析
from sklearn.decomposition import PCA
def scale_PCA():
    """
    主成分分析进行数据降维
    n_components:
            小数: 保留比例
            整数: 保留下来的特征数量
            字符串: 指定解析方法
    :return:
    """
    print("数据降维主成分分析法:")
    data = [
        [2, 8, 4, 5],
        [6, 3, 0, 8],
        [5, 4, 9, 1]
    ]
    value = PCA(n_components=0.9).fit_transform(data)
    print(value)
# end def
```