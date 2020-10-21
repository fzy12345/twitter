import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import re
import seaborn as sns
import matplotlib.pyplot as plt

# 导入数据
train = pd.read_csv('C:/Users/86186/Desktop/大三下/大数据实训/train.csv')
test = pd.read_csv('C:/Users/86186/Desktop/大三下/大数据实训/test.csv')

print('训练数据', train['text'].head())
print('测试数据', test['text'].head())

# 查看训练集中标签为0，1的推文数量
sns.countplot(y=train['target'])
plt.show()


# 写一个方法用来将数据集中杂乱的标点符号去掉
def clean_text(text):
    temp = text.lower()  # 文档转换为小写
    temp = re.sub('\n', ' ', temp)  # 删除换行符
    temp = re.sub('\'', '', temp)  # 删除引号
    temp = re.sub('-', ' ', temp)  # 删除‘-’
    temp = re.sub(r'(http|https|pic.)\S', ' ', temp)  # 删除网址
    temp = re.sub(r'[^\w\s]', ' ', temp)  # 删除可见及不可见符号

    return temp


# 去掉一般的英文停用词
def remove_stopwords(text):
    tokenized_words = word_tokenize(text)
    stop_words = stopwords.words('english')
    temp = [word for word in tokenized_words if word not in stop_words]
    temp = ' '.join(temp)
    return temp


# 清洗训练数据
train['clean'] = train['text'].apply(clean_text)
train['clean'] = train['clean'].apply(remove_stopwords)

# 清洗测试数据
test['clean'] = test['text'].apply(clean_text)
test['clean'] = test['clean'].apply(remove_stopwords)

print('清洗后的训练数据', train['clean'].head())
print('清洗后的测试数据', test['clean'].head())


# 创建一个结合属性方程，将数据内text和keyword两列合并在一起
def combine_attributes(text, keyword):
    var_list = [text, keyword]
    combined = ' '.join(x for x in var_list if x)  # 列表解析 把循环的写进一行里
    return combined


train.fillna('', inplace=True)  # 用空格填充缺失值 inplace=True总是返回被填充对象的引用
# 使用匿名函数返回函数方法，将clean和keyword合并在一起 axis=1表示沿着每一行或者列标签横向执行对应的方法（水平方向）
train['combine'] = train.apply(lambda x: combine_attributes(x['clean'], x['keyword']), axis=1)

X = train['combine']
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# 输出数据集大小
print('原始数据集特征：', X.shape,
      '训练数据集特征', X_train.shape,
      '测试数据集特征：', X_test.shape)

print('原始数据集标签：', y.shape,
      '训练数据集标签', y_train.shape,
      '测试数据集标签：', y_test.shape)

# 通过“词频-逆文本频率”（TF-IDF）处理，给文本中的单词添加权重。（词频统计）

tfidf = TfidfVectorizer()
X_train_vect = tfidf.fit_transform(X_train)
X_test_vect = tfidf.transform(X_test)

print("输出添加权重后的清洗数据训练集： ", X_train_vect.shape)
print("输出添加权重后的清洗数据测试集： ", X_test_vect.shape)

# 通过“向量虚拟机”SVC创建模型
clf = SVC(kernel='linear')
clf.fit(X_train_vect, y_train)
y_pred = clf.predict(X_test_vect)

# 多项式模型
mnb = MultinomialNB()
mnb.fit(X_train_vect, y_train)
y_mnb = mnb.predict(X_test_vect)
# 评估模型
mnbpre = accuracy_score(y_test, y_mnb)
print('多项式模型准确率:', mnbpre)

# 评估模型
pre = accuracy_score(y_test, y_pred)
print('向量机模型准确率:', pre)

# 使用向量机预测测试数据
# 向量化用于预测的数据
y_test_vect = tfidf.transform(test['clean'])
y_pred_all = clf.predict(y_test_vect)
print('预测测试数据结果:', y_pred_all)

# 使用多项式模型预测测试数据
# 向量化用于预测的数据
y_test_vect = tfidf.transform(test['clean'])
y_pred_all = mnb.predict(y_test_vect)
print('预测测试数据结果:', y_pred_all)

# 可视化结果
sns.countplot(y=y_pred_all)
plt.show()

# 将结果写到csv中
# 将data和label并在一起
y = list(zip(test['id'], y_pred_all))
name = ['id', 'target']
# 将匹配结果写入csv文件
result = pd.DataFrame(columns=name, data=y)
result.to_csv('C:/Users/86186/Desktop/大三下/大数据实训/sample_submission.csv')
print('文件写入成功')

